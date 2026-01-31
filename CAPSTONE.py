import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from albumentations import Compose, BboxParams, RandomResizedCrop, RandomRotate90, HorizontalFlip, VerticalFlip, ColorJitter, ShiftScaleRotate, GaussianBlur, Cutout
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import mobilenet_v2
from torch.cuda.amp import autocast, GradScaler

# Paths to the data directories
train_images_dir = '/content/drive/MyDrive/Classroom/CAPSTONE/craters/train/images'
train_labels_dir = '/content/drive/MyDrive/Classroom/CAPSTONE/craters/train/labels'
valid_images_dir = '/content/drive/MyDrive/Classroom/CAPSTONE/craters/valid/images'
valid_labels_dir = '/content/drive/MyDrive/Classroom/CAPSTONE/craters/valid/labels'

# Number of images and labels
num_train_images = len(os.listdir(train_images_dir))
num_train_labels = len(os.listdir(train_labels_dir))
num_valid_images = len(os.listdir(valid_images_dir))
num_valid_labels = len(os.listdir(valid_labels_dir))

print(f'Number of training images: {num_train_images}')
print(f'Number of training labels: {num_train_labels}')
print(f'Number of validation images: {num_valid_images}')
print(f'Number of validation labels: {num_valid_labels}')

# Define the CraterDataset class
class CraterDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.annots = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        annot_path = os.path.join(self.root, "labels", self.annots[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if os.path.getsize(annot_path) != 0:
            bboxs = np.loadtxt(annot_path, ndmin=2)
            bboxs = self.convert_box_cord(bboxs, 'normxywh', 'xyminmax', img.shape)
            num_objs = len(bboxs)
            bboxs = torch.as_tensor(bboxs, dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        else:
            bboxs = torch.as_tensor([[0, 0, 640, 640]], dtype=torch.float32)
            labels = torch.zeros((1,), dtype=torch.int64)
            iscrowd = torch.zeros((1,), dtype=torch.int64)

        area = (bboxs[:, 3] - bboxs[:, 1]) * (bboxs[:, 2] - bboxs[:, 0])
        image_id = torch.tensor([idx])

        target = {"boxes": bboxs, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        if self.transforms:
            sample = self.transforms(image=img, bboxes=target['boxes'], labels=labels)
            img = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'])
            target['labels'] = torch.tensor(sample['labels'])

        return img, target

    def __len__(self):
        return len(self.imgs)

    def convert_box_cord(self, bboxs, format_from, format_to, img_shape):
        if format_from == 'normxywh':
            if format_to == 'xyminmax':
                xw = bboxs[:, (1, 3)] * img_shape[1]
                yh = bboxs[:, (2, 4)] * img_shape[0]
                xmin = xw[:, 0] - xw[:, 1] / 2
                xmax = xw[:, 0] + xw[:, 1] / 2
                ymin = yh[:, 0] - yh[:, 1] / 2
                ymax = yh[:, 0] + yh[:, 1] / 2
                coords_converted = np.column_stack((xmin, ymin, xmax, ymax))
        return coords_converted

# Define transformations with more augmentations and resized images
train_transforms = Compose([
    RandomResizedCrop(height=256, width=256, p=1.0),  # Resize images
    RandomRotate90(p=0.5),
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    ColorJitter(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    GaussianBlur(p=0.5),
    Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.5),
    ToTensorV2(p=1.0)
], bbox_params=BboxParams(format='pascal_voc', label_fields=['labels']))

valid_transforms = Compose([
    RandomResizedCrop(height=256, width=256, p=1.0),
    ToTensorV2(p=1.0)
], bbox_params=BboxParams(format='pascal_voc', label_fields=['labels']))

# Dataset and DataLoader
train_dataset = CraterDataset(root='/content/drive/MyDrive/Classroom/CAPSTONE/craters/train', transforms=train_transforms)
valid_dataset = CraterDataset(root='/content/drive/MyDrive/Classroom/CAPSTONE/craters/valid', transforms=valid_transforms)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Define the MobileNetV2 backbone for object detection
backbone = mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280

# Generate anchors using RPN (Region Proposal Network)
rpn_anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

# Define the RoI (Region of Interest) pooling
roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'], output_size=7, sampling_ratio=2
)

# Create the Faster R-CNN model
model = FasterRCNN(
    backbone,
    num_classes=2,  # 1 class (crater) + background
    rpn_anchor_generator=rpn_anchor_generator,
    box_roi_pool=roi_pooler
)

# Move the model to the right device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define the optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Mixed precision training
scaler = GradScaler()

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += losses.item()

    epoch_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch} - Loss: {epoch_loss}")
    return epoch_loss

# Evaluation function with accuracy calculation
def evaluate(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    correct_detections = 0
    total_images = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]

            accuracy = calculate_accuracy(outputs, targets, iou_threshold)
            correct_detections += accuracy * len(images)
            total_images += len(images)

    accuracy = correct_detections / total_images

    print(f"Accuracy: {accuracy}")

    return accuracy

# Training and evaluation loop
num_epochs = 10
for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
    accuracy = evaluate(model, valid_loader, device)
    lr_scheduler.step(train_loss)