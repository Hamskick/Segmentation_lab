import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
import cv2
import pandas as pd
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn.functional as F

class diceLoss(nn.Module):
    def __init__(self):
        super(diceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)  # sigmoid를 통과한 출력이면 주석처리

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_WORKERS = 4
IMAGE_HEIGHT = 224  # 1280 originally
IMAGE_WIDTH = 224  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
# TRAIN_IMG_DIR = "./cropfile/png/train"
# TRAIN_MASK_DIR = "./cropfile/png/train_labels"
# VAL_IMG_DIR = "./cropfile/png/val"
# VAL_MASK_DIR = "./cropfile/png/val_labels"


TRAIN_IMG_DIR = "C:\\Users\\admin\\Desktop\\Segmentation_lab\\real_datasets\\train"
TRAIN_MASK_DIR = "C:\\Users\\admin\\Desktop\\Segmentation_lab\\real_datasets\\train_labels"
VAL_IMG_DIR = "C:\\Users\\admin\\Desktop\\Segmentation_lab\\real_datasets\\val"
VAL_MASK_DIR = "C:\\Users\\admin\\Desktop\\Segmentation_lab\\real_datasets\\val_labels"

TEST_IMG_DIR="./test_img"

def save_checkpoint(state, filename="./my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def get_loaders2(
    test_dir,
    batch_size,
    test_transform,
    num_workers=4,
    pin_memory=True,
):
    test_ds = CarvanaDataset2(
        image_dir=test_dir,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return test_loader
def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(loader, model, folder="./saved_images2", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        for i in range(len(preds)):
            torchvision.utils.save_image(preds[i], f"{folder}/pred_{idx}_{i}.png")
            torchvision.utils.save_image(y[i].unsqueeze(0), f"{folder}/target_{idx}_{i}.png")

    model.train()

#########################################################################################################################
class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        img_name = os.path.splitext(self.images[index])[0]
        mask_name = f"{img_name}.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

class CarvanaDataset2(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image

######################################################################################################################### Hyperparameters etc.

def train_fn(loader, model, optimizer, dice_loss_fn,bce_loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = dice_loss_fn(predictions, targets) + bce_loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
def main():
    train_transform = A.Compose(
        [
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    test_transforms = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = CustomDeepLabv3(in_channels=3, out_channels=1).to(DEVICE)
    dice_loss_fn = diceLoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("./my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, dice_loss_fn,bce_loss_fn, scaler)

        # save modela
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="./saved_images2", device=DEVICE
        )

    test_loader = get_loaders2(
        TEST_IMG_DIR,
        BATCH_SIZE,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_loader):
            images = images.float().to(DEVICE)

            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.5).astype(np.uint8)

            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '':  # 예측된 건물 픽셀이 아예 없는 경우 -1
                    result.append(-1)
                else:
                    result.append(mask_rle)

    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result
    submit.to_csv('./pls22.csv', index=False)

#######################################################################################################################


class CustomDeepLabv3(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(CustomDeepLabv3, self).__init__()

        self.model = deeplabv3_resnet50(weights=None, num_classes=out_channels)

        if in_channels != 3:
            self.model.backbone.conv1 = nn.Conv2d(
                in_channels,
                self.model.backbone.conv1.out_channels,
                kernel_size=self.model.backbone.conv1.kernel_size,
                stride=self.model.backbone.conv1.stride,
                padding=self.model.backbone.conv1.padding,
                bias=False,
            )

    def forward(self, x):
        return self.model(x)["out"]

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
# class UNET(nn.Module):
#     def __init__(
#             self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
#     ):
#         super(UNET, self).__init__()
#         self.ups = nn.ModuleList()
#         self.downs = nn.ModuleList()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         # Down part of UNET
#         for feature in features:
#             self.downs.append(DoubleConv(in_channels, feature))
#             in_channels = feature
#
#         # Up part of UNET
#         for feature in reversed(features):
#             self.ups.append(
#                 nn.ConvTranspose2d(
#                     feature*2, feature, kernel_size=2, stride=2,
#                 )
#             )
#             self.ups.append(DoubleConv(feature*2, feature))
#
#         self.bottleneck = DoubleConv(features[-1], features[-1]*2)
#         self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
#
#     def forward(self, x):
#         skip_connections = []
#
#         for down in self.downs:
#             x = down(x)
#             skip_connections.append(x)
#             x = self.pool(x)
#
#         x = self.bottleneck(x)
#         skip_connections = skip_connections[::-1]
#
#         for idx in range(0, len(self.ups), 2):
#             x = self.ups[idx](x)
#             skip_connection = skip_connections[idx//2]
#
#             if x.shape != skip_connection.shape:
#                 x = TF.resize(x, size=skip_connection.shape[2:])
#
#             concat_skip = torch.cat((skip_connection, x), dim=1)
#             x = self.ups[idx+1](concat_skip)
#
#         return self.final_conv(x)

# def test():
#     x = torch.randn((3, 1, 161, 161))
#     model = UNET(in_channels=1, out_channels=1)
#     preds = model(x)
#     assert preds.shape == x.shape

if __name__ == "__main__":
    main()