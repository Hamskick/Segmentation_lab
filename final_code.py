
#######################################################################################################################
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
from sklearn.ensemble import VotingClassifier

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


LEARNING_RATE = 0.5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 224  # 1280 originally
IMAGE_WIDTH = 224  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "C:\\Users\\admin\\Desktop\\Segmentation_lab\\real_datasets\\train"
TRAIN_MASK_DIR = "C:\\Users\\admin\\Desktop\\Segmentation_lab\\real_datasets\\train_labels"
VAL_IMG_DIR = "C:\\Users\\admin\\Desktop\\Segmentation_lab\\real_datasets\\val"
VAL_MASK_DIR = "C:\\Users\\admin\\Desktop\\Segmentation_lab\\real_datasets\\val_labels"

TEST_IMG_DIR = "./test_img"

def save_checkpoint(state, filename="./KSJ_val2.pth.tar"):
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
class conv2DGroupNorm(nn.Module):
    def __init__(
        self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16
    ):
        super(conv2DGroupNorm, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        self.cg_unit = nn.Sequential(conv_mod, nn.GroupNorm(n_groups, int(n_filters)))

    def forward(self, inputs):
        outputs = self.cg_unit(inputs)
        return outputs
class conv2DBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNorm, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cb_unit = nn.Sequential(conv_mod, nn.BatchNorm2d(int(n_filters)))
        else:
            self.cb_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class RU(nn.Module):
    """
    Residual Unit for FRRN
    """

    def __init__(self, channels, kernel_size=3, strides=1, group_norm=False, n_groups=None):
        super(RU, self).__init__()
        self.group_norm = group_norm
        self.n_groups = n_groups

        if self.group_norm:
            self.conv1 = conv2DGroupNormRelu(
                channels,
                channels,
                k_size=kernel_size,
                stride=strides,
                padding=1,
                bias=False,
                n_groups=self.n_groups,
            )
            self.conv2 = conv2DGroupNorm(
                channels,
                channels,
                k_size=kernel_size,
                stride=strides,
                padding=1,
                bias=False,
                n_groups=self.n_groups,
            )

        else:
            self.conv1 = conv2DBatchNormRelu(
                channels, channels, k_size=kernel_size, stride=strides, padding=1, bias=False
            )
            self.conv2 = conv2DBatchNorm(
                channels, channels, k_size=kernel_size, stride=strides, padding=1, bias=False
            )

    def forward(self, x):
        incoming = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + incoming
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
            A.Rotate(always_apply=False, p=0.5, limit=(-90, 90), interpolation=4, border_mode=4, value=(0, 0, 0),
                   mask_value=None, rotate_method='largest_box', crop_border=False),
            A.OneOf([
                A.RandomFog(always_apply=False, p=0.7, fog_coef_lower=0.1, fog_coef_upper=0.5, alpha_coef=0.17),
                A.RandomBrightnessContrast(always_apply=False, p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.4, 0.2), brightness_by_max=True),
            ], p=0.7),
            #A.RandomCrop(always_apply=False, p=0.5, height=140, width=200),
            A.HorizontalFlip(always_apply=False, p=0.5),
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

    model = frrn().to(DEVICE)
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
        load_checkpoint(torch.load("./KSJ_val2.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, dice_loss_fn,bce_loss_fn, scaler)

        # save model
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
    submit.to_csv('./pls0726_1.csv', index=False)

#######################################################################################################################
class FRRU(nn.Module):
    """
    Full Resolution Residual Unit for FRRN
    """

    def __init__(self, prev_channels, out_channels, scale, group_norm=False, n_groups=None):
        super(FRRU, self).__init__()
        self.scale = scale
        self.prev_channels = prev_channels
        self.out_channels = out_channels
        self.group_norm = group_norm
        self.n_groups = n_groups

        if self.group_norm:
            conv_unit = conv2DGroupNormRelu
            self.conv1 = conv_unit(
                prev_channels + 32,
                out_channels,
                k_size=3,
                stride=1,
                padding=1,
                bias=False,
                n_groups=self.n_groups,
            )
            self.conv2 = conv_unit(
                out_channels,
                out_channels,
                k_size=3,
                stride=1,
                padding=1,
                bias=False,
                n_groups=self.n_groups,
            )

        else:
            conv_unit = conv2DBatchNormRelu
            self.conv1 = conv_unit(
                prev_channels + 32, out_channels, k_size=3, stride=1, padding=1, bias=False
            )
            self.conv2 = conv_unit(
                out_channels, out_channels, k_size=3, stride=1, padding=1, bias=False
            )

        self.conv_res = nn.Conv2d(out_channels, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, y, z):
        x = torch.cat([y, nn.MaxPool2d(self.scale, self.scale)(z)], dim=1)
        y_prime = self.conv1(x)
        y_prime = self.conv2(y_prime)

        x = self.conv_res(y_prime)
        upsample_size = torch.Size([_s * self.scale for _s in y_prime.shape[-2:]])
        x = F.upsample(x, size=upsample_size, mode="nearest")
        z_prime = z + x

        return y_prime, z_prime
class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs

class conv2DGroupNormRelu(nn.Module):
    def __init__(
        self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, n_groups=16
    ):
        super(conv2DGroupNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        self.cgr_unit = nn.Sequential(
            conv_mod, nn.GroupNorm(n_groups, int(n_filters)), nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        outputs = self.cgr_unit(inputs)
        return outputs

frrn_specs_dic = {
    "A": {
        "encoder": [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16]],
        "decoder": [[2, 192, 8], [2, 192, 4], [2, 48, 2]],
    },
    "B": {
        "encoder": [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16], [2, 384, 32]],
        "decoder": [[2, 192, 16], [2, 192, 8], [2, 192, 4], [2, 48, 2]],
    },
}
class frrn(nn.Module):
    """
    Full Resolution Residual Networks for Semantic Segmentation
    URL: https://arxiv.org/abs/1611.08323

    References:
    1) Original Author's code: https://github.com/TobyPDE/FRRN
    2) TF implementation by @kiwonjoon: https://github.com/hiwonjoon/tf-frrn
    """

    def __init__(self, n_classes=1, model_type="B", group_norm=False, n_groups=16):
        super(frrn, self).__init__()
        self.n_classes = n_classes
        self.model_type = model_type
        self.group_norm = group_norm
        self.n_groups = n_groups

        if self.group_norm:
            self.conv1 = conv2DGroupNormRelu(3, 48, 5, 1, 2)
        else:
            self.conv1 = conv2DBatchNormRelu(3, 48, 5, 1, 2)

        self.up_residual_units = []
        self.down_residual_units = []
        for i in range(3):
            self.up_residual_units.append(
                RU(
                    channels=48,
                    kernel_size=3,
                    strides=1,
                    group_norm=self.group_norm,
                    n_groups=self.n_groups,
                )
            )
            self.down_residual_units.append(
                RU(
                    channels=48,
                    kernel_size=3,
                    strides=1,
                    group_norm=self.group_norm,
                    n_groups=self.n_groups,
                )
            )

        self.up_residual_units = nn.ModuleList(self.up_residual_units)
        self.down_residual_units = nn.ModuleList(self.down_residual_units)

        self.split_conv = nn.Conv2d(48, 32, kernel_size=1, padding=0, stride=1, bias=False)

        # each spec is as (n_blocks, channels, scale)
        self.encoder_frru_specs = frrn_specs_dic[self.model_type]["encoder"]

        self.decoder_frru_specs = frrn_specs_dic[self.model_type]["decoder"]

        # encoding
        prev_channels = 48
        self.encoding_frrus = {}
        for n_blocks, channels, scale in self.encoder_frru_specs:
            for block in range(n_blocks):
                key = "_".join(map(str, ["encoding_frru", n_blocks, channels, scale, block]))
                setattr(
                    self,
                    key,
                    FRRU(
                        prev_channels=prev_channels,
                        out_channels=channels,
                        scale=scale,
                        group_norm=self.group_norm,
                        n_groups=self.n_groups,
                    ),
                )
            prev_channels = channels

        # decoding
        self.decoding_frrus = {}
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = "_".join(map(str, ["decoding_frru", n_blocks, channels, scale, block]))
                setattr(
                    self,
                    key,
                    FRRU(
                        prev_channels=prev_channels,
                        out_channels=channels,
                        scale=scale,
                        group_norm=self.group_norm,
                        n_groups=self.n_groups,
                    ),
                )
            prev_channels = channels

        self.merge_conv = nn.Conv2d(
            prev_channels + 32, 48, kernel_size=1, padding=0, stride=1, bias=False
        )

        self.classif_conv = nn.Conv2d(
            48, self.n_classes, kernel_size=1, padding=0, stride=1, bias=True
        )

    def forward(self, x):

        # pass to initial conv
        x = self.conv1(x)

        # pass through residual units
        for i in range(3):
            x = self.up_residual_units[i](x)

        # divide stream
        y = x
        z = self.split_conv(x)

        prev_channels = 48
        # encoding
        for n_blocks, channels, scale in self.encoder_frru_specs:
            # maxpool bigger feature map
            y_pooled = F.max_pool2d(y, stride=2, kernel_size=2, padding=0)
            # pass through encoding FRRUs
            for block in range(n_blocks):
                key = "_".join(map(str, ["encoding_frru", n_blocks, channels, scale, block]))
                y, z = getattr(self, key)(y_pooled, z)
            prev_channels = channels

        # decoding
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # bilinear upsample smaller feature map
            upsample_size = torch.Size([_s * 2 for _s in y.size()[-2:]])
            y_upsampled = F.upsample(y, size=upsample_size, mode="bilinear", align_corners=True)
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = "_".join(map(str, ["decoding_frru", n_blocks, channels, scale, block]))
                # print("Incoming FRRU Size: ", key, y_upsampled.shape, z.shape)
                y, z = getattr(self, key)(y_upsampled, z)
                # print("Outgoing FRRU Size: ", key, y.shape, z.shape)
            prev_channels = channels

        # merge streams
        x = torch.cat(
            [F.upsample(y, scale_factor=2, mode="bilinear", align_corners=True), z], dim=1
        )
        x = self.merge_conv(x)

        # pass through residual units
        for i in range(3):
            x = self.down_residual_units[i](x)

        # final 1x1 conv to get classification
        x = self.classif_conv(x)

        return x


if __name__ == "__main__":
    main()