import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

size = 192


class ConvertBHWCtoBCHW(nn.Module):
    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(0, 3, 1, 2)


class ConvertBCHWtoCBHW(nn.Module):
    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


class VideoClassificationPresetTrain_cls:
    def __init__(self):
        trans = [
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ]
        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationPresetTrain:
    def __init__(self):
        trans = [ConvertBHWCtoBCHW(),
                 transforms.ConvertImageDtype(torch.float32),
                 transforms.RandomResizedCrop((size, size), scale=(0.7, 1.0)),
                 ConvertBCHWtoCBHW()
                 ]
        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationPresetTrainEHC:
    def __init__(self):
        trans = [ConvertBHWCtoBCHW(),
                 transforms.ConvertImageDtype(torch.float32),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                 ConvertBCHWtoCBHW()
                 ]
        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationPresetEval_cls:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationPresetEval:
    def __init__(self):
        self.transforms = transforms.Compose([
            ConvertBHWCtoBCHW(),
            transforms.ConvertImageDtype(torch.float32),
            ConvertBCHWtoCBHW()
        ])

    def __call__(self, x):
        return self.transforms(x)
