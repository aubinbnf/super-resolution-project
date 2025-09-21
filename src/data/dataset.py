import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=64, scale_factor=4, 
                 mode="srcnn", transform=None):
        """
        hr_dir: HR image folder
        lr_dir: LR image folder (corresponding to HR)
        patch_size: Size of LR patches extracted for training
        scale_factor: upscaling factor (2, 3, 4, ...)
        mode: "srcnn" (input = bicubic-upscaled LR) or "fsrcnn" (input = LR raw)
        transform: optional torchvision transforms
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.mode = mode

        assert len(self.hr_images) == len(self.lr_images), \
            "Different number of HR and LR images"

        # Default transform
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        hr_patch, lr_patch = self.random_crop(hr, lr, self.patch_size)

        if self.mode == "srcnn":
            # Upscale LR patch to HR size with bicubic
            lr_patch = lr_patch.resize(
                (self.patch_size * self.scale_factor,
                 self.patch_size * self.scale_factor),
                resample=Image.BICUBIC
            )
            # Retourne (lr_up, hr)
            lr_patch = self.transform(lr_patch)
            hr_patch = self.transform(hr_patch)
            return lr_patch, hr_patch

        elif self.mode == "fsrcnn":
            # Retourne directement (lr, hr)
            lr_patch = self.transform(lr_patch)
            hr_patch = self.transform(hr_patch)
            return lr_patch, hr_patch

        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def random_crop(self, hr, lr, patch_size):
        """Extract a patch LR of size patch_size, and the corresponding HR patch"""
        hr_w, hr_h = hr.size
        lr_w, lr_h = lr.size

        scale_w = hr_w // lr_w
        scale_h = hr_h // lr_h
        assert scale_w == scale_h == self.scale_factor, \
            f"Expected scale {self.scale_factor}, got {scale_w}x{scale_h}"

        # Random coords in LR
        lr_x = random.randint(0, lr_w - patch_size)
        lr_y = random.randint(0, lr_h - patch_size)

        # HR coords aligned
        hr_x = lr_x * self.scale_factor
        hr_y = lr_y * self.scale_factor

        lr_patch = lr.crop((lr_x, lr_y, lr_x + patch_size, lr_y + patch_size))
        hr_patch = hr.crop((hr_x, hr_y,
                            hr_x + patch_size*self.scale_factor,
                            hr_y + patch_size*self.scale_factor))

        return hr_patch, lr_patch
