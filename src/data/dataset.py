import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=64, pre_upscaled=True, transform=None):
        """
        hr_dir: HR image folder
        lr_dir: LR image folder (corresponding to HR)
        patch_size: Size of patches extracted for training
        pre_upscale: Should low resolution images be upscaled?
        transform: Transformations to apply to the images (optional)
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))
        self.patch_size = patch_size
        self.pre_upscaled = pre_upscaled
        self.transform = transform

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

        # Extract random patch and scale factor
        hr_patch, lr_patch, scale = self.random_crop(hr, lr, self.patch_size)

        # Upscale the LR image to HR dimension if necessary (ex: SRCNN model)
        if self.pre_upscaled:
            lr_patch = lr_patch.resize((self.patch_size * scale,
                                        self.patch_size * scale),
                                       resample=Image.BICUBIC)

        # Apply transformations
        hr_patch = self.transform(hr_patch)
        lr_patch = self.transform(lr_patch)
        return lr_patch, hr_patch

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

        return hr_patch, lr_patch, scale
