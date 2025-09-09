import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=64, transform=None):
        """
        hr_dir: HR image folder
        lr_dir: LR image folder (corresponding to HR)
        patch_size: Size of patches extracted for training
        transform: Transformations to apply to the images (optional)
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = sorted(os.listdir(hr_dir))
        self.lr_images = sorted(os.listdir(lr_dir))
        self.patch_size = patch_size
        self.transform = transform

        assert len(self.hr_images) == len(self.lr_images), "Different number of HR and LR images"

        # Default transformations if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()  # convert [0,255] -> [0,1] et HWC -> CHW
            ])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        # Launch images
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        # Extract random patch
        hr_patch, lr_patch = self.random_crop(hr, lr, self.patch_size)

        # Apply transformations
        hr_patch = self.transform(hr_patch)
        lr_patch = self.transform(lr_patch)

        return lr_patch, hr_patch

    def random_crop(self, hr, lr, patch_size):
        """Extracts a random patch of size patch_size x patch_size"""
        hr_w, hr_h = hr.size
        lr_w, lr_h = lr.size

        # Calculating the upscaling factor
        scale_w = hr_w // lr_w
        scale_h = hr_h // lr_h
        assert scale_w == scale_h, "Different upscaling factors in width and height"
        scale = scale_w

        # Random choice from the top left corner of the LR patch
        lr_x = random.randint(0, lr_w - patch_size)
        lr_y = random.randint(0, lr_h - patch_size)

        # HR corresponding patch
        hr_x = lr_x * scale
        hr_y = lr_y * scale

        lr_patch = lr.crop((lr_x, lr_y, lr_x + patch_size, lr_y + patch_size))
        hr_patch = hr.crop((hr_x, hr_y, hr_x + patch_size*scale, hr_y + patch_size*scale))

        return hr_patch, lr_patch
