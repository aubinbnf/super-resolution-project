import torch
import torch.nn as nn
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchvision.transforms.functional import rgb_to_ycbcr

class PSNR:
    """
    Class to correctly calculate PSNR for super-resolution
    Uses the Y channel of YCbCr according to research standards
    """
    
    def __init__(self, scale=4, data_range=1.0, device='cuda'):
        self.scale = scale
        self.data_range = data_range
        self.device = device
        self.reset()
    
    def reset(self):
        self.total_psnr = 0.0
        self.num_images = 0
    
    def _convert_rgb_to_y(self, img_rgb):
        img_ycbcr = rgb_to_ycbcr(img_rgb)
        img_y = img_ycbcr[:, 0:1, :, :]
        img_y = (img_y - 16) / (235 - 16)
        img_y = torch.clamp(img_y, 0, 1)
        return img_y
    
    def _crop_borders(self, img):
        border = self.scale
        if border > 0:
            return img[:, :, border:-border, border:-border]
        return img
    
    def _calculate_batch_psnr(self, sr, hr):
        sr_y = self._convert_rgb_to_y(sr)
        hr_y = self._convert_rgb_to_y(hr)
        sr_y = self._crop_borders(sr_y)
        hr_y = self._crop_borders(hr_y)
        
        return peak_signal_noise_ratio(sr_y, hr_y, data_range=self.data_range)
    
    def update(self, sr, hr):
        assert sr.shape == hr.shape, "SR and HR must have same shape"
        assert sr.dim() == 4, "Input must be 4D tensor [B, C, H, W]"
        
        sr = torch.clamp(sr, 0, self.data_range)
        batch_psnr = self._calculate_batch_psnr(sr, hr)
        batch_size = sr.shape[0]
        
        self.total_psnr += batch_psnr * batch_size
        self.num_images += batch_size
    
    def compute(self):
        if self.num_images == 0:
            return 0.0
        return self.total_psnr / self.num_images
    
    def get_psnr(self):
        return self.compute()
    
    def get_stats(self):
        return {
            'psnr': self.compute(),
            'num_images': self.num_images,
            'scale': self.scale,
            'data_range': self.data_range
        }