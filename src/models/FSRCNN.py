import torch
import torch.nn as nn

class FSRCNN(nn.Module):
    def __init__(self, scale_factor=4, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        
        # 1. Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )

        # 2. Shrinking
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )

        # 3. Mapping (m couches)
        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=1))
            mapping_layers.append(nn.PReLU(s))
        self.mapping = nn.Sequential(*mapping_layers)

        # 4. Expanding
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )

        # 5. Deconvolution (learned upsampling)
        self.deconv = nn.ConvTranspose2d(
            d, 3, kernel_size=9, stride=scale_factor, padding=4, output_padding=scale_factor-1
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrink(x)
        x = self.mapping(x)
        x = self.expand(x)
        x = self.deconv(x)
        return x
