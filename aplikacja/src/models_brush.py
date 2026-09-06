#New3

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers_brush import GatedConv2d


class OptimizedSelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = (in_dim // 8) ** 0.5

    def forward(self, x):
        batch, c, h, w = x.size()
        proj_query = self.query(x).view(batch, -1, h * w).permute(0, 2, 1)
        proj_key = self.key(x).view(batch, -1, h * w)

        energy = torch.bmm(proj_query, proj_key) / self.scale
        attention = torch.softmax(energy, dim=-1)

        proj_value = self.value(x).view(batch, -1, h * w)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        return self.gamma * out.view(batch, c, h, w) + x


class Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()

        def gated_block(in_f, out_f, kernel=3, stride=1, pad=1, dilation=1):
            return GatedConv2d(in_f, out_f, kernel, stride, pad, dilation)

        self.enc1 = gated_block(in_ch + 1, 64, 7, 1, 3)
        self.enc2 = gated_block(64, 128, 3, 2, 1)
        self.enc3 = gated_block(128, 256, 3, 2, 1)

        self.attn = OptimizedSelfAttention(256)

        self.dilated = nn.Sequential(
            gated_block(256, 256, 3, 1, 2, dilation=2),
            gated_block(256, 256, 3, 1, 4, dilation=4),
            gated_block(256, 256, 3, 1, 8, dilation=8),
            gated_block(256, 256, 3, 1, 16, dilation=16)
        )

        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec1 = gated_block(128, 128, 3, 1, 1)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec2 = gated_block(64, 64, 3, 1, 1)

        self.final = nn.Sequential(
            nn.Conv2d(64, out_ch, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x, mask):
        x = torch.cat([x, mask], dim=1)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.attn(x)
        x = self.dilated(x)
        x = F.relu(self.up1(x))
        x = self.dec1(x)
        x = F.relu(self.up2(x))
        x = self.dec2(x)
        return self.final(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def create_d():
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
                nn.LeakyReLU(0.2, True),
                nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
                nn.LeakyReLU(0.2, True),
                nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(256, 1, 4, 1, 1)
            )

        self.d1 = create_d()
        self.d2 = create_d()

    def forward(self, x):
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        return self.d1(x), self.d2(x2)