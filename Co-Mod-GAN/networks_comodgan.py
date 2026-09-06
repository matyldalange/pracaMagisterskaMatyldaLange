import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# warstwa splotowa z modulacją wag za pomocą wektora stylu
class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True):
        super().__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.demodulate = demodulate

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.modulation = nn.Linear(style_dim, in_channel)

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.weight.unsqueeze(0) * style

        # demodulacja wag w celu stabilizacji wariancji cech
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=self.kernel_size // 2, groups=batch)
        return out.view(batch, self.out_channel, height, width)


class Generator(nn.Module):
    def __init__(self, size=256, style_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU()
        )
        # sieć mapująca szum na wektor stylu
        self.mapping = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim)
        )
        self.conv1 = ModulatedConv2d(256, 256, 3, style_dim)
        self.to_rgb = nn.Conv2d(256, 1, 1)

    def forward(self, x, mask, z):
        inputs = torch.cat([x, mask], dim=1)
        features = self.encoder(inputs)
        style = self.mapping(z)

        out = self.conv1(features, style)
        out = F.interpolate(out, scale_factor=4, mode='bilinear')
        return torch.tanh(self.to_rgb(out))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.conv(x)