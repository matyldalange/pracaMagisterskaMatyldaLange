import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers_pconv import PartialConv


class Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()

        self.coarse_conv1 = PartialConv(in_ch, 64, kernel_size=7, stride=1, padding=3)
        self.coarse_conv2 = PartialConv(64, 128, kernel_size=3, stride=2, padding=1)
        self.coarse_conv3 = PartialConv(128, 256, kernel_size=3, stride=2, padding=1)

        self.dilated_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=4, dilation=4), nn.ReLU()
        )

        self.upsample1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.upsample2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.coarse_out = nn.Conv2d(64, out_ch, 3, 1, 1)

        self.refine_conv1 = nn.Conv2d(out_ch + in_ch + 1, 64, 3, 1, 1)
        self.refine_out = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, out_ch, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x, mask, z=None):
        out, _ = self.coarse_conv1(x, mask)
        out, _ = self.coarse_conv2(out, mask)
        out, _ = self.coarse_conv3(out, mask)
        out = self.dilated_conv(out)
        out = F.relu(self.upsample1(out))
        out = F.relu(self.upsample2(out))
        coarse_result = self.coarse_out(out)

        refine_input = torch.cat([coarse_result, x, mask], dim=1)
        if z is not None:
            refine_input = refine_input + z

        final_result = self.refine_out(self.refine_conv1(refine_input))
        return coarse_result, final_result


class Discriminator(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()

        def block(in_f, out_f, stride=2):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_f, out_f, 4, stride, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            block(in_ch, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)