import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers_gated import GatedConv2d


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, c, h, w = x.size()
        proj_query = self.query(x).view(batch, -1, h * w).permute(0, 2, 1)
        proj_key = self.key(x).view(batch, -1, h * w)
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch, -1, h * w)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        return self.gamma * out.view(batch, c, h, w) + x


class Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.enc1 = GatedConv2d(in_ch + 1, 64, 7, stride=1, padding=3)
        self.enc2 = GatedConv2d(64, 128, 3, stride=2, padding=1)
        self.enc3 = GatedConv2d(128, 256, 3, stride=2, padding=1)

        self.attn = SelfAttention(256)

        # blok dylatacyjny pozwalający zachować szerokie pole widzenia
        self.dilated = nn.Sequential(
            GatedConv2d(256, 256, 3, dilation=2, padding=2),
            GatedConv2d(256, 256, 3, dilation=4, padding=4),
            GatedConv2d(256, 256, 3, dilation=8, padding=8)
        )

        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec1 = GatedConv2d(128, 128, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec2 = GatedConv2d(64, 64, 3, padding=1)
        self.final = nn.Sequential(
            nn.Conv2d(64, out_ch, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x, mask):
        inp = torch.cat([x, mask], dim=1)
        x = self.enc1(inp)
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