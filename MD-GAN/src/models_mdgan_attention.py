import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers_pconv import PartialConv

# warstwa Self-Attention uchwytująca relacje przestrzenne dalekiego zasięgu
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

# ulepszony generator MD-GAN zawierający moduł Self-Attention w etapie Coarse
class Generator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        # Etap 1: Coarse
        self.coarse_conv1 = PartialConv(in_ch, 64, kernel_size=7, stride=1, padding=3)
        self.coarse_conv2 = PartialConv(64, 128, kernel_size=3, stride=2, padding=1)
        self.coarse_conv3 = PartialConv(128, 256, kernel_size=3, stride=2, padding=1)

        self.attn = SelfAttention(256)

        self.dilated_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=4, dilation=4), nn.ReLU()
        )

        self.upsample1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.upsample2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.coarse_out = nn.Conv2d(64, out_ch, 3, 1, 1)

        self.refine_conv1 = nn.Conv2d(out_ch + in_ch + 1, 64, 3, 1, 1)
        self.refine_out = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, out_ch, 3, 1, 1), nn.Tanh()
        )

    def forward(self, x, mask, z=None):
        out, _ = self.coarse_conv1(x, mask)
        out, _ = self.coarse_conv2(out, mask)
        out, m = self.coarse_conv3(out, mask)

        out = self.attn(out)
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