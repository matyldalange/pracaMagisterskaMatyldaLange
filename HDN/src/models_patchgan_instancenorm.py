import torch
import torch.nn as nn
import torch.nn.functional as F

# Generator pozostaje bez zmian (HDN), bo działa świetnie
from src.models_hdn import HDNGenerator

class PatchGANDiscriminatorNew(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminatorNew, self).__init__()

        def disc_block(in_ch, out_ch, stride=2, use_norm=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1))]
            if use_norm:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            disc_block(3, 64, use_norm=False),
            disc_block(64, 128),
            disc_block(128, 256),
            disc_block(256, 512, stride=1),
            nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, padding=1))
        )

    def forward(self, x):
        return self.model(x)