import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True)
    )


class HDNGenerator(nn.Module):
    def __init__(self):
        super(HDNGenerator, self).__init__()

        self.enc1 = conv_block(4, 64, stride=1)
        self.enc2 = conv_block(64, 128, stride=2)
        self.enc3 = conv_block(128, 256, stride=2)
        self.enc4 = conv_block(256, 512, stride=2)

        self.up3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.dec3 = conv_block(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = conv_block(128 + 128, 128)

        self.coarse_head = nn.Conv2d(128, 3, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec1 = conv_block(64 + 64, 64)

        self.fine_head = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x, mask):
        xin = torch.cat([x, mask], dim=1)

        e1 = self.enc1(xin)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))

        coarse_out = self.coarse_head(d2)
        coarse_out_resized = F.interpolate(coarse_out, size=x.shape[2:], mode='bilinear')

        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        fine_out = self.fine_head(d1)

        return coarse_out_resized, fine_out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def disc_block(in_ch, out_ch, stride=2):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            disc_block(3, 64),
            disc_block(64, 128),
            disc_block(128, 256),
            disc_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)