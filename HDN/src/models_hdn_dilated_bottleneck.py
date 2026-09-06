import torch
import torch.nn as nn
import torch.nn.functional as F


# blok rezydualny z możliwością określenia stopnia rozszerzenia
class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(ch)
        )

    def forward(self, x):
        return x + self.block(x)


def conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True)
    )


# Generator HDN wykorzystujący enkoder-dekoder z skip connections oraz bottleneck opartym na dilated conv
class HDNGenerator(nn.Module):
    def __init__(self):
        super(HDNGenerator, self).__init__()

        self.enc1 = nn.Sequential(nn.Conv2d(4, 64, 3, stride=1, padding=1), nn.LeakyReLU(0.2))
        self.enc2 = conv_block(64, 128, stride=2)
        self.enc3 = conv_block(128, 256, stride=2)
        self.enc4 = conv_block(256, 512, stride=2)

        # bottleneck z rosnącym współczynnikiem dilation dla szerokiego kontekstu
        self.bottleneck = nn.Sequential(
            ResBlock(512, dilation=2),
            ResBlock(512, dilation=4),
            ResBlock(512, dilation=1)
        )

        self.up3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec1 = conv_block(128, 64)

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

        b = self.bottleneck(e4)

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.fine_head(d1)
        return out, out


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