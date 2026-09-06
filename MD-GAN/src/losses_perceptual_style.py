import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class InpaintingLoss(nn.Module):
    def __init__(self):
        super(InpaintingLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.l1 = nn.L1Loss()

    def gram_matrix(self, x):
        (b, ch, h, w) = x.size()
        features = x.view(b, ch, w * h)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (ch * h * w)

    def forward(self, gen_img, real_img):
        loss_l1 = self.l1(gen_img, real_img)

        gen_vgg = self.vgg_layers(gen_img)
        real_vgg = self.vgg_layers(real_img)

        loss_perceptual = self.l1(gen_vgg, real_vgg)
        loss_style = self.l1(self.gram_matrix(gen_vgg), self.gram_matrix(real_vgg))

        # ważona suma składowych loss
        return (loss_l1 * 1.0) + (loss_perceptual * 0.1) + (loss_style * 100.0)