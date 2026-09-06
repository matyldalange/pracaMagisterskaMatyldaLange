import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# rozbudowana funkcja straty oparta na 4 warstwach VGG16 (relu1_2, relu2_2, relu3_3, relu4_3)
class InpaintingLossNew(nn.Module):
    def __init__(self):
        super(InpaintingLossNew, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        self.layers = nn.ModuleDict({
            'relu1_2': nn.Sequential(*list(vgg.children())[:4]),
            'relu2_2': nn.Sequential(*list(vgg.children())[4:9]),
            'relu3_3': nn.Sequential(*list(vgg.children())[9:16]),
            'relu4_3': nn.Sequential(*list(vgg.children())[16:23])
        }).eval()

        for param in self.parameters():
            param.requires_grad = False

        self.l1 = nn.L1Loss()

    def gram_matrix(self, x):
        (b, ch, h, w) = x.size()
        features = x.view(b, ch, w * h)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (ch * h * w + 1e-8)

    def forward(self, gen_img, real_img):
        loss_pixel = self.l1(gen_img, real_img)

        loss_perceptual = 0
        loss_style = 0

        x_gen, x_real = gen_img, real_img
        for name, layer in self.layers.items():
            x_gen = layer(x_gen)
            x_real = layer(x_real)

            loss_perceptual += self.l1(x_gen, x_real)
            loss_style += self.l1(self.gram_matrix(x_gen), self.gram_matrix(x_real))

        return loss_pixel + 0.01 * loss_perceptual + 1.0 * loss_style