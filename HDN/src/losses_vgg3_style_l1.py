import torch
import torch.nn as nn
from torchvision import models

# funkcja straty oparta na rekonstrukcji pikselowej L1 oraz ekstrakcji cech z 3 pierwszych bloków relu VGG16 (relu1_2, relu2_2, relu3_3)
class InpaintingLossNew(nn.Module):
    def __init__(self):
        super(InpaintingLossNew, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers = nn.ModuleDict({
            'relu1_2': nn.Sequential(*list(vgg.children())[:4]),
            'relu2_2': nn.Sequential(*list(vgg.children())[4:9]),
            'relu3_3': nn.Sequential(*list(vgg.children())[9:16])
        }).eval()
        for param in self.parameters():
            param.requires_grad = False
        self.l1 = nn.L1Loss()

    # obliczanie macierzy Grama
    def gram_matrix(self, x):
        (b, ch, h, w) = x.size()
        f = x.view(b, ch, w * h)
        return torch.bmm(f, f.transpose(1, 2)) / (ch * h * w)

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