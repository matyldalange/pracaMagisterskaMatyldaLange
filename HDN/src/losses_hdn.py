import torch
import torch.nn as nn
from torchvision import models

# # funkcja straty rekonstrukcji łącząca bezwzględny błąd pikselowy (L1) oraz stratę percepcyjną (Perceptual Loss) opartą na cechach sieci VGG16
class InpaintingLoss(nn.Module):
    def __init__(self):
        super(InpaintingLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

        # ekstrakcja pierwszych 16 warstw VGG16 z zamrożonymi wagami
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.l1 = nn.L1Loss()

    def forward(self, gen_img, real_img):
        loss_l1 = self.l1(gen_img, real_img)

        # porównanie wysokopoziomowych cech percepcyjnych z modelu VGG
        gen_vgg = self.vgg_layers(gen_img)
        real_vgg = self.vgg_layers(real_img)
        loss_perceptual = self.l1(gen_vgg, real_vgg)

        return loss_l1 + 0.05 * loss_perceptual