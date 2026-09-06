import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

# kombinacja straty odbudowy L1 oraz straty percepcyjnej opartej na sieci VGG19
class InpaintingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:16].eval()

        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg

    def forward(self, gen_img, real_img):
        l1_loss = self.l1(gen_img, real_img)

        vgg_gen = self.vgg(gen_img)
        vgg_real = self.vgg(real_img)
        perceptual_loss = self.l1(vgg_gen, vgg_real)

        return l1_loss + 0.05 * perceptual_loss