import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# rozbudowana funkcja straty zawierająca cztery składowe (L1, Perceptual Loss, Style Loss, Edge Loss)
class InpaintingLoss(nn.Module):
    def __init__(self):
        super(InpaintingLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.l1 = nn.L1Loss()

    # obliczenie macierzy Grama
    def gram_matrix(self, x):
        (b, ch, h, w) = x.size()
        features = x.view(b, ch, w * h)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (ch * h * w)

    def get_edges(self, x):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(x.device)

        ch = x.shape[1]
        grad_x = F.conv2d(x, sobel_x.repeat(ch, 1, 1, 1), groups=ch, padding=1)
        grad_y = F.conv2d(x, sobel_y.repeat(ch, 1, 1, 1), groups=ch, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def forward(self, gen_img, real_img):
        loss_l1 = self.l1(gen_img, real_img)

        gen_vgg = self.vgg_layers(gen_img)
        real_vgg = self.vgg_layers(real_img)

        loss_perceptual = self.l1(gen_vgg, real_vgg)
        loss_style = self.l1(self.gram_matrix(gen_vgg), self.gram_matrix(real_vgg))

        loss_edge = self.l1(self.get_edges(gen_img), self.get_edges(real_img))

        # suma ważona komponentów funkcji straty
        return loss_l1 + 0.05 * loss_perceptual + 120 * loss_style + 0.1 * loss_edge