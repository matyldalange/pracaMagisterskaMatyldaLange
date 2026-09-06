import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# przekształcamy obraz do domeny częstotliwości za pomocą 2D FFT, a następnie dynamicznie ważymy błędy w zależności od odległości macierzy amplitud
class FocalFrequencyLoss(nn.Module):

    def __init__(self, alpha=1.0):
        super(FocalFrequencyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # przejście do dziedziny częstotliwości
        pred_freq = torch.fft.rfft2(pred, norm='ortho')
        target_freq = torch.fft.rfft2(target, norm='ortho')

        # obliczenie widma amplitudy
        pred_amp = torch.abs(pred_freq)
        target_amp = torch.abs(target_freq)

        # odległość kwadratowa macierzy amplitud
        matrix_dist = (pred_amp - target_amp) ** 2

        # adaptacyjna waga focal
        weight = (matrix_dist / (matrix_dist.max() + 1e-8)) ** self.alpha
        loss = weight * matrix_dist

        return torch.mean(loss)


class InpaintingLoss(nn.Module):
    def __init__(self):
        super(InpaintingLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.l1 = nn.L1Loss()
        self.fft_loss = FocalFrequencyLoss(alpha=1.0)

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

        loss_freq = self.fft_loss(gen_img, real_img)

        return (loss_l1 * 1.0) + (loss_perceptual * 0.1) + (loss_style * 100.0) + (loss_freq * 10.0)