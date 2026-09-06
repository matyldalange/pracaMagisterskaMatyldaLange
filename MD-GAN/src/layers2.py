import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.mask_conv = nn.Conv2d(1, out_ch, kernel_size, stride, padding, bias=False)

        nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x, mask):
        if mask.shape[2:] != x.shape[2:]:
            mask = F.interpolate(mask, size=x.shape[2:], mode='nearest')

        output = self.conv(x * mask)

        with torch.no_grad():
            mask_output = self.mask_conv(mask)

        max_mask_val = self.mask_conv.kernel_size[0] * self.mask_conv.kernel_size[1]
        mask_ratio = max_mask_val / (mask_output + 1e-8)

        output = output * mask_ratio
        new_mask = torch.clamp(mask_output, 0, 1)

        new_mask = torch.mean(new_mask, dim=1, keepdim=True)
        new_mask = torch.clamp(new_mask, 0, 1)

        return output, new_mask