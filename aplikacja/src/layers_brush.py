#New2

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, activation=F.elu):
        super(GatedConv2d, self).__init__()
        self.activation = activation
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, stride, padding, dilation)

    def forward(self, x):
        x = self.conv(x)
        phi, gamma = torch.split(x, x.size(1) // 2, dim=1)
        gate = torch.sigmoid(gamma)
        if self.activation:
            phi = self.activation(phi)
        return phi * gate