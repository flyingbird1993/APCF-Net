"""Coordinate attention, extracted from the supplied coordatt.py."""
import torch
from torch import nn


class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv_h = nn.Conv2d(hidden, channels, 1)
        self.conv_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        height, width = x.shape[2:]
        h = self.pool_h(x)
        w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.bn1(self.conv1(torch.cat([h, w], dim=2)))
        y = y * torch.nn.functional.relu6(y + 3) / 6
        h, w = torch.split(y, [height, width], dim=2)
        w = w.permute(0, 1, 3, 2)
        return x * self.conv_h(h).sigmoid() * self.conv_w(w).sigmoid()
