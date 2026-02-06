import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Complex-valued U-Net (DCU-Net) implemented with real-valued ops.
Input and output use stacked real/imag channels:
- Complex channel count C is represented as 2*C real channels.
"""


def _group_norm_groups(num_channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


class ComplexConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
    ):
        super().__init__()
        self.real = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.imag = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real, x_imag = torch.chunk(x, 2, dim=1)
        y_real = self.real(x_real) - self.imag(x_imag)
        y_imag = self.real(x_imag) + self.imag(x_real)
        return torch.cat([y_real, y_imag], dim=1)


class ComplexConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        out_real_channels = 2 * out_channels
        groups = _group_norm_groups(out_real_channels)

        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, out_real_channels)
        self.act1 = nn.PReLU(out_real_channels)

        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, out_real_channels)
        self.act2 = nn.PReLU(out_real_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.gn1(self.conv1(x)))
        x = self.act2(self.gn2(self.conv2(x)))
        return x


class DCUNet(nn.Module):
    def __init__(self, base_channels: int = 16):
        super().__init__()

        self.enc1 = ComplexConvBlock(1, base_channels)
        self.enc2 = ComplexConvBlock(base_channels, base_channels * 2)
        self.enc3 = ComplexConvBlock(base_channels * 2, base_channels * 4)

        self.bottleneck = ComplexConvBlock(base_channels * 4, base_channels * 8)

        self.dec3 = ComplexConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = ComplexConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ComplexConvBlock(base_channels * 2 + base_channels, base_channels)

        self.out_conv = ComplexConv2d(base_channels, 1, kernel_size=1, padding=0)
        self.out_act = nn.Tanh()

        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        b = self.bottleneck(p3)

        d3 = F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        mask = self.out_act(self.out_conv(d1))
        return mask
