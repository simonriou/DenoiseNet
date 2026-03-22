import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.constants import N_FFT


def _group_norm_groups(num_channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


def _downsampled_size(size: int, levels: int) -> int:
    for _ in range(levels):
        size //= 2
    return size


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_bottleneck: bool = False):
        super().__init__()
        dilation = (1, 1)
        pad = (0, 1)

        if is_bottleneck:
            dilation = (1, 2)
            pad = (0, 2)

        groups = _group_norm_groups(out_channels)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, 3),
                padding=pad,
                dilation=dilation,
            ),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion_factor: int = 2,
        kernel_size: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = (kernel_size - 1) // 2

        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_in = nn.Conv1d(dim, inner_dim * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=inner_dim,
        )
        self.batch_norm = nn.BatchNorm1d(inner_dim)
        self.activation = nn.SiLU()
        self.pointwise_out = nn.Conv1d(inner_dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = F.glu(self.pointwise_in(x), dim=1)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_out(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, ff_expansion_factor, dropout)
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(
            dim=dim,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )
        self.ff2 = FeedForwardModule(dim, ff_expansion_factor, dropout)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)

        attn_input = self.self_attn_norm(x)
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.self_attn_dropout(attn_output)

        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class BottleneckConformer(nn.Module):
    def __init__(
        self,
        channels: int,
        freq_bins: int,
        model_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.freq_bins = freq_bins
        flattened_dim = channels * freq_bins

        self.input_norm = nn.LayerNorm(flattened_dim)
        self.input_projection = nn.Linear(flattened_dim, model_dim)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    dim=model_dim,
                    num_heads=num_heads,
                    ff_expansion_factor=ff_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, flattened_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, freq_bins, time_steps = x.shape
        if channels != self.channels or freq_bins != self.freq_bins:
            raise ValueError(
                f"Expected bottleneck shape [B, {self.channels}, {self.freq_bins}, T], "
                f"got [B, {channels}, {freq_bins}, {time_steps}]."
            )

        residual = x
        sequence = x.permute(0, 3, 1, 2).reshape(batch_size, time_steps, channels * freq_bins)
        sequence = self.input_projection(self.input_norm(sequence))

        for layer in self.layers:
            sequence = layer(sequence)

        sequence = self.output_projection(self.output_norm(sequence))
        sequence = sequence.view(batch_size, time_steps, channels, freq_bins)
        sequence = sequence.permute(0, 2, 3, 1).contiguous()

        return residual + sequence


class DenoiseUNetConformer(nn.Module):
    """
    DenoiseUNet-style encoder/decoder with a bottleneck Conformer operating over the
    compressed time sequence. Input/output channels follow the project's real/imag
    stacked complex-spectrogram convention: [B, 2, F, T].
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        freq_bins: int = (N_FFT // 2) + 1,
        conformer_dim: int = 256,
        conformer_layers: int = 2,
        conformer_heads: int = 4,
        conformer_kernel_size: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_channels = (16, 32, 64)
        bottleneck_channels = 128
        pooling_levels = len(encoder_channels)
        bottleneck_freq_bins = _downsampled_size(freq_bins, pooling_levels)

        self.enc1 = ConvBlock(in_channels, encoder_channels[0])
        self.enc2 = ConvBlock(encoder_channels[0], encoder_channels[1])
        self.enc3 = ConvBlock(encoder_channels[1], encoder_channels[2])

        self.bottleneck = ConvBlock(encoder_channels[2], bottleneck_channels, is_bottleneck=True)
        self.bottleneck_conformer = BottleneckConformer(
            channels=bottleneck_channels,
            freq_bins=bottleneck_freq_bins,
            model_dim=conformer_dim,
            num_layers=conformer_layers,
            num_heads=conformer_heads,
            conv_kernel_size=conformer_kernel_size,
            dropout=dropout,
        )

        self.dec3 = ConvBlock(bottleneck_channels + encoder_channels[2], encoder_channels[2])
        self.dec2 = ConvBlock(encoder_channels[2] + encoder_channels[1], encoder_channels[1])
        self.dec1 = ConvBlock(encoder_channels[1] + encoder_channels[0], encoder_channels[0])

        self.out_conv = nn.Conv2d(encoder_channels[0], out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        b = self.bottleneck(p3)
        b = self.bottleneck_conformer(b)

        d3 = F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)
