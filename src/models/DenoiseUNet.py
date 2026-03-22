import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This is the DenoiseNet model definition.
It follows the standard U-Net architecture with an encoder-decoder structure and skip connections.
The model takes as input a normalized complex spectrogram of shape [Batch, 2, Freq, Time]
and predicts a denoised normalized complex spectrogram of the same shape for direct
spectral mapping.
The U-Net consists of:
- Encoder: 3 downsampling blocks, each with Conv2D -> GroupNorm -> ReLU, followed by MaxPooling.
- Bottleneck: A deeper Conv2D block to capture complex features.
- Decoder: 3 upsampling blocks, each with Conv2D -> GroupNorm -> ReLU, followed by interpolation upsampling.
- Skip Connections: Concatenation of encoder outputs to decoder inputs at each level.
- Output Layer: A final Conv2D layer that regresses the denoised complex spectrogram.

(a) Switched from BatchNorm to GroupNorm, potentially allowing for better generalization across varying speakers.
More info: https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html

(b) Used factorised kernels to better model both time and frequency patterns in spectrograms.
"""

class DenoiseUNet(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 2):
        super().__init__()
        
        # --- Encoder (Downsampling) ---
        # We increase channels as we go deeper to capture more complex features
        self.enc1 = self._make_block(in_channels, 16)
        self.enc2 = self._make_block(16, 32)
        self.enc3 = self._make_block(32, 64)
        
        # --- Bottleneck (The deepest representation) ---
        self.bottleneck = self._make_block(64, 128, True)
        
        # --- Decoder (Upsampling) ---
        # Input channels = current_layer + skip_connection_channels
        self.dec3 = self._make_block(128 + 64, 64) 
        self.dec2 = self._make_block(64 + 32, 32)
        self.dec1 = self._make_block(32 + 16, 16)
        
        # --- Output ---
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        
        # Pooling definition
        self.pool = nn.MaxPool2d(2, 2)

    def _make_block(self, in_channels, out_channels, is_bottleneck=False):
        """Helper to create a Conv -> BatchNorm -> ReLU block"""
        dilation = (1, 1)
        pad = (0, 1)
        if is_bottleneck:
            dilation = (1, 2)  # Wider receptive field in time for bottleneck
            pad = (0, 2) # Matching padding for dilation
        return nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,1), padding=(1,0)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), padding=pad, dilation=dilation),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            # Adding a second conv layer per block increases learning capacity per level.
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1)),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # --- Encoder Path ---
        # Save output of each block (e1, e2, e3) for skip connections
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        # --- Bottleneck ---
        b = self.bottleneck(p3)
        
        # --- Decoder Path ---
        
        # Block 3 Upsampling
        # We use interpolation instead of TransposeConv to avoid checkerboard artifacts
        d3 = F.interpolate(b, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1) # Skip Connection: Concatenate
        d3 = self.dec3(d3)
        
        # Block 2 Upsampling
        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1) # Skip Connection
        d2 = self.dec2(d2)
        
        # Block 1 Upsampling
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1) # Skip Connection
        d1 = self.dec1(d1)
        
        # --- Output ---
        spectrogram = self.out_conv(d1)
        
        return spectrogram
