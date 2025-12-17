import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This is the Denoising U-Net model definition.
It follows the standard U-Net architecture with an encoder-decoder structure and skip connections.
The model takes as input a log-magnitude spectrogram of shape [Batch, 1, Freq, Time]
and outputs an ideal binary mask (IBM) of the same shape to be applied to the noisy spectrogram.
The U-Net consists of:
- Encoder: 3 downsampling blocks, each with Conv2D -> BatchNorm -> ReLU, followed by MaxPooling.
- Bottleneck: A deeper Conv2D block to capture complex features.
- Decoder: 3 upsampling blocks, each with Conv2D -> BatchNorm -> ReLU, followed by interpolation upsampling.
- Skip Connections: Concatenation of encoder outputs to decoder inputs at each level.
- Output Layer: A final Conv2D layer followed by a Sigmoid activation to produce the mask in [0, 1].

(a) Switched from BatchNorm to GroupNorm, potentially allowing for better generalization across varying speakers.
More info: https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html

(b) Used factorised kernels to better model both time and frequency patterns in spectrograms.
"""

class DenoiseUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- Encoder (Downsampling) ---
        # We increase channels as we go deeper to capture more complex features
        self.enc1 = self._make_block(2, 16) # Input has 2 channels: real + imag
        self.enc2 = self._make_block(16, 32)
        self.enc3 = self._make_block(32, 64)
        
        # --- Bottleneck (The deepest representation) ---
        self.bottleneck = self._make_block(64, 128)
        
        # --- Decoder (Upsampling) ---
        # Input channels = current_layer + skip_connection_channels
        self.dec3 = self._make_block(128 + 64, 64) 
        self.dec2 = self._make_block(64 + 32, 32)
        self.dec1 = self._make_block(32 + 16, 16)
        
        # --- Output ---
        self.out_conv = nn.Conv2d(16, 2, kernel_size=1) # Output has 2 channels: real + imag
        
        # Pooling definition
        self.pool = nn.MaxPool2d(2, 2)

    def _make_block(self, in_channels, out_channels):
        """Helper to create a Conv -> BatchNorm -> ReLU block"""
        return nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,1), padding=(1,0)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1)),
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
        mask = self.out_conv(d1)
        
        return mask # Output shape: [B, 2, F, T] -> [Re(M), Im(M)]