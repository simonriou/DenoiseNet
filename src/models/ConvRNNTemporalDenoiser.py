import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvRNNTemporalDenoiser(nn.Module):
    def __init__(self, input_channels=1, conv_channels=64, rnn_hidden=256, rnn_layers=2, bidirectional=True):
        super().__init__()
        
        # --- Conv1D frontend ---
        # Extract local temporal features and downsample sequence
        self.conv1 = nn.Conv1d(input_channels, conv_channels, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(conv_channels, conv_channels, kernel_size=5, stride=1, padding=2)
        
        # --- RNN for temporal modeling ---
        self.rnn = nn.LSTM(
            input_size=conv_channels,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # --- Linear output to waveform ---
        self.fc = nn.Linear(rnn_hidden * (2 if bidirectional else 1), 1)

    def forward(self, x):
        """
        x: [B, T, 1] waveform input
        returns: [B, T, 1] predicted clean waveform
        """
        # Conv1D expects [B, C, T]
        x = x.transpose(1, 2)           # [B, 1, T]
        x = F.relu(self.conv1(x))       # [B, C, T/2]
        x = F.relu(self.conv2(x))       # [B, C, T/4]
        x = F.relu(self.conv3(x))       # [B, C, T/4]
        
        # RNN expects [B, T, C]
        x = x.transpose(1, 2)           # [B, T/4, C]
        out, _ = self.rnn(x)            # [B, T/4, H*2]
        out = self.fc(out)               # [B, T/4, 1]
        
        # Upsample to original length
        out = F.interpolate(out.transpose(1, 2), size=x.shape[1]*4, mode='linear', align_corners=False)
        out = out.transpose(1, 2)       # [B, T, 1]
        return out