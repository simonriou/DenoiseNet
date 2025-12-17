import os
import glob
import random
import torch
from torch.utils.data import Dataset

from utils.save_wav import save_wav
from utils.constants import *

if DEBUG:
    import matplotlib.pyplot as plt

"""
This Dataset class loads clean speech files and noise files from specified directories.
It mixes them at a specified SNR to create noisy mixtures.
It computes log-magnitude spectrogram features and ideal binary masks (IBM) for training a denoising model.
The __getitem__ method returns a tuple (features, ibm) where:
- features: Tensor of shape [1, Freq, Time] representing the normalized log-magnitude spectrogram of the noisy mixture.
- ibm: Tensor of shape [1, Freq, Time] representing the ideal binary mask (1 if clean > noise, else 0).

Important notice: Audio files are stored as int16 .pt tensors to save space and loading time.
This is why we convert them to float32 in the __getitem__ method.

The DEBUG flag enables saving intermediate audio files and printing debug information for verification (triggers at each __getitem__ call).
"""

class SpeechNoiseDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, snr_db=5.0, mode='train'):
        self.clean_files = glob.glob(os.path.join(clean_dir, '*.pt'))
        self.noise_files = glob.glob(os.path.join(noise_dir, '*.pt'))
        self.snr_db = snr_db
        self.mode = mode
        
        # Pre-load noise files to memory to speed up training (optional, good for small noise sets)
        self.noises = []
        for nf in self.noise_files:
            try:
                self.noises.append(torch.load(nf))
            except:
                pass
                
        if len(self.clean_files) == 0:
            # Abort if no clean files found
            raise RuntimeError(f"No clean files found in {clean_dir}")
        if len(self.noises) == 0:
            print(f"Warning: No noise files found in {noise_dir}. Using random noise instead.")

    def __len__(self):
        return len(self.clean_files)

    def _compute_rms(self, tensor):
        return torch.sqrt(torch.mean(tensor ** 2) + 1e-8)

    def _get_stft(self, signal):
        window = torch.hann_window(WIN_LENGTH, device=signal.device)
        return torch.stft(
            signal,
            n_fft = N_FFT,
            hop_length = HOP_LENGTH,
            win_length = WIN_LENGTH,
            window = window,
            return_complex=True
        )

    def __getitem__(self, idx):
        # 1. Load Clean
        clean_path = self.clean_files[idx]
        clean_audio = torch.load(clean_path).squeeze(0).float()

        if DEBUG:
            print(f"DEBUG: Loading clean file: {clean_path}")
            print(f"Has shape: {clean_audio.shape}, dtype: {clean_audio.dtype}")
        
        # Flatten if needed
        if clean_audio.dim() > 1: clean_audio = clean_audio.view(-1)
        
        # 2. Get Random Noise
        if self.noises:
            noise_audio = random.choice(self.noises).float()
        else:
            # Fallback if no noise files
            noise_audio = torch.randn_like(clean_audio)

        if noise_audio.dim() > 1: noise_audio = noise_audio.view(-1)

        # 3. Match Length (Loop or Cut Noise)
        clean_len = len(clean_audio)
        noise_len = len(noise_audio)
        
        if noise_len >= clean_len:
            # Pick a random start point in the noise
            start = random.randint(0, noise_len - clean_len)
            noise_segment = noise_audio[start : start + clean_len]
        else:
            # Repeat noise to cover clean file
            repeats = (clean_len // noise_len) + 1
            noise_segment = noise_audio.repeat(repeats)[:clean_len]

        # 4. Mix at Specific SNR
        clean_rms = self._compute_rms(clean_audio)
        noise_rms = self._compute_rms(noise_segment)
        
        # Calculate scaling factor
        snr_linear = 10 ** (self.snr_db / 20.0)
        target_noise_rms = clean_rms / (snr_linear + 1e-8)
        scale_factor = target_noise_rms / (noise_rms + 1e-8)
        
        noise_scaled = noise_segment * scale_factor
        mixture = clean_audio + noise_scaled

        if DEBUG:
            # Save clean to file for listening
            save_dir = "debug_outputs"
            os.makedirs(save_dir, exist_ok=True)
            clean_path_out = os.path.join(save_dir, f"clean_{os.path.basename(clean_path)}.wav")
            save_wav(clean_audio, clean_path_out, sample_rate=SAMPLE_RATE)
            # Save mixture to file for listening
            mix_path = os.path.join(save_dir, f"mixture_{os.path.basename(clean_path)}.wav")
            save_wav(mixture, mix_path, sample_rate=SAMPLE_RATE)

        # Overall normalization to prevent clipping
        max_amp = torch.max(torch.abs(mixture))
        if max_amp > 1.0:
            mixture = mixture / max_amp
            clean_audio = clean_audio / max_amp
            noise_scaled = noise_scaled / max_amp

        # 5. Features are now complex STFTs
        Y = self._get_stft(mixture)
        X = self._get_stft(clean_audio)

        # 6. Prepare network input: real + imag
        # Shape: [2, F, T]
        Y_real = Y.real
        Y_imag = Y.imag
        features = torch.stack([Y_real, Y_imag], dim=0)

        sample = {
            "features": features, # [2, F, T]
            "mix_stft": Y, # complex
            "clean_stft": X, # complex
            "clean_audio": clean_audio # for temporal loss if needed
        }
        
        return sample