import os
import glob
import random
import torch
from torch.utils.data import Dataset

from utils.save_wav import save_wav
from utils.constants import *

if DEBUG:
    import matplotlib.pyplot as plt

class SpeechNoiseDatasetTemporal(Dataset):
    def __init__(self, clean_dir, noise_dir, snr_db=5.0, mode='train', seq_len=None):
        """
        seq_len: if specified, all segments are trimmed or looped to this length (in samples)
        """
        self.clean_files = glob.glob(os.path.join(clean_dir, '*.pt'))
        self.noise_files = glob.glob(os.path.join(noise_dir, '*.pt'))
        self.snr_db = snr_db
        self.mode = mode
        self.seq_len = seq_len
        
        self.noises = []
        for nf in self.noise_files:
            try:
                self.noises.append(torch.load(nf))
            except:
                pass
                
        if len(self.clean_files) == 0:
            raise RuntimeError(f"No clean files found in {clean_dir}")
        if len(self.noises) == 0:
            print(f"Warning: No noise files found in {noise_dir}. Using random noise instead.")

    def __len__(self):
        return len(self.clean_files)

    def _compute_rms(self, tensor):
        return torch.sqrt(torch.mean(tensor ** 2) + 1e-8)

    def _get_noise_segment(self, clean_len):
        if self.noises:
            noise_audio = random.choice(self.noises).float()
        else:
            noise_audio = torch.randn(clean_len)
        
        if noise_audio.dim() > 1: noise_audio = noise_audio.view(-1)
        noise_len = len(noise_audio)
        
        if noise_len >= clean_len:
            start = random.randint(0, noise_len - clean_len)
            return noise_audio[start : start + clean_len]
        else:
            repeats = (clean_len // noise_len) + 1
            return noise_audio.repeat(repeats)[:clean_len]

    def __getitem__(self, idx):
        # 1. Load clean waveform
        clean_path = self.clean_files[idx]
        clean_audio = torch.load(clean_path).squeeze(0).float()
        if clean_audio.dim() > 1: clean_audio = clean_audio.view(-1)

        # 2. Truncate / pad to seq_len if needed
        if self.seq_len is not None:
            if len(clean_audio) >= self.seq_len:
                start = random.randint(0, len(clean_audio) - self.seq_len)
                clean_audio = clean_audio[start:start+self.seq_len]
            else:
                repeats = (self.seq_len // len(clean_audio)) + 1
                clean_audio = clean_audio.repeat(repeats)[:self.seq_len]

        # 3. Get noise segment
        noise_segment = self._get_noise_segment(len(clean_audio))

        # 4. Mix at target SNR
        clean_rms = self._compute_rms(clean_audio)
        noise_rms = self._compute_rms(noise_segment)
        snr_linear = 10 ** (self.snr_db / 20.0)
        target_noise_rms = clean_rms / (snr_linear + 1e-8)
        scale_factor = target_noise_rms / (noise_rms + 1e-8)
        noise_scaled = noise_segment * scale_factor
        mixture = clean_audio + noise_scaled

        # 5. Normalize to [-1, 1] if needed
        max_amp = torch.max(torch.abs(mixture))
        if max_amp > 1.0:
            mixture = mixture / max_amp
            clean_audio = clean_audio / max_amp

        if DEBUG:
            save_dir = "debug_outputs"
            os.makedirs(save_dir, exist_ok=True)
            save_wav(clean_audio, os.path.join(save_dir, f"clean_{os.path.basename(clean_path)}.wav"), SAMPLE_RATE)
            save_wav(mixture, os.path.join(save_dir, f"mixture_{os.path.basename(clean_path)}.wav"), SAMPLE_RATE)

        # Return 1D tensors for temporal model
        return {
            "mixture": mixture,       # [T]
            "clean": clean_audio      # [T]
        }