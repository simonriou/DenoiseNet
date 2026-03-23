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
It computes complex STFT features and normalized clean complex targets for direct
spectral mapping. The __getitem__ method returns a sample where:
- features: Tensor of shape [2, Freq, Time] representing normalized real/imag parts of the noisy mixture STFT.
- clean_complex: Tensor of shape [1, Freq, Time] representing the normalized clean complex STFT target.

Important notice: Audio files are stored as int16 .pt tensors to save space and loading time.
This is why we convert them to float32 in the __getitem__ method.

The DEBUG flag enables saving intermediate audio files and printing debug information for verification (triggers at each __getitem__ call).
"""

class SpeechNoiseDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, snr_db=5.0, mode='train', file_indices=None):
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.pt')))
        self.noise_files = sorted(glob.glob(os.path.join(noise_dir, '*.pt')))
        if file_indices is not None:
            self.clean_files = [self.clean_files[idx] for idx in file_indices]
        self.snr_db = snr_db
        self.mode = mode
        self.test_seed = TEST_RANDOM_SEED
        
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

    def _get_stft_complex(self, tensor):
        window = torch.hann_window(WIN_LENGTH, device=tensor.device)
        return torch.stft(
            tensor,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
            return_complex=True,
        )

    def _sample_snr_db(self, rng):
        if isinstance(self.snr_db, (tuple, list)):
            if len(self.snr_db) != 2:
                raise ValueError("snr_db range must contain exactly two values.")
            lo, hi = sorted(float(v) for v in self.snr_db)
            return rng.uniform(lo, hi)

        return float(self.snr_db)

    def __getitem__(self, idx):
        # 1. Load Clean
        clean_path = self.clean_files[idx]
        clean_audio = torch.load(clean_path).squeeze(0).float()
        is_deterministic = self.mode != 'train'
        rng = random.Random(self.test_seed + idx) if is_deterministic else random

        if DEBUG:
            print(f"DEBUG: Loading clean file: {clean_path}")
            print(f"Has shape: {clean_audio.shape}, dtype: {clean_audio.dtype}")
        
        # Flatten if needed
        if clean_audio.dim() > 1: clean_audio = clean_audio.view(-1)
        
        # 2. Get Random Noise
        if self.noises:
            noise_audio = self.noises[rng.randrange(len(self.noises))].float()
        else:
            # Fallback if no noise files
            if is_deterministic:
                generator = torch.Generator(device=clean_audio.device)
                generator.manual_seed(self.test_seed + idx)
                noise_audio = torch.randn(
                    clean_audio.shape,
                    generator=generator,
                    device=clean_audio.device,
                )
            else:
                noise_audio = torch.randn_like(clean_audio)

        if noise_audio.dim() > 1: noise_audio = noise_audio.view(-1)

        # 3. Match Length (Loop or Cut Noise)
        clean_len = len(clean_audio)
        noise_len = len(noise_audio)
        
        if noise_len >= clean_len:
            # Pick a random start point in the noise
            start = rng.randint(0, noise_len - clean_len)
            noise_segment = noise_audio[start : start + clean_len]
        else:
            # Repeat noise to cover clean file
            repeats = (clean_len // noise_len) + 1
            noise_segment = noise_audio.repeat(repeats)[:clean_len]

        # 4. Mix at Specific SNR
        clean_rms = self._compute_rms(clean_audio)
        noise_rms = self._compute_rms(noise_segment)
        snr_db = self._sample_snr_db(rng)
        
        # Calculate scaling factor
        snr_linear = 10 ** (snr_db / 20.0)
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

        # 5. Compute STFT & Features
        mix_complex = self._get_stft_complex(mixture)
        clean_complex = self._get_stft_complex(clean_audio)

        mix_mag = mix_complex.abs()
        clean_mag = clean_complex.abs()
        mix_phase = torch.angle(mix_complex)

        mix_scale = torch.clamp(mix_mag.max(), min=1e-8)
        mix_complex_norm = mix_complex / mix_scale
        clean_complex_norm = clean_complex / mix_scale

        features = torch.stack(
            [mix_complex_norm.real, mix_complex_norm.imag],
            dim=0,
        )

        if DEBUG:
            print(f"DEBUG: Loaded {os.path.basename(clean_path)}")
            print(f"  Clean RMS: {clean_rms:.4f}, Noise RMS: {noise_rms:.4f}, Scale: {scale_factor:.4f}, SNR: {snr_db:.2f} dB")
            print(f"  Mixture Max Amp: {torch.max(torch.abs(mixture)):.4f}")
            print(f"  Feature Shape: {features.shape}, Clean Complex Shape: {clean_complex_norm.shape}")

            # Plot noisy and clean spectrograms for verification.
            plt.figure(figsize=(12, 6))
            plt.subplot(2,1,1)
            plt.title("Mixture Log-Magnitude Spectrogram")
            plt.imshow(20 * torch.log10(mix_mag + 1e-8).numpy(), origin='lower', aspect='auto', cmap='magma')
            plt.colorbar(format='%+2.0f dB')
            plt.subplot(2,1,2)
            plt.title("Clean Log-Magnitude Spectrogram")
            plt.imshow(20 * torch.log10(clean_mag + 1e-8).numpy(), origin='lower', aspect='auto', cmap='magma')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.show()

        # Input shape needs to be [Channels, Freq, Time] for CNN
        # Current shape is [Freq, Time], unsqueeze to add channel
        
        sample = {
            "features": features,
            "clean_mag": clean_mag.unsqueeze(0),
            "mix_mag": mix_mag.unsqueeze(0),
            "mix_phase": mix_phase.unsqueeze(0),
            "clean_audio": clean_audio.unsqueeze(0),
            "mix_complex": mix_complex_norm.unsqueeze(0),
            "clean_complex": clean_complex_norm.unsqueeze(0),
            "mix_scale": mix_scale,
            "clean_length": clean_audio.shape[0],
            "spec_length": mix_complex.shape[-1],
            "snr_db": snr_db,
            "filename": None
        }

        if self.mode == 'test':
            print("Test mode: adding filename to sample.")
            sample["mix_phase"] = mix_phase.unsqueeze(0)
            sample["clean_audio"] = clean_audio.unsqueeze(0)
            sample["filename"] = os.path.basename(clean_path).split('.')[0]
        
        return sample
