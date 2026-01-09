import torch
import torchaudio
from constants import *

def create_mel_filterbank(
    n_fft,
    n_mels,
    sample_rate
):
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs = n_fft // 2 + 1,
        f_min = 0.0,
        f_max = sample_rate / 2,
        n_mels = n_mels,
        sample_rate = sample_rate,
        norm = "slaney"
    )
    return mel_fb  # (n_mels, F)

mel_fb = create_mel_filterbank(
    n_fft=N_FFT,
    n_mels=N_MELS,
    sample_rate=SAMPLE_RATE
)

torch.save(mel_fb, f'mel_fb_{N_FFT}_{N_MELS}_{SAMPLE_RATE}.pt')