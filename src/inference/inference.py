import os
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.models.ConvRNNTemporalDenoiser import DenoiseUNet
from training.dataset import SpeechNoiseDataset
from utils.constants import *
from utils.save_wav import save_wav
from utils.compute_snr import compute_snr
from utils.pad_collate import pad_collate

# --- 1. Prepare dataset ---
dataset = SpeechNoiseDataset(
    clean_dir=CLEAN_TEST_DIR,
    noise_dir=NOISE_TEST_DIR,
    snr_db=TARGET_SNR,
    mode="test"
)

# --- 2. DataLoader ---
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=pad_collate
)

# --- 3. Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoiseUNet().to(device)
model.load_state_dict(torch.load(MODEL_DIR / f"{MODEL_NAME}.pth", map_location=device))
model.eval()

# --- 4. Output directory ---
denoised_dir = NOISE_ENHANCED_DIR
os.makedirs(denoised_dir, exist_ok=True)

# --- 5. Inference loop ---
with torch.no_grad():
    for idx, batch in enumerate(loader):
        features    = batch["features"].to(device)  # [B, 1, F, T]
        clean_audio = batch["clean_audio"][0].cpu().numpy()

        # Predict complex mask: [B, 2, F, T]
        pred_complex = model(features)

        # Convert to complex STFT: real + i*imag
        # Assuming channel 0 = real, channel 1 = imag
        pred_complex = pred_complex.permute(0, 2, 3, 1)  # [B, F, T, 2]
        enhanced_stft = torch.complex(pred_complex[..., 0], pred_complex[..., 1])  # [B, F, T]

        # Inverse STFT to waveform
        enhanced_audio = torch.istft(
            enhanced_stft[0],
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            length=batch["clean_audio"].shape[1]
        ).cpu().numpy()

        enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)

        # --- Save audio ---
        if SAVE_DENOISED:
            save_wav(enhanced_audio, os.path.join(denoised_dir, f"denoised_{idx}.wav"), sample_rate=SAMPLE_RATE)

        # --- Compute SNR ---
        min_len = min(len(clean_audio), len(enhanced_audio))
        clean_audio = clean_audio[:min_len]
        enhanced_audio = enhanced_audio[:min_len]

        snr_enhanced = compute_snr(clean_audio, enhanced_audio)
        print(f"File {idx}: Enhanced SNR = {snr_enhanced:.2f} dB")

        # --- Logging ---
        log_csv_dir = os.path.join(LOG_DIR, MODEL_NAME)
        os.makedirs(log_csv_dir, exist_ok=True)
        log_csv_path = os.path.join(log_csv_dir, "inference_snr_log.csv")

        if idx == 0:
            with open(log_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["file_index", "enhanced_snr_db"])

        with open(log_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([idx, f"{snr_enhanced:.2f}"])

print("Inference Complete.")