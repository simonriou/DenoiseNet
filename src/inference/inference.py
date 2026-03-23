import torch
from torchaudio.transforms import GriffinLim
from torch.utils.data import DataLoader
from inference.classical_reconstruction import reconstruct_classical_waveform
from inference.neural_vocoder import SpeechHiFiGANVocoder
from models import build_model
from training.dataset import SpeechNoiseDataset
from utils.constants import *
from utils.save_wav import save_wav
from utils.compute_snr import compute_snr
from utils.pad_collate import pad_collate
import os
import csv
import numpy as np
import time

# 1. Prepare dataset
dataset = SpeechNoiseDataset(
    clean_dir=CLEAN_TEST_DIR,
    noise_dir=NOISE_TEST_DIR,
    snr_db=TARGET_SNR,
    mode="test"
)

# 2. DataLoader (IMPORTANT: use pad_collate)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=pad_collate
)

# 3. Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gl = GriffinLim(
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    power=1.0,
    n_iter=GL_ITERS,
    momentum=0.99,
    length=None,
    rand_init=True,
)

model = build_model(MODEL_ARCHITECTURE).to(device)
model.load_state_dict(
    torch.load(MODEL_DIR / f"{MODEL_NAME}.pth", map_location=device)
)
model.eval()
print(f"Loaded architecture: {MODEL_ARCHITECTURE}")

stft_window = torch.hann_window(WIN_LENGTH).to(device)
vocoder = None
if PHASE_MODE.lower() == "vocoder":
    print("Loading SpeechBrain HiFi-GAN vocoder.")
    vocoder = SpeechHiFiGANVocoder(device=device)

# 4. Output directory
denoised_dir = NOISE_ENHANCED_DIR
os.makedirs(denoised_dir, exist_ok=True)

# 5. Inference loop
total_time = 0.0
with torch.no_grad():
    for idx, batch in enumerate(loader):

        features      = batch["features"].to(device)      # [1, 2, F, T]
        clean_audio   = batch["clean_audio"][0].cpu().numpy()
        fname      = batch["filename"][0]
        mix_complex = batch["mix_complex"].to(device).squeeze(1)
        mix_scale = batch["mix_scale"].to(device).view(-1, 1, 1)

        start_time = time.time()

        # Predict the denoised complex spectrogram directly.
        pred_spectrogram = model(features)                 # [1, 2, F, T]
        enhanced_complex_norm = pred_spectrogram[:, 0] + 1j * pred_spectrogram[:, 1]
        enhanced_complex = enhanced_complex_norm * mix_scale
        mix_complex_denorm = mix_complex * mix_scale
        target_length = batch["clean_audio"].shape[1]
        noisy_audio = torch.istft(
            mix_complex_denorm[0],
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=stft_window,
            length=target_length
        )

        phase_mode = PHASE_MODE.lower()
        if phase_mode in {"complex", "raw", "hybrid"}:
            if phase_mode == "hybrid":
                print("Using hybrid classical reconstruction.")
            elif phase_mode == "raw":
                print("Using mixture phase for reconstruction.")

            enhanced_audio = reconstruct_classical_waveform(
                phase_mode,
                enhanced_complex[0],
                mix_complex_denorm[0],
                batch["mix_phase"].to(device).squeeze(1)[0],
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH,
                window=stft_window,
                target_length=target_length,
                mask_smooth_freq=RECON_MASK_SMOOTH_FREQ,
                mask_smooth_time=RECON_MASK_SMOOTH_TIME,
                phase_blend_power=RECON_PHASE_BLEND_POWER,
                mask_ceiling=RECON_MASK_CEILING,
                loudness_mode=RECON_LOUDNESS_MODE,
                loudness_blend=RECON_LOUDNESS_BLEND,
                loudness_max_step_db=RECON_LOUDNESS_MAX_STEP_DB,
            )
        elif phase_mode == 'gl':
            print("Using Griffin-Lim for phase reconstruction.")
            enhanced_mag = enhanced_complex.abs().unsqueeze(1)
            enhanced_audio = gl(enhanced_mag[0, 0])
        elif phase_mode == 'vocoder':
            print("Using neural vocoder for reconstruction.")
            enhanced_mag = enhanced_complex.abs().unsqueeze(1)
            enhanced_audio = vocoder.decode(
                enhanced_mag,
                target_length=target_length,
            ).squeeze(0)
        else:
            raise ValueError(f"Unknown PHASE_MODE: {PHASE_MODE}")
        
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time

        print(f"Processed file {fname} in {inference_time:.3f} seconds.")

        enhanced_audio = enhanced_audio.cpu().numpy()
        enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)

        noisy_audio = noisy_audio.cpu().numpy()
        noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

        # Save audio
        if SAVE_DENOISED:
            save_wav(
                enhanced_audio,
                denoised_dir / f"denoised_{fname}.wav",
                sample_rate=SAMPLE_RATE
            )

        if SAVE_NOISY:
            save_wav(
                noisy_audio,
                denoised_dir / f"noisy_{fname}.wav",
                sample_rate=SAMPLE_RATE
            )

        # --- Metrics ---
        min_len = min(len(clean_audio), len(noisy_audio), len(enhanced_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        enhanced_audio = enhanced_audio[:min_len]

        snr_noisy = compute_snr(clean_audio, noisy_audio)
        snr_enhanced = compute_snr(clean_audio, enhanced_audio)

        print(
            f"File {fname}: "
            f"Noisy SNR = {snr_noisy:.2f} dB | "
            f"Enhanced SNR = {snr_enhanced:.2f} dB"
        )

        # --- Logging ---
        log_csv_dir = os.path.join(LOG_DIR, MODEL_NAME)
        os.makedirs(log_csv_dir, exist_ok=True)
        log_csv_path = os.path.join(log_csv_dir, "inference_snr_log.csv")

        if idx == 0:
            with open(log_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["file", "noisy_snr_db", "enhanced_snr_db"]
                )

        with open(log_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [fname, f"{snr_noisy:.2f}", f"{snr_enhanced:.2f}"]
            )

average_time = total_time / len(loader)
print(f"Average inference time per file: {average_time:.3f} seconds.")
print("Inference Complete.")
