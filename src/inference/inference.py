import torch
from speechbrain.inference.vocoders import HIFIGAN
from torchaudio.transforms import GriffinLim
from torch.utils.data import DataLoader
from models.DenoiseUNet import DenoiseUNet
from training.dataset import SpeechNoiseDataset
from utils.constants import *
from utils.save_wav import save_wav
from utils.compute_snr import compute_snr
from utils.pad_collate import pad_collate
from utils.magnitude_to_mel import magnitude_to_mel
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

model = DenoiseUNet().to(device)
model.load_state_dict(
    torch.load(MODEL_DIR / f"{MODEL_NAME}.pth", map_location=device)
)
model.eval()

# 4. Output directory
denoised_dir = NOISE_ENHANCED_DIR
os.makedirs(denoised_dir, exist_ok=True)

# 5. Inference loop
total_time = 0.0
with torch.no_grad():
    for idx, batch in enumerate(loader):

        features      = batch["features"].to(device)      # [1, 1, F, T]
        mix_mag       = batch["mix_mag"].to(device)       # [1, 1, F, T]
        clean_audio   = batch["clean_audio"][0].cpu().numpy()
        fname      = batch["filename"][0]

        start_time = time.time()

        # Predict mask
        pred_mask = model(features)                        # [1, 1, F, T]

        # Apply mask
        enhanced_mag = pred_mask * mix_mag                 # [1, 1, F, T]

        enhanced_mag = enhanced_mag.to(device)
        mix_mag = mix_mag.to(device)

        if PHASE_MODE.lower() == 'gl':
            print("Using Griffin-Lim for phase reconstruction.")
            enhanced_audio = gl(enhanced_mag[0, 0])
            noisy_audio = gl(mix_mag[0, 0])
        elif PHASE_MODE.lower() == 'raw':
            print("Using mixture phase for reconstruction.")
            # Use mixture phase
            mix_phase = batch["mix_phase"].to(device)     # [1, 1, F, T]
            complex_spec = enhanced_mag.squeeze(0).squeeze(0) * torch.exp(1j * mix_phase.squeeze(0).squeeze(0))
            enhanced_audio = torch.istft(
                complex_spec,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH,
                window=torch.hann_window(WIN_LENGTH).to(device),
                length=batch["clean_audio"].shape[1]
            )
            complex_spec_noisy = mix_mag.squeeze(0).squeeze(0) * torch.exp(1j * mix_phase.squeeze(0).squeeze(0))
            noisy_audio = torch.istft(
                complex_spec_noisy,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WIN_LENGTH,
                window=torch.hann_window(WIN_LENGTH).to(device),
                length=batch["clean_audio"].shape[1]
            )
        elif PHASE_MODE.lower() == 'vocoder':
            raise NotImplementedError("Vocoder mode not implemented yet.")
            print("Using neural vocoder for reconstruction.")
            # Extract mel spectrogram
            mel_fb = torch.load(f'mel_fb_{N_FFT}_{N_MELS}_{SAMPLE_RATE}.pt').to(device)  # (n_mels, F)
            enhanced_mel = magnitude_to_mel(enhanced_mag, mel_fb)  # (1, 1, M, T)
            hifi_gan = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-libritts-16kHz",
                savedir="pretrained_models/tts-hifigan-libritts-16kHz",
                run_opts={"device":"cuda"} if torch.cuda.is_available() else {"device":"cpu"}
            )

            waveform = hifi_gan.decode_batch(enhanced_mel.squeeze(0)) # (1, 1, T)
            enhanced_audio = waveform.squeeze(1) # remove channel dim -> (1, T)

            waveform = hifi_gan.decode_batch(magnitude_to_mel(mix_mag, mel_fb).squeeze(0))
            noisy_audio = waveform.squeeze(1)
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