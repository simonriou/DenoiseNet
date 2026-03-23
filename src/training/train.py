import csv
import glob
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from training.dataset import SpeechNoiseDataset
from training.metrics import batch_validation_metrics
from training.objectives import (
    build_audio_mask,
    build_frame_mask,
    complex_l1_loss,
    intelligibility_weighted_mag_loss,
    linear_l1_loss,
    mel_l1_loss,
    reconstruct_batch_waveforms,
    stacked_channels_to_complex,
    suppression_gain_continuity_loss,
    waveform_objective,
)
from utils.constants import *
from utils.pad_collate import pad_collate

"""
Training entrypoint for the speech denoising model.

The model predicts a normalized clean complex spectrogram from a normalized noisy
complex spectrogram. Training now matches inference more closely by reconstructing
waveforms with the same hybrid classical decoder used at inference, and by
optimizing a richer objective that combines:
- complex-domain L1
- linear/mel magnitude losses
- an intelligibility-weighted magnitude term
- a multi-term waveform objective (time-domain L1 + SI-SDR surrogate + MR-STFT)
- a temporal continuity loss on framewise suppression gain
"""


def custom_loss(complex_term, linear_term, mel_term, waveform_term, continuity_term):
    return (
        LAMBDA * complex_term
        + GAMMA * linear_term
        + OMEGA * mel_term
        + ZETA * waveform_term
        + ETA * continuity_term
    )


def _format_optional_metric(value):
    return "n/a" if value is None else f"{value:.4f}"


def _build_split_indices(dataset_size, val_ratio=0.15, seed=42):
    if dataset_size < 2:
        raise RuntimeError("Training requires at least two clean speech files to create train and validation splits.")

    n_val = max(1, int(dataset_size * val_ratio))
    n_val = min(n_val, dataset_size - 1)
    n_train = dataset_size - n_val

    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(dataset_size, generator=generator).tolist()
    return shuffled[:n_train], shuffled[n_train:]


def _forward_loss_terms(model, batch, mel_fb, device, reconstruction_loudness_mode=None):
    features = batch["features"].to(device)
    clean_audio = batch["clean_audio"].to(device)
    clean_complex = batch["clean_complex"].to(device).squeeze(1)
    mix_complex = batch["mix_complex"].to(device).squeeze(1)
    mix_phase = batch["mix_phase"].to(device).squeeze(1)
    mix_scale = batch["mix_scale"].to(device).view(-1, 1, 1)
    audio_lengths = batch["clean_length"].to(device)
    spec_lengths = batch["spec_length"].to(device)

    pred_spectrogram = model(features)
    pred_complex_norm = stacked_channels_to_complex(pred_spectrogram)
    pred_mag = pred_complex_norm.abs().unsqueeze(1)
    clean_mag = clean_complex.abs().unsqueeze(1)
    mix_mag = mix_complex.abs().unsqueeze(1)

    spec_mask = build_frame_mask(spec_lengths, pred_complex_norm.shape[-1], device)
    audio_mask = build_audio_mask(audio_lengths, clean_audio.shape[1], device)

    reconstructed_audio = reconstruct_batch_waveforms(
        pred_complex_norm,
        mix_complex,
        mix_phase,
        mix_scale,
        audio_lengths,
        loudness_mode=reconstruction_loudness_mode,
    )

    complex_term = complex_l1_loss(pred_complex_norm, clean_complex, spec_mask)
    linear_l1 = linear_l1_loss(pred_mag, clean_mag, spec_mask)
    mel_l1 = mel_l1_loss(pred_mag, clean_mag, mel_fb, spec_mask)
    intelligibility_l1 = intelligibility_weighted_mag_loss(pred_mag, clean_mag, spec_mask)
    continuity = suppression_gain_continuity_loss(pred_mag, mix_mag, clean_mag, spec_mask)
    waveform_total, waveform_l1, si_sdr_component, mrstft_component = waveform_objective(
        reconstructed_audio,
        clean_audio,
        audio_mask,
    )

    linear_term = linear_l1 + INTELLIGIBILITY_LOSS_WEIGHT * intelligibility_l1

    return {
        "audio_lengths": audio_lengths,
        "clean_audio": clean_audio,
        "complex_term": complex_term,
        "linear_l1": linear_l1,
        "linear_term": linear_term,
        "mel_l1": mel_l1,
        "intelligibility_l1": intelligibility_l1,
        "continuity": continuity,
        "reconstructed_audio": reconstructed_audio,
        "waveform_total": waveform_total,
        "waveform_l1": waveform_l1,
        "waveform_si_sdr": si_sdr_component,
        "waveform_mrstft": mrstft_component,
    }


def evaluate(model, dataloader, mel_fb, device):
    model.eval()
    totals = {
        "complex_l1": 0.0,
        "l1_linear": 0.0,
        "l1_mel": 0.0,
        "intelligibility_l1": 0.0,
        "continuity": 0.0,
        "waveform_total": 0.0,
        "waveform_l1": 0.0,
        "waveform_si_sdr_loss": 0.0,
        "waveform_mrstft": 0.0,
        "si_sdr": 0.0,
        "seg_snr": 0.0,
        "lsd": 0.0,
        "stoi": 0.0,
        "pesq": 0.0,
    }
    counts = {key: 0 for key in totals}

    with torch.no_grad():
        for batch in dataloader:
            loss_terms = _forward_loss_terms(model, batch, mel_fb, device)
            batch_size = loss_terms["audio_lengths"].shape[0]

            totals["complex_l1"] += loss_terms["complex_term"].item() * batch_size
            counts["complex_l1"] += batch_size
            totals["l1_linear"] += loss_terms["linear_l1"].item() * batch_size
            counts["l1_linear"] += batch_size
            totals["l1_mel"] += loss_terms["mel_l1"].item() * batch_size
            counts["l1_mel"] += batch_size
            totals["intelligibility_l1"] += loss_terms["intelligibility_l1"].item() * batch_size
            counts["intelligibility_l1"] += batch_size
            totals["continuity"] += loss_terms["continuity"].item() * batch_size
            counts["continuity"] += batch_size
            totals["waveform_total"] += loss_terms["waveform_total"].item() * batch_size
            counts["waveform_total"] += batch_size
            totals["waveform_l1"] += loss_terms["waveform_l1"].item() * batch_size
            counts["waveform_l1"] += batch_size
            totals["waveform_si_sdr_loss"] += loss_terms["waveform_si_sdr"].item() * batch_size
            counts["waveform_si_sdr_loss"] += batch_size
            totals["waveform_mrstft"] += loss_terms["waveform_mrstft"].item() * batch_size
            counts["waveform_mrstft"] += batch_size

            batch_metrics, batch_metric_counts = batch_validation_metrics(
                loss_terms["clean_audio"].detach().cpu(),
                loss_terms["reconstructed_audio"].detach().cpu(),
                loss_terms["audio_lengths"].detach().cpu(),
            )
            for metric_name, metric_value in batch_metrics.items():
                if metric_value is None:
                    continue
                totals[metric_name] += metric_value * batch_metric_counts[metric_name]
                counts[metric_name] += batch_metric_counts[metric_name]

    return {
        key: (totals[key] / counts[key] if counts[key] > 0 else None)
        for key in totals
    }


def train(session_name: str):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(NOISE_DIR, exist_ok=True)
    if not glob.glob(f"{CLEAN_DIR}/*.pt"):
        print("Error: No clean data found. Please add .pt files to the clean data directory.")
        return

    dataset_probe = SpeechNoiseDataset(CLEAN_DIR, NOISE_DIR, snr_db=VAL_SNR_DB, mode="val")
    train_indices, val_indices = _build_split_indices(len(dataset_probe))
    train_dataset = SpeechNoiseDataset(
        CLEAN_DIR,
        NOISE_DIR,
        snr_db=TRAIN_SNR_RANGE_DB,
        mode="train",
        file_indices=train_indices,
    )
    val_dataset = SpeechNoiseDataset(
        CLEAN_DIR,
        NOISE_DIR,
        snr_db=VAL_SNR_DB,
        mode="val",
        file_indices=val_indices,
    )

    print(f"Training on SNR range {TRAIN_SNR_RANGE_DB[0]:.1f} to {TRAIN_SNR_RANGE_DB[1]:.1f} dB.")
    print(f"Validation uses fixed SNR {VAL_SNR_DB:.1f} dB.")

    mel_fb = torch.load(f"{ROOT}/src/training/mel_fb_{N_FFT}_{N_MELS}_{SAMPLE_RATE}.pt").to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_collate,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(MODEL_ARCHITECTURE).to(device)
    print(f"Training architecture: {MODEL_ARCHITECTURE}")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoints_dir = os.path.join(CHECKPOINT_DIR, session_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    log_file_dir = os.path.join(LOG_DIR, session_name)
    os.makedirs(log_file_dir, exist_ok=True)
    log_file_path = os.path.join(log_file_dir, "training_log.csv")

    print(f"Logging training progress to {log_file_path}")
    with open(log_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_complex_l1",
                "val_l1_linear",
                "val_l1_mel",
                "val_intelligibility_l1",
                "val_continuity",
                "val_waveform_total",
                "val_waveform_l1",
                "val_waveform_si_sdr_loss",
                "val_waveform_mrstft",
                "val_si_sdr",
                "val_seg_snr",
                "val_lsd",
                "val_stoi",
                "val_pesq",
            ]
        )

    running = {
        "complex": None,
        "linear": None,
        "mel": None,
        "waveform": None,
        "continuity": None,
    }
    smoothing = 0.99

    print("Starting Training...")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            optimizer.zero_grad()
            loss_terms = _forward_loss_terms(
                model,
                batch,
                mel_fb,
                device,
                reconstruction_loudness_mode=TRAIN_RECON_LOUDNESS_MODE,
            )

            component_values = {
                "complex": loss_terms["complex_term"].item(),
                "linear": loss_terms["linear_term"].item(),
                "mel": loss_terms["mel_l1"].item(),
                "waveform": loss_terms["waveform_total"].item(),
                "continuity": loss_terms["continuity"].item(),
            }

            for key, value in component_values.items():
                if running[key] is None:
                    running[key] = abs(value) + 1e-6
                else:
                    running[key] = smoothing * running[key] + (1.0 - smoothing) * (abs(value) + 1e-6)

            loss = custom_loss(
                loss_terms["complex_term"] / running["complex"],
                loss_terms["linear_term"] / running["linear"],
                loss_terms["mel_l1"] / running["mel"],
                loss_terms["waveform_total"] / running["waveform"],
                loss_terms["continuity"] / running["continuity"],
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, mel_fb, device)

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Complex L1: {val_metrics['complex_l1']:.4f} | "
            f"Val Linear: {val_metrics['l1_linear']:.4f} | "
            f"Val Mel: {val_metrics['l1_mel']:.4f} | "
            f"Val Intelligibility: {val_metrics['intelligibility_l1']:.4f} | "
            f"Val Continuity: {val_metrics['continuity']:.4f} | "
            f"Val Waveform: {val_metrics['waveform_total']:.4f} | "
            f"SI-SDR: {val_metrics['si_sdr']:.4f} | "
            f"SegSNR: {val_metrics['seg_snr']:.4f} | "
            f"LSD: {val_metrics['lsd']:.4f} | "
            f"STOI: {_format_optional_metric(val_metrics['stoi'])} | "
            f"PESQ: {_format_optional_metric(val_metrics['pesq'])}"
        )

        checkpoint_path = os.path.join(checkpoints_dir, f"chkp_{session_name}_epoch{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        with open(log_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    val_metrics["complex_l1"],
                    val_metrics["l1_linear"],
                    val_metrics["l1_mel"],
                    val_metrics["intelligibility_l1"],
                    val_metrics["continuity"],
                    val_metrics["waveform_total"],
                    val_metrics["waveform_l1"],
                    val_metrics["waveform_si_sdr_loss"],
                    val_metrics["waveform_mrstft"],
                    val_metrics["si_sdr"],
                    val_metrics["seg_snr"],
                    val_metrics["lsd"],
                    val_metrics["stoi"],
                    val_metrics["pesq"],
                ]
            )

        if epoch == EPOCHS - 1:
            final_model_path = os.path.join(MODEL_DIR, f"{session_name}.pth")
            torch.save(model.state_dict(), final_model_path)

    print("Training Complete.")


if __name__ == "__main__":
    session_name = input("Enter a session name for this training run: ").strip()
    if not session_name:
        print("Session name cannot be empty. Exiting.")
    elif os.path.exists(os.path.join(CHECKPOINT_DIR, session_name)):
        overwrite = input(f"Session '{session_name}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite == "y":
            train(session_name)
        else:
            print("Exiting without training.")
    else:
        train(session_name)
