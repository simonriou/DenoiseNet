import os
import glob
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.constants import *
from training.dataset import SpeechNoiseDataset
from models.DCUNet import DCUNet
from utils.pad_collate import pad_collate

"""
This is the main training script for the speech denoising model.
It sets up the dataset, dataloader, model, loss function, and optimizer.
It runs the training loop for a specified number of epochs, printing progress and saving model checkpoints.

The model predicts complex ratio masks on STFTs and is trained with a multi-term loss
combining complex L1, linear/mel magnitude L1, and waveform L1.
The Adam optimizer is used for training.
"""

def mel_l1_loss(x, y, mel_fb):
    """
    x, y: (B, 1, F, T)
    mel_fb: (F, M)
    """

    # Move frequency to last axis
    rx = x.permute(0, 1, 3, 2)  # (B, 1, T, F)
    ry = y.permute(0, 1, 3, 2)  # (B, 1, T, F)

    # Apply Mel projection
    pred_mel  = torch.matmul(rx, mel_fb)   # (B, 1, T, M)
    clean_mel = torch.matmul(ry, mel_fb)   # (B, 1, T, M)

    return torch.mean(torch.abs(pred_mel - clean_mel))

def l1_loss(x, y):
    return nn.L1Loss()(x, y)

def complex_l1_loss(pred, target):
    return torch.mean(torch.abs(pred.real - target.real) + torch.abs(pred.imag - target.imag))

def custom_loss(complex_l1, l1_linear, l1_mel, waveform, lambda_, gamma_, omega_, zeta_):
    # lambda Complex L1 + gamma L1 + omega Mel + zeta Waveform
    return lambda_ * complex_l1 + gamma_ * l1_linear + omega_ * l1_mel + zeta_ * waveform

def evaluate(model, dataloader, criterion_l1_linear, criterion_l1_mel, device):
    model.eval()
    total_complex_l1 = 0.0
    total_l1  = 0.0
    total_l1_linear = 0.0
    total_l1_mel = 0.0
    total_waveform = 0.0
    n_batches = 0

    mel_fb = torch.load(f"{ROOT}/src/training/mel_fb_{N_FFT}_{N_MELS}_{SAMPLE_RATE}.pt").to(device)

    with torch.no_grad():
        for batch in dataloader:
            features     = batch["features"].to(device)
            clean_audio  = batch["clean_audio"].to(device)
            mix_complex  = batch["mix_complex"].to(device).squeeze(1)
            clean_complex = batch["clean_complex"].to(device).squeeze(1)
            mix_scale = batch["mix_scale"].to(device).view(-1, 1, 1)

            pred_mask = model(features)
            pred_mask_complex = pred_mask[:, 0] + 1j * pred_mask[:, 1]
            pred_complex_norm = pred_mask_complex * mix_complex
            pred_mag = pred_complex_norm.abs().unsqueeze(1)

            reconstructed_audio = []
            for b in range(pred_mag.shape[0]):
                complex_spec = pred_complex_norm[b] * mix_scale[b]

                audio = torch.istft(
                    complex_spec,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    win_length=WIN_LENGTH,
                    window=torch.hann_window(WIN_LENGTH).to(device),
                    length=clean_audio.shape[1]
                )
                reconstructed_audio.append(audio)
            
            reconstructed_audio = torch.stack(reconstructed_audio, dim=0).to(device)

            complex_l1 = complex_l1_loss(pred_complex_norm, clean_complex)

            clean_mag = clean_complex.abs().unsqueeze(1)
            l1_linear_loss = criterion_l1_linear(pred_mag, clean_mag)
            l1_mel_loss = criterion_l1_mel(pred_mag, clean_mag, mel_fb)
            l1_loss = l1_linear_loss + ALPHA * l1_mel_loss
            waveform_loss = criterion_l1_linear(reconstructed_audio, clean_audio)

            total_complex_l1 += complex_l1.item()
            total_l1  += l1_loss.item()
            total_l1_linear += l1_linear_loss.item()
            total_l1_mel += l1_mel_loss.item()
            total_waveform += waveform_loss.item()
            n_batches += 1

    avg_complex_l1 = total_complex_l1 / n_batches
    avg_l1  = total_l1 / n_batches

    avg_l1_linear = total_l1_linear / n_batches
    avg_l1_mel = total_l1_mel / n_batches

    avg_waveform = total_waveform / n_batches

    return avg_complex_l1, avg_l1_linear, avg_l1_mel, avg_waveform

def train(session_name: str):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(NOISE_DIR, exist_ok=True)
    if not glob.glob(f"{CLEAN_DIR}/*.pt"):
        print("Error: No clean data found. Please add .pt files to the clean data directory.")
        return

    # 2. Load Data
    dataset = SpeechNoiseDataset(CLEAN_DIR, NOISE_DIR, snr_db=TARGET_SNR)
    mel_fb = torch.load(f"{ROOT}/src/training/mel_fb_{N_FFT}_{N_MELS}_{SAMPLE_RATE}.pt").to(device)

    val_ratio = 0.15
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_collate,
        pin_memory=(device.type == 'cuda' )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate,
        pin_memory=(device.type == 'cuda' )
    )
    
    # 3. Model & Loss
    model = DCUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create checkpoint directory for this session
    checkpoints_dir = os.path.join(CHECKPOINT_DIR, session_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    log_file_dir = os.path.join(LOG_DIR, session_name)
    os.makedirs(log_file_dir, exist_ok=True)
    log_file_path = os.path.join(log_file_dir, "training_log.csv")

    # Write CSV header
    print(f"Logging training progress to {log_file_path}")
    with open(log_file_path, mode='w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_complex_l1", "val_l1_linear", "val_l1_mel", "val_waveform"])

    # Initialize running averages for losses
    avg_complex_l1 = 0.0
    avg_l1 = 0.0
    avg_waveform = 0.0
    alpha = 0.99 # smoothing factor for running avg

    print("Starting Training...")
    
    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            features = batch["features"].to(device)
            clean_audio = batch["clean_audio"].to(device)
            mix_complex = batch["mix_complex"].to(device).squeeze(1)
            clean_complex = batch["clean_complex"].to(device).squeeze(1)
            mix_scale = batch["mix_scale"].to(device).view(-1, 1, 1)

            optimizer.zero_grad()
            pred_mask = model(features)
            pred_mask_complex = pred_mask[:, 0] + 1j * pred_mask[:, 1]
            pred_complex_norm = pred_mask_complex * mix_complex
            pred_mag = pred_complex_norm.abs().unsqueeze(1)

            reconstructed_audio = []

            for b in range(pred_mag.shape[0]):
                complex_spec = pred_complex_norm[b] * mix_scale[b]

                audio = torch.istft(
                    complex_spec,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH,
                    win_length=WIN_LENGTH,
                    window=torch.hann_window(WIN_LENGTH).to(device),
                    length=clean_audio.shape[1]
                )
                reconstructed_audio.append(audio)

            reconstructed_audio = torch.stack(reconstructed_audio, dim=0).to(device)

            complex_l1 = complex_l1_loss(pred_complex_norm, clean_complex)
            clean_mag = clean_complex.abs().unsqueeze(1)
            l1_linear = l1_loss(pred_mag, clean_mag)
            l1_mel = mel_l1_loss(pred_mag, clean_mag, mel_fb)
            l1 = l1_linear + ALPHA * l1_mel
            waveform_loss = l1_loss(reconstructed_audio, clean_audio)

            if avg_complex_l1 == 0.0:
                avg_complex_l1 = complex_l1.item()
                avg_l1_linear = l1_linear.item()
                avg_l1_mel = l1_mel.item()
                avg_waveform = waveform_loss.item()
            else:
                avg_complex_l1 = alpha * avg_complex_l1 + (1 - alpha) * complex_l1.item()
                avg_l1_linear = alpha * avg_l1_linear + (1 - alpha) * l1_linear.item()
                avg_l1_mel = alpha * avg_l1_mel + (1 - alpha) * l1_mel.item()
                avg_waveform = alpha * avg_waveform + (1 - alpha) * waveform_loss.item()

            loss = custom_loss(
                (complex_l1 / (avg_complex_l1 + 1e-8)),
                (l1_linear / (avg_l1_linear + 1e-8)),
                (l1_mel / (avg_l1_mel + 1e-8)),
                (waveform_loss / (avg_waveform + 1e-8)),
                LAMBDA,
                GAMMA,
                OMEGA,
                ZETA,
            )
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        val_complex_l1, val_l1_linear, val_l1_mel, val_waveform = evaluate(
            model,
            val_loader,
            criterion_l1_linear=l1_loss,
            criterion_l1_mel=mel_l1_loss,
            device=device,
        )
        
        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Complex L1: {val_complex_l1:.4f}, Val L1 Linear: {val_l1_linear:.4f}, Val L1 Mel: {val_l1_mel:.4f}, Val Waveform: {val_waveform:.4f}"
        )

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoints_dir, f"chkp_{session_name}_epoch{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Log to CSV
        with open(log_file_path, mode='a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_complex_l1, val_l1_linear, val_l1_mel, val_waveform])

        # If final epoch, also save final model
        if epoch == EPOCHS - 1:
            final_model_path = os.path.join(MODEL_DIR, f"{session_name}.pth")
            torch.save(model.state_dict(), final_model_path)

    print("Training Complete.")


if __name__ == "__main__":
    # Ask for session name
    session_name = input("Enter a session name for this training run: ").strip()
    if not session_name:
        print("Session name cannot be empty. Exiting.")
    elif os.path.exists(os.path.join(CHECKPOINT_DIR, session_name)):
        overwrite = input(f"Session '{session_name}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite == 'y':
            train(session_name)
        else:
            print("Exiting without training.")
    else:
        train(session_name)