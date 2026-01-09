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
from models.DenoiseUNet import DenoiseUNet
from utils.pad_collate import pad_collate

"""
This is the main training script for the speech denoising model.
It sets up the dataset, dataloader, model, loss function, and optimizer.
It runs the training loop for a specified number of epochs, printing progress and saving model checkpoints.

The criterion used is Binary Cross-Entropy Loss (BCELoss) since the model outputs ideal binary masks (IBM).
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

def bce_loss(x, y):
    return nn.BCELoss()(x, y)

def l1_loss(x, y):
    return nn.L1Loss()(x, y)

def custom_loss(bce, l1, lambda_, gamma_):
    # lambda BCE + gamma L1
    return lambda_ * bce + gamma_ * l1

def evaluate(model, dataloader, criterion_bce, linear_l1, mel_l1, device):
    model.eval()
    total_bce = 0.0
    total_l1  = 0.0
    n_batches = 0

    mel_fb = torch.load(f"{ROOT}/src/training/mel_fb_{N_FFT}_{N_MELS}_{SAMPLE_RATE}.pt").to(device)

    with torch.no_grad():
        for batch in dataloader:
            features     = batch["features"].to(device)
            ibm_target   = batch["ibm"].to(device)
            mix_mag      = batch["mix_mag"].to(device)
            clean_mag    = batch["clean_mag"].to(device)

            pred_mask = model(features)
            pred_mag  = pred_mask * mix_mag

            bce_loss = criterion_bce(pred_mask, ibm_target)

            l1_linear_loss  = linear_l1(pred_mag, clean_mag)
            l1_mel_loss = mel_l1(pred_mag, clean_mag, mel_fb)
            l1_loss = l1_linear_loss + ALPHA * l1_mel_loss

            total_bce += bce_loss.item()
            total_l1  += l1_loss.item()
            n_batches += 1

    avg_bce = total_bce / n_batches
    avg_l1  = total_l1 / n_batches

    return avg_bce, avg_l1

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
    model = DenoiseUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create checkpoint directory for this session
    checkpoints_dir = os.path.join(CHECKPOINT_DIR, session_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    log_file_dir = os.path.join(LOG_DIR, session_name)
    os.makedirs(log_file_dir, exist_ok=True)
    log_file_path = os.path.join(log_file_dir, "training_log.csv")

    # Write CSV header
    with open(log_file_path, mode='w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_bce", "val_l1"])

    # Initialize running averages for losses
    avg_bce = 0.0
    avg_l1 = 0.0
    alpha = 0.99 # smoothing factor for running avg

    print("Starting Training...")
    
    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            features = batch["features"].to(device)
            mix_mag = batch["mix_mag"].to(device)
            clean_mag = batch["clean_mag"].to(device)
            ibm = batch["ibm"].to(device)

            optimizer.zero_grad()
            pred_mask = model(features)
            pred_mag = pred_mask * mix_mag

            print(pred_mag.shape)
            print(mel_fb.shape)

            bce = bce_loss(pred_mask, ibm)
            l1_linear = l1_loss(pred_mag, clean_mag)
            l1_mel = mel_l1_loss(pred_mag, clean_mag, mel_fb)
            l1 = l1_linear + ALPHA * l1_mel

            if avg_bce == 0.0:
                avg_bce = bce.item()
                avg_l1 = l1.item()
            else:
                avg_bce = alpha * avg_bce + (1 - alpha) * bce.item()
                avg_l1 = alpha * avg_l1 + (1 - alpha) * l1.item()

            loss = custom_loss((bce / (avg_bce + 1e-8)), (l1 / (avg_l1 + 1e-8)), LAMBDA, GAMMA)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        val_bce, val_l1 = evaluate(model, val_loader, criterion_bce=bce_loss, criterion_l1_linear=l1_loss, criterion_l1_mel=mel_l1_loss, device=device)
        
        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val BCE: {val_bce:.4f}, Val L1: {val_l1:.4f}"
        )

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoints_dir, f"chkp_{session_name}_epoch{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Log to CSV
        with open(log_file_path, mode='a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_bce, val_l1])

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