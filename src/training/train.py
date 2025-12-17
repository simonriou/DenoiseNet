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

def custom_loss(x, y, a, b, lambda_, gamma_):
    # lambda BCE + gamma L1
    bce = nn.BCELoss()(x, y)
    l1 = nn.L1Loss()(a, b)
    return lambda_ * bce + gamma_ * l1


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            features = batch["features"].to(device)
            mix_mag = batch["mix_mag"].to(device)
            clean_mag = batch["clean_mag"].to(device)
            ibm = batch["ibm"].to(device)

            pred_mask = model(features)
            est_mag = pred_mask * mix_mag

            loss = criterion(pred_mask, ibm,est_mag, clean_mag)
            total_loss += loss.item()
        
    return total_loss / len(dataloader)

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
    criterion = lambda x, y, a, b: custom_loss(x, y, a, b, lambda_=LAMBDA, gamma_=GAMMA)
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
        writer.writerow(["epoch", "train_loss", "val_loss"])

    print("Starting Training...")
    
    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            features = batch["features"].to(device)
            mix_mag = batch["mix_mag"].to(device)
            clean_mag = batch["clean_mag"].to(device)
            ibm = batch["ibm"].to(device)

            pred_mask = model(features)
            est_mag = pred_mask * mix_mag

            loss = criterion(pred_mask, ibm, est_mag, clean_mag)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoints_dir, f"chkp_{session_name}_epoch{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Log to CSV
        with open(log_file_path, mode='a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])

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