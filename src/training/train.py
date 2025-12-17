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
Training script for complex-mask-based speech denoising.

The model now predicts a complex ratio mask M(f,t).
The enhanced STFT is obtained as: X_hat = M ⊙ Y. (⊙ denotes element-wise multiplication)
The loss is a complex L1 loss between X_hat and the clean STFT X. (to be upgraded with SI-SDR later)
"""

# -----------------------------------------------------------
# Loss
# -----------------------------------------------------------

def complex_l1_loss(X_hat: torch.Tensor, X: torch.Tensor):
    """
    Complex L1 loss: E[ |X_hat - X| ]
    """
    return torch.mean(torch.abs(X_hat - X))


# -----------------------------------------------------------
# Evaluation
# -----------------------------------------------------------

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)      # [B, 2, F, T]
            Y = batch["mix_stft"].to(device)              # complex
            X = batch["clean_stft"].to(device)            # complex

            pred_mask = model(features)                   # [B, 2, F, T]
            Mr = pred_mask[:, 0]
            Mi = pred_mask[:, 1]
            M = torch.complex(Mr, Mi)

            X_hat = M * Y
            loss = complex_l1_loss(X_hat, X)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


# -----------------------------------------------------------
# Training
# -----------------------------------------------------------

def train(session_name: str):

    # 1. Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # 2. Sanity checks
    os.makedirs(CLEAN_DIR, exist_ok=True)
    os.makedirs(NOISE_DIR, exist_ok=True)

    if not glob.glob(f"{CLEAN_DIR}/*.pt"):
        print("Error: No clean data found.")
        return

    # 3. Dataset
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
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate,
        pin_memory=(device.type == "cuda")
    )

    # 4. Model & Optimizer
    model = DenoiseUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Logging & checkpoints
    checkpoints_dir = os.path.join(CHECKPOINT_DIR, session_name)
    os.makedirs(checkpoints_dir, exist_ok=True)

    log_file_dir = os.path.join(LOG_DIR, session_name)
    os.makedirs(log_file_dir, exist_ok=True)
    log_file_path = os.path.join(log_file_dir, "training_log.csv")

    with open(log_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    # 6. Training loop
    print("Starting training...")

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            features = batch["features"].to(device)
            Y = batch["mix_stft"].to(device)
            X = batch["clean_stft"].to(device)

            optimizer.zero_grad()

            pred_mask = model(features)
            Mr = pred_mask[:, 0]
            Mi = pred_mask[:, 1]
            M = torch.complex(Mr, Mi)

            X_hat = M * Y
            loss = complex_l1_loss(X_hat, X)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # Save checkpoint
        checkpoint_path = os.path.join(
            checkpoints_dir, f"chkp_{session_name}_epoch{epoch}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)

        # Log
        with open(log_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])

        # Save final model
        if epoch == EPOCHS - 1:
            final_model_path = os.path.join(MODEL_DIR, f"{session_name}.pth")
            torch.save(model.state_dict(), final_model_path)

    print("Training complete.")


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    session_name = input("Enter a session name for this training run: ").strip()

    if not session_name:
        print("Session name cannot be empty.")
    elif os.path.exists(os.path.join(CHECKPOINT_DIR, session_name)):
        overwrite = input(
            f"Session '{session_name}' already exists. Overwrite? (y/n): "
        ).strip().lower()
        if overwrite == "y":
            train(session_name)
        else:
            print("Exiting without training.")
    else:
        train(session_name)