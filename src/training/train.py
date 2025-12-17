import os
import glob
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.constants import *
from training.dataset import SpeechNoiseDatasetTemporal
from models.ConvRNNTemporalDenoiser import ConvRNNTemporalDenoiser

# -----------------------------------------------------------
# Loss
# -----------------------------------------------------------

def waveform_mse_loss(pred: torch.Tensor, target: torch.Tensor):
    """
    Mean squared error on waveform: E[ |pred - target|^2 ]
    """
    return torch.mean((pred - target) ** 2)

# -----------------------------------------------------------
# Evaluation
# -----------------------------------------------------------

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            mixture = batch["mixture"].to(device).unsqueeze(-1)  # [B, T, 1]
            clean = batch["clean"].to(device).unsqueeze(-1)

            pred_clean = model(mixture)
            loss = waveform_mse_loss(pred_clean, clean)

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
    dataset = SpeechNoiseDatasetTemporal(
        clean_dir=CLEAN_DIR,
        noise_dir=NOISE_DIR,
        snr_db=TARGET_SNR,
        seq_len=SEQ_LEN  # e.g., 16000 samples for 1s segments
    )

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
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=(device.type == "cuda")
    )

    # 4. Model & Optimizer
    model = ConvRNNTemporalDenoiser().to(device)
    criterion = nn.MSELoss()
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
            mixture = batch["mixture"].to(device).unsqueeze(-1)
            clean = batch["clean"].to(device).unsqueeze(-1)

            optimizer.zero_grad()
            pred_clean = model(mixture)
            loss = criterion(pred_clean, clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
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