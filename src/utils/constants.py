from pathlib import Path

"""
These are the constants used throughout the project.
"""

SAMPLE_RATE = 16000
TARGET_SNR = 0.0 # dB
MIN_DB_CLIP = 80.0

N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512

ROOT = Path(__file__).parent.parent.parent.resolve() # Root folder

DATA_TRAIN_DIR = ROOT / "data" / "train"
CLEAN_DIR = DATA_TRAIN_DIR / "speech"
NOISE_DIR = DATA_TRAIN_DIR / "noise"

DATA_TEST_DIR = ROOT / "data" / "test"
RAW_TEST_DIR = DATA_TEST_DIR / "raw" # Raw .wav files for testing
CLEAN_TEST_DIR = DATA_TEST_DIR / "speech" # .pt converted from .wav files
NOISE_TEST_DIR = DATA_TRAIN_DIR / "noise" # Using train noise for test as well
NOISE_ENHANCED_DIR = DATA_TEST_DIR / "enhanced" # Enhanced .wav files after denoising

MODEL_DIR = ROOT / "data" / "models"

LOG_DIR = ROOT / "experiments" / "logs"
CHECKPOINT_DIR = ROOT / "experiments" / "checkpoints"
MODEL_NAME = "waveform-4"
SAVE_DENOISED = True
SAVE_NOISY = True

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

LAMBDA = 1.0 # BCE
GAMMA = 0.8 # L1 Linear
OMEGA = 0.5 # L1 Mel
ZETA = 0.8 # L1 Waveform

N_MELS = 80
ALPHA = 10.0  # Weight for mel-scale L1 loss

PHASE_MODE = "raw"  # Options: "raw", "GL", "vocoder"
GL_ITERS = 64

DEBUG = False