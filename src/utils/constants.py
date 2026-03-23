from pathlib import Path

"""
These are the constants used throughout the project.
"""

SAMPLE_RATE = 16000
TARGET_SNR = 0.0 # dB
TRAIN_SNR_RANGE_DB = (-5.0, 5.0) # dB
VAL_SNR_DB = TARGET_SNR
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
MODEL_NAME = "direct-3"
MODEL_ARCHITECTURE = "conformer"  # Options: "dcunet", "denoise_unet", "denoise_unet_conformer"
SAVE_DENOISED = True
SAVE_NOISY = True

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

LAMBDA = 0.5 # Complex L1
GAMMA = 0.8 # L1 Linear
OMEGA = 1.0 # L1 Mel
ZETA = 2.0 # L1 Waveform
ETA = 0.25 # Temporal continuity

N_MELS = 80
ALPHA = 10.0  # Weight for mel-scale L1 loss
INTELLIGIBILITY_LOSS_WEIGHT = 0.75
INTELLIGIBILITY_BAND_START_HZ = 1000.0
INTELLIGIBILITY_BAND_END_HZ = 4000.0
INTELLIGIBILITY_BAND_BOOST = 1.0
UNVOICED_FRAME_BOOST = 0.75
UNVOICED_VOICED_MAX_HZ = 350.0
UNVOICED_FRICATIVE_MIN_HZ = 2000.0
WAVEFORM_L1_WEIGHT = 0.2
SI_SDR_LOSS_WEIGHT = 0.8
MRSTFT_LOSS_WEIGHT = 0.6
TRAIN_RECON_MODE = "hybrid"
TRAIN_RECON_LOUDNESS_MODE = "none"
ENABLE_STOI_METRIC = True
ENABLE_PESQ_METRIC = True

PHASE_MODE = "hybrid"  # Options: "complex", "raw", "hybrid", "GL", "vocoder"
GL_ITERS = 64
RECON_MASK_SMOOTH_FREQ = 1
RECON_MASK_SMOOTH_TIME = 7
RECON_PHASE_BLEND_POWER = 1.5
RECON_MASK_CEILING = 1.0
RECON_LOUDNESS_MODE = "suppression"  # Options: "none", "output", "suppression"
RECON_LOUDNESS_BLEND = 0.1
RECON_LOUDNESS_MAX_STEP_DB = 3.0
VOCODER_SOURCE = "speechbrain/tts-hifigan-libritts-16kHz"
VOCODER_CACHE_DIR = ROOT / "pretrained_models" / "tts-hifigan-libritts-16kHz"

DEBUG = False
TEST_RANDOM_SEED = 42
