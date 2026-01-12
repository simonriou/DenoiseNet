# DenoiseNet: A U-Net Mask-Based Speech Denoising Model
## Overview
- Model: **DenoiseNet**, a U-Net mask estimator trained on log-magnitude STFT features with multi-objective losses (BCE on IBM, L1 on linear and mel magnitudes, waveform L1).
- Training script: [src/training/train.py](src/training/train.py).
- Inference script: [src/inference/inference.py](src/inference/inference.py).
- Configuration centralised in [src/utils/constants.py](src/utils/constants.py#L1-L60).

## Repository Layout
- Data: [data/train](data/train) (speech `.pt`, noise `.pt`), [data/test](data/test) (raw `.wav`, converted speech `.pt`, enhanced outputs), pre-trained weights in [data/models](data/models).
- Source: [src](src) with subpackages `training`, `inference`, `models`, `utils`.
- Experiments: [experiments/logs](experiments/logs) (CSV logs) and [experiments/checkpoints](experiments/checkpoints) (epoch checkpoints).
- Scripts: [scripts/convert_wav.py](scripts/convert_wav.py) for test-time WAV→PT conversion.
- Docs site (static demo): [docs](docs).

## Environment Setup
- Recommended: Python 3.10+ with `torch`, `torchaudio`, `speechbrain`, `numpy`, `scipy`, `tqdm`, `matplotlib` (optional for debug).
- Example (from repo root):
	- `python -m venv .venv && source .venv/bin/activate`
	- `pip install torch torchaudio numpy scipy tqdm matplotlib`

## Why Run as Modules from `src`
- Imports use package-style paths (e.g., `from utils.constants import *` in [src/training/train.py](src/training/train.py#L1-L20)). Running as a module from inside `src` ensures Python resolves these packages without manual `PYTHONPATH` edits.
- Commands (run from the `src` directory):
	- Training: `python -m training.train`
	- Inference: `python -m inference.inference`
- If you prefer running from the repo root, set `PYTHONPATH=src` (e.g., `PYTHONPATH=src python -m training.train`).

## Data Preparation
- Training expects int16 tensors: `data/train/speech/*.pt` and `data/train/noise/*.pt`. Files provided in the repository follow this format.
- Test audio conversion: place `.wav` files in [data/test/raw](data/test/raw) and run `python -m scripts.convert_wav` from the repo root. Converted `.pt` files are written to [data/test/speech](data/test/speech).
- Mel filterbank: precomputed at [src/training/mel_fb_512_80_16000.pt](src/training/mel_fb_512_80_16000.pt). If you change FFT or mel settings, regenerate with [src/utils/create_filterbank.py](src/utils/create_filterbank.py).

## Configuring Hyperparameters
- Edit [src/utils/constants.py](src/utils/constants.py#L1-L60) to change:
	- Data paths (`ROOT`, `CLEAN_DIR`, `NOISE_DIR`, test directories).
	- STFT params (`N_FFT`, `HOP_LENGTH`, `WIN_LENGTH`, `N_MELS`).
	- Training params (`EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, loss weights `LAMBDA`, `GAMMA`, `OMEGA`, `ZETA`, mel weight `ALPHA`).
	- Phase reconstruction (`PHASE_MODE` in {`raw`, `GL`, `vocoder`} and `GL_ITERS`).
	- Logging/output toggles (`SAVE_DENOISED`, `SAVE_NOISY`) and model selection (`MODEL_NAME`).

## Training Procedure
- Working directory: `src` (module mode).
- Command: `python -m training.train`
- The script prompts for `session_name` (used to namespace logs and checkpoints).
- Internals (see [src/training/train.py](src/training/train.py#L62-L190)):
	- Dataset: [SpeechNoiseDataset](src/training/dataset.py) mixes clean and noise at `TARGET_SNR`, computes log-magnitude features, IBM labels, and phases.
	- Dataloaders: random 85/15 train/val split with seed 42, padding via [utils/pad_collate.py](src/utils/pad_collate.py).
	- Model: DenoiseNet implementation in [models/DenoiseUNet.py](src/models/DenoiseUNet.py) predicting time-frequency masks.
	- Loss: normalized multi-term loss combining BCE, linear L1, mel L1, and waveform L1.
	- Checkpoints: saved each epoch to [experiments/checkpoints/<session_name>](experiments/checkpoints). Final weights: [data/models/<session_name>.pth](data/models).
	- Logs: per-epoch CSV at [experiments/logs/<session_name>/training_log.csv](experiments/logs).
- To resume or continue with different hyperparameters, edit `constants.py`, keep the same `session_name` if you wish to append logs, or choose a new one to avoid overwrite.

## Inference Pipeline
- Working directory: `src` (module mode).
- Ensure `MODEL_NAME` in [src/utils/constants.py](src/utils/constants.py#L26-L40) points to a weight file in [data/models](data/models) (e.g., `waveform-3.pth`).
- Ensure test `.pt` speech files exist in [data/test/speech](data/test/speech); use the conversion script if starting from `.wav`.
- Command: `python -m inference.inference`
- Internals (see [src/inference/inference.py](src/inference/inference.py#L1-L150)):
	- Loads `SpeechNoiseDataset` in `test` mode (adds filenames), batch size 1 with padding.
	- Predicts mask, reconstructs magnitude; phase via `PHASE_MODE` (`raw` uses mixture phase and `torch.istft`, `GL` uses Griffin-Lim, `vocoder` placeholder not implemented).
	- Saves enhanced (and optionally noisy) audio to [data/test/enhanced](data/test/enhanced) and logs SNR per file to [experiments/logs/<MODEL_NAME>/inference_snr_log.csv](experiments/logs).
	- Reports per-file and average inference time.

## Reproducibility Notes
- Randomness: validation split uses a fixed seed (42); other loaders follow PyTorch defaults (set `torch.manual_seed` externally if stricter determinism is required).
- Data: training uses the provided `.pt` tensors; ensure any new data follows the same int16 tensor convention and sample rate (`SAMPLE_RATE` in constants).
- Hyperparameters and architecture: fully specified in [src/utils/constants.py](src/utils/constants.py#L1-L60) and [src/models/DenoiseUNet.py](src/models/DenoiseUNet.py).
- Artifacts: checkpoints and logs are versioned by session/model names; retain these along with the exact `constants.py` snapshot to reproduce results.

## Troubleshooting
- Import errors (e.g., `No module named utils`): run commands from `src` or set `PYTHONPATH=src`.
- Missing data: verify `.pt` files in [data/train/speech](data/train/speech) and [data/train/noise](data/train/noise); for test, populate [data/test/raw](data/test/raw) and reconvert.
- Phase mode errors: `PHASE_MODE='vocoder'` is not implemented; use `raw` or `GL`.

## Citing
If you build on this work, please cite the repository and describe DenoiseNet as “a U-Net mask-based speech denoising model trained with combined BCE, linear/mel L1, and waveform losses.”