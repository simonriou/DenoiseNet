# DenoiseNet: A Complex-Valued U-Net Speech Denoising Model
## Overview
**A demo of the model's performances is available at the [docs site](https://simonriou.github.io/DenoiseNet/).**

- Model: **DCU-Net**, a complex-valued U-Net direct spectral mapping model trained to predict clean complex STFTs from noisy complex STFTs with multi-objective losses (complex L1, linear/mel magnitude losses, intelligibility weighting, continuity regularisation, and waveform objectives aligned with the hybrid decoder used at inference).
- Training corpus: ~2.5 hours of English speech mixed with babble noise at controlled SNR.
- Inference: streaming-ready; runs in (near) real time on a standard laptop CPU.
- Results: significant SNR improvements on test set; subjective quality gains observed. Near state-of-the-art performance among lightweight denoising models.
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
- Recommended: Python 3.10+ with `torch`, `torchaudio`, `speechbrain`, `numpy`, `scipy`, `tqdm`, `matplotlib` (optional for debug), plus optional `pystoi` and `pesq` if you want those validation metrics logged.
- Example (from repo root):
  - `python -m venv .venv && source .venv/bin/activate`
  - `pip install torch torchaudio speechbrain numpy scipy tqdm matplotlib`
  - Optional validation extras: `pip install pystoi pesq`

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
  - Training params (`EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, loss weights `LAMBDA`, `GAMMA`, `OMEGA`, `ZETA`, `ETA`, the train SNR range, intelligibility weighting controls, waveform-objective weights, and optional STOI/PESQ toggles).
  - Phase reconstruction (`PHASE_MODE` in {`complex`, `raw`, `hybrid`, `GL`, `vocoder`}), Griffin-Lim iterations, the hybrid mask/phase blending controls, and optional loudness-stabilisation settings for reducing frame-to-frame gain pumping. `TRAIN_RECON_MODE`/`TRAIN_RECON_LOUDNESS_MODE` let training reuse the same classical decoder while keeping the non-differentiable loudness limiter out of the backward pass.
  - Logging/output toggles (`SAVE_DENOISED`, `SAVE_NOISY`), checkpoint selection (`MODEL_NAME`), and architecture selection (`MODEL_ARCHITECTURE`).
  - Deterministic test mixing via `TEST_RANDOM_SEED`, which fixes the noisy mixture used during inference/evaluation so reconstruction A/B comparisons are repeatable.

## Training Procedure
- Working directory: `src` (module mode).
- Command: `python -m training.train`
- The script prompts for `session_name` (used to namespace logs and checkpoints).
- Internals (see [src/training/train.py](src/training/train.py)):
  - Dataset: [SpeechNoiseDataset](src/training/dataset.py) now mixes training samples across `TRAIN_SNR_RANGE_DB`, keeps validation deterministic at `VAL_SNR_DB`, and carries exact waveform/STFT lengths so padded regions can be masked out of the losses.
  - Dataloaders: fixed 85/15 train/val split with seed 42, padding via [utils/pad_collate.py](src/utils/pad_collate.py).
  - Model: selected via `MODEL_ARCHITECTURE`, with `dcunet` in [src/models/DCUNet.py](src/models/DCUNet.py), a direct-mapping baseline in [src/models/DenoiseUNet.py](src/models/DenoiseUNet.py), and a bottleneck-Conformer variant in [src/models/DenoiseUNetConformer.py](src/models/DenoiseUNetConformer.py). All predict denoised complex spectrograms directly.
  - Reconstruction-aware waveform loss: [src/training/objectives.py](src/training/objectives.py) rebuilds training audio with the same hybrid classical decoder used at inference, then applies masked time-domain L1, an SI-SDR surrogate, and multi-resolution STFT loss.
  - Spectral/intelligibility loss: the same helper file adds an intelligibility-weighted magnitude loss that emphasizes the 1-4 kHz speech band and frames that look more unvoiced/fricative-heavy, plus a continuity loss on frame-to-frame suppression gain.
  - Validation metrics: [src/training/metrics.py](src/training/metrics.py) logs SI-SDR, segmental SNR, and log-spectral distance every epoch, and will also log STOI/PESQ automatically when `pystoi`/`pesq` are installed.
  - Checkpoints: saved each epoch to [experiments/checkpoints/<session_name>](experiments/checkpoints). Final weights: [data/models/<session_name>.pth](data/models).
  - Logs: per-epoch CSV at [experiments/logs/<session_name>/training_log.csv](experiments/logs).
- To resume or continue with different hyperparameters, edit `constants.py`, keep the same `session_name` if you wish to append logs, or choose a new one to avoid overwrite.

## Inference Pipeline
- Working directory: `src` (module mode).
- Ensure `MODEL_NAME` in [src/utils/constants.py](src/utils/constants.py#L26-L40) points to a weight file in [data/models](data/models) (e.g., `waveform-3.pth`) and that `MODEL_ARCHITECTURE` matches the checkpoint you want to load.
- Ensure test `.pt` speech files exist in [data/test/speech](data/test/speech); use the conversion script if starting from `.wav`.
- Command: `python -m inference.inference`
- Internals (see [src/inference/inference.py](src/inference/inference.py#L1-L170)):
	- Loads `SpeechNoiseDataset` in `test` mode (adds filenames), batch size 1 with padding.
	- Predicts a denoised complex spectrogram directly and, by default, reconstructs it with a hybrid classical path (`PHASE_MODE='hybrid'`) that smooths the estimated mask, blends predicted phase with mixture phase, and lightly stabilises frame-to-frame suppression gain.
	- Direct complex iSTFT, mixture-phase (`raw`), Griffin-Lim (`GL`), and the optional neural vocoder path remain available for comparison.
	- Saves enhanced (and optionally noisy) audio to [data/test/enhanced](data/test/enhanced) and logs SNR per file to [experiments/logs/<MODEL_NAME>/inference_snr_log.csv](experiments/logs).
	- Reports per-file and average inference time.

## Reproducibility Notes
- Randomness: validation split uses a fixed seed (42). Test-time noisy mixtures are also deterministic through `TEST_RANDOM_SEED`, so repeated inference runs compare the same noisy inputs.
- Data: training uses the provided `.pt` tensors; ensure any new data follows the same int16 tensor convention and sample rate (`SAMPLE_RATE` in constants).
- Hyperparameters and architecture: fully specified in [src/utils/constants.py](src/utils/constants.py#L1-L60) and the model definitions under [src/models](src/models).
- Artifacts: checkpoints and logs are versioned by session/model names; retain these along with the exact `constants.py` snapshot to reproduce results.

## Troubleshooting
- Import errors (e.g., `No module named utils`): run commands from `src` or set `PYTHONPATH=src`.
- Missing data: verify `.pt` files in [data/train/speech](data/train/speech) and [data/train/noise](data/train/noise); for test, populate [data/test/raw](data/test/raw) and reconvert.
- Vocoder errors: install `speechbrain` and allow the first inference run to download the public `speechbrain/tts-hifigan-libritts-16kHz` checkpoint into `pretrained_models/`.
- macOS/Conda OpenMP errors (`libomp.dylib already initialized`): run inference with `KMP_DUPLICATE_LIB_OK=TRUE` in the environment before `python -m inference.inference`.

## Citing
If you build on this work, please cite the repository and describe DenoiseNet as “a complex-valued U-Net speech denoising model trained with direct spectral mapping and combined complex, linear/mel L1, and waveform losses.”
