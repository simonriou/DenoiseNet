from pathlib import Path
import inspect

import torch
import torch.nn.functional as F
import torchaudio

from utils.constants import (
    N_FFT,
    SAMPLE_RATE,
    VOCODER_CACHE_DIR,
    VOCODER_SOURCE,
)


VOCODER_N_MELS = 80
VOCODER_MEL_EPS = 1e-5


def _device_string(device):
    if device.index is None:
        return device.type
    return f"{device.type}:{device.index}"


def _patch_hf_hub_download_for_speechbrain():
    """
    SpeechBrain 1.0.x still passes `use_auth_token`, while newer versions of
    `huggingface_hub` renamed that argument to `token`. Newer hub versions also
    raise `RemoteEntryNotFoundError` for missing optional files, while
    SpeechBrain expects `ValueError` for the optional `custom.py`.
    """

    import huggingface_hub
    from huggingface_hub.errors import RemoteEntryNotFoundError

    if "use_auth_token" in inspect.signature(
        huggingface_hub.hf_hub_download
    ).parameters:
        return

    original_download = huggingface_hub.hf_hub_download

    def compat_hf_hub_download(*args, use_auth_token=None, **kwargs):
        if use_auth_token is not None and "token" not in kwargs:
            kwargs["token"] = use_auth_token
        try:
            return original_download(*args, **kwargs)
        except RemoteEntryNotFoundError as exc:
            filename = kwargs.get("filename")
            if filename is None and len(args) >= 2:
                filename = args[1]
            if filename == "custom.py":
                raise ValueError("Optional custom.py not found on the Hub.") from exc
            raise

    huggingface_hub.hf_hub_download = compat_hf_hub_download


class SpeechHiFiGANVocoder:
    """
    Wrapper around SpeechBrain's 16 kHz HiFi-GAN vocoder.

    The denoiser predicts a linear STFT magnitude, so we project it to the
    80-bin log-mel representation expected by the vocoder before decoding.
    """

    def __init__(
        self,
        device,
        source=VOCODER_SOURCE,
        cache_dir=VOCODER_CACHE_DIR,
        n_fft=N_FFT,
        n_mels=VOCODER_N_MELS,
        sample_rate=SAMPLE_RATE,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            from speechbrain.inference.vocoders import HIFIGAN
        except ImportError as exc:
            raise RuntimeError(
                "PHASE_MODE='vocoder' requires the public SpeechBrain HiFi-GAN "
                "package. Install `speechbrain` to enable neural-vocoder reconstruction."
            ) from exc

        _patch_hf_hub_download_for_speechbrain()

        self.mel_transform = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            f_min=0.0,
            f_max=sample_rate / 2,
            n_stft=n_fft // 2 + 1,
            norm="slaney",
            mel_scale="slaney",
        ).to(device)

        try:
            self.vocoder = HIFIGAN.from_hparams(
                source=source,
                savedir=str(self.cache_dir),
                run_opts={"device": _device_string(device)},
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the SpeechBrain HiFi-GAN vocoder checkpoint. "
                f"Expected to cache it under '{self.cache_dir}'."
            ) from exc

    def magnitude_to_log_mel(self, magnitude):
        if magnitude.dim() != 4 or magnitude.size(1) != 1:
            raise ValueError(
                "Expected magnitude spectrogram with shape [B, 1, F, T]."
            )

        mel = self.mel_transform(magnitude.squeeze(1).clamp_min(0.0))
        return torch.log(torch.clamp(mel, min=VOCODER_MEL_EPS))

    def decode(self, magnitude, target_length=None):
        log_mel = self.magnitude_to_log_mel(magnitude)
        waveform = self.vocoder.decode_batch(log_mel)

        if waveform.dim() == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)
        elif waveform.dim() != 2:
            raise RuntimeError(
                f"Unexpected vocoder output shape: {tuple(waveform.shape)}"
            )

        if target_length is not None:
            waveform = self._match_length(waveform, target_length)

        return waveform

    @staticmethod
    def _match_length(waveform, target_length):
        current_length = waveform.shape[-1]
        if current_length > target_length:
            return waveform[..., :target_length]
        if current_length < target_length:
            return F.pad(waveform, (0, target_length - current_length))
        return waveform
