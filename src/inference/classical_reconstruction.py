import torch
import torch.nn.functional as F
import numpy as np


def _istft_waveform(
    complex_spec,
    *,
    n_fft,
    hop_length,
    win_length,
    window,
    target_length,
):
    return torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=target_length,
    )


def _frame_rms(magnitude):
    return torch.sqrt(torch.mean(magnitude ** 2, dim=0) + 1e-8)


def _limit_log_gain_delta(log_gain, max_step_db):
    max_step = max_step_db / 20.0 * np.log(10.0)

    forward = log_gain.clone()
    for idx in range(1, forward.numel()):
        lo = forward[idx - 1] - max_step
        hi = forward[idx - 1] + max_step
        forward[idx] = torch.clamp(forward[idx], lo, hi)

    backward = forward.clone()
    for idx in range(backward.numel() - 2, -1, -1):
        lo = backward[idx + 1] - max_step
        hi = backward[idx + 1] + max_step
        backward[idx] = torch.clamp(backward[idx], lo, hi)

    return 0.5 * (forward + backward)


def reconstruct_classical_waveform(
    mode,
    enhanced_complex,
    mix_complex,
    mix_phase,
    *,
    n_fft,
    hop_length,
    win_length,
    window,
    target_length,
    mask_smooth_freq=3,
    mask_smooth_time=5,
    phase_blend_power=1.5,
    mask_ceiling=1.0,
    loudness_mode="none",
    loudness_blend=0.0,
    loudness_max_step_db=3.0,
):
    if mode == "complex":
        return _istft_waveform(
            enhanced_complex,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            target_length=target_length,
        )

    if mode == "raw":
        enhanced_mag = enhanced_complex.abs()
        complex_spec = enhanced_mag * torch.exp(1j * mix_phase)
        return _istft_waveform(
            complex_spec,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            target_length=target_length,
        )

    if mode != "hybrid":
        raise ValueError(f"Unsupported classical reconstruction mode: {mode}")

    eps = 1e-8
    pred_mag = enhanced_complex.abs()
    mix_mag = mix_complex.abs()

    # Use the denoiser as a soft mask estimator, then smooth isolated TF spikes.
    mask = torch.clamp(pred_mag / (mix_mag + eps), 0.0, mask_ceiling)
    if mask_smooth_freq > 1 or mask_smooth_time > 1:
        mask = F.avg_pool2d(
            mask.unsqueeze(0).unsqueeze(0),
            kernel_size=(mask_smooth_freq, mask_smooth_time),
            stride=1,
            padding=(mask_smooth_freq // 2, mask_smooth_time // 2),
        ).squeeze(0).squeeze(0)
        mask = torch.clamp(mask, 0.0, mask_ceiling)

    refined_mag = mask * mix_mag

    if loudness_mode in {"output", "suppression"} and loudness_blend > 0.0:
        mix_env = _frame_rms(mix_mag)
        out_env = _frame_rms(refined_mag)

        if loudness_mode == "output":
            base_log_gain = torch.log(out_env + eps)
        else:
            base_log_gain = torch.log(out_env + eps) - torch.log(mix_env + eps)

        limited_log_gain = _limit_log_gain_delta(
            base_log_gain,
            max_step_db=loudness_max_step_db,
        )
        target_log_gain = (
            (1.0 - loudness_blend) * base_log_gain
            + loudness_blend * limited_log_gain
        )
        refined_mag = refined_mag * torch.exp(
            target_log_gain - base_log_gain
        ).unsqueeze(0)

    pred_unit = enhanced_complex / (pred_mag + eps)
    mix_unit = torch.exp(1j * mix_phase)

    # Trust the model phase more in high-confidence bins and the mixture phase
    # elsewhere to reduce broadband phase noise.
    alpha = mask.pow(phase_blend_power)
    phase = alpha * pred_unit + (1.0 - alpha) * mix_unit
    phase = phase / (phase.abs() + eps)

    complex_spec = refined_mag * phase
    return _istft_waveform(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        target_length=target_length,
    )
