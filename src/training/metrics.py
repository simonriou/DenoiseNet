import math

import torch

from utils.constants import ENABLE_PESQ_METRIC, ENABLE_STOI_METRIC, HOP_LENGTH, N_FFT, SAMPLE_RATE, WIN_LENGTH

try:
    from pystoi import stoi as stoi_metric_fn
except ImportError:
    stoi_metric_fn = None

try:
    from pesq import pesq as pesq_metric_fn
except ImportError:
    pesq_metric_fn = None


def si_sdr_metric(reference, estimate):
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()

    projection = (
        torch.dot(estimate, reference) / (torch.dot(reference, reference) + 1e-8)
    ) * reference
    noise = estimate - projection
    ratio = torch.dot(projection, projection) / (torch.dot(noise, noise) + 1e-8)
    return 10.0 * torch.log10(ratio + 1e-8).item()


def segmental_snr(reference, estimate, frame_length=512, hop_length=256):
    if reference.numel() < frame_length:
        reference = torch.nn.functional.pad(reference, (0, frame_length - reference.numel()))
        estimate = torch.nn.functional.pad(estimate, (0, frame_length - estimate.numel()))

    ref_frames = reference.unfold(0, frame_length, hop_length)
    est_frames = estimate.unfold(0, frame_length, hop_length)
    noise_frames = est_frames - ref_frames

    signal_power = ref_frames.pow(2).mean(dim=1)
    noise_power = noise_frames.pow(2).mean(dim=1)
    snr = 10.0 * torch.log10(signal_power / (noise_power + 1e-8) + 1e-8)
    snr = torch.clamp(snr, min=-10.0, max=35.0)
    return snr.mean().item()


def log_spectral_distance(reference, estimate):
    window = torch.hann_window(WIN_LENGTH, device=reference.device)
    ref_spec = torch.stft(
        reference,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
    )
    est_spec = torch.stft(
        estimate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
    )

    ref_log = torch.log(ref_spec.abs() + 1e-8)
    est_log = torch.log(est_spec.abs() + 1e-8)
    return torch.sqrt(torch.mean((ref_log - est_log) ** 2)).item()


def maybe_stoi(reference, estimate):
    if not ENABLE_STOI_METRIC or stoi_metric_fn is None:
        return None

    ref_np = reference.detach().cpu().numpy()
    est_np = estimate.detach().cpu().numpy()
    return float(stoi_metric_fn(ref_np, est_np, SAMPLE_RATE, extended=False))


def maybe_pesq(reference, estimate):
    if not ENABLE_PESQ_METRIC or pesq_metric_fn is None:
        return None

    ref_np = reference.detach().cpu().numpy()
    est_np = estimate.detach().cpu().numpy()
    try:
        return float(pesq_metric_fn(SAMPLE_RATE, ref_np, est_np, "wb"))
    except Exception:
        return None


def batch_validation_metrics(reference_batch, estimate_batch, audio_lengths):
    totals = {
        "si_sdr": 0.0,
        "seg_snr": 0.0,
        "lsd": 0.0,
    }
    counts = {key: 0 for key in totals}
    optional_totals = {"stoi": 0.0, "pesq": 0.0}
    optional_counts = {key: 0 for key in optional_totals}

    for sample_idx, length in enumerate(audio_lengths.tolist()):
        reference = reference_batch[sample_idx, :length]
        estimate = estimate_batch[sample_idx, :length]

        totals["si_sdr"] += si_sdr_metric(reference, estimate)
        counts["si_sdr"] += 1
        totals["seg_snr"] += segmental_snr(reference, estimate)
        counts["seg_snr"] += 1
        totals["lsd"] += log_spectral_distance(reference, estimate)
        counts["lsd"] += 1

        stoi_value = maybe_stoi(reference, estimate)
        if stoi_value is not None and math.isfinite(stoi_value):
            optional_totals["stoi"] += stoi_value
            optional_counts["stoi"] += 1

        pesq_value = maybe_pesq(reference, estimate)
        if pesq_value is not None and math.isfinite(pesq_value):
            optional_totals["pesq"] += pesq_value
            optional_counts["pesq"] += 1

    metrics = {
        key: totals[key] / max(counts[key], 1)
        for key in totals
    }
    metrics["stoi"] = (
        optional_totals["stoi"] / optional_counts["stoi"]
        if optional_counts["stoi"] > 0 else None
    )
    metrics["pesq"] = (
        optional_totals["pesq"] / optional_counts["pesq"]
        if optional_counts["pesq"] > 0 else None
    )
    metric_counts = {**counts, **optional_counts}
    return metrics, metric_counts
