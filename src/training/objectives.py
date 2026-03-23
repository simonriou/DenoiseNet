import torch

from inference.classical_reconstruction import reconstruct_classical_waveform
from utils.constants import (
    HOP_LENGTH,
    INTELLIGIBILITY_BAND_BOOST,
    INTELLIGIBILITY_BAND_END_HZ,
    INTELLIGIBILITY_BAND_START_HZ,
    MRSTFT_LOSS_WEIGHT,
    N_FFT,
    PHASE_MODE,
    RECON_LOUDNESS_BLEND,
    RECON_LOUDNESS_MAX_STEP_DB,
    RECON_LOUDNESS_MODE,
    RECON_MASK_CEILING,
    RECON_MASK_SMOOTH_FREQ,
    RECON_MASK_SMOOTH_TIME,
    RECON_PHASE_BLEND_POWER,
    SAMPLE_RATE,
    SI_SDR_LOSS_WEIGHT,
    TRAIN_RECON_MODE,
    UNVOICED_FRICATIVE_MIN_HZ,
    UNVOICED_FRAME_BOOST,
    UNVOICED_VOICED_MAX_HZ,
    WAVEFORM_L1_WEIGHT,
    WIN_LENGTH,
)


def stacked_channels_to_complex(x):
    return x[:, 0] + 1j * x[:, 1]


def build_frame_mask(lengths, max_len, device):
    frame_ids = torch.arange(max_len, device=device).unsqueeze(0)
    return frame_ids < lengths.unsqueeze(1)


def build_audio_mask(lengths, max_len, device):
    sample_ids = torch.arange(max_len, device=device).unsqueeze(0)
    return sample_ids < lengths.unsqueeze(1)


def masked_mean(values, mask, eps=1e-8):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / (mask.sum() + eps)


def complex_l1_loss(pred, target, spec_mask):
    diff = torch.abs(pred.real - target.real) + torch.abs(pred.imag - target.imag)
    return masked_mean(diff, spec_mask.unsqueeze(1))


def linear_l1_loss(pred_mag, clean_mag, spec_mask):
    return masked_mean(
        torch.abs(pred_mag - clean_mag),
        spec_mask.unsqueeze(1).unsqueeze(1),
    )


def mel_l1_loss(pred_mag, clean_mag, mel_fb, spec_mask):
    pred_mel = torch.matmul(pred_mag.permute(0, 1, 3, 2), mel_fb)
    clean_mel = torch.matmul(clean_mag.permute(0, 1, 3, 2), mel_fb)
    return masked_mean(
        torch.abs(pred_mel - clean_mel),
        spec_mask.unsqueeze(1).unsqueeze(-1),
    )


def _frequency_axis(freq_bins, device):
    return torch.linspace(0.0, SAMPLE_RATE / 2.0, freq_bins, device=device)


def _frequency_weights(freq_bins, device):
    freqs = _frequency_axis(freq_bins, device)
    band_center = 0.5 * (INTELLIGIBILITY_BAND_START_HZ + INTELLIGIBILITY_BAND_END_HZ)
    band_width = max(INTELLIGIBILITY_BAND_END_HZ - INTELLIGIBILITY_BAND_START_HZ, 1.0)
    smooth_band = torch.exp(-0.5 * ((freqs - band_center) / (0.35 * band_width)) ** 2)
    return 1.0 + INTELLIGIBILITY_BAND_BOOST * smooth_band


def _unvoiced_frame_weights(clean_mag, spec_mask):
    magnitude = clean_mag.squeeze(1)
    freqs = _frequency_axis(magnitude.shape[1], clean_mag.device)
    voiced_mask = freqs <= UNVOICED_VOICED_MAX_HZ
    fricative_mask = freqs >= UNVOICED_FRICATIVE_MIN_HZ

    voiced_energy = magnitude[:, voiced_mask].mean(dim=1)
    fricative_energy = magnitude[:, fricative_mask].mean(dim=1)
    total_energy = magnitude.mean(dim=1)

    energy_floor = total_energy.mean(dim=1, keepdim=True) + 1e-8
    activity = torch.clamp(total_energy / energy_floor, 0.0, 1.5)
    unvoiced_score = fricative_energy / (fricative_energy + voiced_energy + 1e-8)
    unvoiced_score = torch.clamp((unvoiced_score - 0.35) / 0.65, 0.0, 1.0)

    weights = 1.0 + UNVOICED_FRAME_BOOST * unvoiced_score * activity
    return torch.where(spec_mask, weights, torch.zeros_like(weights))


def intelligibility_weighted_mag_loss(pred_mag, clean_mag, spec_mask):
    freq_weights = _frequency_weights(pred_mag.shape[2], pred_mag.device).view(1, 1, -1, 1)
    frame_weights = _unvoiced_frame_weights(clean_mag, spec_mask).unsqueeze(1).unsqueeze(1)
    mask = spec_mask.unsqueeze(1).unsqueeze(1).to(pred_mag.dtype)
    weight_map = freq_weights * frame_weights
    diff = torch.abs(pred_mag - clean_mag)
    return (diff * weight_map * mask).sum() / ((weight_map * mask).sum() + 1e-8)


def suppression_gain_continuity_loss(pred_mag, mix_mag, clean_mag, spec_mask):
    pred_env = torch.sqrt(torch.mean(pred_mag.squeeze(1) ** 2, dim=1) + 1e-8)
    mix_env = torch.sqrt(torch.mean(mix_mag.squeeze(1) ** 2, dim=1) + 1e-8)
    clean_env = torch.sqrt(torch.mean(clean_mag.squeeze(1) ** 2, dim=1) + 1e-8)

    log_gain = torch.log(pred_env + 1e-8) - torch.log(mix_env + 1e-8)
    delta = log_gain[:, 1:] - log_gain[:, :-1]

    pair_mask = spec_mask[:, 1:] & spec_mask[:, :-1]
    activity = torch.sqrt(clean_env[:, 1:] * clean_env[:, :-1])
    activity = activity / (activity.mean(dim=1, keepdim=True) + 1e-8)
    activity = torch.clamp(activity, 0.0, 2.0)

    return masked_mean(torch.abs(delta) * activity, pair_mask)


def masked_waveform_l1_loss(pred_audio, clean_audio, audio_mask):
    return masked_mean(torch.abs(pred_audio - clean_audio), audio_mask)


def si_sdr_loss(pred_audio, clean_audio, audio_mask):
    mask = audio_mask.to(pred_audio.dtype)
    valid = mask.sum(dim=1, keepdim=True) + 1e-8

    pred = pred_audio * mask
    clean = clean_audio * mask

    pred = pred - pred.sum(dim=1, keepdim=True) / valid
    clean = clean - clean.sum(dim=1, keepdim=True) / valid

    projection = (
        (pred * clean).sum(dim=1, keepdim=True) / ((clean ** 2).sum(dim=1, keepdim=True) + 1e-8)
    ) * clean
    noise = pred - projection

    ratio = (projection ** 2).sum(dim=1) / ((noise ** 2).sum(dim=1) + 1e-8)
    return torch.log1p(1.0 / (ratio + 1e-8)).mean()


def multi_resolution_stft_loss(pred_audio, clean_audio, audio_mask):
    masked_pred = pred_audio * audio_mask.to(pred_audio.dtype)
    masked_clean = clean_audio * audio_mask.to(clean_audio.dtype)
    total_loss = 0.0

    for n_fft, hop_length, win_length in ((256, 64, 256), (512, 128, 512), (1024, 256, 1024)):
        window = torch.hann_window(win_length, device=pred_audio.device)
        pred_spec = torch.stft(
            masked_pred,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        clean_spec = torch.stft(
            masked_clean,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )

        pred_mag = pred_spec.abs()
        clean_mag = clean_spec.abs()

        spectral_convergence = (
            torch.linalg.vector_norm((clean_mag - pred_mag).flatten(1), dim=1)
            / (torch.linalg.vector_norm(clean_mag.flatten(1), dim=1) + 1e-8)
        ).mean()
        log_mag = torch.mean(torch.abs(torch.log(pred_mag + 1e-8) - torch.log(clean_mag + 1e-8)))

        total_loss = total_loss + spectral_convergence + log_mag

    return total_loss / 3.0


def waveform_objective(reconstructed_audio, clean_audio, audio_mask):
    waveform_l1 = masked_waveform_l1_loss(reconstructed_audio, clean_audio, audio_mask)
    si_sdr = si_sdr_loss(reconstructed_audio, clean_audio, audio_mask)
    mrstft = multi_resolution_stft_loss(reconstructed_audio, clean_audio, audio_mask)

    total = (
        WAVEFORM_L1_WEIGHT * waveform_l1
        + SI_SDR_LOSS_WEIGHT * si_sdr
        + MRSTFT_LOSS_WEIGHT * mrstft
    )
    return total, waveform_l1, si_sdr, mrstft


def resolve_training_reconstruction_mode():
    mode = TRAIN_RECON_MODE or PHASE_MODE
    if mode in {"GL", "vocoder"}:
        return "hybrid"
    return mode


def reconstruct_batch_waveforms(
    pred_complex_norm,
    mix_complex_norm,
    mix_phase,
    mix_scale,
    audio_lengths,
    loudness_mode=None,
):
    mode = resolve_training_reconstruction_mode()
    device = pred_complex_norm.device
    window = torch.hann_window(WIN_LENGTH, device=device)
    max_len = int(audio_lengths.max().item())
    reconstructed = pred_complex_norm.real.new_zeros((pred_complex_norm.shape[0], max_len))
    effective_loudness_mode = RECON_LOUDNESS_MODE if loudness_mode is None else loudness_mode
    effective_loudness_blend = RECON_LOUDNESS_BLEND if effective_loudness_mode != "none" else 0.0

    for batch_idx in range(pred_complex_norm.shape[0]):
        enhanced_complex = pred_complex_norm[batch_idx] * mix_scale[batch_idx]
        mixture_complex = mix_complex_norm[batch_idx] * mix_scale[batch_idx]
        waveform = reconstruct_classical_waveform(
            mode,
            enhanced_complex,
            mixture_complex,
            mix_phase[batch_idx],
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=window,
            target_length=int(audio_lengths[batch_idx].item()),
            mask_smooth_freq=RECON_MASK_SMOOTH_FREQ,
            mask_smooth_time=RECON_MASK_SMOOTH_TIME,
            phase_blend_power=RECON_PHASE_BLEND_POWER,
            mask_ceiling=RECON_MASK_CEILING,
            loudness_mode=effective_loudness_mode,
            loudness_blend=effective_loudness_blend,
            loudness_max_step_db=RECON_LOUDNESS_MAX_STEP_DB,
        )
        reconstructed[batch_idx, : waveform.numel()] = waveform

    return reconstructed
