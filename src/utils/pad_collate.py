import torch

"""
This function pads variable-length spectrograms in a batch to the maximum length in that batch.
It ensures that all tensors in the batch have the same time dimension by zero-padding shorter ones.
This is used for batching in the data loader when dealing with audio data of varying lengths.
"""

def pad_collate(batch):
    batch_size = len(batch)
    collated = {}

    # --- Spectrogram padding ---
    for key in [
        "features",
        "ibm",
        "mix_mag",
        "clean_mag",
        "mix_phase",
        "mix_complex",
        "clean_complex",
    ]:
        if batch[0].get(key) is None:
            collated[key] = None
            continue

        channels = batch[0][key].shape[0]
        freq_bins = batch[0][key].shape[1]
        max_spec_len = max(item[key].shape[2] for item in batch)

        padded = torch.zeros(
            batch_size, channels, freq_bins, max_spec_len,
            dtype=batch[0][key].dtype
        )

        for i, item in enumerate(batch):
            curr_len = item[key].shape[2]
            padded[i, :, :, :curr_len] = item[key]

        collated[key] = padded

    if batch[0].get("mix_scale") is not None:
        collated["mix_scale"] = torch.tensor([item["mix_scale"].item() for item in batch])

    # --- Waveform padding ---
    if batch[0].get("clean_audio") is not None:
        clean_wavs = [item["clean_audio"].view(-1) for item in batch]
        max_wav_len = max(wav.shape[0] for wav in clean_wavs)

        wav_padded = torch.zeros(batch_size, max_wav_len)

        for i, wav in enumerate(clean_wavs):
            wav_padded[i, :wav.shape[0]] = wav

        collated["clean_audio"] = wav_padded
    else:
        collated["clean_audio"] = None

    # --- Metadata (NO padding) ---
    if "filename" in batch[0]:
        collated["filename"] = [item["filename"] for item in batch]

    return collated