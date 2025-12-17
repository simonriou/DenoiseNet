import torch

def pad_collate(batch):
    """
    Pads variable-length tensors along the time dimension.

    Supported shapes:
    - features:    [C, F, T]   (real-valued)
    - mix_stft:    [F, T]      (complex)
    - clean_stft:  [F, T]      (complex)
    - clean_audio: [T]
    """

    collated = {}
    batch_size = len(batch)

    for key in batch[0].keys():

        # --------------------------------------------------
        # CNN inputs: [C, F, T]
        # --------------------------------------------------
        if key == "features":
            C, F, _ = batch[0][key].shape
            max_T = max(item[key].shape[-1] for item in batch)

            padded = torch.zeros(
                batch_size, C, F, max_T,
                dtype=batch[0][key].dtype
            )

            for i, item in enumerate(batch):
                T = item[key].shape[-1]
                padded[i, :, :, :T] = item[key]

            collated[key] = padded

        # --------------------------------------------------
        # Complex STFTs: [F, T]
        # --------------------------------------------------
        elif key in ["mix_stft", "clean_stft"]:
            F, _ = batch[0][key].shape
            max_T = max(item[key].shape[-1] for item in batch)

            padded = torch.zeros(
                batch_size, F, max_T,
                dtype=batch[0][key].dtype
            )

            for i, item in enumerate(batch):
                T = item[key].shape[-1]
                padded[i, :, :T] = item[key]

            collated[key] = padded

        # --------------------------------------------------
        # Waveforms: [T]
        # --------------------------------------------------
        elif key == "clean_audio":
            max_len = max(item[key].shape[-1] for item in batch)

            padded = torch.zeros(batch_size, max_len)

            for i, item in enumerate(batch):
                T = item[key].shape[-1]
                padded[i, :T] = item[key]

            collated[key] = padded

        # --------------------------------------------------
        # Fallback: stack
        # --------------------------------------------------
        else:
            collated[key] = torch.stack([item[key] for item in batch])

    return collated