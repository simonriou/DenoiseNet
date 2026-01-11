import torch

def magnitude_to_mel(mag, mel_fb):
    """
    Docstring for magnitude_to_mel
    
    :param mag: (B, 1, F, T)
    :param mel_fb: (F, M)

    returns:
        mel: (B, 1, M, T)
    """

    # (B, 1, T, F)
    x = mag.permute(0, 1, 3, 2)

    # (B, 1, T, M)
    mel = torch.matmul(x, mel_fb)

    # (B, 1, M, T)
    mel = mel.permute(0, 1, 3, 2)

    return mel