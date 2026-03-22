from models.DCUNet import DCUNet
from models.DenoiseUNetConformer import DenoiseUNetConformer


MODEL_REGISTRY = {
    "dcunet": DCUNet,
    "denoise_unet_conformer": DenoiseUNetConformer,
    "denoiseunet_conformer": DenoiseUNetConformer,
    "conformer": DenoiseUNetConformer,
    "conformer_unet": DenoiseUNetConformer,
    "corformer": DenoiseUNetConformer,
    "corformer_unet": DenoiseUNetConformer,
}


def build_model(model_architecture: str):
    key = model_architecture.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model architecture '{model_architecture}'. "
            f"Available options: {available}."
        )

    return MODEL_REGISTRY[key]()


__all__ = ["DCUNet", "DenoiseUNetConformer", "MODEL_REGISTRY", "build_model"]
