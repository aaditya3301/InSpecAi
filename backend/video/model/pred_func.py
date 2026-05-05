import io
from pathlib import Path

import torch


def _cpu_load_from_bytes(payload):
    return torch.load(io.BytesIO(payload), map_location=torch.device("cpu"), weights_only=False)


torch.storage._load_from_bytes = _cpu_load_from_bytes

device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.empty_cache()


def load_cvit(cvit_weight, net, fp16=False):
    if net != "cvit2":
        raise ValueError(f"Unsupported CViT model variant: {net}")

    from .cvit import CViT

    model = CViT(
        image_size=224,
        patch_size=7,
        num_classes=2,
        channels=512,
        dim=1024,
        depth=6,
        heads=8,
        mlp_dim=2048,
    )
    model.to(device)

    weight_path = Path(cvit_weight)
    if not weight_path.is_absolute():
        weight_path = Path(__file__).resolve().parents[1] / "weight" / cvit_weight

    checkpoint = torch.load(weight_path, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    model.eval()

    if fp16:
        model.half()

    return model
