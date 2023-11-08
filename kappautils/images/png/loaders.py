import einops
import numpy as np
import png
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from .constants import VIRIDIS_PALETTE_NP


def png_loader(path):
    with open(path, "rb") as f:
        return to_tensor(Image.open(f))


def png_loader_viridis(path):
    img = (png_loader(path) * 255).squeeze(0).long().numpy()
    rgb = np.take(VIRIDIS_PALETTE_NP, img, axis=0)
    return einops.rearrange(torch.from_numpy(rgb), "h w c -> c h w")


def png_loader_with_info(path):
    # might also be possible with Image.open (is possibly faster)
    r = png.Reader(filename=str(path))
    w, h, reader, info = r.read()
    data = []
    for row in reader:
        data.append(torch.frombuffer(row, dtype=torch.int8))
    data = torch.stack(data)
    return data.unsqueeze(0) / 255, info
