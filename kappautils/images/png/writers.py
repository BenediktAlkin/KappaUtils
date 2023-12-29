from torchvision.transforms.functional import to_pil_image

from .constants import VIRIDIS_PALETTE_LIST


def png_writer_greyscale(img, fp, save_format=None):
    if img.ndim == 2:
        img = img.unsqueeze(0)
    assert img.ndim == 3 and img.size(0) == 1
    to_pil_image(img, mode="L").save(fp, format=save_format)


def png_writer_viridis(img, fp, save_format=None):
    if img.ndim == 2:
        img = img.unsqueeze(0)
    assert img.ndim == 3 and img.size(0) == 1
    img = to_pil_image(img, mode="L")
    img.putpalette(VIRIDIS_PALETTE_LIST)
    img.save(fp, format=save_format)
