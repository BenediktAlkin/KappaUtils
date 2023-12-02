import torch


def xy_to_image(x, y, resolution, xmin=None, xmax=None, ymin=None, ymax=None, weights=None):
    coords = torch.stack([x, y], dim=1)
    if xmin is not None:
        assert xmax is not None
        coords_min = torch.tensor(xmin, xmax, device=x.device)
    else:
        coords_min = None
    if ymin is not None:
        assert ymax is not None
        coords_max = torch.tensor(xmin, xmax, device=x.device)
    else:
        coords_max = None
    return coords_to_image(
        coords=coords,
        resolution=resolution,
        coords_min=coords_min,
        coords_max=coords_max,
        weights=weights,
    )


def coords_to_image(coords, resolution, coords_min=None, coords_max=None, weights=None):
    if not torch.is_tensor(resolution):
        resolution = torch.tensor(resolution, device=coords.device)
    assert resolution.ndim == 1 and resolution.numel() == 2
    # rescale coords to resolution
    if coords_min is not None:
        if not torch.is_tensor(coords_min):
            coords_min = torch.tensor(coords_min, device=coords.device)
    else:
        coords_min = coords.min(dim=0).values
    assert coords_min.ndim == 1 and coords_min.numel() == 2
    if coords_max is not None:
        if not torch.is_tensor(coords_max):
            coords_max = torch.tensor(coords_max, device=coords.device)
    else:
        coords_max = coords.max(dim=0).values
    assert coords_max.ndim == 1 and coords_max.numel() == 2
    coords_max = coords_max / (resolution - 1)
    coords = (coords - coords_min.unsqueeze(0)) / coords_max.unsqueeze(0)
    coords = coords.round().long()
    # flatten coords (e.g. [4, 6] with resolution=(6, 8) will be flattened to 4 * 8 + 6)
    coords = coords[:, 0] * resolution[1] + coords[:, 1]
    # weighted counting of occourances
    img = torch.bincount(coords, weights=weights, minlength=resolution[0] * resolution[1])
    img = img.view(*resolution)
    return img
