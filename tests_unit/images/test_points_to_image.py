import unittest

import torch

from kappautils.images.points_to_image import xy_to_image


class TestPointsToImage(unittest.TestCase):
    def test_shape(self):
        rng = torch.Generator().manual_seed(8493)
        x = torch.rand(100, generator=rng) * 4
        y = torch.rand(100, generator=rng) * 2
        weights = torch.rand(100, generator=rng)
        img = xy_to_image(
            x=x,
            y=y,
            weights=weights,
            resolution=(32, 16)
        )
        self.assertEqual((32, 16), img.shape)