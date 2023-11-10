import torch
import unittest
from kappautils.amp_utils import get_grad_scaler_and_autocast_context, NoopGradScaler, NoopContext


class TestAmpUtils(unittest.TestCase):
    def test(self):
        grad_scaler, autocast_ctx = get_grad_scaler_and_autocast_context(
            precision=torch.float32,
            device=torch.device("cpu"),
        )
        self.assertIsInstance(grad_scaler, NoopGradScaler)
        self.assertIsInstance(autocast_ctx, NoopContext)