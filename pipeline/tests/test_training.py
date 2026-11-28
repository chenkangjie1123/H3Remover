import importlib.util
import unittest

import numpy as np

from pipeline.training.prepare import condition
from pipeline.training.rewards import full_reward, hps_indices


class FullRewardTests(unittest.TestCase):
    def test_scores_and_unknown(self):
        n, h, w = 124, 64, 80
        depth = np.broadcast_to(np.linspace(0, 1, w, dtype=np.float32), (n, h, w))
        masks = np.zeros((2, n, h, w), bool)
        masks[0, :, 4:12, 10:18] = True
        masks[1, :, 8:16, 34:50] = True
        eligible = np.arange(n) < 60
        valid = np.ones_like(depth, bool)
        rgb = np.zeros((*depth.shape, 3), np.uint8)
        # Only the smaller of the two instances is removed.
        edited_depth = depth - 0.25 * masks[0].astype(np.float32)
        mu = np.zeros(8)
        edit = rgb.copy()
        edit[:, 20:40] = 255
        protected = valid.copy()
        protected[:, 20:40] = False

        def measure(protect, d=edited_depth):
            return full_reward(
                rgb, edit, masks, valid, eligible, depth, d, protect, mu, mu + 2
            )

        result = measure(protected)
        self.assertAlmostEqual(result["T"], 0.5 * 60 / 124)
        self.assertEqual(result["C"], 0)
        self.assertEqual(result["E"], 0)
        self.assertAlmostEqual(result["reward"], 0.5 * 60 / 124 + 0.1)
        self.assertLess(measure(valid)["reward"], result["reward"])
        self.assertIsNone(measure(~valid))
        self.assertIsNone(measure(valid, np.zeros_like(depth)))
        self.assertEqual(hps_indices(eligible).tolist(), [0, 8, 17, 25, 34, 42, 51, 59])

    def test_touching_instances_keep_separate_outlines(self):
        rgb = np.zeros((1, 20, 40, 3), np.uint8)
        masks = np.zeros((2, 1, 20, 40), bool)
        masks[0, 0, 2:18, 3:15] = True
        masks[1, 0, 2:18, 15:36] = True
        result = condition(rgb, masks)
        self.assertTrue((result[0, 8, 14:16] == 255).all())
        self.assertTrue((result[0, 0] == 0).all())
        self.assertTrue((rgb == 0).all())


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("visual_rl"),
    "requires PyTorch and DiffusionART",
)
class ARTTests(unittest.TestCase):
    def test_replay_updates_only_generated_rows(self):
        import torch
        from visual_rl.algorithms.covariance import art_scalar_covariance

        from pipeline.training.__main__ import COV, replay_window

        class Toy:
            def __init__(self):
                self.model = torch.nn.Module()
                self.model.velocity = torch.nn.Parameter(torch.zeros(1, 3, 2))

            def replay(self, cache):
                return (self.model.velocity,)

        backend = Toy()
        ends = torch.arange(4.0).view(4, 1, 1).expand(-1, 2, 2)
        cov = art_scalar_covariance(
            ends, torch.arange(4.0), torch.zeros(4, dtype=torch.long), 1, **COV
        )
        cache = dict(
            condition_rows=1, old_video=torch.zeros(1, 3, 2), sigma=0.6, dt=-0.2
        )
        replay_window(
            backend,
            [{"cache": [cache] * 4}] * 2,
            cov["group_directions"],
            cov["group_guidance_magnitudes"],
        )
        grad = backend.model.velocity.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertEqual(grad[:, 0].abs().sum(), 0)
        self.assertGreater(grad[:, 1:].abs().sum(), 0)
        torch.optim.AdamW(backend.model.parameters(), lr=0.001, weight_decay=0).step()
        self.assertGreater(backend.model.velocity[:, 1:].abs().sum(), 0)
