import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.video import annotate, load_inputs


class VideoTests(unittest.TestCase):
    def test_overlay_preserves_background(self):
        rgb = np.full((2, 20, 40, 3), 100, np.uint8)
        mask = np.zeros((2, 20, 40), bool)
        mask[:, 4:16, 10:30] = True
        result = annotate(rgb, mask)
        np.testing.assert_array_equal(result[~mask], rgb[~mask])
        np.testing.assert_array_equal(result[0, 4, 10], [255, 255, 255])
        self.assertGreater(int(result[0, 10, 20, 0]), 100)
        np.testing.assert_array_equal(rgb, 100)

    def test_mask_and_source_share_canvas(self):
        rgb = np.full((124, 10, 20, 3), 100, np.uint8)
        mask = np.zeros_like(rgb)
        mask[:, 3:8, 5:15] = 255
        with patch('pipeline.video.read_video', side_effect=[(rgb, 24., 124), (mask, 24., 124)]):
            source, reference = load_inputs('video', 'mask')
        self.assertEqual(source.shape, (124, 544, 960, 3))
        np.testing.assert_array_equal(source[:, :32], 0)
        np.testing.assert_array_equal(reference[:, :32], 0)
        np.testing.assert_array_equal(source[:, 100, 100], reference[:, 100, 100])
        self.assertTrue(np.any(source[:, 300, 400] != reference[:, 300, 400]))

    def test_reject_mismatched_clock(self):
        data = np.zeros((124, 10, 20, 3), np.uint8)
        with patch('pipeline.video.read_video', side_effect=[(data, 24., 124), (data, 25., 124)]):
            with self.assertRaises(ValueError):
                load_inputs('video', 'mask')

    def test_reject_short_video(self):
        data = np.zeros((20, 10, 20, 3), np.uint8)
        with patch('pipeline.video.read_video', side_effect=[(data, 24., 20), (data, 24., 20)]):
            with self.assertRaises(ValueError):
                load_inputs('video', 'mask')


if __name__ == '__main__':
    unittest.main()
