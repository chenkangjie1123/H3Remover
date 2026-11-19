import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import auto_remove
import download_weights
from pipeline.common import collect_instances, motion_fraction, validate_motion


class Predictor:
    def __init__(self, frames):
        self.frames = frames

    def handle_stream_request(self, request):
        return iter(self.frames)


def output(i, ids, masks):
    return {'frame_index': i, 'outputs': {'out_obj_ids': np.array(ids), 'out_binary_masks': masks}}


class MotionTests(unittest.TestCase):
    def collect(self, frames, probability, capacity=64):
        return collect_instances(Predictor(frames), 'session', probability, 4, 6,
            threshold=.5, fraction=.5, min_frames=2, min_pixels=2, max_objects=capacity)

    def test_select_whole_instance_across_pauses(self):
        masks = np.zeros((2, 4, 6), bool)
        masks[0, :, :2] = True
        masks[1, :, 4:] = True
        probability = np.zeros((3, 4, 6), np.float32)
        probability[:2, :, :2] = .9
        union, records = self.collect([output(i, [10, 20], masks) for i in range(3)], probability)
        np.testing.assert_array_equal(union, np.repeat(masks[0][None], 3, axis=0))
        self.assertTrue(records['10']['selected'])
        self.assertFalse(records['20']['selected'])

    def test_late_entry_empty_first_frame(self):
        masks = np.ones((1, 4, 6), bool)
        frames = [output(0, [], np.empty((0, 1, 4, 6)))]
        frames += [output(i, [3], masks[:, None]) for i in [1, 2]]
        union, _ = self.collect(frames, np.ones((3, 4, 6)))
        self.assertFalse(union[0].any())
        self.assertTrue(union[1:].all())

    def test_missing_frame_rejected(self):
        with self.assertRaises(RuntimeError):
            self.collect([output(0, [], np.array([]))], np.zeros((3, 4, 6)))

    def test_duplicate_frame_rejected(self):
        frames = [output(0, [], np.array([]))] * 2
        with self.assertRaises(ValueError):
            self.collect(frames, np.zeros((2, 4, 6)))

    def test_capacity_rejected(self):
        with self.assertRaises(RuntimeError):
            self.collect([output(0, [0, 1], np.ones((2, 4, 6)))], np.zeros((1, 4, 6)), 2)

    def test_bad_motion_rejected(self):
        for value in [np.full((3, 4, 6), np.nan), np.full((3, 4, 6), 1.1), np.zeros((3, 0, 6))]:
            with self.assertRaises(ValueError):
                validate_motion(value, 3)

    def test_noninteger_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.collect([output(0, [.5], np.ones((1, 4, 6)))], np.zeros((1, 4, 6)))

    def test_probability_resize_and_empty_mask(self):
        self.assertEqual(motion_fraction(np.ones((4, 6), bool), np.ones((2, 3)), .5), 1)
        self.assertEqual(motion_fraction(np.zeros((4, 6), bool), np.ones((2, 3)), .5), 0)


class DownloadTests(unittest.TestCase):
    def test_pinned_adapter_only_and_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            weight = Path(folder) / 'h3remover.safetensors'
            weight.write_bytes(b'test adapter')
            digest = hashlib.sha256(weight.read_bytes()).hexdigest()
            with patch('huggingface_hub.snapshot_download', return_value=folder) as fetch:
                with patch.object(download_weights, 'ADAPTER_SHA256', digest):
                    download_weights.download(folder, adapter_only=True)
                self.assertEqual(fetch.call_count, 1)
                self.assertEqual(fetch.call_args.kwargs['revision'], download_weights.ADAPTER_REVISION)
                with patch.object(download_weights, 'ADAPTER_SHA256', 'bad'):
                    with self.assertRaises(RuntimeError):
                        download_weights.download(folder, adapter_only=True)


class OrchestrationTests(unittest.TestCase):
    def test_prepare_only_runs_two_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.mp4'
            source.touch()
            output_dir = Path(folder) / 'result'
            argv = ['auto_remove.py', '--video', str(source), '--output-dir', str(output_dir),
                    '--pi4-python', sys.executable, '--sam-python', sys.executable, '--prepare-only']
            with patch.object(sys, 'argv', argv), patch('auto_remove.prepare'), \
                 patch('auto_remove.run_stage', return_value=1.5) as stage:
                auto_remove.main()
            self.assertEqual([call.args[0] for call in stage.call_args_list], ['motion', 'segment'])
            report = json.loads((output_dir / 'run.json').read_text())
            self.assertEqual(report['status'], 'masks_ready')
            self.assertEqual(report['stages_seconds']['segment'], 1.5)

    def test_failed_motion_never_starts_segmentation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input.mp4'
            source.touch()
            output_dir = Path(folder) / 'result'
            argv = ['auto_remove.py', '--video', str(source), '--output-dir', str(output_dir),
                    '--pi4-python', sys.executable, '--sam-python', sys.executable, '--prepare-only']
            with patch.object(sys, 'argv', argv), patch('auto_remove.prepare'), \
                 patch('auto_remove.run_stage', side_effect=RuntimeError('motion failure')) as stage:
                with self.assertRaises(RuntimeError):
                    auto_remove.main()
            self.assertEqual(stage.call_count, 1)
            self.assertEqual(json.loads((output_dir / 'run.json').read_text())['status'], 'failed')

    def test_shared_frame_clock(self):
        rgb = np.full((160, 10, 20, 3), 80, np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            with patch('auto_remove.read_video', return_value=(rgb, 30, 160)), \
                 patch('auto_remove.sha256', return_value='source'), patch('auto_remove.save_video') as save:
                auto_remove.prepare('input.mp4', Path(folder))
            meta = json.loads((Path(folder) / 'source.json').read_text())
            self.assertEqual(meta['source_frame_indices'], np.floor(np.arange(124)*30/24+.5).astype(int).tolist())
            prepared = save.call_args.args[1]
            self.assertEqual(prepared.shape, (124, 544, 960, 3))
            self.assertTrue((prepared[:, :32] == 0).all())

    def test_failed_stage_preserves_log(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            with self.assertRaises(RuntimeError):
                auto_remove.run_stage('test', [sys.executable, '-c', 'print("failure marker"); raise SystemExit(7)'],
                                      folder, folder, None)
            self.assertIn('failure marker', (folder / 'test.log').read_text())


if __name__ == '__main__':
    unittest.main()
