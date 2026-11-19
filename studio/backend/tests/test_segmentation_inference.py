"""Numerical and checkpoint compatibility tests for independent inference."""
import json
from pathlib import Path
import tempfile
import time
import unittest

import nibabel as nib
import numpy as np
import torch

from studio_core.classification_data import inspect_dataset
from studio_core.classification_runtime import run_supervised
from studio_core.segmentation_templates import segmentation_graph
from studio_core.segmentation_inference import _write_views, run_segmentation_inference, sliding_mask, window_starts
from test_segmentation import config_for


def _report_ready_worker(graph, config, messages, stop, artifact_directory):
    directory = Path(artifact_directory)
    directory.mkdir()
    (directory / 'inference.pending.json').write_text('{}', encoding='utf-8')
    stop.wait(10)
    messages.put({'type': 'result', 'result': {'status': 'succeeded', 'artifacts': {
        'inference': str(directory / 'inference.json')}}})


class SegmentationInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sliding_window_overlap_matches_full_accumulator_reference(self):
        native = np.arange(7 * 3 * 11, dtype=np.float32).reshape(1, 7, 3, 11) / 230
        window = (4, 5, 6)
        overlap = .5
        padded = np.pad(native, ((0, 0), (0, 0), (0, 2), (0, 0)), mode='edge')

        def predict(patch):
            # Vary the logits by tile and by a local pixel coordinate.
            return np.stack((.4 - patch[0] + patch.mean(), patch[0] + np.arange(6, dtype=np.float32) / 10))

        axes = [np.maximum(.1, 1 - np.abs(np.linspace(-1, 1, width, dtype=np.float32))) for width in window]
        weights = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
        sums = np.zeros((2, 7, 5, 11), dtype=np.float32)
        self.assertEqual(window_starts(11, 6, .5), (0, 3, 5))
        for depth in (0, 2, 3):
            for row in (0,):
                for col in (0, 3, 5):
                    section = (slice(depth, depth + 4), slice(row, row + 5), slice(col, col + 6))
                    sums[(slice(None), *section)] += predict(padded[(slice(None), *section)]) * weights
        expected = sums[:, :, :3].argmax(0)
        observed = sliding_mask(native, window, 2, overlap, predict, lambda: None)
        np.testing.assert_array_equal(observed, expected)
        self.assertEqual(observed.shape, (7, 3, 11))
        # A 2D case forces an early slab flush and an extra end-aligned tile.
        constant = np.ones((1, 13, 11), dtype=np.float32)
        observed = sliding_mask(constant, (5, 6), 2, overlap, lambda tile: np.stack((tile[0] - 2, tile[0])), lambda: None)
        np.testing.assert_array_equal(observed, np.ones((13, 11), dtype=np.uint8))

    def test_downscaled_slice_preserves_tiny_foreground_region(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'view.npy'
            image = np.zeros((1, 512, 512), dtype=np.float32)
            mask = np.zeros((512, 512), dtype=np.uint8)
            mask[0, 0] = 1
            _write_views(path, image, mask, lambda: None)
            preview = np.load(path, allow_pickle=False)
            self.assertEqual(preview.shape, (1, 224, 224, 4))
            self.assertEqual(int(preview[..., 3].sum()), 1)

    def test_cancellation_wins_after_report_is_prepared(self):
        from server import RunManager

        with tempfile.TemporaryDirectory() as temp:
            manager = RunManager(temp, worker_target=_report_ready_worker)
            try:
                job = manager.submit_segmentation_inference({'run_spec': 'unused', 'source': 'unused'})
                directory = Path(temp) / job['run_id']
                staging = directory / 'inference.pending.json'
                deadline = time.monotonic() + 15
                while not staging.is_file() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(staging.exists())
                manager.cancel(job['run_id'])
                deadline = time.monotonic() + 10
                while manager.get(job['run_id'])['status'] not in ('succeeded', 'failed', 'cancelled') and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertEqual(manager.get(job['run_id'])['status'], 'cancelled')
                self.assertFalse(staging.exists())
                self.assertFalse((directory / 'inference.json').exists())
            finally:
                manager.close()

    def test_fixed_grid_matches_s8_predictions_and_uses_only_trained_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _ = config_for(root)
            manifest = inspect_dataset(config['data'], config['seed'])
            result = run_supervised(segmentation_graph(manifest), config, root / 'trained')
            options = {'run_spec': result['artifacts']['run_spec'], 'source': str(root / 'image3.png')}
            Path(config['data']['path']).unlink()
            (root / 'mask3.png').unlink()
            real = run_segmentation_inference(options, root / 'infer')
            report = json.loads(Path(real['artifacts']['inference']).read_text(encoding='utf-8'))
            prediction = root / 'infer' / report['cases'][0]['mask_file']
            self.assertEqual(prediction.read_bytes(), (root / 'trained/masks/000001.png').read_bytes())
            self.assertEqual(report['cases'][0]['source_shape'], [11, 13])
            self.assertEqual(report['cases'][0]['tile_count'], 1)
            self.assertEqual(report['mask_values'], [0, 255])
            self.assertRegex(report['cases'][0]['source_sha256'], r'^[0-9a-f]{64}$')
            with self.assertRaisesRegex(ValueError, 'two files|mismatch|identity|checkpoint'):
                run_segmentation_inference({**options, 'checkpoint': options['run_spec']}, root / 'bad')
            state = torch.load(result['artifacts']['best_checkpoint'], map_location='cpu', weights_only=True)
            state['model_digest'] = '0' * 64
            mismatch = root / 'mismatch.pt'
            torch.save(state, mismatch)
            with self.assertRaisesRegex(ValueError, 'Checkpoint identity'):
                run_segmentation_inference({**options, 'checkpoint': str(mismatch)}, root / 'bad')

    def test_nifti_batch_preserves_affine_and_preview_slice_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, affine = config_for(root, volume=True)
            manifest = inspect_dataset(config['data'], config['seed'])
            result = run_supervised(segmentation_graph(manifest), config, root / 'trained')
            source = root / 'unlabeled'
            source.mkdir()
            for name, count in [('first', 7), ('second', 9)]:
                intensities = np.broadcast_to(np.arange(count, dtype=np.float32),(14, 15, count)).copy()
                image = nib.Nifti1Image(intensities, affine)
                image.header.set_xyzt_units('mm')
                nib.save(image, source / (name + ('.nii' if name == 'first' else '.nii.gz')))
            (source / 'notes.txt').write_text('unlabeled', encoding='utf-8')
            options = {'run_spec': result['artifacts']['run_spec'], 'source': str(source), 'mode': 'sliding', 'overlap': .5}
            real = run_segmentation_inference(options, root / 'infer')
            report = json.loads(Path(real['artifacts']['inference']).read_text(encoding='utf-8'))
            self.assertEqual(len(report['cases']), 2)
            for case in report['cases']:
                predicted = nib.load(root / 'infer' / case['mask_file'])
                self.assertEqual(list(predicted.shape[::-1]), case['source_shape'])
                np.testing.assert_allclose(predicted.affine, affine)
                self.assertEqual(predicted.header.get_xyzt_units()[0], 'mm')
                self.assertLessEqual(set(np.unique(np.asarray(predicted.dataobj))), {0, 255})
                self.assertGreater(case['tile_count'], 1)
                preview = np.load(root / 'infer' / case['view_file'], allow_pickle=False)
                self.assertEqual(preview.shape, (*case['source_shape'], 4))
                self.assertEqual(int(preview[0, 0, 0, 0]), 0)
                self.assertEqual(int(preview[-1, 0, 0, 0]), 255)
