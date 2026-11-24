"""Synthetic checks for stitched volumes, physical distances and native exports."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
from PIL import Image
import torch

from studio_core import medical_volume_eval
from studio_core.segmentation_task import SegmentationTask


class MedicalVolumeEvaluationTests(unittest.TestCase):
    def test_stitches_overlap_weights_before_computing_voxel_loss(self):
        image = np.broadcast_to(np.arange(5, dtype=np.float32), (1, 2, 2, 5))
        labels = np.broadcast_to(np.array([0, 1, 0, 1, 1]), (2, 2, 5))
        calls = []

        def predict(padded):
            self.assertEqual(padded.shape, (1, 3, 2, 3))
            np.testing.assert_array_equal(padded[:, 2], padded[:, 1])
            calls.append(int(padded[0, 0, 0, 0]))
            logits = np.broadcast_to(np.array([-3, 0, 4], dtype=np.float32), (3, 2, 3))
            return np.stack((np.zeros_like(logits), logits))

        mask, loss_sum = medical_volume_eval._sliding_case(image, labels, (3, 2, 3), 2, .5, predict, lambda: None)
        blended = np.array([-3, -.3 / 1.1, .1 / 1.2, .4 / 1.1, 4])
        expected_loss = (np.logaddexp(0, blended) - labels * blended).sum()
        self.assertEqual(calls, [0, 1, 2])
        np.testing.assert_array_equal(mask, np.broadcast_to(blended > 0, labels.shape))
        self.assertAlmostEqual(loss_sum, expected_loss, places=6)

    def test_stitches_across_depth_and_visits_final_edge(self):
        shape = (9, 3, 3)
        image = np.broadcast_to(np.arange(shape[0], dtype=np.float32)[None, :, None, None], (1, *shape))
        labels = (image[0] >= 4).astype(np.int64)
        calls = []

        def predict(padded):
            calls.append(int(padded[0, 0, 0, 0]))
            logits = padded[0] - 3.5
            return np.stack((np.zeros_like(logits), logits))

        mask, loss_sum = medical_volume_eval._sliding_case(image, labels, (4, 3, 3), 2, .25, predict, lambda: None)
        self.assertEqual(calls, [0, 3, 5])
        np.testing.assert_array_equal(mask, labels)
        logits = image[0].astype(np.float64) - 3.5
        self.assertAlmostEqual(loss_sum, (np.logaddexp(0, logits) - labels * logits).sum(), places=6)

    def test_uses_maximum_directed_surface_percentile_in_anisotropic_millimetres(self):
        target = np.zeros((4, 4, 6), dtype=bool)
        predicted = target.copy()
        target[2, 2, 1:5] = True
        predicted[2, 2, 1] = True
        affine = np.array([[0, -3, 0, 42], [2, 0, 0, -7], [0, 0, -5, 12], [0, 0, 0, 1]])
        hd95 = medical_volume_eval._hd95_mm(target, predicted, affine, lambda: None)
        # Directed distances in millimetres are (0, 2, 4, 6) and (0,).
        self.assertAlmostEqual(hd95, 5.7)

    def test_sheared_affine_uses_voxel_centres_in_world_coordinates(self):
        target = np.zeros((4, 4, 6), dtype=bool)
        predicted = target.copy()
        target[1, 1, 1] = True
        predicted[2, 2, 2] = True
        affine = np.array([[2, 1, 0, 42], [0, 3, 0, -7], [0, 0, -5, 12], [0, 0, 0, 1]])
        hd95 = medical_volume_eval._hd95_mm(target, predicted, affine, lambda: None)
        self.assertAlmostEqual(hd95, (3**2 + 3**2 + 5**2)**.5)

    def test_both_splits_score_full_native_cases_and_export_physical_masks(self):
        native_affine = np.array([[0, -3, 0, 42], [2, 0, 0, -7], [0, 0, -5, 12], [0, 0, 0, 1]])
        model_affine = native_affine.copy()
        model_affine[:3, 3] -= native_affine[:3, 0]
        target = np.zeros((4, 4, 6), dtype=np.int64)
        target[1, 1, 2] = 1
        target[1, 1, 4] = 2
        predicted = np.zeros((4, 4, 8), dtype=np.float32)
        predicted[1, 1, 3] = 1
        predicted[3, 3, 6] = 3
        def grid(shape, affine):
            return {'source_shape_dhw': list(shape), 'affine_xyz': affine.tolist(), 'spatial_unit': 'mm',
                    'affine_code': 1, 'nifti_version': 1}

        case = {'image': predicted[None], 'labels': predicted.astype(np.int64),
                'native_image': np.full((1, *target.shape), .25, dtype=np.float32), 'native_labels': target,
                'reference_grid': grid(target.shape, native_affine),
                'model_grid': grid(predicted.shape, model_affine)}
        spec = {'overlap': .5}
        manifest = {'data_spec': spec, 'input_shape': [None, 1, 3, 3, 4], 'mask_values': [0, 7, 512, 513, 514],
                    'records': [{'id': 'held', 'split': 'val'}, {'id': 'independent', 'split': 'test'}]}
        task = SegmentationTask({'input_port': 'volume'})
        self_assert = self

        class Pointwise(torch.nn.Module):
            calls = 0

            def forward(self, volume):
                self_assert.assertFalse(self.training)
                self_assert.assertFalse(torch.is_grad_enabled())
                self_assert.assertEqual(tuple(volume.shape), (1, 1, 3, 3, 4))
                self.calls += 1
                return {'mask_logits': torch.nn.functional.one_hot(volume[:, 0].long(), 5).movedim(-1, 1).float() * 8}

        model = Pointwise()
        calls = []

        def read(record, received_spec, *, cancelled):
            self.assertIs(received_spec, spec)
            self.assertFalse(cancelled())
            calls.append(record['id'])
            return case

        def check():
            pass

        progress = []
        with patch.object(medical_volume_eval, 'read_medical_case', side_effect=read), tempfile.TemporaryDirectory() as temp:
            val_metrics, val_rows = medical_volume_eval.evaluate_medical_split(model, task, manifest, 'val', 'cpu', check)
            val_tiles = model.calls
            self.assertGreater(val_tiles, 1)
            self.assertEqual(val_rows, [])
            root = Path(temp)
            metrics, rows = medical_volume_eval.evaluate_medical_split(
                model, task, manifest, 'test', 'cpu', check, output_directory=root,
                progress=lambda done, total: progress.append((done, total)))
            self.assertEqual(calls, ['held', 'independent'])
            self.assertEqual(model.calls, 2 * val_tiles)
            self.assertEqual(progress, [(1, 1)])
            self.assertEqual(val_metrics, metrics)
            self.assertEqual(metrics['count'], 1)
            self.assertEqual(metrics['voxels'], target.size)
            self.assertEqual(metrics['loss_voxels'], predicted.size)
            self.assertAlmostEqual(metrics['loss'], np.log1p(4 * np.exp(-8)), places=7)
            self.assertEqual(metrics['space'], 'original')
            self.assertEqual(metrics['original_space']['version'], 1)
            self.assertEqual(metrics['foreground_dice'], 1/3)
            self.assertEqual(metrics['foreground_iou'], 1/3)
            self.assertEqual(metrics['foreground_hd95_mm'], 0)
            self.assertEqual([record['class_value'] for record in metrics['per_class']], manifest['mask_values'])
            _, matched, target_only, predicted_only, absent = metrics['per_class']
            self.assertEqual((matched['dice'], matched['iou'], matched['hd95_mm']), (1, 1, 0))
            self.assertEqual((target_only['dice'], target_only['iou'], target_only['hd95_mm']), (0, 0, None))
            self.assertEqual((predicted_only['dice'], predicted_only['iou'], predicted_only['hd95_mm']), (0, 0, None))
            self.assertEqual((absent['dice'], absent['iou'], absent['hd95_mm']), (None, None, None))
            self.assertEqual((matched['valid_hd95_cases'], target_only['single_empty_cases'], absent['both_empty_cases']), (1, 1, 1))
            self.assertEqual(rows, [{'sample_id': 'independent', 'mask_file': 'masks/000001.nii.gz',
                                     'preview_file': 'masks/000001-preview.png', 'slice_index': 2, 'slice_count': 4}])
            mask = nib.load(root / rows[0]['mask_file'])
            self.assertEqual(mask.shape, target.shape[::-1])
            self.assertEqual(mask.header.get_xyzt_units()[0], 'mm')
            np.testing.assert_allclose(mask.affine, native_affine)
            expected = np.zeros_like(target)
            expected[1, 1, 2] = 7
            expected[3, 3, 5] = 513
            np.testing.assert_array_equal(np.asanyarray(mask.dataobj).transpose(2, 1, 0), expected)
            with Image.open(root / rows[0]['preview_file']) as preview:
                self.assertEqual(preview.size, (26, 4))
                self.assertNotEqual(preview.getpixel((12, 1)), preview.getpixel((2, 1)))
            with (root / 'medical_case_metrics.csv').open(encoding='utf-8', newline='') as stream:
                details = list(csv.DictReader(stream))
            self.assertEqual(len(details), 5)
            self.assertEqual(details[1]['dice'], '1.0')
            self.assertEqual(details[2]['hd95_mm'], '')
            self.assertEqual(details[4]['dice'], '')
            self.assertEqual(details[3]['mask_file'], rows[0]['mask_file'])
            self.assertEqual(details[3]['spacing_dhw_mm'], '[5.0, 3.0, 2.0]')

    def test_real_nifti_cases_aggregate_overlap_per_case_and_skip_undefined_hd95(self):
        from studio_core.medical_volume_data import inspect_medical_dataset, medical_defaults

        class Threshold(torch.nn.Module):
            def forward(self, volume):
                score = (volume[:, 0] - .5) * 8
                return {'mask_logits': torch.stack((torch.zeros_like(score), score), dim=1)}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pairs = root / 'pairs.csv'
            affine = np.diag([2., 2., 3., 1.])
            groups = ('train', 'train', 'val', 'val', 'test')
            with pairs.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(('sample_id', 'patient_id', 'image_path', 'mask_path', 'split'))
                for index, split in enumerate(groups):
                    target = np.zeros((8, 8, 6), dtype=np.uint8)
                    if index == 3:
                        target[3:5, 3:5, 2:4] = 7
                        image = np.full(target.shape, 10, dtype=np.int16)
                    else:
                        target[index + 1, 3, 3] = 7
                        image = target * 20
                    image_file, mask_file = f'image{index}.nii.gz', f'mask{index}.nii.gz'
                    for name, array in ((image_file, image), (mask_file, target)):
                        volume = nib.Nifti1Image(array, affine)
                        volume.header.set_xyzt_units('mm')
                        nib.save(volume, root / name)
                    writer.writerow((f'case{index}', f'patient{index}', image_file, mask_file, split))
            spec = {**medical_defaults('mri_nifti'), 'path': str(pairs), 'split_column': 'split', 'registered': True,
                    'mask_values': [0, 7], 'depth': 2, 'image_size': [8, 8], 'spacing_mm': [3., 2., 2.]}
            manifest = inspect_medical_dataset(spec, seed=5)
            task = SegmentationTask({'input_port': 'volume'})
            progress = []
            metrics, _ = medical_volume_eval.evaluate_medical_split(
                Threshold(), task, manifest, 'val', 'cpu', lambda: None,
                progress=lambda done, total: progress.append((done, total)))
            self.assertEqual(progress, [(1, 2), (2, 2)])
            self.assertEqual(metrics['count'], 2)
            self.assertEqual(metrics['voxels'], 2 * 8 * 8 * 6)
            self.assertEqual(metrics['foreground_dice'], .5)
            self.assertEqual(metrics['foreground_iou'], .5)
            self.assertEqual(metrics['foreground_hd95_mm'], 0)
            foreground = metrics['per_class'][1]
            self.assertEqual(foreground['target_voxels'], 9)
            self.assertEqual(foreground['predicted_voxels'], 1)
            self.assertEqual(foreground['valid_overlap_cases'], 2)
            self.assertEqual(foreground['valid_hd95_cases'], 1)
            self.assertEqual(foreground['single_empty_cases'], 1)

    def test_cancel_is_checked_between_tiles(self):
        image = np.zeros((1, 4, 4, 4), dtype=np.float32)
        labels = np.zeros((4, 4, 4), dtype=np.int64)
        calls = []

        def check():
            if calls:
                raise InterruptedError('fixture cancellation')

        def predict(padded):
            calls.append(padded.shape)
            return np.zeros((2, 3, 3, 3), dtype=np.float32)

        with self.assertRaisesRegex(InterruptedError, 'fixture cancellation'):
            medical_volume_eval._sliding_case(image, labels, (3, 3, 3), 2, .5, predict, check)
        self.assertEqual(calls, [(1, 3, 3, 3)])


if __name__ == '__main__':
    unittest.main()
