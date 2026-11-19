"""Exercise mask labels, paired geometry, graph round trip and real training."""
import csv
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
from PIL import Image
import torch

from studio_core.classification_data import build_datasets, inspect_dataset
from studio_core.classification_runtime import run_supervised
from studio_core.experiment_spec import default_segmentation_config, model_digest
from studio_core.python_io import export_python, import_python
from studio_core.segmentation_templates import segmentation_graph
from studio_core.task_adapters import get_task_adapter


def fixture(directory, volume=False):
    path = directory / 'pairs.csv'
    rows = []
    # Noncanonical axis direction and anisotropic spacing expose accidental reorientation.
    affine = np.array([[0, -0.75, 0, 42], [0.5, 0, 0, -7], [0, 0, -2, 12], [0, 0, 0, 1.]])
    for index, split in enumerate(('train', 'train', 'val', 'test')):
        shape = (13, 11, 5) if volume else (11, 13)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[2:9, 3:10] = 255
        image = (mask * 0.8 + np.random.default_rng(index).integers(0, 48, shape)).astype(np.uint8)
        suffix = '.nii.gz' if volume else '.png'
        names = (f'image{index}{suffix}', f'mask{index}{suffix}')
        for name, array in zip(names, (image, mask)):
            if volume:
                item = nib.Nifti1Image(array, affine)
                item.header.set_xyzt_units('mm')
                nib.save(item, directory / name)
            else:
                Image.fromarray(array).save(directory / name)
        rows.append((f's{index}', f'patient{index}', *names, split))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('sample_id', 'patient_id', 'image_path', 'mask_path', 'split'))
        writer.writerows(rows)
    return path, affine


def config_for(directory, volume=False):
    path, affine = fixture(directory, volume)
    config = default_segmentation_config()
    config.update(epochs=1, batch_size=2, seed=17)
    config['data'].update(path=str(path), split_column='split', image_size=[9, 8], mask_values=[0, 255])
    if volume:
        config['data'].update(format='nifti', depth=4)
    return config, affine


class SegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_image_pair_requires_integral_declared_labels_and_same_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            config, _ = config_for(directory)
            manifest = inspect_dataset(config['data'])
            image, mask, sample = build_datasets(manifest)['train'][0]
            self.assertEqual((tuple(image.shape), tuple(mask.shape), sample), ((1, 9, 8), (9, 8), 's0'))
            self.assertEqual(set(mask.unique().tolist()), {0, 1})
            # Verify nearest sampling at voxel centres for a non-integer resize ratio.
            expected = np.asarray(Image.open(directory / 'mask0.png'))
            rows = ((np.arange(9) + .5) * 11 / 9).astype(int)
            cols = ((np.arange(8) + .5) * 13 / 8).astype(int)
            np.testing.assert_array_equal(mask, expected[np.ix_(rows, cols)] // 255)
            Image.fromarray(np.zeros((10, 13), dtype='uint8')).save(directory / 'mask0.png')
            with self.assertRaisesRegex(ValueError, 'changed'):
                build_datasets(manifest)['train'][0]
            with self.assertRaisesRegex(ValueError, 'grid'):
                inspect_dataset(config['data'])
            Image.fromarray(np.full((11, 13), 7, dtype='uint8')).save(directory / 'mask0.png')
            with self.assertRaisesRegex(ValueError, 'undeclared'):
                inspect_dataset(config['data'])

    def test_rejects_patient_and_duplicate_image_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            config, _ = config_for(directory)
            path = Path(config['data']['path'])
            original = path.read_text(encoding='utf-8')
            path.write_text(original.replace('patient3,', 'patient0,'), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'group leakage'):
                inspect_dataset(config['data'])
            path.write_text(original, encoding='utf-8')
            with Image.open(directory / 'image0.png') as image:
                image.save(directory / 'image3.png', compress_level=0)
            with self.assertRaisesRegex(ValueError, 'content leakage'):
                inspect_dataset(config['data'])

    def test_nifti_compares_voxel_geometry_and_ignores_no_failed_label_values(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            config, affine = config_for(directory, volume=True)
            manifest = inspect_dataset(config['data'])
            self.assertEqual(manifest['input_shape'], [None, 1, 4, 9, 8])
            path = directory / 'mask0.nii.gz'
            shifted = affine.copy()
            shifted[0, 3] += 1
            item = nib.Nifti1Image(np.zeros((13, 11, 5), dtype=np.uint8), shifted)
            item.header.set_xyzt_units('mm')
            nib.save(item, path)
            with self.assertRaisesRegex(ValueError, 'affine'):
                inspect_dataset(config['data'])
            item = nib.Nifti1Image(np.full((13, 11, 5), .5, dtype=np.float32), affine)
            item.header.set_xyzt_units('mm')
            nib.save(item, path)
            with self.assertRaisesRegex(ValueError, 'integer'):
                inspect_dataset(config['data'])

    def test_jpeg_compressed_tiff_is_rejected_as_a_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            config, _ = config_for(directory)
            path = Path(config['data']['path'])
            body = path.read_text(encoding='utf-8').replace('mask0.png', 'mask0.tiff')
            path.write_text(body, encoding='utf-8')
            mask = np.zeros((11, 13), dtype=np.uint8)
            Image.fromarray(mask).save(directory / 'mask0.tiff', compression='jpeg')
            with self.assertRaisesRegex(ValueError, 'lossless TIFF'):
                inspect_dataset(config['data'])

    def test_nifti_rejects_one_voxel_shift_even_with_submillimetre_spacing(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            config, _ = config_for(directory, volume=True)
            grid = np.diag([.0005, .0005, .0005, 1.0])
            shifted = grid.copy()
            shifted[0, 3] = .0005
            for name, affine in (('image0', grid), ('mask0', shifted)):
                path = directory / (name + '.nii.gz')
                original = nib.load(path)
                image = nib.Nifti1Image(np.asarray(original.dataobj), affine)
                image.header.set_xyzt_units('mm')
                nib.save(image, path)
            with self.assertRaisesRegex(ValueError, 'affine'):
                inspect_dataset(config['data'])

    def test_metrics_exclude_background_and_report_undefined(self):
        task = get_task_adapter('segmentation', 1)(default_segmentation_config())
        target = torch.tensor([[[0, 1], [1, 0]]])
        predicted = torch.tensor([[[0, 1], [0, 0]]])
        logits = torch.nn.functional.one_hot(predicted, 3).movedim(-1, 1).float()
        batch = {'targets': target, 'sample_ids': ['test'], 'features': {}}
        summary = task.finalize(task.summarize({'mask_logits': logits}, batch, torch.tensor(.2)))
        self.assertAlmostEqual(summary['foreground_dice'], 2 / 3)
        self.assertAlmostEqual(summary['foreground_iou'], 1 / 2)
        self.assertIsNone(summary['per_class'][2]['dice'])

    def test_supervised_2d_and_3d_restore_original_grid_after_python_round_trip(self):
        for volume in (False, True):
            with self.subTest(volume=volume), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                config, affine = config_for(directory, volume=volume)
                manifest = inspect_dataset(config['data'], config['seed'])
                graph = segmentation_graph(manifest)
                restored = import_python(export_python(graph))
                self.assertEqual(model_digest(graph), model_digest(restored))
                config['expected_dataset_digest'] = manifest['dataset_digest']
                result = run_supervised(restored, config, directory / 'run')
                self.assertEqual(result['metrics']['test']['count'], 1)
                self.assertEqual(result['metrics']['test']['voxels'], (4 if volume else 1) * 9 * 8)
                self.assertTrue(any(p['update_norm'] > 0 for p in result['parameters']))
                with Path(result['artifacts']['predictions']).open(encoding='utf-8', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([row['sample_id'] for row in rows], ['s3'])
                mask_path = directory / 'run' / rows[0]['mask_file']
                if volume:
                    prediction = nib.load(mask_path)
                    self.assertEqual(prediction.shape, (13, 11, 5))
                    np.testing.assert_allclose(prediction.affine, affine)
                    self.assertEqual(prediction.header.get_xyzt_units()[0], 'mm')
                    values = set(np.unique(np.asanyarray(prediction.dataobj)))
                else:
                    with Image.open(mask_path) as prediction:
                        self.assertEqual(prediction.size, (13, 11))
                        values = set(np.unique(prediction))
                self.assertLessEqual(values, {0, 255})
                self.assertTrue((directory / 'run' / rows[0]['preview_file']).is_file())
