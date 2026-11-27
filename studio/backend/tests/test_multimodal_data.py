"""Case alignment and physical preprocessing using synthetic sources."""
import csv
from pathlib import Path
import shutil
import tempfile
import unittest

import nibabel as nib
import numpy as np
from PIL import Image, PngImagePlugin
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
import torch

from studio_core.multimodal_data import (build_multimodal_datasets, inspect_multimodal_dataset,
                                         multimodal_defaults, validate_multimodal_spec)


FIELDS = ('case_id', 'patient_id', 'label', 'split', 'volume_path', 'image_path', 't2_path', 'age', 'score')


def write_ct(path, offset):
    path.mkdir()
    series, frame = generate_uid(), generate_uid()
    for depth, raw in enumerate((-5, 5, 15, 35, 45)):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        filename = path / f'slice-{4 - depth}.dcm'
        image = FileDataset(filename, {}, file_meta=meta, preamble=b'\0' * 128)
        image.SOPClassUID, image.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        image.Modality, image.SeriesInstanceUID, image.FrameOfReferenceUID = 'CT', series, frame
        image.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
        image.Rows, image.Columns = 8, 8
        image.SamplesPerPixel, image.PhotometricInterpretation = 1, 'MONOCHROME2'
        image.PixelRepresentation, image.BitsAllocated, image.BitsStored, image.HighBit = 1, 16, 16, 15
        image.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        image.ImagePositionPatient = [12., -3., 40. + depth * 2.5]
        image.PixelSpacing, image.SliceThickness = [.8, .5], 2.5
        image.RescaleSlope, image.RescaleIntercept = 2, -100
        image.PixelData = np.full((8, 8), raw + offset, dtype='<i2').tobytes()
        image.save_as(filename, enforce_file_format=True)


def save_nifti(path, values, affine):
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units('mm')
    nib.save(image, path)


class MultimodalDataTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.csv = self.root / 'cases.csv'
        self.rows = []
        for index, split in enumerate(('train', 'train', 'val', 'val', 'test', 'test')):
            self.rows.append(dict(case_id=f'case-{index}', patient_id=f'patient-{index}',
                                  split=split, label=('control', 'positive')[index % 2],
                                  volume_path=f'volume-{index}', image_path=f'old image {index}.png',
                                  t2_path=f't2-{index}.nii.gz', age=(10, 20, 10, 99, 10, 999)[index], score=5))
        self.spec = {**multimodal_defaults(), 'path': str(self.csv), 'split_column': 'split',
                     'modalities': ['image', 'tabular'], 'image_size': [8, 8], 'image_channels': 3,
                     'feature_columns': ['age', 'score']}

    def write_rows(self):
        with self.csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

    def write_images(self):
        for index, row in enumerate(self.rows):
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            image[0, 0] = (index * 27, 0, 255)
            Image.fromarray(image).save(self.root / row['image_path'])

    def write_volumes(self, fmt):
        for index, row in enumerate(self.rows):
            if fmt == 'dicom_ct':
                write_ct(self.root / row['volume_path'], index * 2)
                continue
            row['volume_path'] += '.nii.gz'
            affine = np.diag([1., 1., 2., 1.])
            x, y, z = np.indices((8, 8, 5), dtype=np.float32)
            save_nifti(self.root / row['volume_path'], x + (index + 1) * y + (index + 2) * z, affine)
            affine[0, 0] = .5
            x, y, z = np.indices((15, 8, 5), dtype=np.float32)
            save_nifti(self.root / row['t2_path'], 7 - x * .5 + (index + 1) * y + (index + 2) * z, affine)
        self.write_rows()

    def test_image_and_tabular_use_same_row_and_train_only_statistics(self):
        self.write_images()
        self.write_rows()
        manifest = inspect_multimodal_dataset(self.spec, 13)
        self.assertEqual(manifest['counts'], {'train': 2, 'val': 2, 'test': 2})
        self.assertEqual(manifest['class_names'], ['control', 'positive'])
        self.assertEqual(manifest['input_shapes'], {'image': [None, 3, 8, 8], 'tabular': [None, 2]})
        self.assertEqual(manifest['input_axes']['tabular'], ['batch', 'feature'])
        self.assertEqual(manifest['csv_relative_path'], 'cases.csv')
        self.assertEqual(manifest['preprocessing']['tabular']['mean'], [15., 5.])
        self.assertEqual(manifest['preprocessing']['tabular']['std'], [5., 1.])
        datasets = build_multimodal_datasets(manifest)
        tensors, label, identifier = datasets['val'][1]
        self.assertEqual((list(tensors), label, identifier), (['image', 'tabular'], 1, 'case-3'))
        torch.testing.assert_close(tensors['tabular'], torch.tensor([16.8, 0.]))
        torch.testing.assert_close(tensors['image'][:, 0, 0], torch.tensor([81 / 255, 0., 1.]))

    def test_portable_sources_are_relative_and_unselected_images_are_not_decoded_again(self):
        from unittest.mock import patch

        self.write_images()
        relative = self.rows[0]['image_path']
        self.rows[0]['image_path'] = str(self.root / relative)
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'case-0.*image_path.*relative'):
            inspect_multimodal_dataset(self.spec, 13)
        self.rows[0]['image_path'] = relative
        self.write_rows()
        manifest = inspect_multimodal_dataset(self.spec, 13)
        with patch('studio_core.multimodal_data._read_raster', side_effect=AssertionError('inactive image decoded')):
            features, label, case_id = build_multimodal_datasets(manifest, modalities=['tabular'])['test'][0]
        self.assertEqual((list(features), label, case_id), (['tabular'], 0, 'case-4'))

    def test_errors_identify_case_and_missing_column_or_conflicting_label(self):
        self.write_images()
        self.rows[2]['age'] = ''
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'case-2.*age'):
            inspect_multimodal_dataset(self.spec, 13)
        self.rows[2].update(age=10, case_id='case-1')
        self.write_rows()
        with self.assertRaisesRegex(ValueError, '[Cc]onflicting label.*case-1'):
            inspect_multimodal_dataset(self.spec, 13)
        self.rows[2].update(label='positive')
        self.write_rows()
        with self.assertRaisesRegex(ValueError, '[Dd]uplicate.*case-1'):
            inspect_multimodal_dataset(self.spec, 13)

    def test_editable_draft_allows_unfinished_modality_or_feature_selection(self):
        for draft in ({**self.spec, 'modalities': []}, {**self.spec, 'feature_columns': []}):
            with self.subTest(draft=draft):
                self.assertEqual(validate_multimodal_spec(draft, editable=True)['modalities'], draft['modalities'])
                with self.assertRaises(ValueError):
                    validate_multimodal_spec(draft)
        with self.assertRaisesRegex(ValueError, 'ct_window'):
            validate_multimodal_spec({**self.spec, 'ct_window': [float('nan'), 400.]}, editable=True)

    def test_checks_patient_path_and_decoded_image_independently_of_table(self):
        self.write_images()
        baseline = dict(self.rows[2])
        for update, error in (({'patient_id': 'patient-0'}, 'group'),
                              ({'image_path': self.rows[0]['image_path']}, 'image path')):
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, error + '(?=.*case-0)(?=.*case-2)'):
                self.rows[2] = {**baseline, 'age': 71, **update}
                self.write_rows()
                inspect_multimodal_dataset(self.spec, 13)
        self.rows[2] = {**baseline, 'age': 71}
        self.write_rows()
        with Image.open(self.root / self.rows[0]['image_path']) as image:
            extra = PngImagePlugin.PngInfo()
            extra.add_text('synthetic', 'different file, same pixels')
            image.save(self.root / self.rows[2]['image_path'], pnginfo=extra)
        with self.assertRaisesRegex(ValueError, 'image content.*case-0.*case-2'):
            inspect_multimodal_dataset(self.spec, 13)

    def test_rejects_source_outside_root_and_late_raster_or_csv_replacement(self):
        self.write_images()
        self.write_rows()
        manifest = inspect_multimodal_dataset(self.spec, 13)
        dataset = build_multimodal_datasets(manifest)['train']
        shutil.copyfile(self.root / self.rows[1]['image_path'], self.root / self.rows[0]['image_path'])
        with self.assertRaisesRegex(ValueError, 'case-0.*changed'):
            dataset[0]
        self.rows[0]['age'] = 11
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'CSV.*changed'):
            dataset[1]
        self.rows[0]['image_path'] = '../outside.png'
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'case-0.*image_path.*source root'):
            inspect_multimodal_dataset(self.spec, 13)

    def test_automatic_split_and_digest_survive_moving_grouped_cases(self):
        self.write_images()
        self.rows.append({**self.rows[0], 'case_id': 'second-visit'})
        self.write_rows()
        spec = {**self.spec, 'split_column': None}
        manifest = inspect_multimodal_dataset(spec, 29)
        first, repeated = (manifest['records'][index] for index in (0, -1))
        self.assertEqual(first['split'], repeated['split'])
        copied = self.root / 'copy'
        copied.mkdir()
        for path in self.root.iterdir():
            if path.is_file():
                shutil.copyfile(path, copied / path.name)
        relocated = inspect_multimodal_dataset({**spec, 'path': str(copied / self.csv.name)}, 29)
        self.assertEqual(relocated['records'], manifest['records'])
        self.assertEqual(relocated['dataset_digest'], manifest['dataset_digest'])

    def test_ct_projection_and_slice_use_uncropped_patient_grid(self):
        self.write_volumes('dicom_ct')
        settings = {**self.spec, 'modalities': ['tabular', 'volume', 'image'], 'volume_shape': [3, 10, 8],
                    'spacing_mm': [2.5, .8, .5], 'ct_window': [-100., 0.], 'image_channels': 1}
        for source, expected in (('center_slice', .3), ('mean_projection', .4)):
            with self.subTest(source=source):
                manifest = inspect_multimodal_dataset({**settings, 'image_source': source}, 13)
                self.assertEqual(manifest['input_axes']['volume'], ['batch', 'channel', 'depth', 'height', 'width'])
                tensors, label, identifier = build_multimodal_datasets(manifest)['train'][0]
                self.assertEqual((list(tensors), label, identifier), (['tabular', 'volume', 'image'], 0, 'case-0'))
                self.assertEqual(tensors['volume'].shape, (1, 3, 10, 8))
                torch.testing.assert_close(tensors['volume'][0, :, 1, 0], torch.tensor([.1, .3, .7]))
                torch.testing.assert_close(tensors['volume'][0, :, 0, 0], torch.zeros(3))
                torch.testing.assert_close(tensors['image'], torch.full((1, 8, 8), expected))
                volume = manifest['records'][0]['geometry']['volume']
                self.assertEqual(volume['model_grid']['source_shape_dhw'], [5, 8, 8])
                self.assertEqual(volume['output_grid']['source_shape_dhw'], [3, 10, 8])
                self.assertAlmostEqual(volume['output_grid']['affine_xyz'][2][3], 42.5)

        derived_only = {**settings, 'modalities': ['image'], 'image_source': 'center_slice',
                        'spacing_mm': [3.75, .8, .5]}
        manifest = inspect_multimodal_dataset(derived_only, 13)
        tensors, _, _ = build_multimodal_datasets(manifest)['train'][0]
        self.assertEqual(list(tensors), ['image'])
        # In the 3.75 mm grid, depth index 2 samples native index 2.75.
        torch.testing.assert_close(tensors['image'], torch.full((1, 8, 8), .6))

    def test_mri_sequence_order_alignment_and_independent_volume_leakage(self):
        self.write_images()
        self.write_volumes('mri_nifti')
        settings = {**self.spec, 'modalities': ['volume', 'image'], 'volume_format': 'mri_nifti',
                    'sequence_columns': ['t2_path'], 'registered': True, 'ct_window': None,
                    'volume_shape': [5, 8, 8], 'spacing_mm': [2., 1., 1.]}
        with self.assertRaisesRegex(ValueError, 'registered'):
            validate_multimodal_spec({**settings, 'registered': False})
        manifest = inspect_multimodal_dataset(settings, 19)
        first = manifest['records'][0]
        self.assertEqual([source['relative_path'] for source in first['sources']['volume']],
                         ['volume-0.nii.gz', 't2-0.nii.gz'])
        dataset = build_multimodal_datasets(manifest)['train']
        tensors, _, _ = dataset[0]
        torch.testing.assert_close(tensors['volume'][:, 0, 0, 0], torch.tensor([0., 7 / 22]))
        displaced = np.diag([.5, 1., 2., 1.])
        displaced[0, 3] = 20.
        save_nifti(self.root / self.rows[0]['t2_path'], np.ones((15, 8, 5), dtype=np.float32), displaced)
        with self.assertRaisesRegex(ValueError, 'case-0.*changed'):
            dataset[0]
        with self.assertRaisesRegex(ValueError, 'case-0.*t2_path.*cover'):
            inspect_multimodal_dataset(settings, 19)
        # Only the raw reference voxels repeat, with different bytes and an independent raster/sequence.
        self.rows[0]['t2_path'] = self.rows[1]['t2_path']
        first_image = nib.load(self.root / self.rows[0]['volume_path'])
        relocated = first_image.affine.copy()
        relocated[1, 3] = .01
        save_nifti(self.root / self.rows[2]['volume_path'], first_image.get_fdata(dtype=np.float32), relocated)
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'volume content.*case-0.*case-2'):
            inspect_multimodal_dataset(settings, 19)


if __name__ == '__main__':
    unittest.main()
