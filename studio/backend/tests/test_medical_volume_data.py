"""Exercise S13 medical data using only synthetic volumes and metadata."""
import csv
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

from studio_core.medical_volume_data import (build_medical_datasets, inspect_medical_dataset,
                                            medical_defaults, preview_medical_patch, read_medical_case,
                                            read_medical_input, validate_medical_spec)


AFFINE = np.diag([-1., 1., 2., 1.])
AFFINE[:3, 3] = [26., -15., 4.]
SHAPE = (17, 15, 9)
CSV_COLUMNS = ('sample_id', 'patient_id', 'image_path', 'mask_path', 'split', 't2_path')


def save_nifti(path, array, affine=AFFINE):
    image = nib.Nifti1Image(array, affine)
    image.header.set_xyzt_units('mm')
    nib.save(image, path)


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def mri_fixture(directory):
    rows = []
    for case, split in enumerate(('train', 'train', 'val', 'test')):
        # T2 samples the same physical field twice as finely as the reference.
        grids = [np.indices(shape, dtype=np.float32) for shape in
                 (SHAPE, tuple(2 * n + 1 for n in SHAPE))]
        for index, (x, y, z) in enumerate(grids):
            factor = 2 ** index
            origin = .5 if index else 0
            x, y, z = [np.clip(axis / factor - origin, 0, size - 1)
                       for axis, size in zip((x, y, z), SHAPE)]
            ramp = x + (case + 1) * y + (case + 2) * z
            affine = AFFINE.copy()
            affine[:3, :3] /= factor
            affine[:3, 3] -= origin * np.diag(AFFINE)[:3]
            save_nifti(directory / f'case{case}-{index}.nii.gz', ramp, affine)
        labels = np.zeros(SHAPE, dtype=np.uint8)
        labels[10:13, 3:6, 4:7] = 7
        save_nifti(directory / f'mask{case}.nii.gz', labels)
        rows.append([f's{case}', f'p{case}', f'case{case}-0.nii.gz', f'mask{case}.nii.gz', split,
                     f'case{case}-1.nii.gz'])
    csv_path = directory / 'pairs.csv'
    write_rows(csv_path, rows)
    return {**medical_defaults('mri_nifti'), 'path': str(csv_path), 'depth': 4, 'image_size': [8, 8],
            'split_column': 'split', 'registered': True, 'channels': 2, 'sequence_columns': ['t2_path'],
            'mask_values': [0, 7], 'spacing_mm': [1., 1., 1.], 'patches_per_case': 8}, rows



def ct_fixture(directory):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    # CT world axes use LPS. The mask uses the corresponding RAS affine.
    affine = np.diag([-.5, -.8, 2.5, 1.])
    affine[:3, 3] = [-12., 3., 40.]
    rows, depth, height, width = [], 3, 6, 7
    expected = None
    for case, split in enumerate(('train', 'val', 'test')):
        series = directory / f'ct{case}'
        series.mkdir()
        series_uid, frame_uid = generate_uid(), generate_uid()
        raw = (np.arange(depth * height * width).reshape((depth, height, width)) - 20 + case * 8).astype('<i2')
        for z in range(depth):
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = CTImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            # Filename order opposes physical slice order. The first file is hidden.
            path = series / (f'.{depth - z}.dcm' if z == depth - 1 else f'{depth - z}.dcm')
            item = FileDataset(path, {}, file_meta=meta, preamble=b'\0' * 128)
            item.SOPClassUID, item.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
            item.Modality, item.SeriesInstanceUID, item.FrameOfReferenceUID = 'CT', series_uid, frame_uid
            item.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
            item.Rows, item.Columns = height, width
            item.SamplesPerPixel, item.PhotometricInterpretation = 1, 'MONOCHROME2'
            item.PixelRepresentation, item.BitsAllocated, item.BitsStored, item.HighBit = 1, 16, 16, 15
            item.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            item.ImagePositionPatient = [12., -3., 40. + z * 2.5]
            item.PixelSpacing, item.SliceThickness = [.8, .5], 2.5
            item.RescaleSlope, item.RescaleIntercept = 2, -100
            item.PixelData = raw[z].tobytes()
            item.save_as(path, enforce_file_format=True)
        if case == 0:
            expected = np.clip(2 * raw, 0, 100) / 100
        labels = np.zeros((width, height, depth), dtype=np.uint8)
        labels[4:6, 2:4, :2] = 7
        mask = f'ct-mask{case}.nii.gz'
        save_nifti(directory / mask, labels, affine)
        rows.append([f'ct{case}', f'patient{case}', series.name, mask, split, ''])
    csv_path = directory / 'ct-pairs.csv'
    write_rows(csv_path, rows)
    spec = {**medical_defaults('dicom_ct'), 'path': str(csv_path), 'depth': 4, 'image_size': [8, 8],
            'split_column': 'split', 'mask_values': [0, 7], 'spacing_mm': [2.5, .8, .5],
            'ct_window': [-100., 0.]}
    return spec, expected

class MedicalVolumeDataTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec, self.rows = mri_fixture(self.root)
        self.csv = Path(self.spec['path'])

    def test_medical_recipe_validates_format_specific_inputs(self):
        no_channel = {key: value for key, value in self.spec.items() if key != 'channels'}
        self.assertEqual(validate_medical_spec(no_channel)['channels'], 2)
        for override, message in (({'registered': False}, 'registered'),
                                  ({'channels': 1}, 'channel'),
                                  ({'sequence_columns': ['mask_path']}, 'distinct'),
                                  ({'spacing_mm': [1, 0, 1]}, 'spacing'),
                                  ({'foreground_fraction': True}, 'foreground'),
                                  ({'patches_per_case': 0}, 'patches'),
                                  ({'overlap': 1}, 'overlap')):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, message):
                validate_medical_spec({**self.spec, **override})
        ct = medical_defaults('dicom_ct')
        with self.assertRaisesRegex(ValueError, 'window'):
            validate_medical_spec({**ct, 'ct_window': [200, -100]})

    def test_mri_aligns_multiple_grids_and_shares_independent_inference_transform(self):
        manifest = inspect_medical_dataset(self.spec, 17)
        self.assertEqual(manifest['version'], 2)
        self.assertEqual(manifest['counts'], {'train': 2, 'val': 1, 'test': 1})
        self.assertEqual(manifest['input_shape'], [None, 2, 4, 8, 8])
        self.assertEqual(manifest['target_shape'], [None, 4, 8, 8])
        record = manifest['records'][0]
        self.assertEqual(record['relative_path'], 'case0-0.nii.gz')
        self.assertEqual(record['sequence_relative_paths'], ['case0-1.nii.gz'])
        self.assertEqual(len(record['sequence_source_digests'][0]), 64)
        case = read_medical_case(record, manifest['data_spec'])
        np.testing.assert_allclose(case['image'][0], case['image'][1], rtol=1e-6, atol=1e-6)
        self.assertEqual(case['image'].dtype, np.float32)
        self.assertEqual(case['native_image'].shape, (2, *SHAPE[::-1]))
        self.assertEqual(case['native_labels'].shape, SHAPE[::-1])
        self.assertEqual(int(case['native_labels'].sum()), 27)
        self.assertEqual(set(np.unique(case['labels'])), {0, 1})
        self.assertGreater(case['labels'].shape[0], case['native_labels'].shape[0])
        unlabelled = read_medical_input(self.root / record['relative_path'],
                                       [self.root / path for path in record['sequence_relative_paths']],
                                       manifest['data_spec'])
        for name in ('image', 'native_image'):
            np.testing.assert_array_equal(unlabelled[name], case[name])
        for name in ('reference_grid', 'model_grid'):
            self.assertEqual(unlabelled[name], case[name])

    def test_rejects_uncovered_mri_sequence_and_shifted_mask(self):
        shifted = AFFINE.copy()
        shifted[0, 3] += 10
        save_nifti(self.root / 'case0-1.nii.gz', np.ones(SHAPE, dtype=np.float32), shifted)
        with self.assertRaisesRegex(ValueError, 'cover|extent'):
            inspect_medical_dataset(self.spec, 17)
        self.spec['sequence_columns'] = []
        self.spec['channels'] = 1
        save_nifti(self.root / 'mask0.nii.gz', np.ones(SHAPE, dtype=np.uint8), shifted)
        with self.assertRaisesRegex(ValueError, 'grid|affine'):
            inspect_medical_dataset(self.spec, 17)

    def test_patient_split_and_duplicate_channels_are_checked_across_splits(self):
        self.rows[3][1] = self.rows[0][1]
        write_rows(self.csv, self.rows)
        with self.assertRaisesRegex(ValueError, 'group leakage'):
            inspect_medical_dataset(self.spec, 17)
        self.rows[3][1] = 'p3'
        write_rows(self.csv, self.rows)
        # A reused secondary MRI sequence contaminates the partition even with a unique primary image.
        train_sequence = self.root / self.rows[0][-1]
        test_sequence = self.root / self.rows[3][-1]
        shutil.copyfile(train_sequence, test_sequence)
        with self.assertRaisesRegex(ValueError, 'source.*leakage'):
            inspect_medical_dataset(self.spec, 17)
        image = nib.load(train_sequence)
        image.header['descrip'] = 'Changed header, same decoded image'
        nib.save(image, test_sequence)
        with self.assertRaisesRegex(ValueError, 'decoded.*leakage'):
            inspect_medical_dataset(self.spec, 17)

    def test_generated_splits_are_by_patient_and_csv_paths_stay_within_root(self):
        self.rows[1][1] = self.rows[0][1]
        write_rows(self.csv, self.rows)
        spec = {**self.spec, 'split_column': None}
        first = inspect_medical_dataset(spec, 17)
        self.assertEqual(first['records'][0]['split'], first['records'][1]['split'])
        self.assertEqual(first['dataset_digest'], inspect_medical_dataset(spec, 17)['dataset_digest'])
        self.rows[0][-1] = '../other.nii.gz'
        write_rows(self.csv, self.rows)
        with self.assertRaisesRegex(ValueError, 'inside.*source root'):
            inspect_medical_dataset(spec, 17)

    def test_foreground_patch_replay_matches_dataset_and_varies_by_epoch(self):
        manifest = inspect_medical_dataset({**self.spec, 'foreground_fraction': 1}, 17)
        datasets = build_medical_datasets(manifest)
        train = datasets['train']
        self.assertEqual(len(train), 16)
        image, mask, sample = train[0]
        preview, annotation, details = preview_medical_patch(manifest, case_id='s0', epoch=1, patch_index=0)
        self.assertEqual(sample, details['case_id'])
        self.assertEqual(details['split'], 'train')
        self.assertEqual(details['reference_spacing_mm'], [2., 1., 1.])
        self.assertEqual(details['model_spacing_mm'], [1., 1., 1.])
        self.assertEqual((image.dtype, mask.dtype), (torch.float32, torch.int64))
        self.assertEqual(tuple(image.shape), (2, 4, 8, 8))
        self.assertTrue(mask.any())
        np.testing.assert_array_equal(preview, image)
        np.testing.assert_array_equal(annotation, mask)
        self.assertEqual(datasets['val'][0][2], 's2')
        self.assertEqual(len(datasets['val']), 1)
        starts = [preview_medical_patch(manifest, case_id='s0', epoch=epoch)[2]['patch_start_dhw']
                  for epoch in range(1, 4)]
        self.assertGreater(len({tuple(start) for start in starts}), 1)
        train.set_epoch(2)
        resumed = build_medical_datasets(manifest)['train']
        resumed.set_epoch(2)
        torch.testing.assert_close(train[0][0], resumed[0][0])
        torch.testing.assert_close(train[0][1], resumed[0][1])

    def test_source_and_recipe_mutation_invalidate_frozen_data(self):
        manifest = inspect_medical_dataset(self.spec, 17)
        changed_recipe = inspect_medical_dataset({**self.spec, 'overlap': .5}, 17)
        self.assertNotEqual(changed_recipe['dataset_digest'], manifest['dataset_digest'])
        forged = deepcopy(manifest)
        forged['data_spec']['foreground_fraction'] = .99
        with self.assertRaisesRegex(ValueError, 'manifest digest'):
            build_medical_datasets(forged)
        train = build_medical_datasets(manifest)['train']
        train[0]
        path = self.root / self.rows[0][2]
        image = nib.load(path)
        save_nifti(path, np.asarray(image.dataobj) + 17.)
        with self.assertRaisesRegex(ValueError, 'changed since inspection'):
            train[1]
        self.assertNotEqual(inspect_medical_dataset(self.spec, 17)['dataset_digest'], manifest['dataset_digest'])

    def test_ct_window_and_mask_follow_physical_series_order_into_padded_patch(self):
        spec, expected = ct_fixture(self.root)
        manifest = inspect_medical_dataset(spec, 17)
        self.assertEqual(manifest['class_pixels']['train'], [118, 8])
        record = manifest['records'][0]
        case = read_medical_case(record, manifest['data_spec'])
        np.testing.assert_allclose(case['native_image'][0], expected, atol=1e-7)
        np.testing.assert_allclose(case['image'][0], expected, atol=1e-7)
        image, labels, details = preview_medical_patch(manifest)
        self.assertEqual(details['patch_start_dhw'], [0, 0, 0])
        self.assertEqual(details['pad_after_dhw'], [1, 2, 1])
        np.testing.assert_allclose(image[0, :3, :6, :7], expected, atol=1e-7)
        np.testing.assert_array_equal(labels[:3, :6, :7], case['native_labels'])
        self.assertEqual(int(labels.sum()), 8)
        np.testing.assert_array_equal(image[:, 3, :6, :7], image[:, 2, :6, :7])
        self.assertFalse(labels[3:].any())
    def test_ct_source_digest_covers_every_series_file_even_after_caching(self):
        from pydicom import dcmread

        spec, _ = ct_fixture(self.root)
        manifest = inspect_medical_dataset(spec, 17)
        dataset = build_medical_datasets(manifest)['train']
        dataset[0]
        path = self.root / 'ct0' / '.1.dcm'
        item = dcmread(path)
        item.PatientName = 'Synthetic Metadata Change'
        item.save_as(path, enforce_file_format=True)
        with self.assertRaisesRegex(ValueError, 'changed since inspection'):
            dataset[1]

    def test_enhanced_ct_series_can_use_a_single_volume_file(self):
        from pydicom import dcmread
        from pydicom.dataset import Dataset
        from pydicom.uid import EnhancedCTImageStorage

        spec, expected = ct_fixture(self.root)
        series = [dcmread(path) for path in sorted((self.root / 'ct0').iterdir(), reverse=True)]
        series.sort(key=lambda item: item.ImagePositionPatient[2])
        payload = b''.join(item.PixelData for item in series)
        item = series[0]
        item.file_meta.MediaStorageSOPClassUID = item.SOPClassUID = EnhancedCTImageStorage
        item.PerFrameFunctionalGroupsSequence = []
        for source in series:
            frame, position, transform = Dataset(), Dataset(), Dataset()
            position.ImagePositionPatient = source.ImagePositionPatient
            frame.PlanePositionSequence = [position]
            transform.RescaleSlope = 2
            transform.RescaleIntercept = -100
            transform.RescaleType = 'HU'
            frame.PixelValueTransformationSequence = [transform]
            item.PerFrameFunctionalGroupsSequence.append(frame)
        item.NumberOfFrames = len(series)
        item.PixelData = payload
        item.save_as(self.root / 'ct0.dcm', enforce_file_format=True)
        path = Path(spec['path'])
        with path.open('r', encoding='utf-8', newline='') as stream:
            rows = list(csv.reader(stream))[1:]
        rows[0][2] = 'ct0.dcm'
        write_rows(path, rows)
        manifest = inspect_medical_dataset(spec, 17)
        case = read_medical_case(manifest['records'][0], manifest['data_spec'])
        np.testing.assert_allclose(case['native_image'][0], expected, atol=1e-7)
    def test_cancellation_propagates(self):
        with self.assertRaises(InterruptedError):
            inspect_medical_dataset(self.spec, 17, cancelled=lambda: True)
