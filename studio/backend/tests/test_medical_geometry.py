"""Synthetic volumes exercise physical grids without patient data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


def grid(shape, affine, unit='mm', *, version=1):
    return {'source_shape_dhw': list(shape), 'affine_xyz': np.asarray(affine).tolist(),
            'spatial_unit': unit, 'affine_code': 1, 'nifti_version': version}


class MedicalGeometryTests(unittest.TestCase):
    def test_grids_compare_in_mm_and_reject_subvoxel_misalignment(self):
        from studio_core.medical_geometry import affine_mm, assert_same_grid, spacing_mm

        affine = np.array([[-2, 0, 0, 13], [0, 3, 0, -4], [0, 0, 5, 72], [0, 0, 0, 1.]])
        in_microns = affine.copy()
        in_microns[:3] *= 1000
        reference = grid((5, 4, 3), affine)
        labels = grid((5, 4, 3), in_microns, 'micron')
        np.testing.assert_array_equal(affine_mm(labels), affine)
        self.assertEqual(spacing_mm(labels), [5, 3, 2])
        assert_same_grid(reference, labels)

        flipped = affine.copy()
        flipped[:, 0] *= -1
        flipped[0, 3] -= 4
        with self.assertRaisesRegex(ValueError, 'align|affine|grid'):
            assert_same_grid(reference, grid((5, 4, 3), flipped))
        with self.assertRaisesRegex(ValueError, 'shape|grid'):
            assert_same_grid(reference, grid((5, 4, 4), affine))

        tiny = np.diag([1e-5, 1e-5, 1e-5, 1.])
        shifted = tiny.copy()
        shifted[1, 3] += 1e-5
        with self.assertRaisesRegex(ValueError, 'align|affine|grid'):
            assert_same_grid(grid((4, 4, 4), tiny), grid((4, 4, 4), shifted))

    def test_contains_accepts_shared_physical_fov_with_different_sampling(self):
        from studio_core.medical_geometry import assert_contains

        reference = grid((4, 4, 4), np.eye(4))
        coarse = np.diag([2., 2., 2., 1.])
        coarse[:3, 3] = .5
        assert_contains(reference, grid((2, 2, 2), coarse))

        reversed_x = coarse.copy()
        reversed_x[0, 0], reversed_x[0, 3] = -2, 2.5
        assert_contains(reference, grid((2, 2, 2), reversed_x))

        cropped = coarse.copy()
        cropped[0, 3] += .6
        with self.assertRaisesRegex(ValueError, 'cover|field of view'):
            assert_contains(reference, grid((2, 2, 2), cropped))

    def test_model_grid_covers_fov_and_keeps_center_and_physical_directions(self):
        from studio_core.medical_geometry import affine_mm, make_model_grid, spacing_mm

        affine = np.array([[0, -3, 0, 11], [-2, 0, 0, 4], [0, 0, 5, 72], [0, 0, 0, 1.]])
        native = affine.copy()
        native[:3] /= 1000
        reference = grid((3, 5, 7), native, 'meter')
        model = make_model_grid(reference, (4, 2, 3))
        self.assertEqual(model['source_shape_dhw'], [4, 8, 5])
        np.testing.assert_allclose(spacing_mm(model), [4, 2, 3])
        target = affine_mm(model)
        np.testing.assert_allclose(target[:3, :3], [[0, -2, 0], [-3, 0, 0], [0, 0, 4]])
        np.testing.assert_allclose(affine @ [3, 2, 1, 1], target @ [2, 3.5, 1.5, 1])
        with self.assertRaisesRegex(ValueError, 'budget|limit'):
            make_model_grid(reference, (0.001, 0.001, 0.001))
        with self.assertRaisesRegex(ValueError, 'spacing'):
            make_model_grid(reference, (1, float('inf'), 1))

    def test_model_grid_reusing_native_spacing_reuses_the_exact_grid(self):
        from studio_core.medical_geometry import make_model_grid, spacing_mm

        affine = [[1., .1, 0., 23.3], [0., .1, 0., -4.2], [0., .1, 1., -19.7], [0., 0., 0., 1.]]
        reference = grid((33, 56, 124), affine)
        self.assertEqual(make_model_grid(reference, spacing_mm(reference)), reference)

    def test_resample_linear_uses_world_transform_and_extends_nearest_edge(self):
        from studio_core.medical_geometry import resample_volume

        source = grid((2, 3, 4), np.eye(4))
        transformed = np.eye(4)
        transformed[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
        transformed[:3, 3] = [-0.25, 2.25, .25]
        target = grid((1, 5, 4), transformed)
        d, h, w = np.indices((2, 3, 4))
        values = 100 * d + 10 * h + w
        result = resample_volume(values, source, target, order=1)
        self.assertEqual(result.dtype, np.float32)
        expected = 25 + 10 * np.array([2, 1.25, .25, 0]) + np.array([0, .75, 1.75, 2.75, 3])[:, None]
        np.testing.assert_allclose(result[0], expected)

    def test_resample_nearest_retains_stored_large_ids_through_flip_and_ties(self):
        from studio_core.medical_geometry import resample_volume

        source = grid((2, 2, 3), np.eye(4))
        transformed = np.diag([-1., 1., 1., 1.])
        transformed[0, 3] = 1.5
        target = grid((2, 2, 3), transformed)
        labels = (np.arange(12, dtype=np.uint64) + 2**63).reshape((2, 2, 3))
        result = resample_volume(labels, source, target, order=0)
        self.assertEqual(result.dtype, np.uint64)
        np.testing.assert_array_equal(result, labels[..., [2, 1, 0]])
        with self.assertRaisesRegex(ValueError, 'shape|grid'):
            resample_volume(labels[:1], source, target, order=0)

    def test_affine_rejects_malformed_bottom_row_rank_and_units(self):
        from studio_core.medical_geometry import affine_mm

        for affine, unit in [(np.diag([1., 1., 1., 2.]), 'mm'),
                             (np.diag([1., 1., 0., 1.]), 'mm'),
                             (np.diag([1., 1., np.nan, 1.]), 'mm'),
                             (np.eye(4), 'unknown')]:
            with self.subTest(affine=affine, unit=unit), self.assertRaisesRegex(ValueError, 'affine|unit'):
                affine_mm(grid((2, 2, 2), affine, unit))


@unittest.skipUnless(importlib.util.find_spec('nibabel'), 'Requires nibabel')
class NiftiMedicalGeometryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_scaled_nifti_retains_native_flip_and_reference_units(self):
        import nibabel as nib
        from studio_core.medical_geometry import read_nifti, spacing_mm

        values = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        affine = np.diag([-2400., 3000., 4100., 1.])
        affine[:3, 3] = [61000, -1000, 5000]
        image = nib.Nifti2Image(values, affine)
        image.header.set_xyzt_units('micron')
        image.header.set_slope_inter(2.5, -7)
        path = self.root / 'volume.nii.gz'
        nib.save(image, path)
        result, geometry = read_nifti(path)
        np.testing.assert_array_equal(result, values.transpose(2, 1, 0) * 2.5 - 7)
        np.testing.assert_array_equal(geometry['affine_xyz'], affine)
        np.testing.assert_allclose(spacing_mm(geometry), [4.1, 3, 2.4])
        self.assertEqual(geometry['spatial_unit'], 'micron')
        self.assertEqual(geometry['nifti_version'], 2)

    def test_malformed_nifti_and_unknown_units_raise_value_error(self):
        import nibabel as nib
        from studio_core.medical_geometry import read_nifti

        path = self.root / 'volume.nii'
        image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.int16), np.eye(4))
        nib.save(image, path)
        with self.assertRaisesRegex(ValueError, 'unit'):
            read_nifti(path)

        image.header.set_xyzt_units('mm')
        nib.save(image, path)
        original = path.read_bytes()
        # Write an invalid on-disk scaling slope. nibabel would silently treat
        # this as unscaled data, so validate the raw header in this workflow.
        slope_offset = image.header.structarr.dtype.fields['scl_slope'][1]
        invalid = bytearray(original)
        invalid[slope_offset:slope_offset + 4] = np.asarray(np.inf, dtype='<f4').tobytes()
        path.write_bytes(invalid)
        with self.assertRaisesRegex(ValueError, 'scal'):
            read_nifti(path)

        intercept_offset = image.header.structarr.dtype.fields['scl_inter'][1]
        invalid = bytearray(original)
        invalid[intercept_offset:intercept_offset + 4] = np.asarray(np.nan, dtype='<f4').tobytes()
        path.write_bytes(invalid)
        with self.assertRaisesRegex(ValueError, 'scal'):
            read_nifti(path)

        image.set_sform(np.diag([1., 0., 1., 1.]), code=1)
        nib.save(image, path)
        with self.assertRaisesRegex(ValueError, 'affine'):
            read_nifti(path)

    def test_writes_native_values_without_copying_source_metadata(self):
        import nibabel as nib
        from studio_core.medical_geometry import read_nifti, write_native_labels

        affine = np.array([[-2, 0, .2, 24], [0, 3, 0, -43], [0, 0, 4, 21], [0, 0, 0, 1.]])
        labels = np.array([0, 700, 23, 0, 4, 0, 700, 0], dtype=np.uint16).reshape(2, 2, 2)
        source = nib.Nifti2Image(labels.transpose(2, 1, 0), affine)
        source.header.set_xyzt_units('micron')
        source.header['descrip'] = 'Synthetic name to exclude'
        input_path = self.root / 'source.nii'
        nib.save(source, input_path)
        _, geometry = read_nifti(input_path)

        path = self.root / 'native.nii.gz'
        write_native_labels(path, labels, geometry)
        output = nib.load(path)
        self.assertEqual(output.header.get_xyzt_units()[0], 'micron')
        self.assertEqual(output.get_data_dtype().kind, 'u')
        np.testing.assert_array_equal(np.asarray(output.dataobj), labels.transpose(2, 1, 0))
        np.testing.assert_array_equal(output.affine, affine)
        self.assertEqual(bytes(output.header['descrip']).rstrip(b'\x00'), b'')
        self.assertEqual(len(output.header.extensions), 0)
        with self.assertRaisesRegex(ValueError, 'shape|grid'):
            write_native_labels(path, labels[:, :, :1], geometry)

    def test_qform_only_native_affine_can_round_trip_without_precision_loss(self):
        import nibabel as nib
        from studio_core.medical_geometry import read_nifti, write_native_labels

        labels = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        angle = .37
        affine = np.eye(4)
        affine[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        affine[:3, 3] = [-4.5, 7.75, 30.]
        source = nib.Nifti1Image(labels.transpose(2, 1, 0), affine)
        source.set_qform(affine, code=1)
        source.set_sform(None, code=0)
        source.header.set_xyzt_units('mm')
        source_path = self.root / 'qform_source.nii'
        nib.save(source, source_path)
        data, geometry = read_nifti(source_path)
        path = self.root / 'qform_labels.nii'
        write_native_labels(path, data, geometry)
        output = nib.load(path)
        np.testing.assert_array_equal(output.affine, nib.load(source_path).affine)
        np.testing.assert_array_equal(np.asarray(output.dataobj), labels.transpose(2, 1, 0))


@unittest.skipUnless(importlib.util.find_spec('pydicom'), 'Requires pydicom')
class CTMedicalGeometryTests(unittest.TestCase):
    def setUp(self):
        from pydicom.uid import generate_uid

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.series_uid, self.frame_uid = generate_uid(), generate_uid()
        self.first_row = np.array([.6, .8, 0.])
        self.first_column = np.array([0., 0., 1.])
        self.origin = np.array([11., -16., 5.75])
        self.normal = np.array([.8, -.6, 0.])

    def write_slice(self, name, index, *, value=None, slope=2, intercept=-1000,
                    rescale_type=None, image_type=('ORIGINAL', 'PRIMARY', 'AXIAL'),
                    multi_energy=None, lateral=0.):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

        path = self.root / name
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        dataset = FileDataset(path, {}, file_meta=meta, preamble=b'\0' * 128)
        dataset.SOPClassUID, dataset.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        dataset.SeriesInstanceUID, dataset.FrameOfReferenceUID = self.series_uid, self.frame_uid
        dataset.Modality, dataset.PatientName = 'CT', 'Synthetic Patient'
        dataset.Rows, dataset.Columns = 2, 3
        dataset.SamplesPerPixel, dataset.PhotometricInterpretation = 1, 'MONOCHROME2'
        dataset.PixelRepresentation, dataset.BitsAllocated, dataset.BitsStored, dataset.HighBit = 1, 16, 16, 15
        dataset.ImageOrientationPatient = np.r_[self.first_row, self.first_column].tolist()
        dataset.ImagePositionPatient = (self.origin + index * 2.5 * self.normal + lateral * self.first_row).tolist()
        dataset.PixelSpacing = [1.8, .35]
        dataset.SliceThickness = 17.0  # Slice positions determine the depth spacing.
        if image_type is not None:
            dataset.ImageType = list(image_type)
        if slope is not None:
            dataset.RescaleSlope = slope
        if intercept is not None:
            dataset.RescaleIntercept = intercept
        if rescale_type is not None:
            dataset.RescaleType = rescale_type
        if multi_energy is not None:
            dataset.MultienergyCTAcquisition = multi_energy
        values = np.arange(6, dtype='<i2').reshape((2, 3)) + (index * 10 if value is None else value)
        dataset.PixelData = values.astype('<i2').tobytes()
        dataset.save_as(path, enforce_file_format=True)
        return path

    def test_oblique_ct_uses_sorted_slice_origins_correct_pixel_axes_and_hu(self):
        from studio_core.medical_geometry import affine_mm, read_ct, spacing_mm

        self.write_slice('first.dcm', 1, slope=.5, intercept=10)
        self.write_slice('second.dcm', 0, value=-10)
        data, geometry = read_ct(self.root)
        self.assertEqual(data.dtype, np.float32)
        voxels = np.arange(6).reshape(2, 3)
        np.testing.assert_array_equal(data[0], (voxels - 10) * 2 - 1000)
        np.testing.assert_array_equal(data[1], (voxels + 10) * .5 + 10)
        np.testing.assert_allclose(spacing_mm(geometry), [2.5, 1.8, .35])

        # DICOM maps column indices along the first row direction. The row
        # spacing multiplies the first column direction. Check in world RAS.
        affine = affine_mm(geometry)
        ras_from_lps = [-1., -1., 1.]
        np.testing.assert_allclose(affine[:3, 3], self.origin * ras_from_lps)
        lps = self.origin + 2 * .35 * self.first_row + 1.8 * self.first_column + 2.5 * self.normal
        np.testing.assert_allclose(affine[:3] @ [2, 1, 1, 1], lps * ras_from_lps)
        self.assertNotIn('Synthetic Patient', str(geometry))
        self.assertNotIn(self.series_uid, str(geometry))
        with self.assertRaises(InterruptedError):
            read_ct(self.root, cancelled=lambda: True)

    def test_strict_ct_requires_documented_hu_calibration_for_each_slice(self):
        from studio_core.medical_geometry import read_ct

        self.write_slice('first.dcm', 0)
        unsupported = [{'slope': None, 'intercept': None}, {'intercept': None}, {'slope': 0},
                       {'rescale_type': 'OD'}, {'image_type': None},
                       {'image_type': ('DERIVED', 'PRIMARY', 'AXIAL')}, {'multi_energy': 'YES'},
                       {'image_type': ('ORIGINAL', 'PRIMARY', 'LOCALIZER'), 'rescale_type': 'HU'}]
        for settings in unsupported:
            with self.subTest(settings=settings):
                self.write_slice('second.dcm', 1, **settings)
                with self.assertRaisesRegex(ValueError, 'HU|calibrat|rescale|LOCALIZER'):
                    read_ct(self.root)

        self.write_slice('second.dcm', 1, image_type=('DERIVED', 'PRIMARY', 'AXIAL'),
                         multi_energy='YES', rescale_type='HU')
        voxels, _ = read_ct(self.root)
        self.assertEqual(voxels[1, 0, 0], -980)

    def test_enhanced_ct_uses_explicit_per_frame_calibration(self):
        from pydicom import dcmread
        from pydicom.dataset import Dataset
        from pydicom.uid import EnhancedCTImageStorage
        from studio_core.medical_geometry import read_ct

        path = self.write_slice('enhanced.dcm', 0)
        dataset = dcmread(path)
        dataset.file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID = EnhancedCTImageStorage
        dataset.NumberOfFrames = 2
        dataset.PerFrameFunctionalGroupsSequence = []
        for index, slope, intercept in ((1, 4, 3), (0, 2, -100)):
            frame, position, transform = Dataset(), Dataset(), Dataset()
            position.ImagePositionPatient = (self.origin + index * 2.5 * self.normal).tolist()
            frame.PlanePositionSequence = [position]
            transform.RescaleSlope = slope
            transform.RescaleIntercept = intercept
            transform.RescaleType = 'HU'
            frame.PixelValueTransformationSequence = [transform]
            dataset.PerFrameFunctionalGroupsSequence.append(frame)
        dataset.PixelData = np.full((2, 2, 3), 7, dtype='<i2').tobytes()
        dataset.save_as(path, enforce_file_format=True)
        data, _ = read_ct(path)
        np.testing.assert_array_equal(data[:, 0, 0], [-86, 31])

        transform = dataset.PerFrameFunctionalGroupsSequence[0].PixelValueTransformationSequence[0]
        del transform.RescaleIntercept
        dataset.save_as(path, enforce_file_format=True)
        with self.assertRaisesRegex(ValueError, 'HU|calibrat|rescale'):
            read_ct(path)

    def test_ct_requires_multiple_slices_and_regular_affine_geometry(self):
        from studio_core.medical_geometry import read_ct

        path = self.write_slice('first.dcm', 0)
        with self.assertRaisesRegex(ValueError, 'two|2 slices'):
            read_ct(path)
        self.write_slice('second.dcm', 1, lateral=.01)
        with self.assertRaisesRegex(ValueError, 'displacement|affine|tilted'):
            read_ct(self.root)
