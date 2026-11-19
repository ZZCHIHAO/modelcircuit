import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from studio_core.research_decoder import decode


def config(format, size, depth=1, channels=1, **extra):
    return {'kind': 'research_csv', 'format': format, 'image_size': size, 'depth': depth,
            'channels': channels, 'intensity_min': None, 'intensity_max': None, **extra}


class ResearchDecoderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_image_grayscale_and_gray16_keep_distinct_high_values(self):
        jpg = self.root / 'ordinary.jpg'
        Image.new('RGB', (4, 3), (60, 60, 60)).save(jpg)
        image, meta = decode(jpg, config('image', [3, 4]))
        self.assertEqual(image.shape, (1, 1, 3, 4))
        self.assertEqual(image.dtype, np.float32)
        np.testing.assert_allclose(image, 60 / 255)
        self.assertEqual(meta['axis_order'], ['C', 'D', 'H', 'W'])

        high = self.root / 'original.png'
        original = np.array([[0, 500], [1000, 4095]], dtype=np.uint16)
        Image.fromarray(original).save(high)
        spec = dict(config('gray16', [2, 2]), intensity_min=0, intensity_max=4095)
        image, meta = decode(high, spec)
        np.testing.assert_allclose(image[0, 0], original.astype(np.float32) / 4095)
        self.assertEqual(meta['intensity_bounds'], [0.0, 4095.0])
        with self.assertRaisesRegex(ValueError, 'gray16'):
            decode(high, config('image', [2, 2]))
        with self.assertRaises(InterruptedError):
            decode(high, spec, cancelled=lambda: True)

    @unittest.skipUnless(importlib.util.find_spec('nibabel'), 'Requires nibabel')
    def test_nifti_orientation_preserves_intensity_and_geometry(self):
        import nibabel as nib

        path = self.root / 'volume.nii.gz'
        data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
        affine = np.diag([-2., 3., 4., 1.])
        nib.save(nib.Nifti1Image(data, affine), path)
        output, meta = decode(path, config('nifti', [3, 2], depth=4))
        self.assertEqual(output.shape, (1, 4, 3, 2))
        np.testing.assert_allclose(output[0], data[::-1].transpose(2, 1, 0) / 23, atol=1e-6)
        self.assertEqual(meta['orientation'], 'RAS')
        self.assertEqual(meta['source_spacing_dhw'], [4, 3, 2])

    @unittest.skipUnless(importlib.util.find_spec('pydicom'), 'Requires pydicom')
    def test_dicom_series_sorted_by_position_and_rescaled_to_hu(self):
        from pydicom.uid import generate_uid

        uid = generate_uid()
        folder = self.root / 'series'
        folder.mkdir()
        self.write_slice(folder / 'first.dcm', uid, z=6, value=550)
        self.write_slice(folder / 'last.dcm', uid, z=3, value=500)
        spec = dict(config('dicom', [2, 2], depth=2), intensity_min=0, intensity_max=200)
        output, meta = decode(folder, spec)
        np.testing.assert_allclose(output[0, :, 0, 0], [0, 0.5])
        self.assertEqual(meta['source_spacing_dhw'], [3, 0.7, 0.4])
        self.assertEqual(meta['modality'], 'CT')
        self.assertNotIn(uid, str(meta))
        self.assertNotIn('Private Name', str(meta))
        self.write_slice(folder / 'mixed.dcm', generate_uid(), z=9, value=600)
        with self.assertRaisesRegex(ValueError, 'series'):
            decode(folder, spec)

    @unittest.skipUnless(importlib.util.find_spec('pydicom'), 'Requires pydicom')
    def test_oct_multiframe_preserves_scan_order_when_geometry_is_unavailable(self):
        import pydicom
        from pydicom.uid import OphthalmicTomographyImageStorage, generate_uid

        path = self.root / 'oct.dcm'
        self.write_slice(path, generate_uid(), z=0, value=0)
        ds = pydicom.dcmread(path)
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID = OphthalmicTomographyImageStorage
        ds.Modality = 'OPT'
        ds.NumberOfFrames = 3
        for field in ('ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing', 'RescaleSlope', 'RescaleIntercept'):
            delattr(ds, field)
        ds.PixelData = np.repeat(np.asarray([40, 10, 90], dtype='<u2'), 4).tobytes()
        ds.save_as(path, enforce_file_format=True)
        spec = dict(config('dicom', [2, 2], depth=3), intensity_min=0, intensity_max=100, modality='OCT')
        output, meta = decode(path, spec)
        np.testing.assert_allclose(output[0, :, 0, 0], [.4, .1, .9])
        self.assertEqual(meta['geometry'], 'frame_order_only')
        self.assertEqual(meta['modality'], 'OCT')
        with self.assertRaisesRegex(ValueError, 'modality'):
            decode(path, dict(spec, modality='CT'))

    @unittest.skipUnless(importlib.util.find_spec('pydicom'), 'Requires pydicom')
    def test_enhanced_dicom_applies_each_frame_slope_and_orders_by_geometry(self):
        import pydicom
        from pydicom.dataset import Dataset
        from pydicom.uid import EnhancedCTImageStorage, generate_uid

        path = self.root / 'enhanced.dcm'
        self.write_slice(path, generate_uid(), z=0, value=0)
        ds = pydicom.dcmread(path)
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID = EnhancedCTImageStorage
        ds.NumberOfFrames = 2
        ds.PerFrameFunctionalGroupsSequence = []
        for z, slope in ((2, 3), (0, 2)):
            frame, position, rescale = Dataset(), Dataset(), Dataset()
            position.ImagePositionPatient = [0, 0, z]
            rescale.RescaleSlope, rescale.RescaleIntercept = slope, 0
            frame.PlanePositionSequence = [position]
            frame.PixelValueTransformationSequence = [rescale]
            ds.PerFrameFunctionalGroupsSequence.append(frame)
        ds.PixelData = np.full((2, 2, 2), 10, dtype='<u2').tobytes()
        ds.save_as(path, enforce_file_format=True)
        output, meta = decode(path, dict(config('dicom', [2, 2], depth=2), intensity_min=0, intensity_max=100))
        np.testing.assert_allclose(output[0, :, 0, 0], [.2, .3])
        self.assertEqual(meta['source_spacing_dhw'], [2, .7, .4])

    @staticmethod
    def write_slice(path, series_uid, z, value):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

        file_meta = FileMetaDataset()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        image = FileDataset(path, {}, file_meta=file_meta, preamble=b'\0' * 128)
        image.SOPClassUID = file_meta.MediaStorageSOPClassUID
        image.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        image.SeriesInstanceUID = series_uid
        image.PatientName = 'Private Name'
        image.Modality = 'CT'
        image.Rows = image.Columns = 2
        image.SamplesPerPixel = 1
        image.PhotometricInterpretation = 'MONOCHROME2'
        image.PixelRepresentation = 0
        image.BitsAllocated = image.BitsStored = 16
        image.HighBit = 15
        image.ImagePositionPatient = [5, 4, z]
        image.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        image.PixelSpacing = [0.7, 0.4]
        image.RescaleSlope, image.RescaleIntercept = 2, -1000
        image.PixelData = np.full((2, 2), value, dtype='<u2').tobytes()
        image.save_as(path, enforce_file_format=True)

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'Requires OpenCV')
    def test_video_uniform_frame_sampling(self):
        import cv2

        path = self.root / 'clip.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 6, (16, 16))
        if not writer.isOpened():
            self.skipTest('MJPG encoder unavailable')
        try:
            for i in range(5):
                writer.write(np.full((16, 16, 3), 40 * i, dtype=np.uint8))
        finally:
            writer.release()
        output, meta = decode(path, config('video', [8, 8], depth=3, channels=3))
        self.assertEqual(output.shape, (3, 3, 8, 8))
        self.assertEqual(meta['frame_indices'], [0, 2, 4])
        np.testing.assert_allclose(output[0, :, 0, 0], [0, 80/255, 160/255], atol=0.03)
        self.assertEqual(meta['fps'], 6)
