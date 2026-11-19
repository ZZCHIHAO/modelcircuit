"""Bounded previews and original-grid segmentation masks for test predictions."""
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from .classification_runtime import _atomic_write
from .segmentation_data import resize_labels


_COLORS = np.array([(249, 115, 22), (16, 185, 129), (96, 165, 250), (217, 70, 239),
                    (234, 179, 8), (239, 68, 68), (34, 211, 238), (167, 139, 250)], dtype=np.float32)


def _overlay(base, labels):
    colors = _COLORS[(labels - 1) % len(_COLORS)]
    return np.where((labels != 0)[..., None], base * .5 + colors * .5, base).astype(np.uint8)


def _preview(path, features, labels, predicted):
    slice_index, slice_count = 0, 1
    if predicted.ndim == 3:
        slice_count = len(predicted)
        foreground = (labels != 0).sum(axis=(1, 2))
        slice_index = int(foreground.argmax()) if foreground.any() else slice_count // 2
        features, labels, predicted = features[:, slice_index], labels[slice_index], predicted[slice_index]
    image = (np.clip(features, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    if image.shape[-1] == 1:
        image = image.repeat(3, axis=2)
    panels = [Image.fromarray(array) for array in (image, _overlay(image, labels), _overlay(image, predicted))]
    for panel in panels:
        panel.thumbnail((256, 192))
    width, height = panels[0].size
    preview = Image.new('RGB', (3 * width + 8, height), (25, 32, 44))
    for index, panel in enumerate(panels):
        preview.paste(panel, (index * (width + 4), 0))
    _atomic_write(path, lambda stream: preview.save(stream, format='PNG'))
    return slice_index + 1, slice_count


def _write_native_mask(path, predicted, geometry, mask_values):
    native_shape = geometry.get('source_shape_dhw', geometry.get('source_shape_hw'))
    dtype = np.uint8 if max(mask_values) <= 255 else np.uint16
    values = np.array(mask_values, dtype=dtype)[resize_labels(predicted, native_shape)]
    if path.suffix == '.png':
        _atomic_write(path, lambda stream: Image.fromarray(values).save(stream, format='PNG'))
        return

    import nibabel as nib

    image_type = nib.Nifti2Image if geometry['nifti_version'] == 2 else nib.Nifti1Image
    affine = np.asarray(geometry['affine_xyz'], dtype=np.float64)
    image = image_type(values.transpose(2, 1, 0), affine)
    image.set_sform(affine, code=geometry['affine_code'])
    image.header.set_xyzt_units(geometry['spatial_unit'])
    # NIfTI's writer selects compression from the path suffix.
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix='.nii.gz')
    os.close(descriptor)
    try:
        nib.save(image, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class SegmentationPredictions:
    def __init__(self, directory, manifest, check):
        self.directory = directory / 'masks'
        self.directory.mkdir(exist_ok=True)
        self.records = {record['id']: record['geometry'] for record in manifest['records'] if record['split'] == 'test'}
        self.mask_values = manifest['mask_values']
        self.check = check
        self.sequence = 0

    def rows(self, outputs, batch):
        features = next(iter(batch['features'].values()))
        predictions = outputs['mask_logits'].argmax(dim=1)
        for identifier, image, target, predicted in zip(batch['sample_ids'], features, batch['targets'], predictions):
            self.check()
            geometry = self.records[identifier]
            self.sequence += 1
            name = f'{self.sequence:06d}'
            suffix = '.nii.gz' if 'source_shape_dhw' in geometry else '.png'
            native = self.directory / (name + suffix)
            preview = self.directory / (name + '-preview.png')
            predicted = predicted.cpu().numpy()
            _write_native_mask(native, predicted, geometry, self.mask_values)
            index, count = _preview(preview, image.cpu().numpy(), target.cpu().numpy(), predicted)
            yield {'sample_id': identifier, 'mask_file': 'masks/' + native.name, 'preview_file': 'masks/' + preview.name,
                   'slice_index': index, 'slice_count': count}
