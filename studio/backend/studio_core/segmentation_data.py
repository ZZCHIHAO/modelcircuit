"""Paired raster and NIfTI segmentation sources in their native voxel grid."""
import hashlib
from itertools import product
import math
from pathlib import Path
import random

import numpy as np

from .classification_data import (IMAGE_EXTENSIONS, MAX_IMAGE_BYTES, SPLITS, _cancel,
                                  _check_partition, _csv_records, _manifest_digest)
from .research_data import MAX_CAPTURE_BYTES, _fingerprint
from .research_decoder import _normalise, _resize_volume, _source_budget, _voxel_budget


COLUMNS = ('id_column', 'group_column', 'image_column', 'mask_column', 'split_column')
# TIFF tag 259: none, fax 1D/3/4, LZW, Deflate, PackBits, and old Deflate.
_LOSSLESS_TIFF = frozenset({1, 2, 3, 4, 5, 8, 32773, 32946})


def defaults():
    return {'kind': 'segmentation_csv', 'path': '', 'source_root': '', 'format': 'image',
            'id_column': 'sample_id', 'group_column': 'patient_id', 'image_column': 'image_path',
            'mask_column': 'mask_path', 'split_column': None, 'validation_fraction': .2,
            'test_fraction': .2, 'image_size': [32, 32], 'depth': 1, 'channels': 1,
            'mask_values': [0, 1]}


def validate_segmentation_spec(data):
    spec = defaults()
    if not isinstance(data, dict) or not data.keys() <= spec.keys() or data.get('kind') != spec['kind']:
        raise ValueError('Unsupported segmentation data settings')
    spec.update(data)
    for key in ('path', 'source_root'):
        if not isinstance(spec[key], str) or len(spec[key]) > 4096:
            raise ValueError(f'{key} must be a path string')
    columns = [spec[key] for key in COLUMNS if key != 'split_column' or spec[key] is not None]
    if any(not isinstance(name, str) or not name.strip() for name in columns) or len(set(columns)) != len(columns):
        raise ValueError('Segmentation CSV requires distinct sample, patient, image, mask and optional split columns')
    fmt, size, depth, channels = (spec[key] for key in ('format', 'image_size', 'depth', 'channels'))
    if fmt not in ('image', 'nifti') or not isinstance(size, list) or len(size) != 2 or any(
            type(axis) is not int or not 8 <= axis <= 512 for axis in size):
        raise ValueError('Choose raster or NIfTI pairs and height/width from 8 to 512')
    if type(depth) is not int or type(channels) is not int or (fmt == 'image' and (depth != 1 or channels not in (1, 3))) or (
            fmt == 'nifti' and (not 2 <= depth <= 128 or channels != 1)):
        raise ValueError('Raster requires depth 1 with 1 or 3 channels, NIfTI requires one channel and depth 2..128')
    values = spec['mask_values']
    if not isinstance(values, list) or not 2 <= len(values) <= 32 or any(type(value) is not int or not 0 <= value <= 65535 for value in values):
        raise ValueError('Declare 2..32 distinct integer mask values, including background 0')
    if values[0] != 0 or any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError('Mask values must start at background 0 and be strictly increasing')
    if depth * channels * math.prod(size) > 16 * 1024 ** 2:
        raise ValueError('Requested segmentation grid exceeds the 16 million input value budget')
    val, test = spec['validation_fraction'], spec['test_fraction']
    if any(type(value) not in (int, float) or not 0 < value < 1 for value in (val, test)) or val + test >= 1:
        raise ValueError('Split fractions must leave a training partition')
    return spec


def _pair_path(value, root):
    if not value or len(value) > 4096:
        raise ValueError('Missing or invalid image/mask path')
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Image/mask must be an existing file inside the selected source root')
    return path


def _raster(path, *, mask=False, channels=1):
    from PIL import Image

    extensions = {'.png', '.tif', '.tiff'} if mask else IMAGE_EXTENSIONS
    if path.suffix.lower() not in extensions or path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError('Raster pair requires supported images and lossless PNG/TIFF masks within the source budget')
    with Image.open(path) as image:
        if mask and image.format not in ('PNG', 'TIFF'):
            raise ValueError('Raster mask must be a PNG or TIFF with integer labels')
        if mask and image.format == 'TIFF' and image.tag_v2.get(259, 1) not in _LOSSLESS_TIFF:
            raise ValueError('Raster masks require a lossless TIFF compression (such as LZW or Deflate), or PNG')
        if getattr(image, 'n_frames', 1) != 1 or image.getexif().get(274, 1) != 1:
            raise ValueError('Raster pairs must be single-frame with orientation 1. Normalize orientation of both files before import.')
        _voxel_budget((image.height, image.width, 1 if mask else channels))
        if mask:
            if image.mode not in ('1', 'L', 'P', 'I', 'I;16', 'I;16L', 'I;16B', 'I;16N'):
                raise ValueError('Raster mask needs one plane of integer labels or palette indexes')
            return np.asarray(image)
        if image.mode in ('I', 'F') or image.mode.startswith('I;16'):
            if channels != 1:
                raise ValueError('High bit depth raster requires one channel')
            return np.asarray(image, dtype=np.float32)[None]
        converted = np.asarray(image.convert('RGB' if channels == 3 else 'L'), dtype=np.float32)
        return converted.transpose(2, 0, 1) if channels == 3 else converted[None]


def _nifti(path):
    import nibabel as nib

    if not path.name.lower().endswith(('.nii', '.nii.gz')):
        raise ValueError('Volume pairs require .nii or .nii.gz files')
    _source_budget(path)
    image = nib.load(path)
    if not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)) or len(image.shape) != 3:
        raise ValueError('Segmentation expects a single three-dimensional NIfTI volume')
    _voxel_budget(image.shape)
    affine = image.affine
    code = int(image.header['sform_code'] or image.header['qform_code'])
    if not code or affine is None or not np.isfinite(affine).all() or np.linalg.matrix_rank(affine[:3, :3]) != 3:
        raise ValueError('NIfTI needs a valid coded spatial affine (sform or qform)')
    values = np.asanyarray(image.dataobj)
    if values.dtype.kind not in 'biuf':
        raise ValueError('NIfTI must store real numeric voxels')
    geometry = {'source_shape_dhw': list(image.shape[::-1]), 'affine_xyz': affine.tolist(),
                'spatial_unit': image.header.get_xyzt_units()[0], 'affine_code': code,
                'nifti_version': 2 if isinstance(image, nib.Nifti2Image) else 1}
    return values.transpose(2, 1, 0), geometry


def resize_labels(labels, size):
    """Nearest sampling at output voxel centres with deterministic integer ties."""
    if labels.shape == tuple(size):
        return labels
    indices = [(2 * np.arange(target) + 1) * source // (2 * target)
               for source, target in zip(labels.shape, size)]
    return labels[np.ix_(*indices)]


def _digest(array):
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256(f'{values.shape}:{values.dtype.str}\n'.encode('ascii'))
    digest.update(memoryview(values).cast('B'))
    return digest.hexdigest()


def _load_pair(image_path, mask_path, spec, cancelled):
    _cancel(cancelled)
    if spec['format'] == 'nifti':
        image, geometry = _nifti(image_path)
        mask, mask_geometry = _nifti(mask_path)
        if image.shape != mask.shape:
            raise ValueError('Image/mask grid shape differs')
        affine = np.array(geometry['affine_xyz'])
        mask_affine = np.array(mask_geometry['affine_xyz'])
        corners = np.array([(*corner, 1) for corner in product(*[(0, max(1, n - 1)) for n in image.shape[::-1]])])
        unit = geometry['spatial_unit']
        tolerance = {'meter': 1e-6, 'micron': 1.}.get(unit, 1e-3)
        difference = (mask_affine - affine)[:3, :]
        deltas = corners @ difference.T
        voxel_deltas = np.linalg.solve(affine[:3, :3], deltas.T)
        # Physical tolerance alone permits a full-voxel shift in tiny voxel grids.
        if (mask_geometry['spatial_unit'] != unit or np.any(np.abs(deltas) > tolerance)
                or np.any(np.abs(voxel_deltas) > 1e-3)):
            raise ValueError('Image/mask affine or spatial units differ. Align the mask to the original image grid before import.')
        image = image[None]
        target = (spec['depth'], *spec['image_size'])
        scaled_raster = False
    else:
        image, mask = _raster(image_path, channels=spec['channels']), _raster(mask_path, mask=True)
        if image.shape[1:] != mask.shape:
            raise ValueError('Image/mask grid shape differs')
        geometry = {'source_shape_hw': list(mask.shape)}
        target = tuple(spec['image_size'])
        scaled_raster = float(image.min()) >= 0 and float(image.max()) <= 255
    _cancel(cancelled)
    if not np.isfinite(mask).all() or (mask.dtype.kind == 'f' and not np.equal(mask, np.floor(mask)).all()):
        raise ValueError('Mask must contain finite integer label values')
    values = np.array(spec['mask_values'])
    if not np.isin(np.unique(mask), values).all():
        raise ValueError('Mask contains undeclared labels. Set the mask values to match the source.')
    image = np.ascontiguousarray(image, dtype='<f4')
    if not np.isfinite(image).all():
        raise ValueError('Image contains non-finite or out-of-range values')
    image_digest = _digest(image)
    mask = np.ascontiguousarray(mask, dtype='<u2')
    mask_digest = _digest(mask)
    label = np.searchsorted(values, resize_labels(mask, target)).astype(np.int64)
    if scaled_raster:
        image /= 255.0
    else:
        image, _ = _normalise(image, None, None)
    # Resize only spatial axes, using the same voxel-centre convention as the labels.
    if len(target) == 2:
        image = np.stack([_resize_volume(channel[None], (1, *target), cancelled)[0] for channel in image])
    else:
        image = _resize_volume(image[0], target, cancelled)[None]
    return np.ascontiguousarray(image, dtype=np.float32), label, geometry, image_digest, mask_digest


def _assign_groups(records, spec, seed):
    groups = sorted({record['group'] for record in records})
    if len(groups) < 3:
        raise ValueError('Segmentation requires at least three distinct patients for train/val/test')
    random.Random(seed).shuffle(groups)
    val = min(len(groups) - 2, max(1, round(len(groups) * spec['validation_fraction'])))
    test = min(len(groups) - val - 1, max(1, round(len(groups) * spec['test_fraction'])))
    assignments = dict(zip(groups, ['val'] * val + ['test'] * test + ['train'] * (len(groups) - val - test)))
    for record in records:
        record['split'] = assignments[record['group']]


def inspect_segmentation_dataset(data, seed, cancelled=None):
    spec = validate_segmentation_spec(data)
    if not spec['path'].strip():
        raise ValueError('Select a segmentation CSV')
    source = Path(spec['path']).expanduser().resolve()
    root = Path(spec['source_root']).expanduser().resolve() if spec['source_root'] else source.parent
    spec.update(path=str(source), source_root=str(root))
    rows = _csv_records({**spec, 'target_column': None, 'feature_columns': []}, cancelled,
                        (spec['id_column'], spec['image_column'], spec['mask_column']))
    records, identifiers, paths, bytes_seen = [], set(), set(), 0
    for row in rows:
        _cancel(cancelled)
        identifier = row['extra_values'][spec['id_column']].strip()
        if not identifier or len(identifier) > 256 or identifier in identifiers:
            raise ValueError('Each segmentation sample needs a unique ID of 1..256 characters')
        identifiers.add(identifier)
        try:
            image = _pair_path(row['extra_values'][spec['image_column']].strip(), root)
            mask = _pair_path(row['extra_values'][spec['mask_column']].strip(), root)
            if image == mask or image in paths or mask in paths:
                raise ValueError('Image and mask files must be unique to each pair')
            paths.update((image, mask))
            image_source, image_bytes = _fingerprint(image, root, cancelled)
            mask_source, mask_bytes = _fingerprint(mask, root, cancelled)
            bytes_seen += image_bytes + mask_bytes
            if bytes_seen > MAX_CAPTURE_BYTES:
                raise ValueError('Paired data exceeds the 8 GiB inspection budget')
            _, labels, geometry, content, mask_content = _load_pair(image, mask, spec, cancelled)
        except (ValueError, OSError) as exc:
            raise ValueError(f'Segmentation CSV row {row["row_number"]}: {exc}') from exc
        records.append({'id': identifier, 'group': row['group'], 'split': row['split'], 'source_path': str(image),
                        'mask_path': str(mask), 'source_digest': image_source, 'mask_digest': mask_source,
                        'content_digest': content, 'mask_content_digest': mask_content, 'geometry': geometry,
                        'class_pixels': np.bincount(labels.ravel(), minlength=len(spec['mask_values'])).tolist()})
    if not records:
        raise ValueError('Segmentation CSV contains no pairs')
    if spec['split_column'] is None:
        _assign_groups(records, spec, seed)
    if {record['split'] for record in records} != set(SPLITS):
        raise ValueError('Segmentation requires nonempty train, val and test splits')
    counts, _ = _check_partition(records)
    class_pixels = {split: [0] * len(spec['mask_values']) for split in SPLITS}
    for record in records:
        totals = class_pixels[record['split']]
        for index, count in enumerate(record['class_pixels']):
            totals[index] += count
    if not any(class_pixels['train'][1:]):
        raise ValueError('Training masks contain no foreground on the model grid. Inspect masks, split or resize settings.')
    shape = [spec['depth'], *spec['image_size']] if spec['format'] == 'nifti' else spec['image_size']
    missing = [spec['mask_values'][i] for i, count in enumerate(class_pixels['train']) if not count]
    warnings = [f'Training grid has no voxels for declared mask values {missing}'] if missing else []
    manifest = {'kind': spec['kind'], 'version': 1, 'data_spec': spec, 'seed': seed, 'counts': counts,
                'records': records, 'input_shape': [None, spec['channels'], *shape], 'target_shape': [None, *shape],
                'mask_values': spec['mask_values'], 'class_pixels': class_pixels, 'warnings': warnings,
                'preprocessing': {'geometry': 'native_grid', 'image_resampling': 'linear_voxel_centres',
                                  'mask_resampling': 'nearest_voxel_centres', 'size': shape,
                                  'mask_values': spec['mask_values'], 'intensity': 'byte_range_or_per_image_minmax'}}
    manifest['dataset_digest'] = _manifest_digest(manifest)
    return manifest


def read_frozen_pair(record, spec, cancelled=None):
    root = Path(spec['source_root'])
    image = _pair_path(record['source_path'], root)
    mask = _pair_path(record['mask_path'], root)
    for path, key in ((image, 'source_digest'), (mask, 'mask_digest')):
        if _fingerprint(path, root, cancelled)[0] != record[key]:
            raise ValueError('Segmentation source changed since inspection: ' + path.name)
    features, labels, geometry, image_digest, mask_digest = _load_pair(image, mask, spec, cancelled)
    if (geometry, image_digest, mask_digest) != (record['geometry'], record['content_digest'], record['mask_content_digest']):
        raise ValueError('Segmentation pair changed while being read')
    return features, labels


def build_segmentation_datasets(manifest, cancelled=None):
    import torch
    from torch.utils.data import Dataset

    if manifest.get('version') != 1 or _manifest_digest(manifest) != manifest.get('dataset_digest'):
        raise ValueError('Segmentation manifest digest does not match its contents')
    spec = manifest['data_spec']

    class FrozenSegmentationDataset(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            record = self.records[index]
            features, labels = read_frozen_pair(record, spec, cancelled)
            return torch.from_numpy(features), torch.from_numpy(labels), record['id']

    return {split: FrozenSegmentationDataset([item for item in manifest['records'] if item['split'] == split]) for split in SPLITS}
