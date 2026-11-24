"""Versioned paired medical volumes and patient-level deterministic patch sampling."""
import math
from pathlib import Path
import random

import numpy as np

from .classification_data import SPLITS, _cancel, _check_partition, _csv_records, _json_digest, _manifest_digest
from .research_data import MAX_CAPTURE_BYTES, MAX_SOURCE_BYTES, _fingerprint
from .research_decoder import MAX_SERIES_FILES, SOURCE_VOXELS_LIMIT, _normalise
from .segmentation_data import COLUMNS, _assign_groups, _digest, defaults, float_image, validate_segmentation_spec


MEDICAL_FORMATS = ('dicom_ct', 'mri_nifti')
MEDICAL_MANIFEST_VERSION = 2
_PATCH_SAMPLER = 'patient_split_case_epoch_index_v1'
_MAX_CACHED_BYTES = 128 * 1024 ** 2


def medical_defaults(fmt):
    if fmt not in MEDICAL_FORMATS:
        raise ValueError('Choose DICOM CT or MRI NIfTI for medical volume data')
    return {**defaults(), 'format': fmt, 'depth': 16, 'spacing_mm': [1., 1., 1.],
            'ct_window': [-1000., 400.] if fmt == 'dicom_ct' else None,
            'sequence_columns': [], 'registered': False, 'foreground_fraction': .5,
            'patches_per_case': 4, 'overlap': .25}


def _number(value, low, high):
    return type(value) in (int, float) and low <= value <= high


def _patch_shape(spec):
    return (spec['depth'], *spec['image_size'])


def validate_medical_spec(data):
    if not isinstance(data, dict) or data.get('kind') != 'segmentation_csv':
        raise ValueError('Medical volume data requires segmentation CSV pairs')
    spec = medical_defaults(data.get('format'))
    if not data.keys() <= spec.keys():
        raise ValueError('Unsupported medical volume data settings')
    spec.update(data)
    legacy_fields = {key: spec[key] for key in defaults()}
    validate_segmentation_spec({**legacy_fields, 'format': 'nifti', 'channels': 1})
    columns = spec['sequence_columns']
    if (not isinstance(columns, list) or len(columns) > 3
            or any(not isinstance(name, str) or not name.strip() for name in columns)):
        raise ValueError('Declare up to three additional MRI sequence columns')
    names = [spec[key] for key in COLUMNS if spec[key] is not None] + columns
    if len(names) != len(set(names)):
        raise ValueError('Medical CSV requires distinct sample, patient, image, mask, split and sequence columns')
    if type(spec['registered']) is not bool:
        raise ValueError('The registered setting must be a boolean')
    window = spec['ct_window']
    if spec['format'] == 'dicom_ct':
        if columns or spec['registered']:
            raise ValueError('MRI sequence columns and registration acknowledgement apply to MRI NIfTI')
        if (not isinstance(window, list) or len(window) != 2
                or not all(_number(value, -1e12, 1e12) for value in window) or window[0] >= window[1]):
            raise ValueError('CT window must contain a finite lower HU bound below its upper bound')
        spec['ct_window'] = list(map(float, window))
    else:
        if columns and not spec['registered']:
            raise ValueError('Confirm that all MRI sequences are already registered to the reference')
        if window is not None:
            raise ValueError('The CT window applies only to DICOM CT')
    channels = len(columns) + 1
    if 'channels' in data and (type(data['channels']) is not int or data['channels'] != channels):
        raise ValueError('Image channel count must match the reference and additional MRI sequences')
    spec['channels'] = channels
    if math.prod(_patch_shape(spec)) * channels > 16 * 1024 ** 2:
        raise ValueError('Medical patch exceeds the 16 million input value budget')
    spacing = spec['spacing_mm']
    if (not isinstance(spacing, list) or len(spacing) != 3
            or not all(_number(axis, 1e-6, 1e6) for axis in spacing)):
        raise ValueError('Specify three finite, positive model spacing values in millimetres (depth, height, width)')
    spec['spacing_mm'] = list(map(float, spacing))
    if not _number(spec['foreground_fraction'], 0., 1.):
        raise ValueError('The foreground sampling probability must be between 0 and 1')
    if type(spec['patches_per_case']) is not int or not 1 <= spec['patches_per_case'] <= 128:
        raise ValueError('Training patches per case must be an integer from 1 to 128')
    if not _number(spec['overlap'], 0., 1.) or spec['overlap'] == 1:
        raise ValueError('Evaluation overlap must be at least 0 and below 1')
    spec['foreground_fraction'] = float(spec['foreground_fraction'])
    spec['overlap'] = float(spec['overlap'])
    spec['image_size'] = list(spec['image_size'])
    spec['mask_values'] = list(spec['mask_values'])
    spec['sequence_columns'] = list(columns)
    return spec


def medical_recipe(spec):
    fields = ('format', 'channels', 'spacing_mm', 'mask_values', 'ct_window', 'sequence_columns',
              'registered', 'foreground_fraction', 'patches_per_case', 'overlap')
    intensity = ('ct_hu_clip_before_resampling' if spec['format'] == 'dicom_ct'
                 else 'per_volume_channel_minmax_before_resampling')
    return {**{key: spec[key] for key in fields}, 'geometry': 'physical_affine_mm',
            'size': list(_patch_shape(spec)), 'image_resampling': 'linear_voxel_centres',
            'mask_resampling': 'nearest_voxel_centres', 'intensity': intensity,
            'patch_sampler': _PATCH_SAMPLER, 'patch_padding': 'image_edge_mask_background_after'}


def _source_path(value, root, *, allow_directory=False):
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError('Missing or invalid image, mask or MRI sequence path')
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not (path.is_file() or (allow_directory and path.is_dir())):
        expected = ('DICOM CT series directory or multiframe volume file' if allow_directory
                    else 'NIfTI image, mask or sequence file')
        raise ValueError(f'Select an existing {expected} inside the source root')
    return path


def _image_volume_budget(grid, channels):
    if math.prod(grid['source_shape_dhw']) * channels > SOURCE_VOXELS_LIMIT:
        raise ValueError('Medical volume exceeds the decoded image value budget across all channels')


def _read_input(image_path, sequence_paths, spec, cancelled, *, keep_native, digest):
    from .medical_geometry import assert_contains, make_model_grid, read_ct, read_nifti, resample_volume

    _cancel(cancelled)
    is_ct = spec['format'] == 'dicom_ct'
    reference, reference_grid = read_ct(image_path, cancelled) if is_ct else read_nifti(image_path)
    model_grid = make_model_grid(reference_grid, spec['spacing_mm'])
    channels = spec['channels']
    _image_volume_budget(reference_grid, channels)
    _image_volume_budget(model_grid, channels)
    model_image = np.empty((channels, *model_grid['source_shape_dhw']), dtype=np.float32)
    native_image = (np.empty((channels, *reference_grid['source_shape_dhw']), dtype=np.float32)
                    if keep_native else None)
    content_digests = []

    def process(channel, raw, source_grid):
        values = float_image(raw)
        if digest:
            content_digests.append(_digest(values))
        bounds = spec['ct_window'] if is_ct else (None, None)
        normalised, _ = _normalise(values, *bounds)
        _cancel(cancelled)
        model_image[channel] = resample_volume(normalised, source_grid, model_grid, order=1)
        if native_image is not None:
            native_image[channel] = (normalised if channel == 0 else resample_volume(
                normalised, source_grid, reference_grid, order=1))

    process(0, reference, reference_grid)
    del reference
    for channel, path in enumerate(sequence_paths, 1):
        _cancel(cancelled)
        volume, grid = read_nifti(path)
        assert_contains(reference_grid, grid)
        process(channel, volume, grid)
        del volume
    _cancel(cancelled)
    return {'image': model_image, 'native_image': native_image, 'reference_grid': reference_grid,
            'model_grid': model_grid, 'decoded_digests': content_digests}


def read_medical_input(image_path, sequence_paths, spec, cancelled=None):
    """Read an unlabeled CT series or MRI reference and ordered extra sequence paths."""
    spec = validate_medical_spec(spec)
    _cancel(cancelled)
    if not isinstance(sequence_paths, (tuple, list)) or len(sequence_paths) != spec['channels'] - 1:
        raise ValueError('Select the ordered MRI sequence files declared by the training recipe')
    try:
        paths = [Path(path).expanduser().resolve() for path in (image_path, *sequence_paths)]
    except TypeError as exc:
        raise ValueError('Select a medical reference and sequence paths') from exc
    if len(set(paths)) != len(paths):
        raise ValueError('Medical input requires distinct reference and sequence paths')
    digests = [_fingerprint_medical(path, path.parent, cancelled)[0] for path in paths]
    case = _read_input(paths[0], paths[1:], spec, cancelled, keep_native=True, digest=False)
    if digests != [_fingerprint_medical(path, path.parent, cancelled)[0] for path in paths]:
        raise ValueError('Medical input changed during decoding')
    return {**case, 'source_digests': digests}


def _load_case(image_paths, mask_path, spec, cancelled, *, keep_native):
    from .medical_geometry import assert_same_grid, read_nifti, resample_volume

    case = _read_input(image_paths[0], image_paths[1:], spec, cancelled, keep_native=keep_native, digest=True)
    _cancel(cancelled)
    mask, mask_grid = read_nifti(mask_path)
    assert_same_grid(case['reference_grid'], mask_grid)
    values = np.array(spec['mask_values'])
    present = np.unique(mask)
    if (not np.isfinite(present).all() or not np.equal(present, np.floor(present)).all()):
        raise ValueError('Mask must contain finite integer label values')
    if not np.isin(present, values).all():
        raise ValueError('Mask contains undeclared labels. Set mask values to match the source.')
    stored = np.ascontiguousarray(mask, dtype='<u2')
    del mask
    lookup = np.zeros(65536, dtype=np.uint8)
    lookup[values] = np.arange(len(values), dtype=np.uint8)
    native_labels = lookup[stored]
    model_labels = resample_volume(native_labels, case['reference_grid'], case['model_grid'], order=0)
    _cancel(cancelled)
    case.update(labels=np.ascontiguousarray(model_labels), native_labels=native_labels if keep_native else None,
                mask_content_digest=_digest(stored))
    return case


def _fingerprint_medical(path, root, cancelled):
    if not path.is_dir():
        return _fingerprint(path, root, cancelled)
    # The strict CT reader consumes every file in the flat series directory.
    digests, series_bytes = [], 0
    for entry in path.iterdir():
        _cancel(cancelled)
        if not entry.is_file() or entry.is_symlink() or len(digests) == MAX_SERIES_FILES:
            raise ValueError('DICOM series must be a flat directory within the file limit')
        digest, size = _fingerprint(entry, root, cancelled)
        digests.append(digest)
        series_bytes += size
        if series_bytes > MAX_SOURCE_BYTES:
            raise ValueError('DICOM series exceeds the 1 GiB raw source limit')
    if not digests:
        raise ValueError('DICOM series contains no slices')
    return _json_digest(sorted(digests)), series_bytes

def _check_sources(image_paths, mask_path, root, cancelled, record=None):
    expected = ([record['source_digest'], *record['sequence_source_digests'], record['mask_digest']]
                if record is not None else None)
    digests, size = [], 0
    for index, path in enumerate((*image_paths, mask_path)):
        value, byte_count = _fingerprint_medical(path, root, cancelled)
        if expected is not None and value != expected[index]:
            raise ValueError('Medical volume source changed since inspection: ' + path.name)
        digests.append(value)
        size += byte_count
    return digests, size


def _case_paths(record, spec, root):
    reference = _source_path(record['source_path'], root, allow_directory=spec['format'] == 'dicom_ct')
    sequences = [_source_path(path, root) for path in record['sequence_paths']]
    mask = _source_path(record['mask_path'], root)
    return (reference, *sequences), mask


def _source_partitions(records):
    seen_paths = {}
    for record in records:
        for path in (record['source_path'], *record['sequence_paths'], record['mask_path']):
            other = seen_paths.setdefault(path, record)
            if other['split'] != record['split']:
                raise ValueError('Cross-split medical source path leakage between cases ' + other['id'] + ' and ' + record['id'])
    seen_digests = {'source': {}, 'decoded': {}}
    for record in records:
        for kind, digests in (('source', [record['source_digest'], *record['sequence_source_digests']]),
                              ('decoded', [record['content_digest'], *record['sequence_content_digests']])):
            for digest in digests:
                other = seen_digests[kind].setdefault(digest, record)
                if other['split'] != record['split']:
                    raise ValueError(f'Cross-split medical {kind} volume leakage between cases {other["id"]} and {record["id"]}')


def inspect_medical_dataset(data, seed, cancelled=None):
    spec = validate_medical_spec(data)
    if not spec['path'].strip():
        raise ValueError('Select a medical segmentation CSV')
    source = Path(spec['path']).expanduser().resolve()
    root = Path(spec['source_root']).expanduser().resolve() if spec['source_root'] else source.parent
    spec.update(path=str(source), source_root=str(root))
    columns = [spec[key] for key in ('id_column', 'image_column', 'mask_column')] + spec['sequence_columns']
    rows = _csv_records({**spec, 'target_column': None, 'feature_columns': []}, cancelled, columns)
    records, identifiers = [], set()
    for row in rows:
        _cancel(cancelled)
        fields = row['extra_values']
        identifier = fields[spec['id_column']].strip()
        if not identifier or len(identifier) > 256 or identifier in identifiers:
            raise ValueError('Each medical case needs a unique sample ID of 1..256 characters')
        identifiers.add(identifier)
        try:
            image = _source_path(fields[spec['image_column']].strip(), root, allow_directory=spec['format'] == 'dicom_ct')
            sequences = [_source_path(fields[column].strip(), root) for column in spec['sequence_columns']]
            mask = _source_path(fields[spec['mask_column']].strip(), root)
            if len({image, mask, *sequences}) != 2 + len(sequences):
                raise ValueError('Choose distinct image, mask and sequence files for each case')
        except (OSError, ValueError) as exc:
            raise ValueError(f'Medical CSV row {row["row_number"]}: {exc}') from exc
        records.append({'id': identifier, 'group': row['group'], 'split': row['split'], 'source_path': str(image),
                        'mask_path': str(mask), 'relative_path': image.relative_to(root).as_posix(),
                        'mask_relative_path': mask.relative_to(root).as_posix(), 'sequence_paths': list(map(str, sequences)),
                        'sequence_relative_paths': [path.relative_to(root).as_posix() for path in sequences],
                        'content_digest': None})
    if not records:
        raise ValueError('Medical segmentation CSV contains no cases')
    if spec['split_column'] is None:
        _assign_groups(records, spec, seed)
    if {record['split'] for record in records} != set(SPLITS):
        raise ValueError('Medical segmentation requires nonempty train, val and test patient splits')
    counts, _ = _check_partition(records)
    class_count = len(spec['mask_values'])
    class_pixels = {split: [0] * class_count for split in SPLITS}
    bytes_seen = 0
    for record, row in zip(records, rows):
        _cancel(cancelled)
        try:
            image_paths, mask = _case_paths(record, spec, root)
            digests, size = _check_sources(image_paths, mask, root, cancelled)
            bytes_seen += size
            if bytes_seen > MAX_CAPTURE_BYTES:
                raise ValueError('Medical data exceeds the 8 GiB inspection budget')
            case = _load_case(image_paths, mask, spec, cancelled, keep_native=False)
        except (InterruptedError, TimeoutError):
            raise
        except (OSError, ValueError) as exc:
            raise ValueError(f'Medical CSV row {row["row_number"]}: {exc}') from exc
        pixels = np.bincount(case['labels'].ravel(), minlength=class_count).tolist()
        record.update(source_digest=digests[0], sequence_source_digests=digests[1:-1], mask_digest=digests[-1],
                      content_digest=case['decoded_digests'][0], sequence_content_digests=case['decoded_digests'][1:],
                      mask_content_digest=case['mask_content_digest'], geometry=case['reference_grid'],
                      model_grid=case['model_grid'], class_pixels=pixels)
        class_pixels[record['split']] = [total + value for total, value in zip(class_pixels[record['split']], pixels)]
        del case
    _source_partitions(records)
    if not any(class_pixels['train'][1:]):
        raise ValueError('Training masks contain no foreground on the model grid. Inspect masks, split or spacing settings.')
    absent = [value for value, count in zip(spec['mask_values'], class_pixels['train']) if not count]
    warnings = [f'Training grid has no voxels for declared mask values {absent}'] if absent else []
    shape = list(_patch_shape(spec))
    preprocessing = medical_recipe(spec)
    manifest = {'kind': spec['kind'], 'version': MEDICAL_MANIFEST_VERSION, 'data_spec': spec, 'seed': seed,
                'counts': counts, 'records': records, 'input_shape': [None, spec['channels'], *shape],
                'target_shape': [None, *shape], 'mask_values': spec['mask_values'], 'class_pixels': class_pixels,
                'warnings': warnings, 'preprocessing': preprocessing}
    _cancel(cancelled)
    manifest['dataset_digest'] = _manifest_digest(manifest)
    return manifest


def _read_frozen_case(record, spec, cancelled, *, keep_native):
    root = Path(spec['source_root'])
    image_paths, mask = _case_paths(record, spec, root)
    _check_sources(image_paths, mask, root, cancelled, record)
    case = _load_case(image_paths, mask, spec, cancelled, keep_native=keep_native)
    if (case['reference_grid'] != record['geometry'] or case['model_grid'] != record['model_grid']
            or case['decoded_digests'] != [record['content_digest'], *record['sequence_content_digests']]
            or case['mask_content_digest'] != record['mask_content_digest']):
        raise ValueError('Medical volume pair changed while being read')
    return case


def read_medical_case(record, spec, cancelled=None):
    """Return normalized images and compact class indices on the reference and model grids."""
    return _read_frozen_case(record, spec, cancelled, keep_native=True)


def _check_manifest(manifest):
    if (manifest.get('kind') != 'segmentation_csv' or manifest.get('version') != MEDICAL_MANIFEST_VERSION
            or _manifest_digest(manifest) != manifest.get('dataset_digest')):
        raise ValueError('Medical volume manifest digest does not match its contents')


def _foreground_voxel(labels, rng):
    # Select uniformly over all foreground voxels without holding an index for
    # every one. Only the chosen slice needs its foreground coordinates.
    cumulative = np.cumsum(np.count_nonzero(labels, axis=(1, 2)))
    if not cumulative[-1]:
        return None
    rank = rng.randrange(int(cumulative[-1]))
    z = int(np.searchsorted(cumulative, rank, side='right'))
    local_rank = rank - (int(cumulative[z - 1]) if z else 0)
    y, x = divmod(int(np.flatnonzero(labels[z])[local_rank]), labels.shape[2])
    return z, y, x


def _patch_start(labels, patch, record, spec, seed, epoch, index):
    limits = [max(0, axis - size) for axis, size in zip(labels.shape, patch)]
    if record['split'] != 'train':
        return [limit // 2 for limit in limits], 'center'
    entropy = _json_digest([_PATCH_SAMPLER, seed, record['id'], epoch, index])
    rng = random.Random(int(entropy, 16))
    point = _foreground_voxel(labels, rng) if rng.random() < spec['foreground_fraction'] else None
    if point is None:
        return [rng.randrange(limit + 1) for limit in limits], 'uniform'
    return [rng.randint(max(0, coordinate - size + 1), min(limit, coordinate))
            for coordinate, size, limit in zip(point, patch, limits)], 'foreground'


def _crop_case(case, record, spec, seed, epoch, index):
    patch = _patch_shape(spec)
    start, sampling = _patch_start(case['labels'], patch, record, spec, seed, epoch, index)
    valid = [min(size, axis - left) for size, axis, left in zip(patch, case['labels'].shape, start)]
    region = tuple(slice(left, left + size) for left, size in zip(start, valid))
    placement = tuple(slice(size) for size in valid)
    # Use the same edge padding as sliding-window inference. Mask padding is background.
    padding = [size - voxels for size, voxels in zip(patch, valid)]
    crop = case['image'][(slice(None), *region)]
    if any(padding):
        image = np.ascontiguousarray(np.pad(crop, ((0, 0), *((0, after) for after in padding)), mode='edge'))
    else:
        image = np.array(crop, copy=True, order='C')
    labels = np.zeros(patch, dtype=np.int64)
    labels[placement] = case['labels'][region]
    return image, labels, start, padding, sampling


def _check_epoch(epoch):
    if type(epoch) is not int or epoch < 1:
        raise ValueError('Training patch epoch must be a positive integer')


def preview_medical_patch(manifest, *, case_id=None, epoch=1, patch_index=0, cancelled=None):
    """Return the exact training crop and geometry available to the desktop preview."""
    from .medical_geometry import spacing_mm

    _check_manifest(manifest)
    _check_epoch(epoch)
    spec = manifest['data_spec']
    if type(patch_index) is not int or not 0 <= patch_index < spec['patches_per_case']:
        raise ValueError('Patch index must be within the training patches per case')
    record = next((record for record in manifest['records'] if record['split'] == 'train'
                   and (case_id is None or case_id == record['id'])), None)
    if record is None:
        raise ValueError('Choose an existing training case for patch preview')
    case = _read_frozen_case(record, spec, cancelled, keep_native=False)
    image, labels, start, padding, sampling = _crop_case(case, record, spec, manifest['seed'], epoch, patch_index)
    _cancel(cancelled)
    return image, labels, {'case_id': record['id'], 'split': record['split'], 'epoch': epoch, 'patch_index': patch_index,
                           'patch_start_dhw': start, 'patch_shape_dhw': list(labels.shape), 'pad_after_dhw': padding,
                           'sampling': sampling, 'contains_foreground': bool(labels.any()),
                           'reference_spacing_mm': spacing_mm(case['reference_grid']),
                           'model_spacing_mm': spacing_mm(case['model_grid'])}


def build_medical_datasets(manifest, cancelled=None):
    """Build lazy patient-partitioned datasets with one cache entry for training."""
    from copy import deepcopy
    import torch
    from torch.utils.data import Dataset

    _check_manifest(manifest)
    spec = validate_medical_spec(manifest['data_spec'])
    seed = manifest['seed']
    records = deepcopy(manifest['records'])

    class MedicalPatchDataset(Dataset):
        def __init__(self, split):
            self.records = [record for record in records if record['split'] == split]
            self.epoch = 1
            self.training = split == 'train'
            self.patches = spec['patches_per_case'] if self.training else 1
            self._cached = None

        def __len__(self):
            return len(self.records) * self.patches

        def set_epoch(self, epoch):
            _check_epoch(epoch)
            self.epoch = epoch

        def __getitem__(self, index):
            _cancel(cancelled)
            if not -len(self) <= index < len(self):
                raise IndexError(index)
            row, patch = divmod(index % len(self), self.patches)
            record = self.records[row]
            if self._cached is not None and self._cached[0] == row:
                # A cache hit avoids decoding/resampling, but the source is still frozen by its hash.
                root = Path(spec['source_root'])
                paths, mask = _case_paths(record, spec, root)
                _check_sources(paths, mask, root, cancelled, record)
                case = self._cached[1]
            else:
                self._cached = None
                case = _read_frozen_case(record, spec, cancelled, keep_native=False)
                if self.training and case['image'].nbytes + case['labels'].nbytes <= _MAX_CACHED_BYTES:
                    self._cached = (row, case)
            image, labels, _, _, _ = _crop_case(case, record, spec, seed, self.epoch, patch)
            _cancel(cancelled)
            return torch.from_numpy(image), torch.from_numpy(labels), record['id']

    return {split: MedicalPatchDataset(split) for split in SPLITS}
