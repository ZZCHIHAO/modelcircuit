"""Patient-isolated, content-verified classification data bound by case CSV rows."""
import csv
import hashlib
import io
import math
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath

import numpy as np

from .classification_data import (
    IMAGE_EXTENSIONS, MAX_CLASSES, MAX_CSV_BYTES, MAX_FEATURES, MAX_FLOAT32, MAX_IMAGE_BYTES,
    MAX_SAMPLES, SPLITS, _cancel, _check_partition, _decode_image, _group_split,
    _json_digest, _manifest_digest, _prepare_image, _read_source,
)
from .medical_geometry import assert_contains, make_model_grid, read_ct, read_nifti, resample_volume
from .medical_volume_data import MEDICAL_FORMATS, _fingerprint_medical, _image_volume_budget
from .research_data import MAX_CAPTURE_BYTES
from .research_decoder import _normalise, _resize_volume
from .segmentation_data import _digest, float_image


MULTIMODAL_MANIFEST_VERSION = 1
MODALITIES = ('volume', 'image', 'tabular')
IMAGE_SOURCES = ('raster', 'center_slice', 'mean_projection')
_AXES = {'volume': ('batch', 'channel', 'depth', 'height', 'width'),
         'image': ('batch', 'channel', 'height', 'width'), 'tabular': ('batch', 'feature')}


def multimodal_defaults():
    return {'kind': 'multimodal_csv', 'path': '', 'source_root': '', 'modalities': ['volume'],
            'id_column': 'case_id', 'group_column': 'patient_id', 'target_column': 'label', 'split_column': None,
            'volume_column': 'volume_path', 'image_column': 'image_path', 'feature_columns': [],
            'volume_format': 'dicom_ct', 'volume_shape': [16, 32, 32], 'spacing_mm': [1., 1., 1.],
            'ct_window': [-1000., 400.], 'sequence_columns': [], 'registered': False,
            'image_source': 'raster', 'image_size': [32, 32], 'image_channels': 1, 'standardize': True,
            'validation_fraction': .2, 'test_fraction': .2}


def _number(value, lower, upper):
    return type(value) in (float, int) and lower <= value <= upper


def _dimensions(value, maximum):
    return (isinstance(value, list) and len(value) == len(maximum)
            and all(type(axis) is int and 1 <= axis <= limit for axis, limit in zip(value, maximum)))


def _has_derived_image(spec):
    return 'image' in spec['modalities'] and spec['image_source'] != 'raster'


def _source_columns(spec):
    result = {}
    if 'volume' in spec['modalities'] or _has_derived_image(spec):
        result['volume'] = [spec['volume_column'], *spec['sequence_columns']]
    if 'image' in spec['modalities'] and not _has_derived_image(spec):
        result['image'] = [spec['image_column']]
    return result


def _required_columns(spec):
    columns = [spec[key] for key in ('id_column', 'group_column', 'target_column')]
    if spec['split_column'] is not None:
        columns.append(spec['split_column'])
    columns.extend(column for names in _source_columns(spec).values() for column in names)
    if 'tabular' in spec['modalities']:
        columns.extend(spec['feature_columns'])
    return columns


def validate_multimodal_spec(data, *, editable=False):
    """Validate settings independently of the local CSV or image files."""
    spec = multimodal_defaults()
    if not isinstance(data, dict) or data.get('kind') != spec['kind'] or not data.keys() <= spec.keys():
        raise ValueError('Unsupported multimodal CSV settings')
    if data.get('volume_format') == 'mri_nifti':
        spec['ct_window'] = None
    spec.update(data)
    for key in ('path', 'source_root'):
        if not isinstance(spec[key], str) or len(spec[key]) > 4096:
            raise ValueError(f'{key} must be a path string')
    modalities = spec['modalities']
    if (not isinstance(modalities, list) or (not editable and not modalities)
            or any(item not in MODALITIES for item in modalities)
            or len(set(modalities)) != len(modalities)):
        raise ValueError('Choose an ordered list of distinct volume, image or tabular modalities')
    if spec['image_source'] not in IMAGE_SOURCES or spec['volume_format'] not in MEDICAL_FORMATS:
        raise ValueError('Choose supported image_source and volume_format settings')
    if not _dimensions(spec['image_size'], (512, 512)):
        raise ValueError('image_size must be [height,width], each 1..512')
    if type(spec['image_channels']) is not int or spec['image_channels'] not in (1, 3):
        raise ValueError('image_channels must be 1 or 3')
    if not _dimensions(spec['volume_shape'], (128, 512, 512)):
        raise ValueError('volume_shape must be [depth,height,width] within 128 × 512 × 512')
    spacing = spec['spacing_mm']
    if (not isinstance(spacing, list) or len(spacing) != 3
            or not all(_number(value, 1e-6, 1e6) for value in spacing)):
        raise ValueError('spacing_mm requires three positive finite values in depth, height, width order')
    for key, limit in (('feature_columns', MAX_FEATURES), ('sequence_columns', 3)):
        columns = spec[key]
        if (not isinstance(columns, list) or len(columns) > limit
                or any(not isinstance(column, str) or not column.strip() for column in columns)
                or len(set(columns)) != len(columns)):
            raise ValueError(f'{key} requires up to {limit} distinct column names')
    if not editable and 'tabular' in modalities and not spec['feature_columns']:
        raise ValueError('Tabular input requires numeric feature_columns')
    if type(spec['registered']) is not bool or type(spec['standardize']) is not bool:
        raise ValueError('registered and standardize must be boolean')
    for key in ('volume_column', 'image_column'):
        if spec[key] is not None and not isinstance(spec[key], str):
            raise ValueError(f'{key} must be a column name')
    window = spec['ct_window']
    if window is not None:
        if (not isinstance(window, list) or len(window) != 2
                or not all(_number(value, -1e12, 1e12) for value in window) or window[0] >= window[1]):
            raise ValueError('ct_window requires two finite HU bounds with lower < upper')
        spec['ct_window'] = list(map(float, window))
    if 'volume' in _source_columns(spec):
        sequences = spec['sequence_columns']
        if spec['volume_format'] == 'dicom_ct':
            if sequences or spec['registered']:
                raise ValueError('Additional sequences and the registered setting apply to MRI NIfTI')
            if window is None:
                raise ValueError('DICOM CT requires a finite ct_window')
        else:
            if sequences and not spec['registered']:
                raise ValueError('Confirm the additional MRI sequences are already registered to the reference')
            if window is not None:
                raise ValueError('ct_window applies only to DICOM CT')
        if 'volume' in modalities and math.prod(spec['volume_shape']) * (len(sequences) + 1) > 16 * 1024 ** 2:
            raise ValueError('volumes exceed the 16 million float32 value budget')
    columns = _required_columns(spec)
    if (any(not isinstance(column, str) or not column.strip() for column in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError('Case, patient, label, split and enabled modality columns must be distinct column names')
    fractions = [spec[key] for key in ('validation_fraction', 'test_fraction')]
    if (any(type(value) not in (int, float) or not 0 < value < 1 for value in fractions)
            or sum(fractions) >= 1):
        raise ValueError('Validation and test fractions must leave a training partition')
    for key in ('modalities', 'image_size', 'volume_shape', 'feature_columns', 'sequence_columns'):
        spec[key] = list(spec[key])
    spec['spacing_mm'] = list(map(float, spacing))
    return spec


def _local_spec(data):
    spec = validate_multimodal_spec(data)
    if not spec['path'].strip():
        raise ValueError('Select a multimodal case CSV')
    source = Path(spec['path']).expanduser().resolve()
    root = Path(spec['source_root']).expanduser() if spec['source_root'] else source.parent
    if not root.is_absolute():
        root = source.parent / root
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f'Multimodal source root is unavailable: {root}')
    spec.update(path=str(source), source_root=str(root))
    return spec, source, root


def _cell(row, column, case, limit=256):
    value = (row.get(column) or '').strip()
    if not value or len(value) > limit:
        raise ValueError(f'Missing or oversized value for {case}, column {column}')
    return value


@contextmanager
def _case_column(case_id, column):
    try:
        yield
    except (InterruptedError, TimeoutError):
        raise
    except (OSError, ValueError) as exc:
        raise ValueError(f'Case {case_id}, column {column}: {exc}') from exc


def _asset_path(root, reference, *, directory=False):
    if Path(reference).is_absolute() or PureWindowsPath(reference).anchor:
        raise ValueError('Use a relative source path inside the selected dataset root')
    try:
        path = (root / reference.replace('\\', '/')).resolve()
    except RuntimeError as exc:
        raise ValueError('Path cannot be resolved inside the source root') from exc
    if not path.is_relative_to(root) or not (path.is_file() or (directory and path.is_dir())):
        raise ValueError('Select an existing source inside the source root')
    return path


def _csv_cases(raw, spec, root, cancelled):
    try:
        decoded = raw.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError('Multimodal CSV must be UTF-8') from exc
    reader = csv.DictReader(io.StringIO(decoded, newline=''), strict=True)
    records, seen_ids = [], {}
    source_columns = _source_columns(spec)
    try:
        fields = reader.fieldnames or []
        missing = [name for name in _required_columns(spec) if name not in fields]
        if missing:
            raise ValueError(f'Missing columns in multimodal CSV: {", ".join(missing)}')
        if len(fields) != len(set(fields)):
            raise ValueError('Duplicate column names in multimodal CSV')
        for row in reader:
            _cancel(cancelled)
            if len(records) == MAX_SAMPLES:
                raise ValueError('Multimodal case count exceeds the 20000 sample limit')
            case_id = _cell(row, spec['id_column'], f'CSV row {reader.line_num}')
            case = f'case {case_id}'
            if None in row:
                raise ValueError(f'Malformed CSV: extra fields for {case}')
            class_name = _cell(row, spec['target_column'], case)
            previous_label = seen_ids.get(case_id)
            if previous_label is not None:
                if previous_label != class_name:
                    raise ValueError(f'Conflicting label for duplicate {case}, column {spec["target_column"]}')
                raise ValueError(f'Duplicate {case}, column {spec["id_column"]}')
            seen_ids[case_id] = class_name
            group = _cell(row, spec['group_column'], case)
            split = None
            if spec['split_column'] is not None:
                split = _cell(row, spec['split_column'], case)
                if split not in SPLITS:
                    raise ValueError(f'Invalid split for {case}, column {spec["split_column"]}. Use train, val or test')
            features = []
            if 'tabular' in spec['modalities']:
                for column in spec['feature_columns']:
                    value = _cell(row, column, case)
                    with _case_column(case_id, column):
                        value = float(value)
                        if not math.isfinite(value) or abs(value) > MAX_FLOAT32:
                            raise ValueError('Numeric feature must be a finite float32 value')
                    features.append(value)
            sources = {}
            for modality, columns in source_columns.items():
                assets, paths = [], set()
                for column in columns:
                    reference = _cell(row, column, case, limit=4096)
                    with _case_column(case_id, column):
                        path = _asset_path(root, reference, directory=modality == 'volume' and spec['volume_format'] == 'dicom_ct')
                        if path in paths:
                            raise ValueError('Reference and MRI sequence paths must be distinct')
                        paths.add(path)
                    assets.append({'column': column, 'relative_path': path.relative_to(root).as_posix()})
                sources[modality] = assets
            records.append({'id': case_id, 'group': group, 'split': split, 'class_name': class_name,
                            'features': features, 'sources': sources})
    except csv.Error as exc:
        raise ValueError(f'Malformed multimodal CSV at row {reader.line_num}') from exc
    return records


def _fingerprint_volume(path, root, entry, cancelled):
    digest, size = _fingerprint_medical(path, root, cancelled)
    if digest != entry.get('source_digest', digest):
        raise ValueError('Volume source changed since inspection')
    return digest, size


def _read_volume(entries, case_id, spec, root, cancelled):
    is_ct = spec['volume_format'] == 'dicom_ct'
    channels = len(entries)
    sources, grids, bytes_seen = [], [], 0
    for index, entry in enumerate(entries):
        _cancel(cancelled)
        with _case_column(case_id, entry['column']):
            path = _asset_path(root, entry['relative_path'], directory=is_ct)
            source_digest, size = _fingerprint_volume(path, root, entry, cancelled)
            bytes_seen += size
            raw, grid = read_ct(path, cancelled) if is_ct else read_nifti(path)
            if index == 0:
                reference_grid = grid
                model_grid = make_model_grid(reference_grid, spec['spacing_mm'])
                _image_volume_budget(reference_grid, channels)
                _image_volume_budget(model_grid, channels)
                model = np.empty((channels, *model_grid['source_shape_dhw']), dtype=np.float32)
            else:
                assert_contains(reference_grid, grid)
            values = float_image(raw)
            content_digest = _digest(values)
            bounds = spec['ct_window'] if is_ct else (None, None)
            normalized, _ = _normalise(values, *bounds)
            _cancel(cancelled)
            model[index] = resample_volume(normalized, grid, model_grid, order=1)
            del raw, values, normalized
            if _fingerprint_medical(path, root, cancelled)[0] != source_digest:
                raise ValueError('Volume source changed during decoding')
            sources.append({**entry, 'source_digest': source_digest, 'content_digest': content_digest})
            grids.append(grid)
    geometry = {'reference_grid': reference_grid, 'sequence_grids': grids[1:], 'model_grid': model_grid}
    return model, sources, geometry, bytes_seen


def _crop_volume(model, grid, target):
    shape = model.shape[1:]
    start = [max(0, (source - size) // 2) for source, size in zip(shape, target)]
    before = [max(0, (size - source) // 2) for source, size in zip(shape, target)]
    after = [max(0, size - source) - pad for source, size, pad in zip(shape, target, before)]
    crop = tuple(slice(offset, offset + size) for offset, size in zip(start, target))
    padding = list(zip(before, after))
    output = model[(slice(None), *crop)]
    if any(before) or any(after):
        output = np.pad(output, [(0, 0), *padding], constant_values=0)
    affine = np.array(grid['affine_xyz'])
    offset_xyz = [crop_start - pad for crop_start, pad in zip(start[::-1], before[::-1])]
    affine[:3, 3] += affine[:3, :3] @ offset_xyz
    geometry = {'output_grid': {**grid, 'source_shape_dhw': target, 'affine_xyz': affine.tolist()},
                'crop_start_dhw': start, 'padding_dhw': [list(pair) for pair in padding]}
    return np.ascontiguousarray(output), geometry


def _derive_image(model, spec, cancelled):
    source = spec['image_source']
    reference = model[0]
    geometry = {'source': source, 'source_shape_hw': list(reference.shape[1:]), 'volume_channel': 0,
                'volume_axis': 'depth'}
    if source == 'center_slice':
        geometry['slice_index'] = reference.shape[0] // 2
        frame = reference[geometry['slice_index']]
    else:
        frame = np.mean(reference, axis=0, dtype=np.float64).astype(np.float32)
    resized = _resize_volume(frame[None, ...], (1, *spec['image_size']), cancelled)
    return np.repeat(resized, spec['image_channels'], axis=0), geometry


def _read_raster(entry, case_id, spec, root, cancelled):
    with _case_column(case_id, entry['column']):
        path = _asset_path(root, entry['relative_path'])
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError('Choose a supported raster image extension')
        raw = _read_source(path, MAX_IMAGE_BYTES, cancelled)
        source_digest = hashlib.sha256(raw).hexdigest()
        if source_digest != entry.get('source_digest', source_digest):
            raise ValueError('Raster image source changed since inspection')
        image, content_digest = _decode_image(raw, path)
        geometry = {'source': 'raster', 'source_shape_hw': [image.height, image.width]}
        processed = _prepare_image(image, spec['image_size'], spec['image_channels'])
        values = np.asarray(processed, dtype=np.float32).reshape(*spec['image_size'], spec['image_channels'])
        image = np.ascontiguousarray(values.transpose(2, 0, 1) / np.float32(255.))
    return image, {**entry, 'source_digest': source_digest, 'content_digest': content_digest}, geometry, len(raw)


def _tabular_values(record, recipe):
    values = [(value - mean) / std for value, mean, std in zip(record['features'], recipe['mean'], recipe['std'])]
    for column, value in zip(recipe['feature_columns'], values):
        if not math.isfinite(value) or abs(value) > MAX_FLOAT32:
            raise ValueError(f'Case {record["id"]}, column {column}: preprocessed float32 value is out of range')
    return np.array(values, dtype=np.float32)


def _read_inputs(record, spec, root, recipe, cancelled, *, modalities=None):
    modalities = spec['modalities'] if modalities is None else modalities
    sources, geometry, arrays, bytes_seen = {}, {}, {}, 0
    derived = 'image' in modalities and _has_derived_image(spec)
    if 'volume' in modalities or derived:
        entries = record['sources']['volume']
        model, sources['volume'], geometry['volume'], size = _read_volume(
            entries if 'volume' in modalities else entries[:1], record['id'], spec, root, cancelled)
        bytes_seen += size
        if derived:
            arrays['image'], geometry['image'] = _derive_image(model, spec, cancelled)
        if 'volume' in modalities:
            arrays['volume'], crop = _crop_volume(model, geometry['volume']['model_grid'], spec['volume_shape'])
            geometry['volume'].update(crop)
    if 'image' in modalities and 'image' in record['sources']:
        arrays['image'], entry, geometry['image'], size = _read_raster(
            record['sources']['image'][0], record['id'], spec, root, cancelled)
        sources['image'] = [entry]
        bytes_seen += size
    if 'tabular' in modalities:
        arrays['tabular'] = _tabular_values(record, recipe['tabular'])
    _cancel(cancelled)
    return arrays, sources, geometry, bytes_seen


def _fit_preprocessing(records, spec):
    recipe = {}
    if 'volume' in _source_columns(spec):
        recipe['volume'] = {'kind': 'physical_affine_mm', 'format': spec['volume_format'],
                            'spacing_mm': spec['spacing_mm'], 'ct_window': spec['ct_window'],
                            'channels': 1 + len(spec['sequence_columns']), 'sequence_columns': spec['sequence_columns'],
                            'registered': spec['registered'], 'resampling': 'linear_voxel_centres',
                            'intensity': ('ct_hu_clip_before_resampling' if spec['volume_format'] == 'dicom_ct'
                                          else 'per_volume_channel_minmax_before_resampling')}
        if 'volume' in spec['modalities']:
            recipe['volume'].update(volume_shape=spec['volume_shape'], crop='center', padding='zero')
    if 'image' in spec['modalities']:
        recipe['image'] = {'kind': spec['image_source'], 'image_size': spec['image_size'],
                           'channels': spec['image_channels'], 'resampling': 'bilinear'}
        if _has_derived_image(spec):
            recipe['image'].update(grid='full_model_grid', volume_channel=0, volume_axis='depth')
        else:
            recipe['image']['scale'] = 1 / 255.
    if 'tabular' in spec['modalities']:
        columns = spec['feature_columns']
        mean, std = [0.] * len(columns), [1.] * len(columns)
        if spec['standardize']:
            train = [record['features'] for record in records if record['split'] == 'train']
            for column in range(len(columns)):
                mean[column] = math.fsum(row[column] / len(train) for row in train)
                std[column] = math.sqrt(math.fsum((row[column] - mean[column]) ** 2 / len(train) for row in train)) or 1.
        recipe['tabular'] = {'kind': 'standardize' if spec['standardize'] else 'identity', 'fit_split': 'train',
                             'dtype': 'float32', 'feature_columns': columns, 'mean': mean, 'std': std}
    return recipe


def _input_shapes(spec):
    shapes = {'volume': [None, 1 + len(spec['sequence_columns']), *spec['volume_shape']],
              'image': [None, spec['image_channels'], *spec['image_size']],
              'tabular': [None, len(spec['feature_columns'])]}
    return {key: shapes[key] for key in spec['modalities']}


def _content_digest(record, sources, tensors, spec):
    return _json_digest({'sources': [(modality, [entry['content_digest'] for entry in entries])
                                     for modality, entries in sources.items()],
                         'tensors': [(modality, tensors[modality]) for modality in spec['modalities']],
                         'features': record['features']})


def _check_source_partitions(record, tensors, seen):
    def check(kind, modality, identity):
        other = seen.setdefault((modality, kind, identity), record)
        if other['split'] != record['split']:
            raise ValueError(f'Cross-split {modality} {kind} leakage between cases {other["id"]} and {record["id"]}')

    for modality, entries in record['sources'].items():
        for entry in entries:
            check('path', modality, entry['relative_path'])
            for key in ('source_digest', 'content_digest'):
                check('content', modality, entry[key])
    for modality in ('volume', 'image'):
        if modality in tensors:
            check('content', modality, tensors[modality])


def _portable_digest(manifest):
    spec = {key: value for key, value in manifest['data_spec'].items() if key not in ('path', 'source_root')}
    return _manifest_digest({**manifest, 'data_spec': spec})


def _split_cases(records, spec, seed, cancelled):
    try:
        _group_split(records, spec, seed, cancelled)
    except ValueError:
        # Small cohorts can need equal patient allocations to get every class
        # in every split, especially when a patient has repeated visits.
        patient_labels = sorted({(record['group'], record['label']) for record in records})
        patients = [{'group': group, 'label': label} for group, label in patient_labels]
        _group_split(patients, {'validation_fraction': 1 / 3, 'test_fraction': 1 / 3}, seed, cancelled)
        by_group = {patient['group']: patient['split'] for patient in patients}
        for record in records:
            record['split'] = by_group[record['group']]
        return ['Split fractions adjusted to include patients of every class in train, val and test']
    return []


def inspect_multimodal_dataset(data, seed, cancelled=None):
    _cancel(cancelled)
    if type(seed) is not int or not 0 <= seed < 2 ** 31:
        raise ValueError('seed must be an integer in 0..2^31-1')
    spec, csv_path, root = _local_spec(data)
    raw = _read_source(csv_path, MAX_CSV_BYTES, cancelled)
    csv_digest = hashlib.sha256(raw).hexdigest()
    records = _csv_cases(raw, spec, root, cancelled)
    del raw
    class_names = sorted({record['class_name'] for record in records})
    if not 2 <= len(class_names) <= MAX_CLASSES:
        raise ValueError('Multimodal classification requires 2..128 classes')
    labels = {label: index for index, label in enumerate(class_names)}
    for record in records:
        record['label'] = labels[record['class_name']]
    warnings = _split_cases(records, spec, seed, cancelled) if spec['split_column'] is None else []
    # The shared classifier check enforces the same group and class partition policy.
    # Individual imaging sources are checked below, so equal clinical vectors are allowed.
    counts, class_counts = _check_partition(
        [{**record, 'source_path': f'case {record["id"]}', 'content_digest': None} for record in records], class_names)
    recipe = _fit_preprocessing(records, spec)
    bytes_seen, seen = 0, {}
    for record in records:
        arrays, sources, geometry, size = _read_inputs(record, spec, root, recipe, cancelled)
        bytes_seen += size
        if bytes_seen > MAX_CAPTURE_BYTES:
            raise ValueError('Multimodal sources exceed the 8 GiB inspection budget')
        tensors = {modality: _digest(value) for modality, value in arrays.items()}
        record.update(sources=sources, geometry=geometry, tensor_digests=tensors,
                      content_digest=_content_digest(record, sources, tensors, spec))
        _check_source_partitions(record, tensors, seen)
        del arrays
    manifest = {'kind': spec['kind'], 'version': MULTIMODAL_MANIFEST_VERSION, 'seed': seed, 'data_spec': spec,
                'counts': counts, 'class_counts': class_counts, 'class_names': class_names, 'records': records,
                'input_shapes': _input_shapes(spec), 'input_axes': {key: list(_AXES[key]) for key in spec['modalities']},
                'preprocessing': recipe, 'warnings': warnings, 'csv_source_digest': csv_digest,
                'csv_relative_path': csv_path.relative_to(root).as_posix() if csv_path.is_relative_to(root) else None}
    manifest['dataset_digest'] = _portable_digest(manifest)
    _cancel(cancelled)
    return manifest


def _file_stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def build_multimodal_datasets(manifest, cancelled=None, *, modalities=None):
    if (manifest.get('kind') != 'multimodal_csv' or manifest.get('version') != MULTIMODAL_MANIFEST_VERSION
            or _portable_digest(manifest) != manifest.get('dataset_digest')):
        raise ValueError('Multimodal dataset manifest digest does not match its contents')
    spec, csv_path, root = _local_spec(manifest['data_spec'])
    selected = set(spec['modalities'] if modalities is None else modalities)
    if not selected or not selected <= set(spec['modalities']):
        raise ValueError('Choose one or more of the inspected modalities')
    selected = [name for name in spec['modalities'] if name in selected]
    recipe = manifest['preprocessing']
    stamp = None

    def verify_csv():
        nonlocal stamp
        try:
            current = _file_stamp(csv_path)
            if current == stamp:
                return
            digest = hashlib.sha256(_read_source(csv_path, MAX_CSV_BYTES, cancelled)).hexdigest()
            if digest != manifest['csv_source_digest'] or _file_stamp(csv_path) != current:
                raise ValueError('Multimodal CSV source changed since inspection')
            stamp = current
        except OSError as exc:
            raise ValueError('Multimodal CSV source changed or is unavailable') from exc

    _cancel(cancelled)
    verify_csv()
    import torch
    from torch.utils.data import Dataset

    class CaseDataset(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            _cancel(cancelled)
            verify_csv()
            record = self.records[index]
            arrays, sources, geometry, _ = _read_inputs(record, spec, root, recipe, cancelled, modalities=selected)
            if (any(entries != record['sources'][name][:len(entries)] for name, entries in sources.items())
                    or any(_digest(arrays[name]) != record['tensor_digests'][name] for name in selected)
                    or any(geometry[name] != record['geometry'][name] for name in selected if name in geometry)):
                raise ValueError(f'Case {record["id"]} changed since inspection')
            return ({modality: torch.from_numpy(arrays[modality]) for modality in selected},
                    record['label'], record['id'])

    return {split: CaseDataset([record for record in manifest['records'] if record['split'] == split]) for split in SPLITS}
