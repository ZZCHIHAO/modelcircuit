"""Grouped CSV manifests for bounded image, volume and video classification."""
import hashlib
from pathlib import Path

from .classification_data import (MAX_CLASSES, SPLITS, _cancel, _check_partition, _csv_records,
                                  _group_split, _json_digest, _manifest_digest)

RESEARCH_FORMATS = ('image', 'gray16', 'dicom', 'nifti', 'video')
STILL_FORMATS = ('image', 'gray16')
MAX_SOURCE_BYTES = 1024 ** 3
MAX_CAPTURE_BYTES = 8 * 1024 ** 3


def defaults():
    return {'kind': 'research_csv', 'path': '', 'format': 'image', 'source_root': '',
            'path_column': 'path', 'id_column': 'sample_id', 'group_column': 'patient_id',
            'target_column': 'label', 'split_column': None, 'validation_fraction': .2,
            'test_fraction': .2, 'image_size': [32, 32], 'depth': 1, 'channels': 1,
            'intensity_min': None, 'intensity_max': None, 'modality': 'auto'}


def validate_research_spec(data):
    import math
    result = defaults()
    if not isinstance(data, dict) or not data.keys() <= result.keys() or data.get('kind') != 'research_csv':
        raise ValueError('Unsupported research source settings')
    result.update(data)
    for key in ('path', 'source_root'):
        if not isinstance(result[key], str) or len(result[key]) > 4096:
            raise ValueError(f'{key} must be a path string')
    keys = [result[k] for k in ('path_column','id_column','group_column','target_column')]
    if result['split_column'] is not None:
        keys.append(result['split_column'])
    if any(not isinstance(key, str) or not key.strip() for key in keys) or len(keys) != len(set(keys)):
        raise ValueError('Research CSV requires distinct path, sample, group, label and optional split columns')
    val, test = result['validation_fraction'], result['test_fraction']
    if any(type(x) not in (int, float) or not 0 < x < 1 for x in (val,test)) or val + test >= 1:
        raise ValueError('Research split fractions must leave a training partition')
    fmt, size, depth, channels = (result[k] for k in ('format','image_size','depth','channels'))
    if fmt not in RESEARCH_FORMATS or not isinstance(size,list) or len(size) != 2 or any(type(x) is not int or not 8 <= x <= 512 for x in size):
        raise ValueError('Choose a supported research format and height/width from 8 to 512')
    if type(depth) is not int or not 1 <= depth <= 128 or type(channels) is not int or channels not in (1,3):
        raise ValueError('Research depth must be 1..128 and channels must be 1 or 3')
    if (fmt in STILL_FORMATS and depth != 1) or (fmt in ('gray16','dicom','nifti') and channels != 1):
        raise ValueError('Still images require depth 1 and medical grayscale requires one channel')
    if depth * channels * size[0] * size[1] > 16 * 1024 ** 2:
        raise ValueError('Requested sample exceeds 16 million float32 values')
    low, high = result['intensity_min'], result['intensity_max']
    if (low is not None or high is not None):
        if fmt in ('image', 'video') or not all(type(x) in (float,int) and abs(x) <= 1e12 and math.isfinite(x) for x in (low,high)) or low >= high:
            raise ValueError('Medical intensity normalization requires a finite lower bound below the upper bound')
    if result['modality'] not in ('auto','OCT','CT','MR'):
        raise ValueError('Choose auto, OCT, CT or MR')
    return result


def _fingerprint(path, root, cancelled):
    if path.is_dir():
        files = sorted((x for x in path.iterdir() if x.is_file() and not x.name.startswith('.')), key=lambda x:x.name)
    elif path.is_file():
        files = [path]
    else:
        raise ValueError(f'Research source does not exist: {path}')
    if not files or len(files) > 2048:
        raise ValueError('Research source directory must contain 1..2048 files')
    total = 0
    payload = hashlib.sha256()
    for file in files:
        if file.is_symlink() or not file.resolve().is_relative_to(root):
            raise ValueError('Research sample references a file outside its source root')
        _cancel(cancelled)
        per_file = hashlib.sha256()
        size = 0
        with file.open('rb') as stream:
            while block := stream.read(1024 ** 2):
                _cancel(cancelled)
                total += len(block)
                size += len(block)
                if total > MAX_SOURCE_BYTES:
                    raise ValueError('Research sample exceeds the 1 GiB raw source limit')
                per_file.update(block)
        payload.update(f'{size}:{per_file.hexdigest()}\n'.encode('ascii'))
    return payload.hexdigest(), total


def _tensor(path, spec, cancelled):
    import numpy as np
    from .research_decoder import decode
    array, metadata = decode(path, spec, cancelled)
    expected = (spec['channels'], spec['depth'], *spec['image_size'])
    if array.shape != expected or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f'Research decoder did not return finite float32 {expected}')
    return (array[:, 0] if spec['format'] in STILL_FORMATS else array), metadata


def inspect_research_dataset(data_spec, seed, cancelled=None):
    spec = validate_research_spec(data_spec)
    if not spec['path'].strip():
        raise ValueError('Select a research CSV manifest')
    csv = Path(spec['path']).expanduser().resolve(strict=True)
    root = Path(spec['source_root']).expanduser().resolve(strict=True) if spec['source_root'] else csv.parent
    if not root.is_dir():
        raise ValueError('Research source root must be a directory')
    spec.update(path=str(csv), source_root=str(root))
    records = _csv_records({**spec, 'feature_columns': []}, cancelled, (spec['path_column'], spec['id_column']))
    if not records:
        raise ValueError('Research CSV has no samples')
    class_names = sorted({r['class_name'] for r in records})
    if not 2 <= len(class_names) <= MAX_CLASSES:
        raise ValueError('Research classification needs 2..128 classes')
    labels = {label:i for i,label in enumerate(class_names)}
    consumed = 0
    for record in records:
        _cancel(cancelled)
        values = record.pop('extra_values')
        sample_id = values[spec['id_column']].strip()
        ref = Path(values[spec['path_column']].strip())
        if not sample_id or len(sample_id) > 256 or ref.is_absolute() or str(ref) == '.':
            raise ValueError('Each research row needs a sample ID and a relative source path')
        path = (root / ref).resolve(strict=True)
        if not path.is_relative_to(root) or path == root or (path.is_dir() and spec['format'] != 'dicom'):
            raise ValueError('Source must stay under the research root. Only DICOM may use a series directory')
        file_digest, amount = _fingerprint(path, root, cancelled)
        consumed += amount
        if consumed > MAX_CAPTURE_BYTES:
            raise ValueError('Research inspection exceeds the 8 GiB raw input budget')
        tensor, metadata = _tensor(path, spec, cancelled)
        record.update(id=sample_id, relative_path=path.relative_to(root).as_posix(),
                      source_path=str(path), source_digest=file_digest,
                      content_digest=hashlib.sha256(tensor.tobytes()).hexdigest(),
                      label=labels[record['class_name']], metadata=metadata)
        record.pop('features')
    for key in ('id', 'source_path'):
        if len({record[key] for record in records}) != len(records):
            raise ValueError(f'Research CSV contains repeated {key}')
    if spec['split_column'] is None:
        _group_split(records, spec, seed, cancelled)
    counts, class_counts = _check_partition(records, class_names)
    for column in ('source_digest',):
        seen = {}
        for record in records:
            if seen.setdefault(record[column],record['split']) != record['split']:
                raise ValueError('Cross-split raw source duplication')
    shape = [None, spec['channels']]
    if spec['format'] not in STILL_FORMATS:
        shape.append(spec['depth'])
    shape.extend(spec['image_size'])
    manifest = {'version': 1, 'kind': 'research_csv', 'data_spec': spec, 'seed': seed,
                'class_names': class_names, 'input_shape': shape, 'counts': counts,
                'class_counts': class_counts, 'records': records,
                'preprocessing': {k:spec[k] for k in ('format','modality','image_size','depth','channels','intensity_min','intensity_max')}}
    manifest['dataset_digest'] = _manifest_digest(manifest)
    return manifest


def build_research_datasets(manifest, cancelled=None):
    import torch
    from torch.utils.data import Dataset
    if manifest.get('version') != 1 or _manifest_digest(manifest) != manifest.get('dataset_digest'):
        raise ValueError('Research manifest integrity check failed')
    spec = manifest['data_spec']
    root = Path(spec['source_root'])

    class ResearchDataset(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            _cancel(cancelled)
            record = self.records[index]
            path = Path(record['source_path'])
            digest, _ = _fingerprint(path, root, cancelled)
            if digest != record['source_digest']:
                raise ValueError('Research sample changed since inspection: ' + record['id'])
            array, metadata = _tensor(path, spec, cancelled)
            if hashlib.sha256(array.tobytes()).hexdigest() != record['content_digest'] or metadata != record['metadata']:
                raise ValueError('Research decoded sample changed since inspection: ' + record['id'])
            return torch.from_numpy(array.copy()), record['label'], record['id']

    return {split:ResearchDataset([r for r in manifest['records'] if r['split'] == split]) for split in SPLITS}
