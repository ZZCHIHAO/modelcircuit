"""Portable experiment files and binding to verified local datasets."""
from collections.abc import Sequence
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import tempfile
from uuid import UUID, uuid4

from .classification_data import _manifest_digest, inspect_dataset
from .execution_policy import check_config_batch
from .medical_volume_data import MEDICAL_FORMATS
from .experiment_spec import checkpoint_signature, model_digest, validate_supervised_config
from .python_io import export_python, import_python
from .schema import prepare_graph
from .task_adapters import get_task_adapter


PROJECT_FORMAT = 'ml-forge-research-project'
PROJECT_VERSION = 1
_TASKS = {
    ('classification-v2', 'csv'): 'classification',
    ('regression-v1', 'regression_csv'): 'regression',
    ('unsupervised-v1', 'unsupervised_csv'): 'unsupervised',
    ('classification-v2', 'image_folder'): 'classification',
    ('segmentation-v1', 'segmentation_csv'): 'segmentation',
    ('classification-v2', 'table_csv'): 'classical',
    ('regression-v1', 'table_csv'): 'classical',
    ('classification-v2', 'multimodal_csv'): 'classification',
}
_NUMERIC_CSV_KINDS = frozenset({'csv', 'regression_csv'})
EMBEDDABLE_CSV_KINDS = _NUMERIC_CSV_KINDS | {'table_csv', 'unsupervised_csv'}
_BUNDLED_CSV = 'data/source.csv'
_CASE_CSV_KINDS = frozenset({'table_csv', 'multimodal_csv'})
_SOURCE_ROOT_KINDS = frozenset({'segmentation_csv', 'multimodal_csv'})
_CASE_FIELDS = ('id', 'group', 'split', 'label', 'content_digest')
_SOURCE_FIELDS = ('column', 'relative_path', 'source_digest', 'content_digest')
_PATH_FIELDS = ('path', 'source_root')
_HEX_DIGEST = re.compile(r'[0-9a-f]{64}')
_MAX_PROJECT = 8 * 1024 ** 2
_MAX_DATASET = 64 * 1024 ** 2
_MAX_MODEL = 8 * 1024 ** 2  # The source bridge also limits Unicode character count.
_PROJECT_FILES = {'model': 'model.py', 'dataset': 'dataset.json'}
_PAIRED_CSV = 'dataset-pairs.csv'
_WEIGHT_FILES = {'run_spec': 'weights/run_spec.json', 'checkpoint': 'weights/best.pt'}
_RUN_ENTRY = '''"""Run this exported ModelCircuit Studio research project."""
from pathlib import Path
import sys


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_dir / 'runtime'))
    from studio_core.project_runner import main

    raise SystemExit(main(project_dir=project_dir))
'''
_README = '''# ModelCircuit Studio research project

Use Python 3.12. From the project directory, install the shared dependencies:

```
python -m pip install -r requirements.txt
```

For the tested CPU runtime, install the PyTorch wheel separately:

```
python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
```

A CUDA environment can install the matching PyTorch 2.6.0 wheel for its
platform and choose an available `cuda:N` device explicitly.

Use an ImageFolder directory or the root of a segmentation dataset for `DATA_PATH`.
The project includes `dataset-pairs.csv` for segmentation, so a moved dataset can
bind using the relative paths in that file. You can also pass a local segmentation
CSV and specify its image and mask root with `--source-root` when needed.
The file contents, preprocessing, and pairing order must match `dataset.json`.
After a move the dataset gets a fresh local training digest.

```
python run.py check --data DATA_PATH
python run.py train --data DATA_PATH --out run-01
python run.py predict --source IMAGE_OR_FOLDER --out predictions-01 --weights run-01
```

If the project includes weights, `predict` can use them with `--weights` omitted.
Each training command starts a new run and expects a new output directory.
The workbench and runner statically parse supported edits to `model.py`.
'''
_TABLE_README = '''# ModelCircuit Studio table research project

Use Python 3.12 and install the CPU runtime from `requirements.txt`.

```
python -m pip install -r requirements.txt
python run.py check --data cases.csv
python run.py train --data cases.csv --out run-01
```

Use a local copy of the original CSV. The project locks the CSV digest,
patient groups, split assignments, preprocessing recipe and fold members.
Each run fits preprocessing again on each fold's training cases and fits the
final estimator on all development cases. The run writes fold results and
predictions for the frozen test cases.
'''
_MULTIMODAL_README = '''# ModelCircuit Studio multimodal research project

Use Python 3.12. Install the shared dependencies and a matching PyTorch
wheel. For CPU training, use the following commands.

```
python -m pip install -r requirements.txt
python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu
python run.py check --data cases.csv
python run.py train --data cases.csv --out run-01
```

Use a local copy of the original case CSV and sources. To move the dataset,
use relative source paths in the original CSV. They're relative to the CSV
directory by default. Add `--source-root` with the source directory if the
CSV is stored separately. The project verifies ordered cases, source digests,
geometry and preprocessing before training. Training uses the saved input
port mapping and writes test case predictions.
Select `--device cuda:0` for the first GPU in a compatible CUDA environment.
'''
_HELDOUT_README = '''
Keep the run report, heldout.json, metrics.json and predictions.csv together
to compare runs. The reader checks that the report and results describe the
same test cases. These files carry checksums for consistency and are unsigned.
Verify their source separately when authorship matters.
'''


def _json(value, *, pretty=False):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      indent=2 if pretty else None, separators=None if pretty else (',', ':'))


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _read_bytes(path, maximum):
    try:
        with path.open('rb') as stream:
            raw = stream.read(maximum + 1)
    except OSError as error:
        raise ValueError(f'Could not read {path}: {error}') from error
    if len(raw) > maximum:
        raise ValueError(f'{path.name} exceeds its size limit')
    return raw


def _read_json(path, maximum):
    return _decode_json(_read_bytes(path, maximum), path)


def _decode_json(raw, path):
    def finite(number):
        value = float(number)
        if not math.isfinite(value):
            raise ValueError('JSON numbers must be finite')
        return value

    def object_pairs(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError('Duplicate JSON field')
        return result

    try:
        value = json.loads(raw.decode('utf-8-sig'), parse_float=finite,
                           parse_constant=finite, object_pairs_hook=object_pairs)
        if not isinstance(value, dict):
            raise ValueError('Expected a JSON object')
        return value
    except (ValueError, RecursionError) as error:
        raise ValueError(f'Invalid {path.name}: {error}') from error


def _write_json(path, value, maximum, *, pretty=True):
    blob = (_json(value, pretty=pretty) + '\n').encode('utf-8')
    if len(blob) > maximum:
        raise ValueError(f'{path.name} exceeds its size limit')
    path.write_bytes(blob)


def _is_digest(value):
    return isinstance(value, str) and _HEX_DIGEST.fullmatch(value) is not None


def _portable_relative(path):
    return (isinstance(path, str) and not PureWindowsPath(path).drive
            and not any(character in path for character in '\\:<>"|?*\x00')
            and all(part and part not in ('.', '..') for part in path.split('/')))


def _project_file(root, reference, field):
    if not _portable_relative(reference):
        raise ValueError(f'Invalid {field} path reference in project')
    path = (root / reference).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'{field} reference must name an existing file inside the project: {reference}')
    return path


def _project_config(config):
    if not isinstance(config, dict) or not isinstance(config.get('data'), dict):
        raise ValueError('Project requires supervised experiment and data settings')
    profile = config.get('profile')
    kind = config['data'].get('kind')
    if not isinstance(profile, str) or not isinstance(kind, str) or (profile, kind) not in _TASKS:
        raise ValueError('This task and data format cannot be exported as an independent project')
    if profile == 'unsupervised-v1':
        from .unsupervised_task import validate_unsupervised_config
        return validate_unsupervised_config(config)
    return validate_supervised_config(config)


def _portable_spec(spec):
    return {key: value if key not in _PATH_FIELDS else '' for key, value in spec.items()}


def _detach_config(config):
    return {**config, 'data': _portable_spec(config['data']),
            'expected_dataset_digest': None, 'resume_checkpoint': None}


def _validate_detached(config):
    if not isinstance(config, dict) or not isinstance(config.get('data'), dict):
        raise ValueError('Project config must contain data settings')
    data = config['data']
    if (data.get('path') != '' or data.get('source_root', '') != '' or config.get('resume_checkpoint') is not None
            or config.get('expected_dataset_digest') is not None):
        raise ValueError('Project config must leave local data paths and resume settings unbound')
    # Shared validation expects a selected path but never opens it.
    return _detach_config(_project_config({**config, 'data': {**data, 'path': 'dataset'}}))


def _validate_task(graph, config, manifest):
    from .classical_task import classical_contract, contains_classical

    task = _TASKS[config['profile'], config['data']['kind']]
    if task == 'unsupervised':
        from .unsupervised_task import unsupervised_contract
        unsupervised_contract(graph, manifest)
        return
    if task == 'classical' or contains_classical(graph):
        classical_contract(graph, config, manifest)
        return
    check_config_batch(graph, config['batch_size'])
    resolved, shapes, _ = prepare_graph(graph)
    get_task_adapter(task, 1)(config).validate_contract(resolved, shapes, manifest)


def _portable_csv_cases(manifest, payload):
    kind = manifest['kind']
    records = [{field: record[field] for field in _CASE_FIELDS} for record in manifest['records']]
    if kind == 'table_csv':
        task = manifest['task']
        label_field = 'class_names' if task == 'classification' else 'target_column'
        payload.update(task=task, input_shape=manifest['input_shape'], source_digest=manifest['source_digest'],
                       fold_plan=[{key: fold[key] for key in ('fold', 'train_ids', 'val_ids')}
                                  for fold in manifest['fold_plan']])
        payload[label_field] = manifest[label_field]
    else:
        payload.update({key: manifest[key] for key in ('class_names', 'class_counts', 'input_shapes', 'input_axes',
                                                       'csv_source_digest')})
        for record, local in zip(records, manifest['records']):
            record['sources'] = {modality: [{key: source[key] for key in _SOURCE_FIELDS} for source in sources]
                                 for modality, sources in local['sources'].items()}
            record['geometry'] = local['geometry']
    payload['records'] = records
    _verify_csv_case_records(payload)
    return {**payload, 'digest': _digest(payload)}


def _verify_csv_case_records(lock):
    for record in lock['records']:
        if (not isinstance(record, dict) or not set(_CASE_FIELDS) <= record.keys()
                or not _is_digest(record['content_digest'])):
            raise ValueError('Invalid case record in the dataset lock')
    if lock['kind'] == 'table_csv':
        if not _is_digest(lock.get('source_digest')) or not isinstance(lock.get('fold_plan'), list):
            raise ValueError('Invalid table CSV source or fold plan in the dataset lock')
        return
    if not _is_digest(lock.get('csv_source_digest')):
        raise ValueError('Invalid multimodal CSV source digest in the dataset lock')
    for record in lock['records']:
        sources = record.get('sources')
        if not isinstance(sources, dict) or not isinstance(record.get('geometry'), dict):
            raise ValueError('Invalid modality sources or geometry in the dataset lock')
        for entries in sources.values():
            if not isinstance(entries, list) or not all(
                    isinstance(source, dict) and _portable_relative(source.get('relative_path'))
                    and _is_digest(source.get('source_digest')) and _is_digest(source.get('content_digest'))
                    for source in entries):
                raise ValueError('Invalid relative path or source digest in the multimodal dataset lock')


def portable_dataset(manifest):
    """Return an ordered, location-independent lock from an inspected local manifest."""
    kind = manifest.get('kind')
    if kind not in ('image_folder', 'segmentation_csv', 'multimodal_csv', *EMBEDDABLE_CSV_KINDS):
        raise ValueError('Dataset kind is not supported by research projects')
    settings = _portable_spec(manifest['data_spec'])
    payload = {'version': 1, 'kind': kind, 'logical_root': 'dataset', 'settings': settings,
               'seed': manifest['seed'], 'counts': manifest['counts'], 'preprocessing': manifest['preprocessing']}
    if kind in _CASE_CSV_KINDS:
        return _portable_csv_cases(manifest, payload)
    payload['input_shape'] = manifest['input_shape']
    if kind == 'unsupervised_csv':
        payload.update(source_digest=manifest['source_digest'], records=[
            {key: record[key] for key in ('id', 'content_digest')} for record in manifest['records']])
        return {**payload, 'digest': _digest(payload)}
    if kind in _NUMERIC_CSV_KINDS:
        fields = ('class_names', 'class_counts') if kind == 'csv' else ('target_column',)
        payload.update({field: manifest[field] for field in fields})
        keys = ('row_number', 'group', 'split', 'label', 'content_digest', 'source_digest')
        payload['records'] = [{key: row[key] for key in keys} for row in manifest['records']]
        return {**payload, 'digest': _digest(payload)}
    records = []
    if kind == 'image_folder':
        payload.update(class_names=manifest['class_names'], class_counts=manifest['class_counts'])
        keys = ('id', 'split', 'relative_path', 'source_digest', 'decoded_digest', 'content_digest',
                'source_size', 'class_name', 'label')
        for local in manifest['records']:
            records.append({key: local[key] for key in keys})
    else:
        payload.update(target_shape=manifest['target_shape'], mask_values=manifest['mask_values'],
                       class_pixels=manifest['class_pixels'])
        root = Path(manifest['data_spec']['source_root'])
        keys = ('id', 'group', 'split', 'source_digest', 'mask_digest', 'content_digest',
                'mask_content_digest', 'geometry', 'class_pixels')
        if settings.get('format') in MEDICAL_FORMATS:
            keys += ('model_grid', 'sequence_relative_paths', 'sequence_source_digests', 'sequence_content_digests')
        for local in manifest['records']:
            records.append({**{key: local[key] for key in keys},
                            'relative_path': Path(local['source_path']).relative_to(root).as_posix(),
                            'mask_relative_path': Path(local['mask_path']).relative_to(root).as_posix()})
    if any(not _portable_relative(record['relative_path']) or
           (kind == 'segmentation_csv' and not _portable_relative(record['mask_relative_path']))
           or any(not _portable_relative(path) for path in record.get('sequence_relative_paths', ()))
           for record in records):
        raise ValueError('Dataset contains a file name that cannot travel as a relative path')
    payload['records'] = records
    return {**payload, 'digest': _digest(payload)}


def _write_pairs(path, lock):
    columns = {'id_column': 'id', 'group_column': 'group', 'image_column': 'relative_path',
               'mask_column': 'mask_relative_path', 'split_column': 'split'}
    active = [(lock['settings'][setting], field) for setting, field in columns.items() if lock['settings'][setting] is not None]
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        sequences = lock['settings'].get('sequence_columns', [])
        writer.writerow([column for column, _ in active] + sequences)
        writer.writerows([record[field] for _, field in active] + record.get('sequence_relative_paths', [])
                         for record in lock['records'])


def _verify_dataset_lock(lock, config, expected):
    digest = lock.get('digest')
    if (not _is_digest(digest) or digest != expected
            or digest != _digest({key: value for key, value in lock.items() if key != 'digest'})):
        raise ValueError('Dataset lock digest does not match the project')
    if (type(lock.get('version')) is not int or lock['version'] != 1 or lock.get('logical_root') != 'dataset'
            or lock.get('kind') != config['data']['kind'] or lock.get('settings') != config['data']
            or lock.get('seed') != config['seed'] or not isinstance(lock.get('records'), list)):
        raise ValueError('Dataset lock does not match the project data settings')
    if lock['kind'] in _CASE_CSV_KINDS:
        _verify_csv_case_records(lock)
        return
    if lock['kind'] == 'unsupervised_csv':
        if not _is_digest(lock.get('source_digest')) or not lock['records'] or any(
                not isinstance(row, dict) or not isinstance(row.get('id'), str) or not _is_digest(row.get('content_digest'))
                for row in lock['records']):
            raise ValueError('Invalid unlabeled CSV dataset lock')
        return
    if lock['kind'] in _NUMERIC_CSV_KINDS:
        rows = lock['records']
        if not rows or any(not isinstance(row, dict) or type(row.get('row_number')) is not int
                           or not _is_digest(row.get('content_digest')) or not _is_digest(row.get('source_digest'))
                           or row.get('split') not in ('train', 'val', 'test') for row in rows):
            raise ValueError('Invalid numeric CSV row in the dataset lock')
        return
    paths = ('relative_path', 'mask_relative_path') if lock['kind'] == 'segmentation_csv' else ('relative_path',)
    if any(not isinstance(record, dict) or any(not _portable_relative(record.get(key)) for key in paths)
           for record in lock['records']):
        raise ValueError('Invalid relative path in the dataset lock')
    if config['data'].get('format') in MEDICAL_FORMATS:
        from .medical_geometry import affine_mm
        from .medical_volume_data import medical_recipe

        if lock.get('preprocessing') != medical_recipe(config['data']):
            raise ValueError('Medical preprocessing differs from the project config')
        count = len(config['data']['sequence_columns'])
        for record in lock['records']:
            for field, validate in [('sequence_relative_paths', _portable_relative),
                                    ('sequence_source_digests', _is_digest), ('sequence_content_digests', _is_digest)]:
                values = record.get(field)
                if not isinstance(values, list) or len(values) != count or not all(map(validate, values)):
                    raise ValueError(f'Invalid {field} in the medical dataset lock')
            for field in ('geometry', 'model_grid'):
                affine_mm(record.get(field))


def _file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _weight_identity(graph, training, dataset_id):
    task = _TASKS[training['profile'], training['data']['kind']]
    version = 3 if task == 'segmentation' or graph['schema_version'] == 2 else 2
    identity = {'version': version, 'profile': training['profile'], 'dataset_digest': dataset_id,
                'model_digest': model_digest(graph), 'checkpoint_signature': checkpoint_signature(graph, training, dataset_id)}
    if version == 3:
        identity['adapter'] = {'id': task, 'version': 1}
    return identity


def _check_weight_identity(candidate, identity):
    fields = set(identity) | {'adapter'}
    if any(candidate.get(key) != identity.get(key) for key in fields):
        raise ValueError('Checkpoint model or dataset identity does not match the saved project run')


class _CheckpointMismatch(ValueError):
    """The training run describes a different project model or dataset."""


def _check_origin_dataset(source, dataset_id, manifest, lock):
    if not _is_digest(dataset_id):
        raise ValueError('Invalid training dataset identity')
    if dataset_id == manifest['dataset_digest']:
        return
    # Re-export from a rebound dataset is safe for inference if the saved run's
    # lock has the original local digest and the exact same portable identity.
    try:
        digest_manifest = _manifest_digest
        if manifest['kind'] == 'multimodal_csv':
            from .multimodal_data import _portable_digest

            digest_manifest = _portable_digest
        original = _read_json(source.with_name('manifest.json'), _MAX_DATASET)
        if original.get('dataset_digest') != dataset_id or digest_manifest(original) != dataset_id:
            raise ValueError('Saved dataset manifest digest does not match the run specification')
        original_digest = portable_dataset(original)['digest']
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f'Could not verify the training dataset identity: {error}') from error
    if original_digest != lock['digest']:
        raise _CheckpointMismatch('Training and project datasets do not share the same portable identity')


def _training_run(source, config, manifest, lock, expected_model):
    source = Path(source).expanduser().resolve()
    raw = _read_bytes(source, _MAX_PROJECT)
    saved = _decode_json(raw, source)
    try:
        profile = saved['profile']
        training = saved['config']
        if profile != training['profile']:
            raise ValueError(f'Saved training profile does not match its run specification: {source}')
        if profile != config['profile'] or training['data']['kind'] != manifest['kind']:
            raise _CheckpointMismatch('Saved run uses a different task or data kind')
        training = _project_config(training)
        source_graph = saved['graph']
        dataset_id = saved['dataset_digest']
        identity = _weight_identity(source_graph, training, dataset_id)
        _check_weight_identity(saved, identity)
        if identity['model_digest'] != expected_model or _portable_spec(training['data']) != lock['settings']:
            raise _CheckpointMismatch('Saved run model or dataset settings do not match the project')
        if manifest['kind'] == 'multimodal_csv' and training['input_ports'] != config['input_ports']:
            raise _CheckpointMismatch('Saved run input port mapping does not match the project')
        if _portable_spec(saved['data_spec']) != lock['settings']:
            raise ValueError(f'Saved effective data settings do not match the run configuration: {source}')
        _check_origin_dataset(source, dataset_id, manifest, lock)
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError(f'Invalid saved run specification: {source}') from error
    return source, raw, source_graph, training, identity


def _select_training_run(candidates, config, manifest, lock, expected_model):
    if isinstance(candidates, (str, os.PathLike)):
        return _training_run(candidates, config, manifest, lock, expected_model)
    if not isinstance(candidates, Sequence) or not all(isinstance(candidate, (str, os.PathLike)) for candidate in candidates):
        raise ValueError('Checkpoint must be a run specification path or an ordered sequence of run specification paths')
    for candidate in candidates:
        try:
            return _training_run(candidate, config, manifest, lock, expected_model)
        except _CheckpointMismatch:
            continue
    raise ValueError('No saved training run matches the project model and dataset')


def _export_weights(training_run, directory, config, manifest, lock):
    import torch

    source, raw, source_graph, training, identity = training_run
    checkpoint = source.with_name('best.pt')
    try:
        with checkpoint.open('rb') as stream:
            before = os.fstat(stream.fileno())
            original_digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            stream.seek(0)
            state = torch.load(stream, map_location='cpu', weights_only=True)
            after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Checkpoint changed while it was being read')
    except Exception as error:
        raise ValueError(f'Could not read the saved checkpoint {checkpoint}: {error}') from error
    if not isinstance(state, dict):
        raise ValueError('Checkpoint must contain a state mapping')
    _check_weight_identity(state, identity)
    multimodal = manifest['kind'] == 'multimodal_csv'
    fields = ('input_shapes', 'input_axes', 'preprocessing') if multimodal else ('input_shape', 'preprocessing')
    fields += ('mask_values', 'target_shape') if config['profile'] == 'segmentation-v1' else ('class_names',)
    model_state = state.get('model_state')
    if (any(state.get(key) != manifest[key] for key in fields)
            or (multimodal and state.get('input_ports') != training['input_ports']) or not isinstance(model_state, dict)
            or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in model_state.items())):
        raise ValueError('Checkpoint weights or preprocessing do not match the dataset')
    if multimodal:
        fields += ('input_ports',)
    directory.mkdir()
    run_spec = directory / 'run_spec.json'
    weights = directory / 'best.pt'
    # The S9 reader validates a nonempty config path; its inference code uses
    # only the saved preprocessing. Keep a logical placeholder and the old digest.
    training = _detach_config(training)
    training['data']['path'] = 'external-data'
    _write_json(run_spec, {**identity, 'graph': source_graph, 'config': training,
                           'data_spec': lock['settings']}, _MAX_PROJECT)
    torch.save({**identity, **{key: state[key] for key in fields}, 'model_state': model_state}, weights)
    return {**_WEIGHT_FILES, 'format': 'pytorch-weights-only', 'model_digest': identity['model_digest'],
            'training_dataset_digest': identity['dataset_digest'], 'checkpoint_signature': identity['checkpoint_signature'],
            'run_spec_sha256': _file_digest(run_spec), 'checkpoint_sha256': _file_digest(weights),
            'source_run_spec_sha256': hashlib.sha256(raw).hexdigest(), 'source_checkpoint_sha256': original_digest}


def _read_weights(root, weights, initial_model):
    if weights is None:
        return None
    digest_names = ('run_spec_sha256', 'checkpoint_sha256', 'source_run_spec_sha256', 'source_checkpoint_sha256',
                    'training_dataset_digest', 'checkpoint_signature')
    if (not isinstance(weights, dict) or weights.get('format') != 'pytorch-weights-only'
            or weights.get('model_digest') != initial_model or not all(_is_digest(weights.get(name)) for name in digest_names)):
        raise ValueError('Invalid project weights metadata')
    result = dict(weights)
    for field, reference in _WEIGHT_FILES.items():
        if weights.get(field) != reference:
            raise ValueError(f'Invalid {field} path reference in project weights')
        path = _project_file(root, reference, field)
        if _file_digest(path) != weights[field + '_sha256']:
            raise ValueError(f'Project {field} digest does not match the weights metadata')
        result[field] = path
    return result


def _copy_runtime(destination, runtime_source, kind):
    source = Path(runtime_source).resolve() if runtime_source is not None else Path(__file__).resolve().parent
    required = ('__init__.py', 'research_project.py', 'project_runner.py', 'classification_runtime.py')
    if kind == 'table_csv':
        required += ('table_data.py', 'table_cv.py')
    elif kind == 'unsupervised_csv':
        required += ('unsupervised_task.py', 'unsupervised_python.py')
    elif kind == 'multimodal_csv':
        required += ('multimodal_data.py',)
    if not all((source / name).is_file() for name in required):
        raise ValueError('Could not locate the standalone studio_core source directory')
    runtime = destination / 'runtime' / 'studio_core'
    runtime.mkdir(parents=True)
    for module in sorted(source.glob('*.py')):
        if module.is_file():
            shutil.copyfile(module, runtime / module.name)
    for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'VERSION'):
        for parent in (source, source.parent, source.parent.parent, source.parent.parent.parent):
            reference = parent / name
            if reference.is_file():
                shutil.copyfile(reference, destination / name)
                break


def _runtime_requirements(data, *, classical=False):
    kind = data['kind']
    if kind == 'table_csv' or classical:
        return ('numpy==2.3.5\nscipy==1.15.3\nscikit-learn==1.7.2\n'
                'joblib==1.5.2\nthreadpoolctl==3.6.0\n')
    requirements = 'numpy==2.3.5\nPillow==12.3.0\n'
    fmt = data.get('format')
    if kind == 'multimodal_csv':
        modalities = data['modalities']
        if 'volume' in modalities or ('image' in modalities and data['image_source'] != 'raster'):
            fmt = data['volume_format']
    if fmt in ('nifti', 'mri_nifti') or (kind == 'segmentation_csv' and fmt == 'dicom_ct'):
        requirements += 'nibabel==5.3.2\n'
    if kind == 'segmentation_csv' and fmt in MEDICAL_FORMATS:
        requirements += 'scipy==1.15.3\n'
    if fmt == 'dicom_ct':
        requirements += 'pydicom==3.0.1\n'
    return requirements


def export_project(destination, graph, config, *, workbench=None, checkpoint=None, runtime_source=None,
                   include_data=False):
    """Create a standalone project; optional run specifications are in priority order."""
    requested = Path(destination).expanduser().absolute()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f'Project destination already exists: {requested}')
    destination = requested.parent.resolve() / requested.name
    config = _project_config(config)
    kind = config['data']['kind']
    from .classical_task import contains_classical

    classical = contains_classical(graph)
    if (kind == 'table_csv' or classical) and checkpoint is not None:
        raise ValueError('Table projects retrain from the locked CSV. Estimator serialization is unavailable.')
    if include_data and kind not in EMBEDDABLE_CSV_KINDS:
        raise ValueError('Embedding data is available for single-file numeric and table CSV projects')
    manifest = inspect_dataset(config['data'], seed=config['seed'])
    if config['expected_dataset_digest'] not in (None, manifest['dataset_digest']):
        raise ValueError('Dataset digest changed since inspection')
    config['data'] = manifest['data_spec']
    if manifest['kind'] == 'image_folder' and destination.is_relative_to(Path(manifest['data_spec']['path'])):
        raise ValueError('Place the project outside the ImageFolder dataset')
    if workbench is not None and not isinstance(workbench, dict):
        raise ValueError('Project workbench state must be an object')
    _validate_task(graph, config, manifest)
    code = export_python(graph)
    parsed = import_python(code)
    _validate_task(parsed, config, manifest)
    initial_digest = model_digest(parsed)
    lock = portable_dataset(manifest)
    payload = {'format': PROJECT_FORMAT, 'version': PROJECT_VERSION, 'project_id': str(uuid4()),
               'config': _detach_config(config), 'model_digest': initial_digest, **_PROJECT_FILES,
               'dataset_digest': lock['digest'], 'workbench': workbench, 'weights': None}

    # Stage first and claim the requested directory exclusively. Publish the
    # manifest last, so readers cannot observe a half-written project.
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.ml-forge-project-') as temporary:
        stage = Path(temporary)
        if checkpoint is not None:
            if initial_digest != model_digest(graph):
                raise ValueError('The Python round trip changed the saved checkpoint model identity')
            training_run = _select_training_run(checkpoint, config, manifest, lock, initial_digest)
            payload['weights'] = _export_weights(training_run, stage / 'weights', config, manifest, lock)
        (stage / 'model.py').write_text(code, encoding='utf-8', newline='\n')
        _write_json(stage / 'dataset.json', lock, _MAX_DATASET, pretty=False)
        if include_data:
            bundled = stage / _BUNDLED_CSV
            bundled.parent.mkdir()
            shutil.copyfile(manifest['data_spec']['path'], bundled)
            expected = manifest.get('source_digest') or manifest['records'][0]['source_digest']
            if _file_digest(bundled) != expected:
                raise ValueError('CSV contents changed while the project was being exported')
            payload['bundled_data'] = _BUNDLED_CSV
        if manifest['kind'] == 'segmentation_csv':
            _write_pairs(stage / _PAIRED_CSV, lock)
            payload['paired_csv'] = _PAIRED_CSV
        _copy_runtime(stage, runtime_source, kind)
        (stage / 'run.py').write_text(_RUN_ENTRY, encoding='utf-8', newline='\n')
        readme = {'table_csv': _TABLE_README, 'multimodal_csv': _MULTIMODAL_README}.get(kind, _README)
        if kind in _NUMERIC_CSV_KINDS:
            readme = (_TABLE_README if classical else _README) + (
                '\nUse the matching numeric CSV with `run.py check --data FILE` and '
                '`run.py train --data FILE --out DIRECTORY`.\n')
        if kind == 'unsupervised_csv':
            readme = ('# ModelCircuit Studio PCA / K-means project\n\n'
                      'Install `requirements.txt`. Run `python run.py check --data FILE` to verify the CSV, '
                      'then `python run.py train --data FILE --out DIRECTORY` to fit the pipeline and write '
                      '`projection.csv` and `metrics.json`. Open `project.json` in ModelCircuit Studio to edit the graph.\n')
        if include_data:
            readme += '\nThis project includes its CSV at `data/source.csv`. Omit `--data` to use that copy.\n'
        if kind in _CASE_CSV_KINDS:
            readme += _HELDOUT_README
        (stage / 'README.md').write_text(readme, encoding='utf-8', newline='\n')
        requirements = _runtime_requirements(config['data'], classical=classical)
        (stage / 'requirements.txt').write_text(requirements, encoding='utf-8', newline='\n')
        if kind != 'table_csv' and not classical:
            (stage / 'requirements-torch.txt').write_text('torch==2.6.0\n', encoding='utf-8', newline='\n')
        _write_json(stage / 'project.json', payload, _MAX_PROJECT)
        destination.mkdir()
        moved = []
        try:
            for item in sorted(stage.iterdir(), key=lambda path: path.name == 'project.json'):
                item.rename(destination / item.name)
                moved.append(item)
        except OSError:
            for item in reversed(moved):
                (destination / item.name).rename(item)
            destination.rmdir()
            raise
    return destination / 'project.json'


def read_project(path):
    """Open a project and statically parse the current model.py into a graph."""
    path = Path(path).expanduser()
    if path.is_dir():
        path /= 'project.json'
    root = path.parent.resolve()
    path = _project_file(root, path.name, 'project')
    project_bytes = _read_bytes(path, _MAX_PROJECT)
    payload = _decode_json(project_bytes, path)
    if (payload.get('format') != PROJECT_FORMAT or type(payload.get('version')) is not int
            or payload['version'] != PROJECT_VERSION):
        raise ValueError('Unsupported research project format or version')
    project_id = payload.get('project_id')
    try:
        if not isinstance(project_id, str) or str(UUID(project_id)) != project_id:
            raise ValueError('Invalid project identity')
    except ValueError as error:
        raise ValueError('Invalid research project identity') from error
    config = _validate_detached(payload.get('config'))
    workbench = payload.get('workbench')
    if workbench is not None and not isinstance(workbench, dict):
        raise ValueError('Invalid project workbench state')
    initial = payload.get('model_digest')
    if not _is_digest(initial):
        raise ValueError('Invalid initial project model digest')
    for field, reference in _PROJECT_FILES.items():
        if payload.get(field) != reference:
            raise ValueError(f'Invalid {field} path reference in project')
    lock = _read_json(_project_file(root, payload['dataset'], 'dataset'), _MAX_DATASET)
    _verify_dataset_lock(lock, config, payload.get('dataset_digest'))
    if lock['kind'] in ('table_csv', 'unsupervised_csv') and payload.get('weights') is not None:
        raise ValueError('Table projects retrain from the locked CSV. Estimator serialization is unavailable.')
    pair_reference = payload.get('paired_csv')
    paired_csv = None
    bundled_data = None
    if payload.get('bundled_data') is not None:
        if lock['kind'] not in EMBEDDABLE_CSV_KINDS or payload['bundled_data'] != _BUNDLED_CSV:
            raise ValueError('Invalid bundled data path in project')
        bundled_data = _project_file(root, _BUNDLED_CSV, 'bundled CSV')
    if pair_reference is not None:
        if lock['kind'] != 'segmentation_csv' or pair_reference != _PAIRED_CSV:
            raise ValueError('Invalid paired CSV path reference in project')
        paired_csv = _project_file(root, pair_reference, 'paired CSV')
    model_path = _project_file(root, payload['model'], 'model')
    model_bytes = _read_bytes(model_path, _MAX_MODEL)
    try:
        graph = import_python(model_bytes.decode('utf-8-sig'))
    except ValueError as error:
        raise ValueError(f'{model_path}: {error}') from error
    try:
        _validate_task(graph, config, lock)
        graph_error = None
    except (ValueError, KeyError, TypeError) as error:
        graph_error = f'{model_path}: {error}'
    return {'project_id': project_id, 'config': config, 'dataset': lock, 'workbench': workbench,
            'graph': graph, 'graph_error': graph_error, 'path': path, 'model_path': model_path, 'paired_csv': paired_csv,
            'bundled_data': bundled_data,
            'model_digest': model_digest(graph), 'initial_model_digest': initial,
            'weights': _read_weights(root, payload.get('weights'), initial),
            'source_digests': {'project_manifest_sha256': hashlib.sha256(project_bytes).hexdigest(),
                               'model_source_sha256': hashlib.sha256(model_bytes).hexdigest()}}


def bind_project(project, data_path=None, *, source_root=None, cancelled=None):
    """Verify a local dataset matches the project and return runnable config and manifest."""
    if data_path is None:
        data_path = project.get('bundled_data')
    if data_path is None or not os.fspath(data_path):
        raise ValueError('Select a local dataset path')
    settings = deepcopy(project['config'])
    data_path = Path(data_path).expanduser().resolve()
    kind = settings['data']['kind']
    if kind == 'segmentation_csv' and data_path.is_dir():
        paired_csv = project.get('paired_csv')
        if paired_csv is None:
            raise ValueError('This project has no bundled pair list. Select the segmentation CSV instead.')
        source_root, data_path = data_path, paired_csv
    if kind in _SOURCE_ROOT_KINDS:
        settings['data']['source_root'] = str(Path(source_root).expanduser().resolve()) if source_root else ''
    elif source_root:
        raise ValueError('A separate source root applies to segmentation and multimodal CSV data')
    settings['data']['path'] = str(data_path)
    config = _project_config(settings)
    manifest = inspect_dataset(config['data'], seed=config['seed'], cancelled=cancelled)
    if portable_dataset(manifest)['digest'] != project['dataset']['digest']:
        raise ValueError('Dataset does not match the project lock. Compare relative paths, contents, and ordered splits.')
    _validate_task(project['graph'], config, manifest)
    config['data'] = manifest['data_spec']
    config['expected_dataset_digest'] = manifest['dataset_digest']
    config['resume_checkpoint'] = None
    return config, manifest
