"""Train, predict and inspect a portable ModelCircuit research project."""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import uuid

from .classification_data import MAX_SAMPLES, _json_digest
from .classification_runtime import _atomic_write, _write_json
from .experiment_spec import checkpoint_signature, model_digest, validate_supervised_config
from .schema import prepare_graph

PROFILE = 'research-project-train-v1'
REPORT_TYPE = 'modelcircuit-project-run'
REPORT_NAME = 'project-run.json'
PENDING_REPORT = 'project-run.pending.json'
MAX_REPORT_BYTES = 12 * 1024 ** 2
_MAX_PREDICTION_BYTES = 128 * 1024 ** 2
_CASE_PROTOCOLS = {
    ('table_csv', 'classification-v2'): 'table-csv-classification-v2-heldout-v1',
    ('table_csv', 'regression-v1'): 'table-csv-regression-v1-heldout-v1',
    ('multimodal_csv', 'classification-v2'): 'multimodal-csv-classification-test-v1',
}
_COMPARISON_FILES = ('metrics', 'predictions')
_COMPARISON_FIELDS = ('run_id', 'model_digest', 'dataset_digest', 'local_dataset_digest',
                      'dataset_kind', 'profile', 'evaluation', 'evaluation_dataset_digest')
_COMPARISON_BUDGETS = {'metrics': MAX_REPORT_BYTES, 'heldout': MAX_REPORT_BYTES, 'predictions': _MAX_PREDICTION_BYTES}


@dataclass(frozen=True, slots=True)
class _VerifiedEvaluation:
    fingerprint: str


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _sha256(path, cancelled=None, *, maximum=None):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 ** 2):
            _check_cancelled(cancelled)
            size += len(block)
            if maximum is not None and size > maximum:
                raise ValueError('Project run artifact exceeds its size budget')
            digest.update(block)
    return digest.hexdigest()


def _check_cancelled(cancelled):
    if cancelled and cancelled():
        raise InterruptedError('Project run cancelled')


def validate_project_job(config):
    allowed = {'profile', 'project_path', 'data_path', 'source_root', 'device', 'max_runtime_seconds', 'expected_model_digest'}
    if not isinstance(config, dict) or not config.keys() <= allowed or config.get('profile', PROFILE) != PROFILE:
        raise ValueError('Invalid project run settings')
    settings = {'profile': PROFILE, 'source_root': None, 'device': 'cpu', 'max_runtime_seconds': 7200,
                'expected_model_digest': None, **config}
    for name in ('project_path', 'data_path', 'source_root'):
        value = settings.get(name)
        if name == 'source_root' and value is None:
            continue
        if not isinstance(value, str) or not value.strip() or len(value) > 4096 or '\0' in value:
            raise ValueError('Select a valid ' + name)
    if not isinstance(settings['device'], str) or not re.fullmatch(r'cpu|cuda:(0|[1-9][0-9]*)', settings['device']):
        raise ValueError('Select CPU or one explicit CUDA device')
    seconds = settings['max_runtime_seconds']
    if type(seconds) is not int or not 10 <= seconds <= 604800:
        raise ValueError('Project run time limit is outside 10..604800 seconds')
    model_id = settings['expected_model_digest']
    if model_id is not None and (not isinstance(model_id, str) or not re.fullmatch('[0-9a-f]{64}', model_id)):
        raise ValueError('Invalid expected model digest')
    return settings


def _limited_json(path):
    with Path(path).open('rb') as stream:
        data = stream.read(MAX_REPORT_BYTES + 1)
    if len(data) > MAX_REPORT_BYTES:
        raise ValueError('Project run report exceeds its size budget')
    try:
        value = json.loads(data)
        json.dumps(value, allow_nan=False)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError('Invalid project run JSON') from exc
    return value


def _artifact_path(root, value):
    if not isinstance(value, str) or '\\' in value or ':' in value:
        raise ValueError('Artifact path must be relative to its run directory')
    raw = PurePosixPath(value)
    if raw.is_absolute() or not raw.parts or '..' in raw.parts:
        raise ValueError('Artifact path must be relative to its run directory')
    path = root.joinpath(*raw.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact path resolves outside its run directory')
    return path


def read_project_run(path, *, verify=False, cancelled=None):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= REPORT_NAME
    report = _limited_json(path)
    if (not isinstance(report, dict) or report.get('format') != REPORT_TYPE or type(report.get('version')) is not int
            or report['version'] not in (1, 2)
            or report.get('operation') not in ('train', 'predict') or not isinstance(report.get('run_id'), str)
            or report.get('status') not in ('succeeded', 'failed', 'cancelled')):
        raise ValueError('Select a supported project-run.json file')
    report.pop('_verified_evaluation', None)
    artifacts = report.get('artifacts')
    if not isinstance(artifacts, dict):
        raise ValueError('Invalid project run artifact index')
    selected = {key: _artifact_path(path.parent, value) for key, value in artifacts.items()}
    report['artifacts'] = {key: item for key, item in selected.items() if item.is_file()}
    report['missing_artifacts'] = sorted(selected.keys() - report['artifacts'].keys())
    if verify:
        _verify_run_outputs(report, path.parent, cancelled)
        if report['operation'] == 'train' and report['status'] == 'succeeded' and _requires_evaluation_proof(report):
            try:
                _verify_evaluation_proof(report, cancelled)
            except InterruptedError:
                raise
            except (OSError, TypeError, ValueError, csv.Error):
                report['artifacts'].pop('heldout', None)
                report['changed_artifacts'] = sorted(set(report['changed_artifacts']) | {'heldout'})
            else:
                report['_verified_evaluation'] = _VerifiedEvaluation(_comparison_fingerprint(report))
    report['path'] = path
    return report


def _verify_run_outputs(report, directory, cancelled):
    fingerprints = report.get('artifact_sha256', {})
    output_records = report.get('output_sha256', {})
    if not isinstance(fingerprints, dict) or not isinstance(output_records, dict):
        raise ValueError('Invalid project run artifact checksums')
    files = report['artifacts']
    changed = []
    budgets = _COMPARISON_BUDGETS if _requires_evaluation_proof(report) else {}
    for name, file in tuple(files.items()):
        try:
            matches = _sha256(file, cancelled, maximum=budgets.get(name)) == fingerprints.get(name)
            if matches and name == 'metrics' and report['operation'] == 'train':
                matches = _limited_json(file).get('metrics') == report.get('metrics')
        except InterruptedError:
            raise
        except (OSError, ValueError, AttributeError):
            matches = False
        if not matches:
            changed.append(name)
            files.pop(name)
    missing_outputs, changed_outputs = [], []
    for reference, expected in output_records.items():
        file = _artifact_path(directory, reference)
        try:
            if _sha256(file, cancelled) != expected:
                changed_outputs.append(reference)
        except InterruptedError:
            raise
        except OSError:
            missing_outputs.append(reference)
    if report.get('profile') == 'segmentation-v1' and (missing_outputs or changed_outputs):
        for name in ('inference', 'predictions'):
            files.pop(name, None)
    report.update(changed_artifacts=sorted(changed), missing_outputs=sorted(missing_outputs),
                  changed_outputs=sorted(changed_outputs))


def _requires_evaluation_proof(report):
    fingerprints = report.get('artifact_sha256')
    artifacts = report.get('artifacts')
    return (report.get('version') == 2 or report.get('profile') == 'regression-v1'
            or report.get('dataset_kind') in ('table_csv', 'multimodal_csv')
            or report.get('evaluation') in _CASE_PROTOCOLS.values()
            or (isinstance(fingerprints, dict) and 'heldout' in fingerprints)
            or (isinstance(artifacts, dict) and 'heldout' in artifacts))


def _evaluation_contract(dataset):
    return {key: dataset[key] for key in ('kind', 'preprocessing', 'input_shape', 'target_shape',
                                         'class_names', 'mask_values', 'task', 'target_column',
                                         'input_shapes', 'input_axes') if key in dataset}


def _heldout_identity(dataset, profile):
    kind = dataset['kind']
    cases = []
    for record in dataset['records']:
        if record['split'] != 'test':
            continue
        group = record['group'] if kind == 'table_csv' else None
        cases.append([_json_digest([record['id'], record['label'], group]), _json_digest(record)])
    # The full case hash includes patient identity, geometry and all enabled
    # sources. The model's selected ports do not change the evaluation cohort.
    return {'kind': kind, 'profile': profile, 'evaluation': _CASE_PROTOCOLS[kind, profile],
            'contract_sha256': _json_digest(_evaluation_contract(dataset)), 'cases': cases}


def _comparison_fingerprint(report):
    return _json_digest({**{key: report.get(key) for key in _COMPARISON_FIELDS},
                         'version': report.get('version'), 'metrics': report.get('metrics')})


def _prediction_rows_match(path, identity, cancelled):
    if path.stat().st_size > _MAX_PREDICTION_BYTES:
        raise ValueError('Held-out predictions exceed their size limit')
    regression = identity['profile'] == 'regression-v1'
    target_column = 'target' if regression else 'label'
    grouped = identity['kind'] == 'table_csv'
    fields = {'sample_id', target_column} | ({'group'} if grouped else set())
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream, strict=True)
        if not fields <= set(reader.fieldnames or ()) or len(reader.fieldnames) != len(set(reader.fieldnames)):
            return False
        for row_hash, _ in identity['cases']:
            _check_cancelled(cancelled)
            row = next(reader, None)
            if row is None or None in row or any(row[key] is None for key in fields):
                return False
            target = float(row[target_column]) if regression else int(row[target_column])
            group = row['group'] if grouped else None
            if _json_digest([row['sample_id'], target, group]) != row_hash:
                return False
        return next(reader, None) is None


def _verify_evaluation_proof(report, cancelled):
    files = report['artifacts']
    if (report.get('version') != 2 or any(key not in files for key in ('heldout', *_COMPARISON_FILES))
            or report['changed_outputs'] or report['missing_outputs']):
        raise ValueError('Held-out run evidence is incomplete')
    proof = _limited_json(files['heldout'])
    if (not isinstance(proof, dict) or type(proof.get('version')) is not int or proof['version'] != 1
            or any(not isinstance(report.get(key), str) or proof.get(key) != report[key] for key in _COMPARISON_FIELDS)):
        raise ValueError('Held-out evidence does not match the report')
    if proof.get('artifact_sha256') != {key: report['artifact_sha256'][key] for key in _COMPARISON_FILES}:
        raise ValueError('The metrics or predictions do not match the held-out evidence')
    identity = proof.get('identity')
    if not isinstance(identity, dict) or _json_digest(identity) != report['evaluation_dataset_digest']:
        raise ValueError('Held-out cohort digest does not match the report')
    kind, profile = identity.get('kind'), identity.get('profile')
    if (kind != report['dataset_kind'] or profile != report['profile']
            or _CASE_PROTOCOLS.get((kind, profile)) != report['evaluation'] or identity.get('evaluation') != report['evaluation']):
        raise ValueError('Held-out evaluation protocol does not match the report')
    cases = identity.get('cases')
    metrics = report.get('metrics')
    test = metrics.get('test') if isinstance(metrics, dict) else None
    if (not isinstance(cases, list) or not 1 <= len(cases) <= MAX_SAMPLES
            or not isinstance(test, dict) or type(test.get('count')) is not int or len(cases) != test['count']):
        raise ValueError('Invalid held-out case or metric count')
    digests = [identity.get('contract_sha256')]
    for pair in cases:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError('Invalid held-out case digest')
        digests.extend(pair)
    if (not all(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) for digest in digests)
            or not _prediction_rows_match(files['predictions'], identity, cancelled)):
        raise ValueError('Prediction rows do not match the held-out cases')


def _write_evaluation_proof(directory, report, identity, origin, cancelled):
    test_count = report['metrics']['test']['count']
    if len(identity['cases']) != test_count or not _prediction_rows_match(Path(origin['predictions']), identity, cancelled):
        raise ValueError('Prediction rows do not match the held-out cases')
    path = directory / 'heldout.json'
    _write_json(path, {'version': 1, **{key: report[key] for key in _COMPARISON_FIELDS}, 'identity': identity,
                       'artifact_sha256': {key: _sha256(origin[key], cancelled, maximum=_COMPARISON_BUDGETS[key])
                                           for key in _COMPARISON_FILES}})
    return path


def evaluation_dataset_digest(dataset):
    """Identify comparable test cases independently of the training seed."""
    if dataset.get('kind') in ('table_csv', 'multimodal_csv'):
        profile = 'regression-v1' if dataset.get('task') == 'regression' else 'classification-v2'
        return _json_digest(_heldout_identity(dataset, profile))

    relevant = _evaluation_contract(dataset)
    relevant['test'] = [record for record in dataset['records'] if record['split'] == 'test']
    return _json_digest(relevant)


def compare_project_runs(left, right):
    def rejected(reason):
        return {'comparable': False, 'reason': reason, 'metrics': {}}

    if any(row.get('operation') != 'train' or row.get('status') != 'succeeded' for row in (left, right)):
        return rejected('请选择两次已完成的训练。')
    requires_proof = any(_requires_evaluation_proof(row) for row in (left, right))
    if requires_proof and any(not isinstance(row.get('_verified_evaluation'), _VerifiedEvaluation)
                              or row['_verified_evaluation'].fingerprint != _comparison_fingerprint(row)
                              for row in (left, right)):
        return rejected('测试病例、预测或指标文件未通过核验。')
    if any('metrics' in (*row.get('changed_artifacts', []), *row.get('missing_artifacts', [])) for row in (left, right)):
        return rejected('指标文件缺失或内容已改变。')
    key = 'evaluation_dataset_digest' if all(row.get('evaluation_dataset_digest') for row in (left, right)) else 'dataset_digest'
    if not left.get(key) or left[key] != right.get(key):
        return rejected('测试样本、标签或预处理不一致。')
    if not left.get('evaluation') or any(left.get(key) != right.get(key) for key in ('profile', 'evaluation')):
        return rejected('任务或评估协议不一致。')
    a, b = (row.get('metrics', {}).get('test', {}) for row in (left, right))
    if not isinstance(a, dict) or not isinstance(b, dict) or a.get('count') != b.get('count'):
        return rejected('测试集样本数不一致。')
    result = {}
    for name in ('accuracy', 'foreground_dice', 'foreground_iou', 'foreground_hd95_mm', 'loss', 'mae', 'rmse', 'r2'):
        x, y = a.get(name), b.get(name)
        if (type(x) in (int, float) and type(y) in (int, float) and math.isfinite(x) and math.isfinite(y)):
            result[name] = {'left': x, 'right': y, 'delta': y - x}
    return {'comparable': bool(result), 'reason': '' if result else '没有共同的测试指标。', 'metrics': result}


def _new_run(directory, operation, command):
    directory = Path(directory).expanduser().resolve()
    try:
        directory.mkdir(parents=True)
    except FileExistsError as exc:
        raise ValueError('Select a new run directory for each operation') from exc
    return directory, {'format': REPORT_TYPE, 'version': 1, 'run_id': uuid.uuid4().hex, 'operation': operation,
                       'started_at': _utc(), 'command': command, 'artifacts': {}}


def _environment(device, *, classical, torch_threads=None):
    from importlib.metadata import PackageNotFoundError, version

    package_names = ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl') if classical else (
        'torch', 'numpy', 'Pillow', 'nibabel', 'pydicom', 'scipy')
    packages = {}
    for name in package_names:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    result = {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}
    if not classical:
        import torch

        if torch_threads is not None:
            torch.set_num_threads(torch_threads)
        result.update(torch_threads=torch.get_num_threads(),
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        if device.startswith('cuda:'):
            result.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(int(device.split(':')[1])),
                          cudnn=torch.backends.cudnn.version())
            return result
    result['cpu'] = platform.processor() or platform.machine()
    return result


def _record_project(report, project, profile, device, *, torch_threads=None):
    classical = project['dataset']['kind'] == 'table_csv'
    report.update(project_id=project['project_id'], model_digest=project['model_digest'],
                  initial_model_digest=project['initial_model_digest'], profile=profile, device=device,
                  dataset_digest=project['dataset']['digest'],
                  environment=_environment(device, classical=classical, torch_threads=torch_threads),
                  source_versions={**project['source_digests'], 'runtime_sha256': _runtime_digest()})


def _runtime_digest():
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    sources = sorted(package.glob('*.py'))
    if not sources:
        # PyInstaller places source alongside the app resources for project export.
        bundle = Path(getattr(sys, '_MEIPASS', ''))
        sources = sorted((bundle / 'assets' / 'project_runtime' / 'studio_core').glob('*.py'))
    for source in sources:
        digest.update(source.name.encode('utf-8') + b'\0' + _sha256(source).encode('ascii'))
    return digest.hexdigest() if sources else None


def _write_result(directory, report, status, log, origin, error=None, cancelled=None, *, defer_report=False):
    report.update(status=status, finished_at=_utc())
    if error is not None:
        report['error'] = str(error)
    log_file = directory / 'progress.jsonl'
    content = ''.join(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n' for item in log)
    _atomic_write(log_file, lambda stream: stream.write(content.encode('utf-8')))
    report['artifacts'] = {'progress_log': log_file.name}
    if status == 'succeeded':
        for name, item in origin.items():
            item = Path(item).resolve()
            if item.is_file() and item.is_relative_to(directory):
                report['artifacts'][name] = item.relative_to(directory).as_posix()
    report['artifact_sha256'] = {name: _sha256(directory / path, cancelled) for name, path in report['artifacts'].items()}
    if status == 'succeeded':
        outputs = {}
        for subdir in ('masks', 'views'):
            for path in sorted((directory / subdir).glob('*')):
                if path.is_file():
                    outputs[path.relative_to(directory).as_posix()] = _sha256(path, cancelled)
        if outputs:
            report['output_sha256'] = outputs
        _check_cancelled(cancelled)
    target = directory / REPORT_NAME
    staged = directory / PENDING_REPORT
    _write_json(staged, report)
    if status == 'succeeded' and defer_report:
        return target
    staged.replace(target)
    return target


def finalize_project_report(directory, record):
    """Publish a managed run only after the supervisor accepts its terminal state."""
    target, staged = directory / REPORT_NAME, directory / PENDING_REPORT
    status = record['status']
    if status == 'succeeded':
        staged.replace(target)
        return
    payload = _limited_json(target if target.exists() else staged) if target.exists() or staged.exists() else None
    if payload is not None and payload.get('status') == status:
        staged.unlink(missing_ok=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    if payload is None:
        payload = {'format': REPORT_TYPE, 'version': 1, 'operation': 'train', 'run_id': record['run_id'],
                   'started_at': record['created_at'], 'device': record['config'].get('device'), 'artifacts': {}}
    payload.update(status=status, finished_at=_utc(), error=record.get('error') or 'Project run cancelled')
    retained = 'progress_log'
    for key in ('metrics', 'output_sha256', 'best_epoch'):
        payload.pop(key, None)
    for key in ('artifacts', 'artifact_sha256'):
        payload[key] = {name: value for name, value in payload.get(key, {}).items() if name == retained}
    _write_json(target, payload)
    staged.unlink(missing_ok=True)


def _progress(log, forward):
    def emit(event):
        # Epochs, phases and case boundaries describe the run without megabytes of minibatch events.
        if event.get('phase') not in ('train', 'val', 'test', 'sliding') or event.get('batch') == 1:
            log.append({'time': _utc(), **event})
        if forward is not None:
            forward(event)
    return emit


def train_project(project_path, data_path, artifact_directory, *, source_root=None, device='cpu',
                  progress=None, cancelled=None, expected_model_digest=None, defer_report=False, torch_threads=None):
    from .research_project import bind_project, read_project

    directory, report = _new_run(artifact_directory, 'train', {'project': str(project_path), 'data_path': str(data_path),
                                                              'source_root': str(source_root) if source_root else None, 'device': device})
    log = []
    try:
        _check_cancelled(cancelled)
        project = read_project(project_path)
        if expected_model_digest is not None and expected_model_digest != project['model_digest']:
            raise ValueError('Project model changed; reopen the project before starting the run')
        kind = project['dataset']['kind']
        _record_project(report, project, project['config']['profile'], device, torch_threads=torch_threads)
        if kind == 'table_csv' and device != 'cpu':
            raise ValueError('Table projects require CPU execution')
        emit = _progress(log, progress)
        emit({'phase': 'bind'})
        config, manifest = bind_project(project, data_path, source_root=source_root, cancelled=cancelled)
        config['device'] = device
        from .medical_volume_data import MEDICAL_FORMATS
        profile = config['profile']
        identity = None
        if (kind, profile) in _CASE_PROTOCOLS:
            identity = _heldout_identity(project['dataset'], profile)
            evaluation = identity['evaluation']
            evaluation_digest = _json_digest(identity)
            report.update(version=2, dataset_kind=kind)
        elif config['data'].get('format') in MEDICAL_FORMATS:
            evaluation = 'segmentation-native-test-v1'
        else:
            evaluation = profile + '-test-v1'
        if identity is None:
            evaluation_digest = evaluation_dataset_digest(project['dataset'])
        report.update(seed=config['seed'], local_dataset_digest=manifest['dataset_digest'],
                      evaluation=evaluation, evaluation_dataset_digest=evaluation_digest)
        if kind == 'table_csv':
            from .classical_task import run_classical

            report['folds'] = len(manifest['fold_plan'])
            result = run_classical(project['graph'], config, directory, emit, cancelled)
            report['algorithm'] = result['algorithm']
        else:
            from .classification_runtime import run_supervised

            report.update(epochs=config['epochs'], batch_size=config['batch_size'])
            result = run_supervised(project['graph'], config, directory, emit, cancelled)
            report.update(best_epoch=result['best_epoch'], global_step=result['global_step'])
        report.update(history=result['history'], metrics=result['metrics'])
        from .python_io import export_python
        snapshot = directory / 'model_snapshot.py'
        snapshot.write_text(export_python(project['graph']), encoding='utf-8', newline='\n')
        origin = {**result['artifacts'], 'model_snapshot': snapshot}
        if identity is not None:
            origin['heldout'] = _write_evaluation_proof(directory, report, identity, origin, cancelled)
        target = _write_result(directory, report, 'succeeded', log, origin, cancelled=cancelled, defer_report=defer_report)
        return {'status': 'succeeded', 'profile': PROFILE, 'operation': 'train', 'device': result['device'],
                'artifacts': {'project_run': str(target)}}
    except (Exception, KeyboardInterrupt) as exc:
        _write_result(directory, report, 'cancelled' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed', log, {}, exc)
        raise


def _select_weights(project, weights, cancelled):
    if weights is None:
        entry = project.get('weights')
        if not entry:
            raise ValueError('Select a completed project run or export this project with trained weights')
        if entry.get('model_digest') != project['model_digest']:
            raise ValueError('Bundled weights describe the original model. Train the edited project to use it for prediction.')
        return entry['run_spec'], entry['checkpoint']
    run = read_project_run(weights)
    if (run['operation'] != 'train' or run['status'] != 'succeeded' or run.get('model_digest') != project['model_digest']
            or run.get('dataset_digest') != project['dataset']['digest']):
        raise ValueError('Selected weights do not match this project model and data')
    verified = []
    for name in ('run_spec', 'best_checkpoint'):
        path = run['artifacts'].get(name)
        if not path or _sha256(path, cancelled) != run.get('artifact_sha256', {}).get(name):
            raise ValueError('Missing or changed checkpoint artifact: ' + name)
        verified.append(path)
    return tuple(verified)


def _classify_images(graph, run_spec, checkpoint, source, directory, device, cancelled, progress):
    import torch
    from .classification_data import MAX_IMAGE_BYTES, _decode_image, _prepare_image, _read_source
    from .classification_task import ClassificationTask
    from .devices import resolve_device
    from .runtime import build_model
    from .segmentation_inference import _source_paths

    saved = _limited_json(run_spec)
    training = validate_supervised_config(saved['config'])
    identity = checkpoint_signature(saved['graph'], training, saved['dataset_digest'])
    mid = model_digest(graph)
    checkpoint_state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (training['profile'] != 'classification-v2' or training['data']['kind'] != 'image_folder'
            or saved.get('profile') != 'classification-v2' or checkpoint_state.get('profile') != 'classification-v2'
            or identity != saved['checkpoint_signature'] or identity != checkpoint_state.get('checkpoint_signature')
            or mid != saved['model_digest'] or model_digest(saved['graph']) != mid
            or mid != checkpoint_state.get('model_digest') or checkpoint_state.get('dataset_digest') != saved['dataset_digest']):
        raise ValueError('Classification checkpoint identity does not match the saved project model')
    spec = training['data']
    channels, (height, width) = spec['channels'], spec['image_size']
    names = checkpoint_state['class_names']
    manifest = {'input_shape': [None, channels, height, width], 'class_names': names}
    resolved, shapes, _ = prepare_graph(graph)
    ClassificationTask(training).validate_contract(resolved, shapes, manifest)
    target = resolve_device(device)
    model = build_model(graph, device=target, capture=False)
    model.load_state_dict(checkpoint_state['model_state'], strict=True)
    model.eval()
    files = _source_paths(str(source), 'image')
    output = io.StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(('source', 'sha256', 'prediction', 'label', 'probabilities'))
    with torch.inference_mode():
        for index, path in enumerate(files, 1):
            _check_cancelled(cancelled)
            raw = _read_source(path, MAX_IMAGE_BYTES, cancelled)
            image, _ = _decode_image(raw, path)
            image = _prepare_image(image, spec['image_size'], channels)
            tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(height, width, channels)
            tensor = tensor.permute(2, 0, 1).to(device=target, dtype=torch.float32).div_(255.0).unsqueeze(0)
            logits = model(**{training['input_port']: tensor})['logits']
            if logits.shape != (1, len(names)) or not bool(torch.isfinite(logits).all()):
                raise ValueError('Invalid classification output during inference')
            label = int(logits.argmax(1).item())
            writer.writerow((path.name, hashlib.sha256(raw).hexdigest(), label, names[label],
                             json.dumps(logits.softmax(1)[0].cpu().tolist(), allow_nan=False)))
            progress({'phase': 'saved', 'step': index, 'total': len(files)})
    prediction_file = directory / 'predictions.csv'
    _atomic_write(prediction_file, lambda stream: stream.write(output.getvalue().encode('utf-8')))
    return {'cases': len(files), 'artifacts': {'predictions': prediction_file}}


def predict_project(project_path, source, artifact_directory, *, weights=None, device='cpu', mode='auto', overlap=None, sequence_paths=(),
                    progress=None, cancelled=None, torch_threads=None):
    from .research_project import read_project

    directory, report = _new_run(artifact_directory, 'predict', {'project': str(project_path), 'source': str(source),
                                                                'weights': str(weights) if weights else None,
                                                                'device': device, 'mode': mode, 'overlap': overlap})
    log = []
    try:
        _check_cancelled(cancelled)
        project = read_project(project_path)
        profile = project['config']['profile']
        if project['dataset']['kind'] not in ('image_folder', 'segmentation_csv'):
            raise ValueError('Use train with the locked CSV to regenerate held-out predictions. Predict accepts ImageFolder and segmentation projects.')
        _record_project(report, project, profile, device, torch_threads=torch_threads)
        run_spec, checkpoint = _select_weights(project, weights, cancelled)
        report['checkpoint_sha256'] = _sha256(checkpoint, cancelled)
        emit = _progress(log, progress)
        if profile == 'segmentation-v1':
            from .segmentation_inference import run_segmentation_inference

            result = run_segmentation_inference({'run_spec': str(run_spec), 'checkpoint': str(checkpoint),
                                                 'source': str(source), 'device': device, 'mode': mode,
                                                 'overlap': overlap, 'sequence_paths': list(map(str, sequence_paths))}, directory, emit, cancelled)
        else:
            if mode not in ('fixed', 'auto') or sequence_paths:
                raise ValueError('Image classification uses fixed-size inference without MRI sequences')
            result = _classify_images(project['graph'], run_spec, checkpoint, source, directory, device, cancelled, emit)
        report['cases'] = result['cases']
        target = _write_result(directory, report, 'succeeded', log, result['artifacts'], cancelled=cancelled)
        return {'status': 'succeeded', 'profile': PROFILE, 'operation': 'predict', 'device': device,
                'artifacts': {'project_run': str(target)}}
    except (Exception, KeyboardInterrupt) as exc:
        _write_result(directory, report, 'cancelled' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed', log, {}, exc)
        raise


def main(project_dir=None, argv=None):
    parser = argparse.ArgumentParser(description='Run a ModelCircuit research project')
    parser.add_argument('--project', type=Path, default=project_dir, required=project_dir is None)
    commands = parser.add_subparsers(dest='operation', required=True)
    validate = commands.add_parser('check', help='Check the model and bind matching data')
    train = commands.add_parser('train', help='Start training in a new run directory')
    predict = commands.add_parser('predict', help='Predict unlabelled images or segmentation volumes with saved weights')
    for command in (validate, train):
        command.add_argument('--data', type=Path, required=True, help='ImageFolder, table or multimodal CSV, or segmentation CSV or root')
        command.add_argument('--source-root', type=Path, help='Root of source paths referenced by the segmentation or multimodal CSV')
    for command in (train, predict):
        command.add_argument('--out', type=Path, required=True)
        command.add_argument('--device', default='cpu')
    predict.add_argument('--source', type=Path, required=True)
    predict.add_argument('--weights', type=Path, help='Training project-run.json or its directory')
    predict.add_argument('--mode', choices=('auto', 'fixed', 'sliding'), default='auto')
    predict.add_argument('--overlap', type=float)
    predict.add_argument('--sequence', type=Path, action='append', default=[], help='Additional MRI NIfTI in training channel order')
    args = parser.parse_args(argv)
    try:
        if args.operation == 'check':
            from .research_project import bind_project, read_project
            project = read_project(args.project)
            _, manifest = bind_project(project, args.data, source_root=args.source_root)
            print(json.dumps({'model_digest': project['model_digest'], 'dataset_digest': project['dataset']['digest'],
                              'counts': manifest['counts']}, ensure_ascii=False))
            return 0
        if args.operation == 'train':
            result = train_project(args.project, args.data, args.out, source_root=args.source_root, device=args.device, torch_threads=1)
        else:
            result = predict_project(args.project, args.source, args.out, weights=args.weights, device=args.device,
                                     mode=args.mode, overlap=args.overlap, sequence_paths=args.sequence, torch_threads=1)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        print('Project run cancelled', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
