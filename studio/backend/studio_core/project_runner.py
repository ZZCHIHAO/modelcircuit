"""Train, predict and inspect a portable ModelCircuit research project."""
import argparse
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

from .classification_runtime import _atomic_write, _write_json
from .experiment_spec import checkpoint_signature, model_digest, validate_supervised_config
from .schema import prepare_graph

PROFILE = 'research-project-train-v1'
REPORT_TYPE = 'modelcircuit-project-run'
REPORT_NAME = 'project-run.json'
PENDING_REPORT = 'project-run.pending.json'
MAX_REPORT_BYTES = 12 * 1024 ** 2


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _sha256(path, cancelled=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 ** 2):
            _check_cancelled(cancelled)
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
    if (not isinstance(report, dict) or report.get('format') != REPORT_TYPE or report.get('version') != 1
            or report.get('operation') not in ('train', 'predict') or not isinstance(report.get('run_id'), str)
            or report.get('status') not in ('succeeded', 'failed', 'cancelled')):
        raise ValueError('Select a version 1 project-run.json file')
    artifacts = report.get('artifacts')
    if not isinstance(artifacts, dict):
        raise ValueError('Invalid project run artifact index')
    selected = {key: _artifact_path(path.parent, value) for key, value in artifacts.items()}
    report['artifacts'] = {key: item for key, item in selected.items() if item.is_file()}
    report['missing_artifacts'] = sorted(selected.keys() - report['artifacts'].keys())
    if verify:
        _verify_run_outputs(report, path.parent, cancelled)
    report['path'] = path
    return report


def _verify_run_outputs(report, directory, cancelled):
    fingerprints = report.get('artifact_sha256', {})
    output_records = report.get('output_sha256', {})
    if not isinstance(fingerprints, dict) or not isinstance(output_records, dict):
        raise ValueError('Invalid project run artifact checksums')
    files = report['artifacts']
    changed = []
    for name, file in tuple(files.items()):
        try:
            matches = _sha256(file, cancelled) == fingerprints.get(name)
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


def evaluation_dataset_digest(dataset):
    """Identify comparable test cases independently of the training seed."""
    from .classification_data import _json_digest

    relevant = {key: dataset[key] for key in ('kind', 'preprocessing', 'input_shape', 'target_shape',
                                             'class_names', 'mask_values') if key in dataset}
    relevant['test'] = [record for record in dataset['records'] if record['split'] == 'test']
    return _json_digest(relevant)


def compare_project_runs(left, right):
    def rejected(reason):
        return {'comparable': False, 'reason': reason, 'metrics': {}}

    if any(row.get('operation') != 'train' or row.get('status') != 'succeeded' for row in (left, right)):
        return rejected('请选择两次已完成的训练。')
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
    for name in ('accuracy', 'foreground_dice', 'foreground_iou', 'loss'):
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


def _environment(device):
    import torch
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in ('torch', 'numpy', 'Pillow', 'nibabel'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    result = {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages,
              'torch_threads': torch.get_num_threads(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
    if device.startswith('cuda:'):
        result.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(int(device.split(':')[1])),
                      cudnn=torch.backends.cudnn.version())
    else:
        result.update(cpu=platform.processor() or platform.machine())
    return result


def _record_project(report, project, profile, device):
    report.update(project_id=project['project_id'], model_digest=project['model_digest'],
                  initial_model_digest=project['initial_model_digest'], profile=profile, device=device,
                  dataset_digest=project['dataset']['digest'], environment=_environment(device),
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
                  progress=None, cancelled=None, expected_model_digest=None, defer_report=False):
    from .classification_runtime import run_supervised
    from .research_project import bind_project, read_project

    directory, report = _new_run(artifact_directory, 'train', {'project': str(project_path), 'data_path': str(data_path),
                                                              'source_root': str(source_root) if source_root else None, 'device': device})
    log = []
    try:
        _check_cancelled(cancelled)
        project = read_project(project_path)
        if expected_model_digest is not None and expected_model_digest != project['model_digest']:
            raise ValueError('Project model changed; reopen the project before starting the run')
        _record_project(report, project, project['config']['profile'], device)
        emit = _progress(log, progress)
        emit({'phase': 'bind'})
        config, manifest = bind_project(project, data_path, source_root=source_root, cancelled=cancelled)
        config['device'] = device
        report.update(seed=config['seed'], local_dataset_digest=manifest['dataset_digest'],
                      evaluation=config['profile'] + '-test-v1', evaluation_dataset_digest=evaluation_dataset_digest(project['dataset']),
                      epochs=config['epochs'], batch_size=config['batch_size'])
        result = run_supervised(project['graph'], config, directory, emit, cancelled)
        report.update(history=result['history'], metrics=result['metrics'], best_epoch=result['best_epoch'],
                      global_step=result['global_step'])
        from .python_io import export_python
        snapshot = directory / 'model_snapshot.py'
        snapshot.write_text(export_python(project['graph']), encoding='utf-8', newline='\n')
        origin = {**result['artifacts'], 'model_snapshot': snapshot}
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


def predict_project(project_path, source, artifact_directory, *, weights=None, device='cpu', mode='fixed', overlap=.25,
                    progress=None, cancelled=None):
    from .research_project import read_project
    from .segmentation_inference import run_segmentation_inference

    directory, report = _new_run(artifact_directory, 'predict', {'project': str(project_path), 'source': str(source),
                                                                'weights': str(weights) if weights else None,
                                                                'device': device, 'mode': mode, 'overlap': overlap})
    log = []
    try:
        _check_cancelled(cancelled)
        project = read_project(project_path)
        profile = project['config']['profile']
        _record_project(report, project, profile, device)
        run_spec, checkpoint = _select_weights(project, weights, cancelled)
        report['checkpoint_sha256'] = _sha256(checkpoint, cancelled)
        emit = _progress(log, progress)
        if profile == 'segmentation-v1':
            result = run_segmentation_inference({'run_spec': str(run_spec), 'checkpoint': str(checkpoint),
                                                 'source': str(source), 'device': device, 'mode': mode,
                                                 'overlap': overlap}, directory, emit, cancelled)
        else:
            if mode != 'fixed':
                raise ValueError('Image classification uses fixed-size inference')
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
    predict = commands.add_parser('predict', help='Predict unlabelled files with saved weights')
    for command in (validate, train):
        command.add_argument('--data', type=Path, required=True, help='ImageFolder directory or segmentation CSV')
        command.add_argument('--source-root', type=Path, help='Root of paths referenced by the segmentation CSV')
    for command in (train, predict):
        command.add_argument('--out', type=Path, required=True)
        command.add_argument('--device', default='cpu')
    predict.add_argument('--source', type=Path, required=True)
    predict.add_argument('--weights', type=Path, help='Training project-run.json or its directory')
    predict.add_argument('--mode', choices=('fixed', 'sliding'), default='fixed')
    predict.add_argument('--overlap', type=float, default=.25)
    args = parser.parse_args(argv)
    try:
        if args.operation == 'check':
            from .research_project import bind_project, read_project
            project = read_project(args.project)
            _, manifest = bind_project(project, args.data, source_root=args.source_root)
            print(json.dumps({'model_digest': project['model_digest'], 'dataset_digest': project['dataset']['digest'],
                              'counts': manifest['counts']}, ensure_ascii=False))
            return 0
        import torch
        torch.set_num_threads(1)
        if args.operation == 'train':
            result = train_project(args.project, args.data, args.out, source_root=args.source_root, device=args.device)
        else:
            result = predict_project(args.project, args.source, args.out, weights=args.weights, device=args.device,
                                     mode=args.mode, overlap=args.overlap)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        print('Project run cancelled', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
