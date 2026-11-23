"""Persistent single-host Linux DDP jobs over a narrow JSON/ZIP command interface."""
from argparse import ArgumentParser
from contextlib import contextmanager
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import tempfile
import time
from uuid import uuid4
from zipfile import ZIP_STORED, ZipFile

from .classification_runtime import _write_json
from .project_runner import REPORT_NAME, _limited_json, read_project_run
from .research_project import bind_project, read_project

PROTOCOL = 1
GPU_UUID = re.compile(r'GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}')
JOB_ID = re.compile(r'[0-9a-f]{32}')
MAX_REQUEST_BYTES = 96 * 1024**2
MAX_FILE_BYTES = {'project.json': 8 * 1024**2, 'dataset.json': 64 * 1024**2, 'model.py': 8 * 1024**2}
MAX_DOWNLOAD_BYTES = 16 * 1024**3 - 1024**2
MAX_LOG_TAIL = 8192
TERMINAL = {'succeeded', 'failed', 'cancelled'}
ARTIFACT_FILES = {
    'progress_log': 'progress.jsonl', 'last_checkpoint': 'last.pt', 'best_checkpoint': 'best.pt',
    'run_spec': 'run_spec.json', 'metrics': 'metrics.json', 'manifest': 'manifest.json',
    'predictions': 'predictions.csv', 'model_snapshot': 'model_snapshot.py',
}


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def _check_job_id(job_id):
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise ValueError('Expected a 32-digit job ID')
    return job_id


def _validate_request(request):
    required = {'files', 'data_path', 'gpu_uuids', 'per_device_batch_size', 'accumulation_steps', 'epochs'}
    if not isinstance(request, dict) or not required <= request.keys() <= required | {'resume_from', 'request_id'}:
        raise ValueError('Invalid remote job request fields')
    spec = {**request, 'resume_from': request.get('resume_from'), 'request_id': request.get('request_id')}
    if spec['request_id'] is not None:
        _check_job_id(spec['request_id'])
    if spec['resume_from'] is not None:
        _check_job_id(spec['resume_from'])
        if spec['resume_from'] == spec['request_id']:
            raise ValueError('A job cannot resume itself')
    files = spec['files']
    if not isinstance(files, dict) or set(files) != set(MAX_FILE_BYTES):
        raise ValueError('Submit a project.json, model.py, and dataset.json only')
    for name, data in files.items():
        if not isinstance(data, str) or not data or len(data.encode('utf-8')) > MAX_FILE_BYTES[name]:
            raise ValueError('Project file is empty or exceeds its size limit: ' + name)
    path = spec['data_path']
    if not isinstance(path, str) or not path.startswith('/') or len(path) > 4096 or any(ord(c) < 32 for c in path):
        raise ValueError('Expected an existing absolute Linux dataset directory')
    location = PurePosixPath(path)
    if location.anchor != '/' or len(location.parts) < 2 or '..' in location.parts or str(location) != path.rstrip('/'):
        raise ValueError('Expected a canonical absolute Linux dataset directory')
    spec['data_path'] = str(location)
    gpus = spec['gpu_uuids']
    if (not isinstance(gpus, list) or len(gpus) < 2 or
            any(not isinstance(item, str) or not GPU_UUID.fullmatch(item) for item in gpus) or
            len({item.lower() for item in gpus}) != len(gpus)):
        raise ValueError('Select at least two different, explicit GPU UUIDs')
    for key, upper in [('per_device_batch_size', 64), ('accumulation_steps', 32), ('epochs', 1000)]:
        if type(spec[key]) is not int or not 1 <= spec[key] <= upper:
            raise ValueError(f'{key} must be an integer from 1 to {upper}')
    return spec


def _parse_gpus(text):
    rows = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 6:
            raise ValueError('Unexpected nvidia-smi inventory format')
        index, device, name, used, total, utilization = (column.strip() for column in row)
        if not GPU_UUID.fullmatch(device):
            raise ValueError('Invalid GPU UUID in nvidia-smi inventory')
        rows.append({'index': int(index), 'uuid': device, 'name': name, 'memory_used_mib': int(used),
                     'memory_total_mib': int(total), 'utilization': int(utilization)})
    if len({row['index'] for row in rows}) != len(rows) or len({row['uuid'].lower() for row in rows}) != len(rows):
        raise ValueError('nvidia-smi returned duplicate device identities')
    return rows


def _probe_gpus():
    query = 'index,uuid,name,memory.used,memory.total,utilization.gpu'
    process = subprocess.run(['nvidia-smi', '--query-gpu=' + query, '--format=csv,noheader,nounits'],
                             capture_output=True, encoding='utf-8', errors='replace', timeout=12, check=True)
    return _parse_gpus(process.stdout)


def _ddp_available():
    import torch
    return bool(torch.version.cuda and torch.distributed.is_available() and torch.distributed.is_nccl_available())


def probe():
    try:
        devices = _probe_gpus()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {'protocol': PROTOCOL, 'gpus': [], 'ddp': False, 'reason': 'GPU inventory unavailable: ' + str(error)}
    try:
        available = _ddp_available()
    except (ImportError, OSError, RuntimeError) as error:
        return {'protocol': PROTOCOL, 'gpus': devices, 'ddp': False, 'reason': 'PyTorch unavailable: ' + str(error)}
    reason = '' if available and len(devices) >= 2 else 'DDP requires the CUDA/NCCL runtime and at least two GPUs'
    return {'protocol': PROTOCOL, 'gpus': devices, 'ddp': available and len(devices) >= 2, 'reason': reason}


def _checked_root(root, *, create=False):
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError('Job root must be an absolute, canonical directory')
    if create:
        root.mkdir(mode=0o700, exist_ok=True)
    if root.exists() and not root.is_dir():
        raise ValueError('Job root must be a directory')
    if os.name == 'posix' and root.exists():
        info = root.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError('Job root must be private to the SSH account (mode 0700)')
    return root


def _job(root, job_id):
    path = _checked_root(root) / _check_job_id(job_id)
    if path.is_symlink() or not path.is_dir() or not (path / 'state.json').is_file():
        raise ValueError('Unknown remote job: ' + job_id)
    if os.name == 'posix' and (path.stat().st_uid != os.getuid() or stat.S_IMODE(path.stat().st_mode) & 0o077):
        raise ValueError('Remote job does not belong to the SSH account')
    return path


def _normal_file(path):
    return not path.is_symlink() and path.is_file()


def _resume_available(job):
    run = job / 'run'
    checkpoint = run / 'last.pt'
    return not run.is_symlink() and _normal_file(checkpoint) and checkpoint.stat().st_size > 0


def _tail(path):
    if not _normal_file(path):
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - MAX_LOG_TAIL))
        return stream.read(MAX_LOG_TAIL).decode('utf-8', errors='replace')


def status(root, job_id):
    job = _job(root, job_id)
    record = _limited_json(job / 'state.json')
    if record.get('job_id') != job_id:
        raise ValueError('Unexpected remote job identity')
    cancelled = (job / 'cancel').exists()
    if cancelled and record['status'] not in TERMINAL:
        record['status'] = 'cancel_requested'
    progress_file = job / 'progress.json'
    progress = _limited_json(progress_file) if _normal_file(progress_file) else {}
    resume_available = _resume_available(job)
    progress['resume_available'] = resume_available
    return {**record, 'protocol': PROTOCOL, 'progress': progress, 'resume_available': resume_available,
            'log': _tail(job / 'worker.log')}


def cancel(root, job_id):
    job = _job(root, job_id)
    current = status(root, job_id)
    if current['status'] not in TERMINAL:
        # The supervisor is the only process authorized to signal the child it started.
        (job / 'cancel').touch(mode=0o600, exist_ok=True)
        return status(root, job_id)
    return current


def _read_project_for_job(project_path, data_path):
    project = read_project(project_path)
    if project['config']['profile'] != 'classification-v2' or project['config']['data']['kind'] != 'image_folder':
        raise ValueError('Remote DDP currently supports ImageFolder classification projects')
    if project.get('weights') or project.get('graph_error'):
        raise ValueError(project.get('graph_error') or 'Submit the portable project without bundled weights')
    if not Path(data_path).is_dir():
        raise ValueError('Select an existing ImageFolder dataset on the server')
    bind_project(project, data_path)
    return project


def _launch_supervisor(root, job_id):
    job = root / job_id
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1]), 'PYTHONDONTWRITEBYTECODE': '1'}
    with (job / 'worker.log').open('ab', buffering=0) as log:
        subprocess.Popen([sys.executable, '-s', '-m', 'studio_core.remote_runner', '--root', str(root),
                          '_supervise', '--job', job_id], cwd=job, env=env, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=log, close_fds=True, start_new_session=True)


def _retry(root, job_id, fingerprint):
    job = _job(root, job_id)
    if _limited_json(job / 'spec.json').get('request_sha256') != fingerprint:
        raise ValueError('Job ID already belongs to a different request')
    return status(root, job_id)


def submit(root, request):
    spec = _validate_request(request)
    fingerprint = hashlib.sha256(_encode(spec)).hexdigest()
    job_id = spec.pop('request_id') or uuid4().hex
    root = _checked_root(root)
    job = root / job_id
    if job.exists():
        return _retry(root, job_id, fingerprint)

    by_uuid = {card['uuid'].lower(): card['uuid'] for card in _probe_gpus()}
    if any(device.lower() not in by_uuid for device in spec['gpu_uuids']):
        raise ValueError('Unavailable GPU selected for this server')
    if not _ddp_available():
        raise ValueError('This Python runtime cannot use CUDA/NCCL DDP')
    spec['gpu_uuids'] = [by_uuid[device.lower()] for device in spec['gpu_uuids']]
    if spec['resume_from'] is not None:
        previous = status(root, spec['resume_from'])
        if previous['status'] not in TERMINAL or not previous['resume_available']:
            raise ValueError('Resume requires a finished job with an intact last checkpoint')
        prior = _limited_json(root / spec['resume_from'] / 'spec.json')
        for key in ('gpu_uuids', 'per_device_batch_size', 'accumulation_steps'):
            if spec[key] != prior[key]:
                raise ValueError('Resume requires the same GPU order, per-device batch and accumulation')

    _checked_root(root, create=True)
    files = spec.pop('files')
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=root) as temporary:
        staging = Path(temporary) / 'job'
        staging.mkdir(mode=0o700)
        project = staging / 'project'
        project.mkdir(mode=0o700)
        for name, content in files.items():
            (project / name).write_bytes(content.encode('utf-8'))
        _read_project_for_job(project / 'project.json', spec['data_path'])
        spec['request_sha256'] = fingerprint
        _write_json(staging / 'spec.json', spec)
        count = len(spec['gpu_uuids'])
        record = {'job_id': job_id, 'status': 'queued', 'created_at': _utc(), 'gpu_uuids': spec['gpu_uuids'],
                  'per_device_batch_size': spec['per_device_batch_size'], 'accumulation_steps': spec['accumulation_steps'],
                  'effective_batch_size': count * spec['per_device_batch_size'] * spec['accumulation_steps'],
                  'epochs': spec['epochs'], 'resume_from': spec['resume_from'], 'error': None}
        _write_json(staging / 'state.json', record)
        try:
            staging.rename(job)
        except OSError:
            if not job.exists():
                raise
            return _retry(root, job_id, fingerprint)
    try:
        _launch_supervisor(root, job_id)
    except OSError as error:
        _write_json(job / 'state.json', {**record, 'status': 'failed', 'finished_at': _utc(), 'error': str(error)})
    return status(root, job_id)


def _worker_command(job, spec):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(spec['gpu_uuids']), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
           'PYTHONPATH': str(Path(__file__).resolve().parents[1]), 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'}
    temporary, cache = job / 'tmp', job / 'cache'
    for directory in (temporary, cache):
        directory.mkdir(mode=0o700, exist_ok=True)
    env.update(TMPDIR=str(temporary), TEMP=str(temporary), TMP=str(temporary), XDG_CACHE_HOME=str(cache),
               TORCH_HOME=str(cache / 'torch'), CUDA_CACHE_PATH=str(cache / 'cuda'), TRITON_CACHE_DIR=str(cache / 'triton'))
    command = [sys.executable, '-s', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--max-restarts=0',
               '--nproc-per-node=' + str(len(spec['gpu_uuids'])), '-m', 'studio_core.distributed_image',
               '--project', str(job / 'project' / 'project.json'), '--data', spec['data_path'],
               '--out', str(job / 'run'), '--per-device-batch', str(spec['per_device_batch_size']),
               '--accumulation', str(spec['accumulation_steps']), '--epochs', str(spec['epochs']),
               '--progress', str(job / 'progress.json')]
    if spec['resume_from'] is not None:
        command += ['--resume', str(job.parent / spec['resume_from'] / 'run')]
    return command, env


@contextmanager
def _reserve_gpus(root, devices):
    import fcntl

    lease_directory = root / '.gpu-leases'
    lease_directory.mkdir(mode=0o700, exist_ok=True)
    _checked_root(lease_directory)
    handles = []
    try:
        for device in sorted(set(devices), key=str.lower):
            if not GPU_UUID.fullmatch(device):
                raise ValueError('Invalid GPU lease identity')
            path = lease_directory / (device.lower() + '.lock')
            handle = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(device + ' is already reserved by another job in this workspace') from error
        yield
    finally:
        for handle in reversed(handles):
            os.close(handle)


def _check_selected_gpu_activity(devices):
    selected = {device.lower() for device in devices}
    command = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']
    process = subprocess.run(command, capture_output=True, encoding='utf-8', errors='replace', timeout=12, check=True)
    for row in csv.reader(process.stdout.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 2 or not GPU_UUID.fullmatch(row[0].strip()) or not row[1].strip().isdecimal():
            raise ValueError('Could not verify active GPU compute processes')
        device = row[0].strip()
        if device.lower() in selected:
            raise ValueError(device + ' already has an active compute process; select an available GPU')


def _subreaper():
    # torchrun workers have their own sessions. Reparent orphaned workers to this
    # supervisor so its cleanup remains effective even if torchrun crashes.
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    if prctl(36, 1, 0, 0, 0) != 0:  # Linux PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _adopted_children():
    children = Path('/proc/thread-self/children')
    if children.is_file():
        return [int(number) for number in children.read_text(encoding='ascii').split()]
    mine = str(os.getpid())
    own = []
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal():
            continue
        try:
            fields = (process / 'stat').read_text(encoding='utf-8', errors='replace').rsplit(')', 1)[1].split()
        except (OSError, IndexError):
            continue
        if len(fields) >= 2 and fields[1] == mine:
            own.append(int(process.name))
    return own


def _clear_adopted_children():
    forced = time.monotonic() + 1
    while children := _adopted_children():
        for pid in children:
            # An unreaped own child cannot have its PID reassigned. No other
            # thread in the supervisor calls waitpid or signals these processes.
            try:
                reaped, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                continue
            if not reaped:
                try:
                    os.kill(pid, signal.SIGKILL if time.monotonic() > forced else signal.SIGTERM)
                except ProcessLookupError:
                    pass
        time.sleep(0.05)


def _stop_worker(process):
    if process.poll() is None:
        process.terminate()  # torchrun first gives each rank a chance to exit.
        try:
            process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _supervise(root, job_id):
    job = _job(root, job_id)
    state_path = job / 'state.json'
    record = _limited_json(state_path)
    if record['status'] != 'queued':
        raise ValueError('Remote supervisor requires a queued job')
    # Exclusive ownership if a receiver accidentally starts the supervisor twice.
    with (job / 'supervisor').open('x', encoding='ascii') as lease:
        lease.write(str(os.getpid()))
    spec = _limited_json(job / 'spec.json')
    stop = job / 'cancel'
    outcome, error, return_code = 'failed', None, None
    try:
        if not stop.exists():
            with _reserve_gpus(root, spec['gpu_uuids']):
                _check_selected_gpu_activity(spec['gpu_uuids'])
                if not stop.exists():
                    _subreaper()
                    command, env = _worker_command(job, spec)
                    worker = subprocess.Popen(command, cwd=job, env=env, stdin=subprocess.DEVNULL,
                                              close_fds=True, start_new_session=True)
                    try:
                        _write_json(state_path, {**record, 'status': 'running', 'started_at': _utc()})
                        while worker.poll() is None and not stop.exists():
                            time.sleep(0.2)
                    finally:
                        _stop_worker(worker)
                        _clear_adopted_children()
                        return_code = worker.returncode
                    if return_code == 0 and not stop.exists():
                        report = read_project_run(job / 'run' / REPORT_NAME)
                        if report['status'] != 'succeeded':
                            raise ValueError('Distributed worker returned an incomplete project report')
                        outcome = 'succeeded'
                    else:
                        error = f'Distributed worker exited with code {return_code}'
    except Exception as failure:
        error = str(failure)
    finally:
        if stop.exists():
            outcome, error = 'cancelled', None
        _write_json(state_path, {**record, 'status': outcome, 'finished_at': _utc(),
                                 'error': error, 'exit_code': return_code})


def fetch(root, job_id, destination):
    job = _job(root, job_id)
    if status(root, job_id)['status'] != 'succeeded':
        raise ValueError('Only succeeded jobs can be downloaded')
    run = job / 'run'
    report_file = run / REPORT_NAME
    if run.is_symlink() or not _normal_file(report_file):
        raise ValueError('Remote job report is missing')
    document = _limited_json(report_file)
    artifacts = document.get('artifacts', {})
    if (not isinstance(artifacts, dict) or any(ARTIFACT_FILES.get(key) != name for key, name in artifacts.items()) or
            document.get('output_sha256')):
        raise ValueError('Unexpected remote job artifact names')
    for name in artifacts.values():
        if not _normal_file(run / name):
            raise ValueError('Remote job artifacts changed or missing')
    report = read_project_run(report_file, verify=True)
    if report['status'] != 'succeeded' or report['missing_artifacts'] or report['changed_artifacts']:
        raise ValueError('Remote job artifacts changed or missing')
    files = [report_file, *report['artifacts'].values()]
    if sum(path.stat().st_size for path in files) > MAX_DOWNLOAD_BYTES:
        raise ValueError('Remote job artifacts exceed the download budget')
    # A normal (non-seekable) SSH stdout uses the ZIP data descriptor format.
    with ZipFile(destination, mode='w', compression=ZIP_STORED, allowZip64=True) as archive:
        for path in files:
            archive.write(path, arcname=path.name)


def _read_request(stream):
    data = stream.read(MAX_REQUEST_BYTES + 1)
    if len(data) > MAX_REQUEST_BYTES:
        raise ValueError('Remote job request exceeds its byte limit')
    return json.loads(data)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('op', choices=('probe', 'submit', 'status', 'cancel', 'fetch', '_supervise'))
    parser.add_argument('--job')
    arguments = parser.parse_args(argv)
    try:
        if sys.platform != 'linux':
            raise ValueError('The remote DDP runner requires Linux')
        root = _checked_root(arguments.root)
        if arguments.op == 'probe':
            result = probe()
        elif arguments.op == 'submit':
            result = submit(root, _read_request(sys.stdin.buffer))
        elif arguments.op == 'status':
            result = status(root, arguments.job)
        elif arguments.op == 'cancel':
            result = cancel(root, arguments.job)
        elif arguments.op == 'fetch':
            fetch(root, arguments.job, sys.stdout.buffer)
            return 0
        else:
            _supervise(root, arguments.job)
            return 0
        sys.stdout.buffer.write(_encode(result) + b'\n')
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
