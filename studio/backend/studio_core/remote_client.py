"""OpenSSH client for portable ImageFolder jobs and verified result transfer."""
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
from uuid import uuid4
import zipfile

from .project_runner import REPORT_NAME, read_project_run
from .research_project import read_project

PROTOCOL = 1
JOB_ID = re.compile(r'[0-9a-f]{32}')
GPU = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
SSH_ALIAS = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}')
SOURCE_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\.py')
ARTIFACT_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}')
MAX_RESULT_FILES = 64
MAX_RESULT_BYTES = 16 * 1024**3
MAX_PROTOCOL_BYTES = 96 * 1024**2


# Source is transferred as data and checked before an installed runtime becomes visible.
_INSTALL = '''import hashlib, json, pathlib, re, sys, tempfile, shutil
base = pathlib.Path(sys.argv[1])
if base.resolve() != base:
    raise ValueError('Use the server workspace canonical path')
base.mkdir(mode=0o700, exist_ok=True)
runtime = base / 'runtime'
runtime.mkdir(mode=0o700, exist_ok=True)
if runtime.resolve() != runtime:
    raise ValueError('The remote runtime folder cannot be a link')
request = json.load(sys.stdin)
digest, files = request['digest'], request['files']
if not re.fullmatch(r'[0-9a-f]{64}', digest) or not 1 <= len(files) <= 200:
    raise ValueError('Invalid source inventory')
checksums = {}
for name, content in sorted(files.items()):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*\\.py', name) or not isinstance(content, str):
        raise ValueError('Unexpected runtime file')
    checksums[name] = hashlib.sha256(content.encode('utf-8')).hexdigest()
identity = json.dumps(checksums, sort_keys=True, separators=(',', ':')).encode()
if hashlib.sha256(identity).hexdigest() != digest:
    raise ValueError('Runtime source identity mismatch')
installed = runtime / digest
if not installed.exists():
    staging = pathlib.Path(tempfile.mkdtemp(prefix='.upload-', dir=runtime))
    try:
        package = staging / 'studio_core'
        package.mkdir()
        for name, content in files.items():
            (package / name).write_bytes(content.encode('utf-8'))
        try:
            staging.rename(installed)
        except OSError:
            if not installed.is_dir():
                raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
package = installed / 'studio_core'
if package.resolve() != package or not package.is_dir():
    raise ValueError('Runtime placement is invalid')
if set(checksums) != {item.name for item in package.iterdir()} or any(
        (package / name).is_symlink() or hashlib.sha256((package / name).read_bytes()).hexdigest() != value
        for name, value in checksums.items()):
    raise ValueError('Installed runtime verification failed')
print(json.dumps({'protocol': 1, 'digest': digest}))
'''


def _remote_path(text):
    if not isinstance(text, str) or not text.startswith('/') or len(text) > 4096 or any(ord(c) < 32 for c in text):
        raise ValueError('服务器路径须为 Linux 绝对路径')
    path = PurePosixPath(text)
    if path.anchor != '/' or len(path.parts) < 2 or '..' in path.parts or str(path) != text.rstrip('/'):
        raise ValueError('请填写规范的 Linux 绝对路径')
    return str(path)


@dataclass(frozen=True)
class RemoteProfile:
    host: str
    python: str
    workspace: str

    def __post_init__(self):
        if not isinstance(self.host, str) or not SSH_ALIAS.fullmatch(self.host):
            raise ValueError('请选择已配置的 SSH 主机别名')
        object.__setattr__(self, 'python', _remote_path(self.python))
        object.__setattr__(self, 'workspace', _remote_path(self.workspace))


def batch_plan(gpu_uuids, per_device, accumulation):
    if (not isinstance(gpu_uuids, (tuple, list)) or len(gpu_uuids) < 2 or
            any(not isinstance(item, str) or not GPU.fullmatch(item) for item in gpu_uuids) or
            len({item.lower() for item in gpu_uuids}) != len(gpu_uuids)):
        raise ValueError('请明确选择至少两张不同的服务器显卡')
    if type(per_device) is not int or not 1 <= per_device <= 64:
        raise ValueError('每卡批次须为 1 至 64 的整数')
    if type(accumulation) is not int or not 1 <= accumulation <= 32:
        raise ValueError('梯度累积须为 1 至 32 的整数')
    step = len(gpu_uuids) * per_device
    return {'per_device': per_device, 'per_step': step, 'effective': step * accumulation}


def check_job_id(job):
    if not isinstance(job, str) or not JOB_ID.fullmatch(job):
        raise ValueError('任务 ID 须为 32 位十六进制字符')
    return job


def _checked_json(raw):
    if len(raw) > MAX_PROTOCOL_BYTES:
        raise ValueError('服务器响应超过协议大小')
    try:
        response = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise ValueError('服务器未返回有效的任务响应') from exc
    if not isinstance(response, dict):
        raise ValueError('服务器任务响应应为 JSON 对象')
    return response


def _project_files(path):
    opened = read_project(path)
    if opened.get('graph_error'):
        raise ValueError(opened['graph_error'])
    if opened['config']['data']['kind'] != 'image_folder' or opened['config']['profile'] != 'classification-v2':
        raise ValueError('远程 DDP 目前运行 ImageFolder 分类科研工程')
    root = opened['path'].parent
    files = {name: (root / name).read_text(encoding='utf-8-sig') for name in ('project.json', 'model.py', 'dataset.json')}
    if opened['weights'] is not None:
        manifest = json.loads(files['project.json'])
        manifest['weights'] = None
        files['project.json'] = json.dumps(manifest, ensure_ascii=False, allow_nan=False)
    return files


def unpack_result(archive, destination, *, cancelled=None):
    """Check names, sizes and checksums before exposing a downloaded result."""
    destination = Path(destination).absolute()
    if destination.exists():
        raise ValueError('下载目标目录已存在')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.remote-', dir=destination.parent) as staging:
        extracted = Path(staging) / 'run'
        extracted.mkdir()
        names = set()
        portability = set()
        size = 0
        try:
            with zipfile.ZipFile(archive) as package:
                members = package.infolist()
                if not 1 <= len(members) <= MAX_RESULT_FILES:
                    raise ValueError('服务器结果文件数量超出范围')
                for member in members:
                    if cancelled and cancelled():
                        raise InterruptedError('下载已取消')
                    name = member.filename
                    # The renderer reads ordinary files in one run folder. Reject
                    # case aliases and device names before touching Windows paths.
                    reserved = ('CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
                                *(f'LPT{i}' for i in range(1, 10)))
                    if (not ARTIFACT_NAME.fullmatch(name) or name.endswith('.') or name.casefold() in portability
                            or name.split('.')[0].upper() in reserved or stat.S_ISLNK(member.external_attr >> 16)):
                        raise ValueError('服务器结果包含无效或重复文件名')
                    portability.add(name.casefold())
                    names.add(name)
                    size += member.file_size
                    if size > MAX_RESULT_BYTES:
                        raise ValueError('服务器结果超过下载容量')
                    with package.open(member) as source, (extracted / name).open('xb') as output:
                        while block := source.read(1024 * 1024):
                            if cancelled and cancelled():
                                raise InterruptedError('下载已取消')
                            output.write(block)
        except zipfile.BadZipFile as exc:
            raise ValueError('服务器结果压缩包已损坏') from exc
        report = read_project_run(extracted / REPORT_NAME, verify=True, cancelled=cancelled)
        if (report['status'] != 'succeeded' or report['missing_artifacts'] or report['changed_artifacts']
                or report['missing_outputs'] or report['changed_outputs']):
            raise ValueError('服务器结果校验失败或文件不完整')
        expected = {REPORT_NAME, *(item.name for item in report['artifacts'].values()),
                    *report.get('output_sha256', {}).keys()}
        if names != expected:
            raise ValueError('下载包含报告未声明的文件')
        if destination.exists():
            raise ValueError('下载目标目录已存在')
        extracted.rename(destination)
    return destination / REPORT_NAME


class RemoteClient:
    def __init__(self, profile, runtime_source):
        self.profile = profile
        self.runtime_source = Path(runtime_source)
        self.runtime_digest = None

    def ssh_command(self, arguments):
        executable = shutil.which('ssh')
        if executable is None:
            raise ValueError('需要系统 OpenSSH 客户端（ssh）')
        return [executable, '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                '-o', 'ConnectTimeout=8', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
                self.profile.host, shlex.join(arguments)]

    def _request(self, arguments, data=None, *, cancelled=None, target=None, timeout=30):
        if cancelled and cancelled():
            raise InterruptedError('远程操作已取消')
        payload = None if data is None else json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
        if payload is not None and len(payload) > MAX_PROTOCOL_BYTES:
            raise ValueError('科研工程清单超过远程提交容量')
        deadline = time.monotonic() + timeout
        with ExitStack() as resources:
            stdin = subprocess.DEVNULL
            if payload is not None:
                stdin = resources.enter_context(tempfile.TemporaryFile())
                stdin.write(payload)
                stdin.seek(0)
            process = resources.enter_context(subprocess.Popen(
                self.ssh_command(arguments), stdin=stdin,
                stdout=target if target is not None else subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)))
            try:
                while True:
                    try:
                        stdout, stderr = process.communicate(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        if cancelled and cancelled():
                            raise InterruptedError('远程操作已取消')
                        if time.monotonic() >= deadline:
                            raise TimeoutError('SSH 操作超时，请检查连接并按任务 ID 恢复')
                        if target is not None and Path(target.name).stat().st_size > MAX_RESULT_BYTES:
                            raise ValueError('远程结果超过下载容量')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
            if process.returncode:
                detail = stderr.decode('utf-8', errors='replace').strip()[-1600:]
                raise ValueError(detail or f'SSH 退出码 {process.returncode}')
            return stdout

    def connect(self, *, cancelled=None):
        files = {item.name: item.read_bytes().decode('utf-8') for item in self.runtime_source.glob('*.py')
                 if SOURCE_NAME.fullmatch(item.name)}
        if not {'__init__.py', 'distributed_image.py', 'remote_runner.py'} <= files.keys():
            raise ValueError('当前安装缺少服务器工作进程源码')
        inventory = {name: hashlib.sha256(value.encode('utf-8')).hexdigest() for name, value in files.items()}
        identity = json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()
        digest = hashlib.sha256(identity).hexdigest()
        receipt = _checked_json(self._request([self.profile.python, '-I', '-B', '-c', _INSTALL, self.profile.workspace],
                                             {'digest': digest, 'files': files}, cancelled=cancelled))
        if receipt != {'protocol': PROTOCOL, 'digest': digest}:
            raise ValueError('服务器工作进程版本或源码身份不一致')
        self.runtime_digest = digest
        return self.probe(cancelled=cancelled)

    def _command(self, operation, job=None):
        if self.runtime_digest is None:
            raise ValueError('请先连接服务器')
        workspace = self.profile.workspace
        return ['env', 'PYTHONPATH=' + workspace + '/runtime/' + self.runtime_digest,
                'PYTHONDONTWRITEBYTECODE=1', self.profile.python, '-s', '-m', 'studio_core.remote_runner',
                '--root', workspace + '/jobs', operation] + ([] if job is None else ['--job', check_job_id(job)])

    def probe(self, *, cancelled=None):
        info = _checked_json(self._request(self._command('probe'), cancelled=cancelled))
        gpus = info.get('gpus')
        if info.get('protocol') != PROTOCOL or not isinstance(gpus, list) or type(info.get('ddp')) is not bool:
            raise ValueError('服务器作业协议版本不兼容')
        seen = set()
        for gpu in gpus:
            if (not isinstance(gpu, dict) or not isinstance(gpu.get('uuid'), str) or not GPU.fullmatch(gpu['uuid'])
                    or gpu['uuid'].lower() in seen or not isinstance(gpu.get('name'), str)
                    or any(type(gpu.get(key)) is not int or gpu[key] < 0 for key in
                           ('index', 'memory_used_mib', 'memory_total_mib', 'utilization'))):
                raise ValueError('服务器显卡清单格式无效')
            seen.add(gpu['uuid'].lower())
        return info

    def submit(self, project, data_path, gpu_uuids, per_device, accumulation, epochs, *,
               resume_from=None, request_id=None, cancelled=None):
        batch_plan(gpu_uuids, per_device, accumulation)
        if type(epochs) is not int or not 1 <= epochs <= 1000:
            raise ValueError('Epochs 须为 1 至 1000 的整数')
        data_path = _remote_path(data_path)
        request_id = uuid4().hex if request_id is None else check_job_id(request_id)
        if resume_from is not None:
            check_job_id(resume_from)
        request = {'files': _project_files(project), 'data_path': data_path, 'gpu_uuids': list(gpu_uuids),
                   'per_device_batch_size': per_device, 'accumulation_steps': accumulation, 'epochs': epochs,
                   'resume_from': resume_from, 'request_id': request_id}
        receipt = _checked_json(self._request(self._command('submit'), request, cancelled=cancelled, timeout=120))
        if receipt.get('job_id') != request_id:
            raise ValueError('服务器返回了意外的任务 ID')
        return receipt

    def status(self, job, *, cancelled=None):
        return _checked_json(self._request(self._command('status', job), cancelled=cancelled))

    def cancel(self, job, *, cancelled=None):
        return _checked_json(self._request(self._command('cancel', job), cancelled=cancelled))

    def fetch(self, job, destination, *, cancelled=None):
        destination = Path(destination).absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.download-', dir=destination.parent) as temporary:
            archive = Path(temporary) / 'result.zip'
            with archive.open('wb') as stream:
                self._request(self._command('fetch', job), cancelled=cancelled, target=stream, timeout=1800)
            if archive.stat().st_size > MAX_RESULT_BYTES:
                raise ValueError('远程结果超过下载容量')
            return unpack_result(archive, destination, cancelled=cancelled)
