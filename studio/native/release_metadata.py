"""Build-only provenance; importing this module never imports Torch or the GUI."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

PRODUCT_NAME = 'ModelCircuit Studio'
PRODUCT_STAGE = 'S4'


def require_runtime_variant(runtime, torch_version):
    valid = (runtime == 'CPU' and torch_version.endswith('+cpu') or
             runtime in ('CUDA', 'Unified') and '+cu' in torch_version)
    if not valid:
        raise ValueError(f'{runtime} build requires the corresponding PyTorch wheel; found {torch_version}')


def build_manifest(repository, runtime, stage=PRODUCT_STAGE):
    repository = Path(repository).resolve()
    if stage not in ('S4', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'):
        raise ValueError('Unsupported desktop release stage')
    if (stage in ('S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13')) != (runtime == 'Unified'):
        raise ValueError('S6–S13 use the unified desktop build; S4 uses the historical CPU or CUDA variants')
    version_file = repository / 'VERSION'
    version = version_file.read_text(encoding='utf-8').strip() if version_file.is_file() else None
    torch_version = metadata.version('torch')
    require_runtime_variant(runtime, torch_version)
    names = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=repository
    ).decode('utf-8').split('\0')
    selected = {}
    for name in sorted(set(names) - {''}):
        path = repository / name
        if path.suffix.lower() in {'.py', '.ps1', '.cmd', '.spec', '.toml'} or name in {
            'studio/requirements.txt', 'studio/requirements-desktop.txt', 'studio/requirements-classical.txt', 'studio/requirements-research.txt', 'LICENSE', 'VERSION',
            'THIRD_PARTY_NOTICES.md', 'licenses/torchvision-BSD-3-Clause.txt',
            'ml_forge/assets/icon.ico', 'ml_forge/assets/icon.png',
        }:
            selected[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = json.dumps(selected, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True, capture_output=True)
    dirty = subprocess.check_output(['git', 'status', '--porcelain=v1', '-z'], cwd=repository)
    packages = ('torch', 'dearpygui', 'numpy', 'Pillow', 'sympy', 'networkx', 'filelock',
                'fsspec', 'Jinja2', 'MarkupSafe', 'mpmath', 'typing_extensions', 'packaging', 'pyinstaller', 'setuptools')
    if stage in ('S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'):
        packages += ('scikit-learn', 'scipy', 'joblib', 'threadpoolctl')
    if stage in ('S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'):
        packages += ('pydicom', 'nibabel', 'opencv-python-headless')
    return {
        'product': PRODUCT_NAME, 'stage': stage, 'runtime': runtime, 'version': version,
        'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(),
        'source_commit': commit.stdout.strip() if commit.returncode == 0 else None,
        'worktree_dirty': bool(dirty), 'source_sha256': selected,
        'snapshot_sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest(),
        'packages': {name: metadata.version(name) for name in packages},
        'scope': 'Local Windows build. Source-byte provenance, not a cross-machine or performance certification.',
    }


def main():
    parser = argparse.ArgumentParser(description='Create desktop build provenance')
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--runtime', choices=('CPU', 'CUDA', 'Unified'), required=True)
    parser.add_argument('--stage', choices=('S4', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13'), default=PRODUCT_STAGE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.repository, args.runtime, args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
