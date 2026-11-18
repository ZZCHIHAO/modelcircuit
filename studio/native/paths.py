"""Read-only assets and user-owned state never share a directory."""
import os
import sys
from pathlib import Path


def resolve_data_directory(override=None, environ=None):
    if override is not None:
        return Path(override).expanduser().resolve()
    environment = os.environ if environ is None else environ
    parent = environment.get('LOCALAPPDATA')
    return Path(parent) / 'MLForgeStudio' if parent else Path.home() / 'AppData' / 'Local' / 'MLForgeStudio'


def resource_root(bundle_root=None):
    if bundle_root is not None:
        return Path(bundle_root) / 'assets'
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) / 'assets'
    return Path(__file__).resolve().parents[2] / 'ml_forge' / 'assets'


def create_runtime_paths(directory):
    data = Path(directory).expanduser().resolve()
    paths = {'data': data, 'runs': data / 'runs', 'logs': data / 'logs', 'draft': data / 'draft.json',
             'workspace': data / 'workspace.json'}
    for name in ('data', 'runs', 'logs'):
        paths[name].mkdir(parents=True, exist_ok=True)
    return paths
