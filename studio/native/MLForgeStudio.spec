# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata

repository = Path(SPECPATH).resolve().parents[1]
studio = repository / 'studio'
assets = repository / 'ml_forge' / 'assets'
data = [(str(assets / 'icon.ico'), 'assets'), (str(assets / 'icon.png'), 'assets'),
        (str(repository / 'LICENSE'), 'licenses/MLForge')]
for package in ('torch', 'dearpygui', 'numpy', 'sympy', 'networkx', 'filelock', 'fsspec', 'jinja2', 'mpmath'):
    data += copy_metadata(package)

analysis = Analysis(
    [str(studio / 'desktop_entry.py')],
    pathex=[str(studio), str(studio / 'backend')],
    binaries=[], datas=data,
    hiddenimports=['native.ui', 'studio_core.runtime', 'worker', 'torch', 'dearpygui.dearpygui'],
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=['pandas', 'scipy', 'matplotlib', 'IPython', 'pytest', 'tensorboard', 'tensorflow',
              'jax', 'torchvision', 'torchaudio', 'artifact_tool_v2', 'notebook', 'lxml',
              'torch.utils.tensorboard'],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name='MLForgeStudio',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False, icon=str(assets / 'icon.ico'))
collect = COLLECT(exe, analysis.binaries, analysis.datas, strip=False, upx=False, name='MLForgeStudio')
