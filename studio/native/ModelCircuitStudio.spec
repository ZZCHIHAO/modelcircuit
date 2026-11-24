# -*- mode: python ; coding: utf-8 -*-
"""S4 uses independent output paths; the legacy S3 spec remains untouched."""
import json
import os
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

repository = Path(SPECPATH).resolve().parents[1]
studio = repository / 'studio'
assets = repository / 'ml_forge' / 'assets'
build_manifest = Path(os.environ['MODELCIRCUIT_RELEASE_MANIFEST'])
python_license = Path(sys.base_prefix) / 'LICENSE.txt'
for required in (build_manifest, python_license, repository / 'THIRD_PARTY_NOTICES.md',
                 repository / 'licenses' / 'torchvision-BSD-3-Clause.txt'):
    if not required.is_file():
        raise FileNotFoundError('S4 build resource missing: ' + str(required))
data = [
    (str(assets / 'icon.ico'), 'assets'), (str(assets / 'icon.png'), 'assets'),
    (str(build_manifest), 'assets'), (str(repository / 'LICENSE'), 'licenses/MLForge'),
    (str(repository / 'THIRD_PARTY_NOTICES.md'), 'licenses'),
    (str(repository / 'licenses' / 'torchvision-BSD-3-Clause.txt'), 'licenses'),
    (str(python_license), 'licenses/Python'),
]
stage = json.loads(build_manifest.read_text(encoding='utf-8')).get('stage')
if stage in ('S10', 'S11', 'S12', 'S13'):
    data += [(str(source), 'assets/project_runtime/studio_core') for source in sorted((studio / 'backend' / 'studio_core').glob('*.py'))]
    data += [(str(repository / name), 'assets/project_runtime') for name in ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'VERSION')]
modern = stage in ('S6','S7','S8','S9','S10','S11','S12', 'S13')
research_packages = ('pydicom','nibabel','opencv-python-headless') if stage in ('S7','S8','S9','S10','S11','S12', 'S13') else ()
classical_packages = ('scikit-learn', 'scipy', 'joblib', 'threadpoolctl') if modern else ()
for package in ('torch', 'dearpygui', 'numpy', 'Pillow', 'sympy', 'networkx', 'filelock',
                'fsspec', 'Jinja2', 'MarkupSafe', 'mpmath', 'typing_extensions', 'packaging',
                'setuptools', 'pyinstaller') + classical_packages + research_packages:
    # Full dist-info includes nested licenses, torch NOTICE and bootloader COPYING.
    data += copy_metadata(package)
for package in ('torch', 'dearpygui', 'numpy', 'PIL', 'sympy', 'networkx', 'filelock',
                'fsspec', 'jinja2', 'markupsafe', 'mpmath', 'packaging'):
    data += collect_data_files(package, includes=['**/LICENSE*', '**/NOTICE*', '**/COPYING*'])

analysis = Analysis(
    [str(studio / 'desktop_entry.py')],
    pathex=[str(studio), str(studio / 'backend')],
    binaries=[], datas=data,
    hiddenimports=['native.ui', 'native.s4_acceptance', 'native.s4_render_acceptance',
                   'studio_core.runtime', 'worker', 'torch', 'dearpygui.dearpygui'] +
                  (['sklearn.svm', 'sklearn.utils._cyutility', 'scipy.special._cdflib'] if modern else []) +
                  (['pydicom.pixels.decoders.native', 'cv2', 'nibabel'] if stage in ('S7','S8','S9','S10','S11','S12', 'S13') else []),
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=([] if modern else ['scipy', 'sklearn']) + ['pandas', 'matplotlib', 'IPython', 'pytest', 'tensorboard', 'tensorflow',
              'jax', 'torchvision', 'torchaudio', 'artifact_tool_v2', 'notebook', 'lxml',
              'torch.utils.tensorboard'],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name='ModelCircuitStudio',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False, icon=str(assets / 'icon.ico'))
collect = COLLECT(exe, analysis.binaries, analysis.datas, strip=False, upx=False, name='ModelCircuitStudio')
