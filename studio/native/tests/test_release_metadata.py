"""Build provenance must describe the selected source bytes and wheel variant."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from native.release_metadata import build_manifest, require_runtime_variant


class ReleaseMetadataTests(unittest.TestCase):
    def test_wrong_wheel_variant_is_rejected(self):
        for runtime, version in [('CPU', '2.6.0+cu124'), ('CUDA', '2.6.0+cpu')]:
            with self.subTest(runtime=runtime), self.assertRaises(ValueError):
                require_runtime_variant(runtime, version)

    def test_valid_variants(self):
        require_runtime_variant('CPU', '2.6.0+cpu')
        require_runtime_variant('CUDA', '2.6.0+cu124')
        require_runtime_variant('Unified', '2.6.0+cu124')
        with self.assertRaisesRegex(ValueError, 'Unified build'):
            require_runtime_variant('Unified', '2.6.0+cpu')
        with self.assertRaises(ValueError):
            require_runtime_variant('OTHER', '2.6.0+cpu')

    def test_snapshot_tracks_current_source_and_excludes_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(['git', 'init', '-q'], cwd=root, check=True)
            (root / '.gitignore').write_text('.venv/\n', encoding='utf-8')
            (root / 'model.py').write_text('value = 1\n', encoding='utf-8')
            (root / '.venv').mkdir()
            (root / '.venv/private.py').write_text('private\n', encoding='utf-8')
            with patch('native.release_metadata.metadata.version', return_value='2.6.0+cpu'):
                before = build_manifest(root, 'CPU')
                (root / 'model.py').write_text('value = 2\n', encoding='utf-8')
                after = build_manifest(root, 'CPU')
            self.assertIn('model.py', before['source_sha256'])
            self.assertNotIn('.venv/private.py', before['source_sha256'])
            self.assertNotEqual(before['snapshot_sha256'], after['snapshot_sha256'])
            self.assertNotEqual(before['source_sha256']['model.py'], after['source_sha256']['model.py'])
            json.dumps(after, allow_nan=False)
