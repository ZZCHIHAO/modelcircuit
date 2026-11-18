import tempfile
import unittest
from pathlib import Path

from native.paths import create_runtime_paths, resolve_data_directory, resource_root


class RuntimePathsTests(unittest.TestCase):
    def test_user_data_does_not_depend_on_install_or_current_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / 'MLForgeStudio'
            self.assertEqual(resolve_data_directory(environ={'LOCALAPPDATA': directory}), expected)
            paths = create_runtime_paths(expected)
            self.assertTrue(paths['runs'].is_dir())
            self.assertTrue(paths['logs'].is_dir())
            self.assertEqual(paths['draft'], expected / 'draft.json')

    def test_explicit_data_directory_and_frozen_assets_are_distinct(self):
        with tempfile.TemporaryDirectory(prefix='ML Forge ') as directory:
            data = Path(directory) / '用户数据'
            bundle = Path(directory) / '只读资源'
            self.assertEqual(resolve_data_directory(data, environ={}), data.resolve())
            self.assertEqual(resource_root(bundle), bundle / 'assets')
            self.assertNotEqual(create_runtime_paths(data)['data'], resource_root(bundle))

    def test_source_assets_resolve_without_changing_working_directory(self):
        self.assertTrue((resource_root() / 'icon.ico').is_file())


if __name__ == '__main__':
    unittest.main()
