"""Local model extensions are finite architecture packages, not Python loaders."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

from native.chip_library import save_bundle
from native.document import Document
from studio_core.component_types import ComponentBundle
from studio_core.module_templates import make_module


def index_type():
    try:
        return importlib.import_module('native.local_components').LocalComponentIndex
    except ModuleNotFoundError as error:
        raise AssertionError('Local architecture component index is missing') from error


def save_weighted(directory):
    bundle = make_module('fusion.weighted_sum', {'features': 4})
    graph = {'definitions': bundle.definitions, 'resource_profile': 'vision-v4', 'resource_batch_bound': 2}
    return save_bundle(graph, bundle.entry['definition_id'], '1.0.0', directory)


class LocalComponentsTests(unittest.TestCase):
    def test_saved_chip_is_searchable_buildable_fresh_and_keeps_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_weighted(directory)
            index = index_type()(directory)
            index.refresh()
            specs = index.list_components('weighted')
            self.assertEqual(len(specs), 1)
            spec = specs[0]
            self.assertTrue(spec.id.startswith('local:'))
            self.assertIn('1.0.0', spec.label)
            self.assertEqual(spec.input_ports, ('a', 'b'))
            self.assertEqual(spec.output_ports, ('features',))
            self.assertEqual(spec.config_defaults, {'alpha': .5})
            self.assertIn(path.name, spec.source_names)
            bundle = index.build(spec.id, {'alpha': .75})
            self.assertIsInstance(bundle, ComponentBundle)
            self.assertEqual(bundle.entry['params'], {'alpha': .75})
            self.assertEqual(bundle.requirements, {'operator_set': 'vision-ops-v4',
                                                  'resource_profile': 'vision-v4', 'resource_batch_bound': 2})
            bundle.definitions[0]['version'] = 'broken'
            spec.config_defaults['alpha'] = .2
            self.assertEqual(index.build(spec.id, {}).definitions[0]['version'], '1.0.0')
            self.assertEqual(index.list_components()[0].config_defaults, {'alpha': .5})
            for config in ({'alpha': True}, {'alpha': float('nan')}, {'unknown': 1}, []):
                with self.subTest(config=config), self.assertRaises(ValueError):
                    index.build(spec.id, config)

    def test_damage_does_not_hide_valid_file_or_execute_adjacent_python(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            save_weighted(directory)
            (directory / 'bad.json').write_text('{oops', encoding='utf-8')
            (directory / 'not_a_plugin.py').write_text("raise RuntimeError('must not execute')", encoding='utf-8')
            index = index_type()(directory)
            index.refresh()
            self.assertEqual(len(index.list_components()), 1)
            self.assertEqual(len(index.diagnostics), 1)
            self.assertIn('bad.json', index.diagnostics[0])
            self.assertIn('JSON', index.diagnostics[0])

    def test_refresh_replaces_snapshot_content_ids_and_rejects_old_drag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_weighted(directory)
            index = index_type()(directory)
            index.refresh()
            old = index.list_components()[0].id
            payload = json.loads(path.read_text(encoding='utf-8'))
            payload['definitions'][0]['version'] = '1.0.1'
            path.write_text(json.dumps(payload), encoding='utf-8')
            self.assertEqual(index.list_components()[0].id, old)
            index.refresh()
            self.assertNotEqual(index.list_components()[0].id, old)
            with self.assertRaises(ValueError):
                index.build(old, {})
            path.unlink()
            index.refresh()
            self.assertEqual(index.list_components(), [])

    def test_duplicate_bytes_deduplicate_but_file_and_total_limits_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_weighted(directory)
            (Path(directory) / 'copy.json').write_bytes(path.read_bytes())
            index = index_type()(directory)
            index.refresh()
            self.assertEqual(len(index.list_components()), 1)
            self.assertEqual(len(index.list_components()[0].source_names), 2)
            bounded = index_type()(directory, max_files=1)
            bounded.refresh()
            self.assertEqual(len(bounded.list_components()), 1)
            self.assertTrue(any('文件' in issue for issue in bounded.diagnostics))
            bytes_bound = index_type()(directory, max_total_bytes=1)
            bytes_bound.refresh()
            self.assertEqual(bytes_bound.list_components(), [])
            self.assertTrue(any('读取预算' in issue for issue in bytes_bound.diagnostics))

    def test_legacy_saved_definition_builds_with_explicit_classification_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            document = Document()
            document.save_chip('encoder_chip', '1.0.0', directory)
            index = index_type()(directory)
            index.refresh()
            spec = index.list_components()[0]
            self.assertEqual(index.build(spec.id, {}).requirements,
                             {'operator_set': 'vision-ops-v4', 'resource_profile': 'classification-v2'})
            before = deepcopy(document.graph)
            document.add_component(index.build(spec.id, {}), apply_policy=True)
            self.assertEqual(document.graph['resource_profile'], 'classification-v2')
            self.assertEqual(len(document.graph['calls']), len(before['calls']) + 1)

    def test_oversize_unread_file_does_not_consume_read_budget_or_hide_valid_chip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_weighted(directory)
            with (Path(directory) / '000-oversized.json').open('wb') as stream:
                stream.seek(16 * 1024 * 1024 - 1)
                stream.write(b'!')
            index = index_type()(directory)
            index.refresh()
            self.assertEqual(len(index.list_components()), 1)
            self.assertIn(path.name, index.list_components()[0].source_names)
            self.assertTrue(any('000-oversized.json' in issue and '4 MiB' in issue for issue in index.diagnostics))


if __name__ == '__main__':
    unittest.main()
