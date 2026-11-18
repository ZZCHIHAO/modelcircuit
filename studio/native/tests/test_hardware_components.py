"""Native hardware policy supports large architectures without allocating them."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from native.chip_library import import_bundle, save_bundle
from native.document import Document
from native.local_components import LocalComponentIndex
from native.workspace_state import read_workspace_full, save_workspace, synthetic_config
from studio_core.component_catalog import build_component
from studio_core.module_templates import make_module
from studio_core.schema import prepare_graph


def hardware_document():
    graph = deepcopy(Document().graph)
    graph.update(resource_profile='hardware-v5', resource_batch_bound=2)
    return Document(graph)


class HardwareComponentsTests(unittest.TestCase):
    def test_existing_project_explicit_policy_upgrade_is_undoable(self):
        document = Document()
        before = deepcopy(document.graph)
        self.assertTrue(hasattr(document, 'set_execution_policy'), 'Existing projects need an explicit policy upgrade')
        document.set_execution_policy({'resource_profile': 'hardware-v5', 'resource_batch_bound': 64})
        self.assertEqual(document.graph['resource_profile'], 'hardware-v5')
        self.assertEqual(document.graph['resource_batch_bound'], 64)
        self.assertEqual(len(document._undo), 1)
        document.undo()
        expected = deepcopy(before)
        expected['revision'] = document.graph['revision']
        self.assertEqual(document.graph, expected)

    def test_large_primitive_architecture_places_without_fixed_parameter_limit(self):
        document = hardware_document()
        primitive = build_component('primitive.Linear', {'in_features': 4096, 'out_features': 8192})
        before = len(document.graph['calls'])
        document.add_component(primitive)
        self.assertEqual(len(document.graph['calls']), before + 1)
        self.assertEqual(document.selected_instance['params']['out_features'], 8192)
        self.assertEqual(document.graph['resource_profile'], 'hardware-v5')
        legacy = deepcopy(document.graph)
        legacy['resource_profile'] = 'vision-v4'
        old = Document(legacy)
        with self.assertRaisesRegex(ValueError, 'budget'):
            old.add_component(primitive)

    def test_v4_chip_in_hardware_parent_retains_hardware_policy_atomically(self):
        document = hardware_document()
        before = deepcopy(document.graph)
        bundle = make_module('fusion.weighted_sum', {'features': 4})
        self.assertEqual(bundle.requirements['resource_profile'], 'vision-v4')
        document.add_component(bundle)
        self.assertEqual(document.graph['resource_profile'], 'hardware-v5')
        self.assertEqual(document.graph['resource_batch_bound'], 2)
        self.assertEqual(len(document._undo), 1)
        document.undo()
        expected = deepcopy(before)
        expected['revision'] = document.graph['revision']
        self.assertEqual(document.graph, expected)
        document.redo()
        self.assertEqual(document.graph['resource_profile'], 'hardware-v5')

    def test_different_batch_contract_requires_explicit_adoption(self):
        document = hardware_document()
        bundle = make_module('fusion.weighted_sum', {'features': 4, 'resource_batch_bound': 3})
        before = deepcopy(document.graph)
        with self.assertRaises(ValueError):
            document.add_component(bundle)
        self.assertEqual(document.graph, before)

    def test_hardware_chip_package_and_workspace_are_pure_roundtrips(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            document = hardware_document()
            path = save_bundle(document.graph, 'encoder_chip', '1.4.0', directory / 'chips')
            definitions, identifier = import_bundle(path, [], resource_policy={
                'resource_profile': 'hardware-v5', 'resource_batch_bound': 2})
            self.assertEqual(identifier, 'encoder_chip')
            self.assertEqual(len(definitions), 1)
            index = LocalComponentIndex(directory / 'chips')
            index.refresh()
            self.assertEqual(index.diagnostics, [])
            bundle = index.build(index.list_components()[0].id, {})
            self.assertEqual(bundle.requirements['resource_profile'], 'hardware-v5')
            workspace = directory / 'workspace.json'
            save_workspace(workspace, document, synthetic_config())
            restored, _, _ = read_workspace_full(workspace)
            self.assertEqual(restored.graph, document.graph)
            self.assertEqual(restored.semantic_key(), document.semantic_key())

    def test_old_chip_import_into_hardware_parent_does_not_downgrade_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = make_module('fusion.weighted_sum', {'features': 4})
            path = save_bundle({'definitions': bundle.definitions, 'resource_profile': 'vision-v4',
                                'resource_batch_bound': 2}, bundle.entry['definition_id'], '1.0.0', temporary)
            definitions, identifier = import_bundle(path, [], resource_policy={
                'resource_profile': 'hardware-v5', 'resource_batch_bound': 2})
            self.assertEqual(identifier, 'WeightedSum')
            self.assertTrue(definitions)

    def test_legacy_v1_chip_can_be_added_without_downgrading_hardware_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            legacy = Document()
            path = legacy.save_chip('encoder_chip', '1.0.0', temporary)
            definitions, identifier = import_bundle(path, [], resource_policy={
                'resource_profile': 'hardware-v5', 'resource_batch_bound': 2})
            self.assertEqual(identifier, 'encoder_chip')
            self.assertEqual(len(definitions), 1)
            index = LocalComponentIndex(temporary)
            index.refresh()
            document = hardware_document()
            document.add_component(index.build(index.list_components()[0].id, {}))
            self.assertEqual(document.graph['resource_profile'], 'hardware-v5')
            self.assertEqual(document.graph['resource_batch_bound'], 2)


if __name__ == '__main__':
    unittest.main()
