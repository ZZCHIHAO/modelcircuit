"""Finite v2 architecture requirements never silently change project policy."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from native.chip_library import import_bundle, save_bundle, validate_bundle
from native.document import Document
from native.workspace_state import read_workspace, save_workspace, synthetic_config


def architecture_graph(*, vision=True, kind='Identity'):
    shape = [None, 3, 224, 224] if vision else [None, 4]
    policy = {'resource_profile': 'vision-v4', 'resource_batch_bound': 2} if vision else {}
    child = {'inputs': [{'id': 'x', 'shape': shape, 'dtype': 'float32'}],
             'instances': [{'id': 'op', 'kind': kind, 'params': {}}],
             'calls': [{'id': 'op', 'instance_id': 'op', 'inputs': {'x': 'input:x'}}],
             'outputs': [{'id': 'features', 'source': 'call:op'}]}
    return {'schema_version': 1, 'id': 'requirements', 'revision': 1, **policy,
            'inputs': deepcopy(child['inputs']),
            'instances': [{'id': 'encode', 'kind': 'chip', 'definition_id': 'encoder', 'params': {}}],
            'calls': [{'id': 'encode', 'instance_id': 'encode', 'inputs': {'x': 'input:x'}}],
            'outputs': [{'id': 'out', 'source': 'call:encode:features'}],
            'definitions': [{'id': 'encoder', 'version': '1.0.0', 'output_mode': 'named',
                             'public_parameters': [], 'bindings': {}, 'graph': child}]}


class ChipRequirementsTests(unittest.TestCase):
    def test_vision_bundle_saves_v2_and_validates_synthetic_root_under_same_policy(self):
        graph = architecture_graph()
        with tempfile.TemporaryDirectory() as directory:
            path = save_bundle(graph, 'encoder', '1.0.0', directory)
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['chip_format_version'], 2)
            self.assertEqual(payload['requirements'], {'operator_set': 'vision-ops-v4',
                             'resource_profile': 'vision-v4', 'resource_batch_bound': 2})
            self.assertEqual(validate_bundle(payload)[1], ['encoder'])
            self.assertEqual(save_bundle(graph, 'encoder', '1.0.0', directory), path)
            changed = deepcopy(graph); changed['resource_batch_bound'] = 3
            with self.assertRaises(ValueError):
                save_bundle(changed, 'encoder', '1.0.0', directory)

    def test_import_requires_explicit_matching_policy_and_keeps_document_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = architecture_graph()
            path = save_bundle(graph, 'encoder', '1.0.0', directory)
            document = Document()
            before = deepcopy(document.graph)
            with self.assertRaisesRegex(ValueError, 'policy|策略'):
                document.import_chip(path)
            self.assertEqual(document.graph, before)
            self.assertFalse(document.can_undo)
            with self.assertRaises(ValueError):
                import_bundle(path, [], resource_policy={'resource_profile': 'vision-v4',
                                                       'resource_batch_bound': 3})
            with self.assertRaises(ValueError):
                import_bundle(path, [])
            # Policy is applied by the caller explicitly before importing.
            document.graph.update(resource_profile='vision-v4', resource_batch_bound=2)
            identifier = document.import_chip(path)
            self.assertEqual(identifier, 'encoder')
            self.assertEqual(document.graph['resource_batch_bound'], 2)
            self.assertTrue(document.can_undo)
            document.undo()
            self.assertEqual(document.graph['definitions'], before['definitions'])
            self.assertEqual(document.graph['resource_batch_bound'], 2)

    def test_new_operators_use_v2_even_without_vision_resource_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_bundle(architecture_graph(vision=False), 'encoder', '1.0.0', directory)
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['chip_format_version'], 2)
            self.assertEqual(payload['requirements'], {'operator_set': 'vision-ops-v4',
                                                      'resource_profile': None})
            definitions, identifier = import_bundle(path, [], resource_policy={})
            self.assertEqual(identifier, 'encoder')
            self.assertEqual(len(definitions), 1)
            forged = deepcopy(payload); forged['chip_format_version'] = 1; forged.pop('requirements')
            with self.assertRaises(ValueError):
                validate_bundle(forged)

    def test_old_bundle_stays_v1_and_import_api_remains_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = architecture_graph(vision=False, kind='ReLU')
            path = save_bundle(graph, 'encoder', '1.0.0', directory)
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['chip_format_version'], 1)
            self.assertNotIn('requirements', payload)
            self.assertEqual(import_bundle(path, [])[1], 'encoder')

    def test_invalid_requirements_and_stored_policies_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_bundle(architecture_graph(), 'encoder', '1.0.0', directory)
            payload = json.loads(path.read_text(encoding='utf-8'))
            requirements = payload['requirements']
            for bad in ({**requirements, 'operator_set': 'untrusted'},
                        {**requirements, 'resource_batch_bound': True},
                        {**requirements, 'resource_batch_bound': 65},
                        {**requirements, 'resource_batch_bound': None},
                        {**requirements, 'extra': 'ignored'},
                        {'operator_set': 'vision-ops-v4', 'resource_profile': 'vision-v4'}):
                invalid = deepcopy(payload); invalid['requirements'] = bad
                with self.subTest(requirements=bad), self.assertRaises(ValueError):
                    validate_bundle(invalid)
            for bound in (True, 0, 65, 2.0, None):
                invalid = architecture_graph(); invalid['resource_batch_bound'] = bound
                with self.subTest(bound=bound), self.assertRaises(ValueError):
                    Document(invalid)
            invalid = architecture_graph(); invalid.pop('resource_batch_bound')
            with self.assertRaises(ValueError):
                Document(invalid)
            invalid = architecture_graph(vision=False); invalid['resource_batch_bound'] = 2
            with self.assertRaises(ValueError):
                Document(invalid)

    def test_workspace_v2_preserves_policy_and_failure_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'workspace.json'
            document = Document(architecture_graph())
            save_workspace(path, document, synthetic_config())
            restored, _ = read_workspace(path)
            self.assertEqual(restored.graph, document.graph)
            self.assertEqual(json.loads(path.read_text())['workspace_version'], 2)
            before = path.read_bytes()
            document.graph['resource_batch_bound'] = False
            with self.assertRaises(ValueError):
                save_workspace(path, document, synthetic_config())
            self.assertEqual(path.read_bytes(), before)

    def test_public_bindings_use_project_batch_policy_instead_of_legacy_64(self):
        graph = architecture_graph(kind='BatchNorm2d')
        graph['definitions'][0]['graph']['instances'][0]['params'] = {'num_features': 3}
        document = Document(graph)
        document.set_public_parameters('encoder',
                                       [{'id': 'epsilon', 'type': 'float', 'default': 0.01}],
                                       {'epsilon': ['op.eps']})
        self.assertTrue(document.validation()['valid'], document.validation())
        self.assertEqual(document.graph['definitions'][0]['bindings'], {'epsilon': ['op.eps']})
        before = deepcopy(document.graph)
        with self.assertRaises(ValueError):
            document.set_public_parameters('encoder',
                                           [{'id': 'epsilon', 'type': 'float', 'default': -1.0}],
                                           {'epsilon': ['op.eps']})
        self.assertEqual(document.graph, before)

    def test_nested_encapsulation_keeps_full_project_execution_policy(self):
        document = Document(architecture_graph())
        document.enter('encode')
        reference = document.encapsulate(['op'], 'Inner identity', 'inner', 'op')
        self.assertTrue(reference.startswith('call:'))
        self.assertEqual(document.graph['resource_batch_bound'], 2)
        self.assertEqual(document.scope_id, 'encoder')
        self.assertTrue(document.validation()['valid'], document.validation())


if __name__ == '__main__':
    unittest.main()
