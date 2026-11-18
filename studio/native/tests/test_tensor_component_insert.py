from copy import deepcopy
from tempfile import TemporaryDirectory
import unittest

from native.chip_library import read_bundle
from native.document import Document
from native.local_components import LocalComponentIndex
from studio_core.component_types import ComponentBundle


TENSOR_REQUIREMENTS = {'schema_version': 2, 'capability_profile': 'tensor-v1',
                       'operator_set': 'tensor-ops-v1', 'resource_profile': 'hardware-v5',
                       'resource_batch_bound': 2}


def chip(tensor=True):
    spec = {'id': 'x', 'shape': [None, 4], 'dtype': 'float32'}
    if tensor:
        spec.update(axes=['batch', 'feature'], batch_axis=0)
    definition = {'id': 'passthrough', 'name': 'Passthrough', 'version': '1.0.0',
                  'public_parameters': [], 'bindings': {},
                  'graph': {'inputs': [spec], 'instances': [], 'calls': [],
                            'outputs': [{'id': 'y', 'source': 'input:x'}]}}
    return ComponentBundle({'kind': 'chip', 'definition_id': definition['id'], 'params': {}}, [definition],
                           deepcopy(TENSOR_REQUIREMENTS) if tensor else
                           {'operator_set': 'vision-ops-v4', 'resource_profile': None})


class TensorComponentInsertTests(unittest.TestCase):
    def test_tensor_upgrade_and_insertion_undo_together(self):
        doc = Document()
        before = deepcopy(doc.graph)
        bundle = chip()
        with self.assertRaisesRegex(ValueError, 'schema|tensor'):
            doc.add_component(bundle)
        self.assertEqual(doc.graph, before)
        self.assertFalse(doc.can_undo)
        ref = doc.add_component(bundle, apply_policy=True)
        self.assertEqual(doc.graph['schema_version'], 2)
        self.assertEqual(doc.graph['capability_profile'], 'tensor-v1')
        self.assertEqual(doc.graph['resource_profile'], 'hardware-v5')
        self.assertEqual(doc.graph['inputs'][0]['batch_axis'], 0)
        self.assertEqual(doc.graph['revision'], before['revision'] + 1)
        self.assertTrue(doc.undo())
        self.assertEqual(doc.graph, {**before, 'revision': doc.graph['revision']})
        self.assertFalse(doc.can_undo)
        self.assertTrue(doc.redo())
        doc.enter(ref)
        self.assertEqual(doc.scope['inputs'][0]['axes'], ['batch', 'feature'])

    def test_existing_tensor_project_accepts_legacy_file_without_rewriting_it(self):
        with TemporaryDirectory() as directory:
            old_doc = Document()
            old_doc.add_component(chip(tensor=False))
            old_path = old_doc.save_chip('passthrough', '1.0.0', directory)
            original = old_path.read_bytes()
            self.assertEqual(read_bundle(old_path)['chip_format_version'], 1)
            doc = Document()
            doc.add_component(chip(), apply_policy=True)
            index = LocalComponentIndex(directory)
            local = index.refresh()[0]
            doc.add_component(index.build(local.id, {}))
            self.assertEqual(doc.selected_instance['definition_id'], 'passthrough')
            self.assertEqual(old_path.read_bytes(), original)
            new_path = doc.save_chip('passthrough', '2.0.0', directory)
            self.assertEqual(read_bundle(new_path)['requirements'], TENSOR_REQUIREMENTS)
            self.assertEqual(read_bundle(new_path)['chip_format_version'], 3)
            self.assertEqual(len(index.refresh()), 2)
            self.assertEqual(index.diagnostics, [])

    def test_broken_tensor_bundle_preserves_graph_navigation_and_history(self):
        doc = Document()
        doc.enter('encode_a')
        before = deepcopy((doc.graph, doc.trail, doc.selected))
        broken = chip()
        broken.definitions[0]['graph']['inputs'][0].pop('axes')
        with self.assertRaisesRegex(ValueError, 'axes'):
            doc.add_component(broken, apply_policy=True)
        self.assertEqual((doc.graph, doc.trail, doc.selected), before)
        self.assertFalse(doc.can_undo)
