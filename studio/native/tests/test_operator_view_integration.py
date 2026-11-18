"""A1/B1 integration: actual new operators, shared definitions and live scopes."""
from copy import deepcopy
import unittest

from native.document import Document
from native.view_document import view_scope
from native.workbench_state import WorkbenchState
from studio_core.runtime import build_model


def mixed_graph():
    shape = [None, 3, 2, 2]
    definition = {
        'id': 'encoder', 'version': '1.0.0', 'output_mode': 'named',
        'public_parameters': [], 'bindings': {},
        'graph': {
            'inputs': [{'id': 'x', 'shape': shape, 'dtype': 'float32'}],
            'instances': [{'id': 'norm', 'kind': 'BatchNorm2d',
                           'params': {'num_features': 3, 'eps': 1e-5}},
                          {'id': 'gate', 'kind': 'Sigmoid', 'params': {}}],
            'calls': [{'id': 'norm', 'instance_id': 'norm', 'inputs': {'x': 'input:x'}},
                      {'id': 'gate', 'instance_id': 'gate', 'inputs': {'x': 'call:norm'}}],
            'outputs': [{'id': 'features', 'source': 'call:gate'}],
        },
    }
    return {
        'schema_version': 1, 'resource_profile': 'classification-v2',
        'id': 'operator_view_integration', 'revision': 1,
        'inputs': [{'id': name, 'shape': list(shape), 'dtype': 'float32'} for name in ('x', 'y')],
        'instances': [{'id': name, 'kind': 'chip', 'definition_id': 'encoder', 'params': {}}
                      for name in ('left', 'right')] + [{'id': 'sum', 'kind': 'Add', 'params': {}}],
        'calls': [{'id': 'encode_a', 'instance_id': 'left', 'inputs': {'x': 'input:x'}},
                  {'id': 'encode_b', 'instance_id': 'right', 'inputs': {'x': 'input:y'}},
                  {'id': 'sum', 'instance_id': 'sum',
                   'inputs': {'a': 'call:encode_a:features', 'b': 'call:encode_b:features'}}],
        'outputs': [{'id': 'features', 'source': 'call:sum'}], 'definitions': [definition],
    }


class OperatorViewIntegrationTests(unittest.TestCase):
    def test_scoped_bn_edit_uses_one_definition_without_mutating_built_snapshot(self):
        doc, workbench = Document(mixed_graph()), WorkbenchState()
        before_key = doc.semantic_key()
        root_instances = deepcopy(doc.graph['instances'])
        snapshot = build_model(doc.graph, capture=False)
        view_id = workbench.open_definition('encoder', ['encode_a'])
        view = workbench.views[view_id]
        view.selected_refs = ['call:norm']
        with view_scope(doc, view):
            self.assertEqual(doc.probe_source(), 'call:encode_a/call:norm')
            doc.set_param('norm', 'eps', 0.03)
        self.assertIsNone(doc.scope_id)
        self.assertEqual(doc.graph['instances'], root_instances)
        self.assertTrue(doc.validation()['valid'])
        for branch in ('left', 'right'):
            self.assertEqual(snapshot.components[branch].components['norm'].eps, 1e-5)
        rebuilt = build_model(doc.graph, capture=False)
        for branch in ('left', 'right'):
            self.assertEqual(rebuilt.components[branch].components['norm'].eps, 0.03)
        self.assertIsNot(rebuilt.components['left'].components['norm'].weight,
                         rebuilt.components['right'].components['norm'].weight)
        self.assertIsNot(rebuilt.components['left'].components['norm'].running_mean,
                         rebuilt.components['right'].components['norm'].running_mean)
        epoch = view.context_epoch
        self.assertEqual(workbench.open_definition('encoder', ['encode_b']), view_id)
        self.assertGreater(view.context_epoch, epoch)
        with view_scope(doc, view):
            self.assertEqual(doc.probe_source(), 'call:encode_b/call:norm')
        self.assertTrue(doc.undo())
        workbench.reconcile(doc)
        self.assertEqual(doc.semantic_key(), before_key)
        self.assertTrue(doc.validation()['valid'])

    def test_bn_edit_after_failed_command_resolves_replaced_graph_by_id(self):
        doc, workbench = Document(mixed_graph()), WorkbenchState()
        view_id = workbench.open_definition('encoder', ['encode_a'])
        before_instances = deepcopy(doc.graph['instances'])
        with view_scope(doc, workbench.views[view_id]):
            with self.assertRaises(ValueError):
                doc.set_param('norm', 'eps', float('nan'))
            self.assertEqual(doc.scope_id, 'encoder')
            doc.set_param('norm', 'eps', 0.02)
        self.assertEqual(doc.graph['instances'], before_instances)
        self.assertEqual(doc.graph['definitions'][0]['graph']['instances'][0]['params']['eps'], 0.02)
        self.assertIsNone(doc.scope_id)
        self.assertTrue(doc.validation()['valid'])


if __name__ == '__main__':
    unittest.main()
