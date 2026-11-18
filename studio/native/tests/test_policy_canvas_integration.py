"""A2/B2 seam: source policy, imported two-port chips and stale UI events."""
import ast
from copy import deepcopy
import tempfile
import unittest

import dearpygui.dearpygui as dpg
import torch

from native.document import Document
from native.paths import resource_root
from native.ui import NativeWorkbench
from studio_core.python_io import export_python, import_python
from studio_core.runtime import build_model


def graph():
    inputs = [{'id': key, 'shape': [None, 2], 'dtype': 'float32'} for key in ('a', 'b')]
    return {
        'schema_version': 1, 'id': 'policy_canvas', 'revision': 1,
        'resource_profile': 'vision-v4', 'resource_batch_bound': 2,
        'inputs': deepcopy(inputs),
        'instances': [{'id': 'shared', 'kind': 'chip', 'definition_id': 'fusion', 'params': {}}],
        'calls': [{'id': key, 'instance_id': 'shared',
                   'inputs': {'a': 'input:a', 'b': 'input:b'}} for key in ('left', 'right')],
        'outputs': [{'id': 'features', 'source': 'call:left:features'}],
        'definitions': [{
            'id': 'fusion', 'version': '1.0.0', 'output_mode': 'named',
            'public_parameters': [], 'bindings': {},
            'graph': {'inputs': deepcopy(inputs),
                      'instances': [{'id': 'product', 'kind': 'Multiply', 'params': {}},
                                    {'id': 'scale', 'kind': 'Scale', 'params': {'factor': 0.5}}],
                      'calls': [{'id': 'product', 'instance_id': 'product',
                                 'inputs': {'a': 'input:a', 'b': 'input:b'}},
                                {'id': 'scale', 'instance_id': 'scale',
                                 'inputs': {'x': 'call:product'}}],
                      'outputs': [{'id': 'features', 'source': 'call:scale'}]}}]}


def source_without_visual_metadata(value):
    tree = ast.parse(export_python(value))
    tree.body = [item for item in tree.body if not (
        isinstance(item, ast.Assign) and any(isinstance(target, ast.Name)
        and target.id == 'STUDIO_VISUAL_METADATA' for target in item.targets))]
    return ast.unparse(tree)


class PolicyCanvasIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.ui = NativeWorkbench(Document(import_python(source_without_visual_metadata(graph()))),
                                  None, self.directory.name, resource_root())
        self.ui.build()

    def tearDown(self):
        self.ui.shutdown()
        self.directory.cleanup()

    def open_fusion(self, call='left'):
        key = self.ui.workbench.open_definition('fusion', [call])
        self.ui.canvas_host.sync(self.ui.canvas_parent_by_pane)
        return key, self.ui.canvas_host.get(key)

    def test_imported_multiply_has_two_real_ports_and_context_owned_links(self):
        key, view = self.open_fusion()
        ports = {port: item for item, (kind, call, port) in view.target_attributes.items()
                 if kind == 'call' and call == 'product'}
        self.assertEqual(set(ports), {'a', 'b'})
        self.assertTrue(all(dpg.does_item_exist(item) for item in ports.values()))
        root_before = deepcopy(self.ui.document.graph['calls'])
        source = next(item for item, ref in view.source_attributes.items() if ref == 'input:a')
        callback = dpg.get_item_callback(view.editor_tag)
        token = self.ui.canvas_host.token(key)
        callback(view.editor_tag, [source, ports['b']], token)
        inner = self.ui.document.graph['definitions'][0]['graph']
        self.assertEqual(inner['calls'][0]['inputs']['b'], 'input:a')
        self.assertEqual(self.ui.document.graph['calls'], root_before)
        self.assertEqual(self.ui.document.graph['resource_batch_bound'], 2)
        self.assertTrue(self.ui.document.validation()['valid'])
        a, b = torch.tensor([[2., -3.]]), torch.tensor([[5., 7.]])
        torch.testing.assert_close(build_model(self.ui.document.graph)(a=a, b=b)['features'],
                                   0.5 * a * a)

    def test_python_import_callback_replaces_model_and_rejects_old_canvas_event(self):
        key, view = self.open_fusion()
        callback = dpg.get_item_callback(view.editor_tag)
        token = self.ui.canvas_host.token(key)
        old_ports = tuple(view.target_attributes)
        edited = graph()
        edited['resource_batch_bound'] = 3
        edited['definitions'][0]['graph']['instances'][1]['params']['factor'] = 2.0
        dpg.set_value('native_python_code', source_without_visual_metadata(edited))
        self.ui._import_python()
        self.assertEqual(dpg.get_value('native_python_error'), '')
        self.assertEqual(list(self.ui.workbench.views), ['root'])
        self.assertFalse(self.ui.canvas_host.accept(token))
        before = deepcopy(self.ui.document.graph)
        callback(view.editor_tag, list(old_ports[:2]), token)
        self.assertEqual(self.ui.document.graph, before)
        self.assertEqual(before['resource_profile'], 'vision-v4')
        self.assertEqual(before['resource_batch_bound'], 3)
        a, b = torch.ones(3, 2), torch.full((3, 2), 4.)
        torch.testing.assert_close(build_model(before)(a=a, b=b)['features'], 2.0 * a * b)


if __name__ == '__main__':
    unittest.main()
