"""User journey: library drop, policy, wiring, chip edit and runnable Python."""
from copy import deepcopy
import ast
import tempfile
import unittest
from unittest.mock import patch

import dearpygui.dearpygui as dpg
import torch

from native.document import Document
from native.paths import resource_root
from native.ui import NativeWorkbench
from studio_core.experiment_spec import model_digest
from studio_core.python_io import export_python, import_python
from studio_core.runtime import build_model


class ComponentWorkbenchIntegrationTests(unittest.TestCase):
    def test_drop_wire_edit_and_python_use_one_real_component(self):
        graph = {'schema_version': 1, 'id': 'component_workbench', 'revision': 1,
                 'inputs': [{'id': key, 'shape': [None, 4], 'dtype': 'float32'} for key in ('a', 'b')],
                 'instances': [], 'calls': [], 'definitions': [],
                 'outputs': [{'id': 'features'}]}
        with tempfile.TemporaryDirectory() as directory:
            ui = NativeWorkbench(Document(graph), None, directory, resource_root())
            try:
                ui.build()
                for _ in range(4):
                    ui.tick(); dpg.render_dearpygui_frame()
                root = ui.canvas_host.get('root')
                node = root.ref_nodes['input:a']
                point = [value + delta for value, delta in zip(dpg.get_item_rect_min(node), (180, 20))]
                payload = {'version': 1, 'component_id': 'fusion.weighted_sum',
                           'config': {'features': 4, 'alpha': 0.25}}
                drop = dpg.get_item_configuration(root.container_tag)['drop_callback']
                before = deepcopy(ui.document.graph)
                with patch('native.catalog_panel.dpg.get_mouse_pos', return_value=point):
                    drop(root.container_tag, payload, ui.canvas_host.token('root'))
                self.assertEqual(ui.document.graph, before)
                self.assertTrue(dpg.is_item_shown('native_catalog_policy'))
                dpg.get_item_callback('native_catalog_policy_apply')(
                    'native_catalog_policy_apply', None, dpg.get_item_user_data('native_catalog_policy_apply'))
                self.assertEqual(ui.document.graph['resource_profile'], 'vision-v4')
                self.assertEqual(ui.document.graph['resource_batch_bound'], 2)
                call_id = ui.document.selected_call['id']

                def connect(source, target):
                    view = ui.canvas_host.get('root')
                    a = next(item for item, ref in view.source_attributes.items() if ref == source)
                    b = next(item for item, ref in view.target_attributes.items() if ref == target)
                    dpg.get_item_callback(view.editor_tag)(view.editor_tag, [a, b], ui.canvas_host.token('root'))

                connect('input:a', ('call', call_id, 'a'))
                connect('input:b', ('call', call_id, 'b'))
                connect('call:' + call_id + ':features', ('output', 'features', 'value'))
                self.assertTrue(ui.document.validation()['valid'], ui.document.validation())
                ui.canvas_host.apply_selection('root', ['call:' + call_id], intentional=True)
                ui._render_inspector()
                widget = ui.param_widgets['alpha']
                dpg.get_item_callback(widget)(widget, '1.25', dpg.get_item_user_data(widget))
                self.assertEqual(ui.document.selected_instance['params']['alpha'], 1.25)
                values = {'a': torch.tensor([[1., 2., 3., 4.]]), 'b': torch.tensor([[5., 6., 7., 8.]])}
                expected = values['a'] + 1.25 * (values['b'] - values['a'])
                torch.testing.assert_close(build_model(ui.document.graph)(**values)['features'], expected)
                tree = ast.parse(export_python(ui.document.graph))
                tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == 'STUDIO_VISUAL_METADATA' for t in n.targets))]
                source = ast.unparse(tree)
                restored = import_python(source)
                torch.testing.assert_close(build_model(restored)(**values)['features'], expected)
                namespace = {}; exec(source, namespace)
                torch.testing.assert_close(namespace['Model']()(**values)['features'], expected)
                identity = model_digest(ui.document.graph)
                key = ui.workbench_panel.open_call([call_id])
                self.assertIn('root', ui.canvas_host.views)
                for mode in ('columns', 'rows', 'single'):
                    ui.workbench_panel.set_layout(mode)
                ui.workbench_panel.close_view(key)
                self.assertEqual(model_digest(ui.document.graph), identity)
                self.assertEqual(ui.workbench.focused_view_id, 'root')
            finally:
                ui.shutdown()


if __name__ == '__main__':
    unittest.main()
