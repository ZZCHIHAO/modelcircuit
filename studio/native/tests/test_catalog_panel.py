"""Catalog gestures use registered native callbacks, without OS input."""
from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch

import dearpygui.dearpygui as dpg
from native.document import Document
from native.paths import resource_root
from native.ui import NativeWorkbench


class CatalogPanelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.ui = NativeWorkbench(Document(), None, self.directory.name, resource_root())
        self.ui.build()

    def tearDown(self):
        self.ui.shutdown()
        self.directory.cleanup()

    def invoke(self, tag, data=None):
        return dpg.get_item_callback(tag)(tag, data, dpg.get_item_user_data(tag))

    def test_search_configuration_and_click_insertion_are_real(self):
        panel = self.ui.catalog_panel
        self.invoke('native_search', '全连接')
        self.assertIn('primitive.Linear', panel.visible_ids)
        self.assertNotIn('primitive.Conv2d', panel.visible_ids)
        self.invoke('native_catalog_config_primitive.Linear')
        self.assertEqual(set(panel.config_widgets), {'in_features', 'out_features'})
        dpg.set_value(panel.config_widgets['in_features'], '4')
        dpg.set_value(panel.config_widgets['out_features'], '12')
        self.invoke('native_catalog_apply')
        call = self.ui.document.selected_call
        self.assertIsNotNone(call)
        self.assertEqual(self.ui.document.selected_instance['params']['out_features'], 12)
        self.assertIn('call:' + call['id'], self.ui.canvas_host.get('root').ref_nodes)

    def test_policy_confirmation_cancel_and_stale_target_are_atomic(self):
        panel = self.ui.catalog_panel
        before = deepcopy(self.ui.document.graph)
        panel.place({'version': 1, 'component_id': 'fusion.weighted_sum', 'config': {}}, self.ui.canvas_host.token('root'))
        self.assertEqual(before, self.ui.document.graph)
        self.assertTrue(dpg.is_item_shown('native_catalog_policy'))
        self.invoke('native_catalog_policy_cancel')
        self.assertEqual(before, self.ui.document.graph)
        panel.place({'version': 1, 'component_id': 'fusion.weighted_sum', 'config': {}}, self.ui.canvas_host.token('root'))
        self.ui.canvas_host.refresh_scope(None); self.ui._render_graph()
        self.invoke('native_catalog_policy_apply')
        self.assertEqual(before, self.ui.document.graph)
        panel.place({'version': 1, 'component_id': 'fusion.weighted_sum', 'config': {}}, self.ui.canvas_host.token('root'))
        self.invoke('native_catalog_policy_apply')
        self.assertEqual(self.ui.document.graph['resource_profile'], 'vision-v4')
        self.assertEqual(self.ui.document.graph['resource_batch_bound'], 2)
        self.ui._undo()
        restored = deepcopy(self.ui.document.graph)
        restored['revision'] = before['revision']
        self.assertEqual(before, restored)

    def test_drop_uses_target_owner_and_real_node_geometry(self):
        panel = self.ui.catalog_panel
        root_calls = len(self.ui.document.graph['calls'])
        key = self.ui.workbench_panel.open_call(['encode_a'])
        self.ui.workbench_panel.set_layout('columns')
        self.ui.workbench.move_to_pane(key, 'secondary')
        self.ui.workbench_panel.activate_view('root')
        for _ in range(3):
            dpg.render_dearpygui_frame()
        view = self.ui.canvas_host.get(key)
        anchor = view.ref_nodes['call:project']
        x, y = dpg.get_item_rect_min(anchor)
        native_x, native_y = dpg.get_item_pos(anchor)
        point = [x + 33, y + 27]
        expected = self.ui._document_position((native_x + 33, native_y + 27))
        callback = dpg.get_item_configuration(view.container_tag)['drop_callback']
        payload = {'version': 1, 'component_id': 'Identity', 'config': {}}
        with patch('native.catalog_panel.dpg.get_mouse_pos', return_value=point):
            callback(view.container_tag, payload, self.ui.canvas_host.token(key))
        self.assertEqual(self.ui.workbench.focused_view_id, key)
        call = self.ui.document.selected_call
        self.assertAlmostEqual(call['position']['x'], expected[0], delta=2)
        self.assertAlmostEqual(call['position']['y'], expected[1], delta=2)
        self.assertEqual(len(self.ui.document.graph['calls']), root_calls)
        ref = 'call:' + call['id']
        for _ in range(3):
            dpg.render_dearpygui_frame()
        rendered = dpg.get_item_rect_min(self.ui.canvas_host.get(key).ref_nodes[ref])
        self.assertTrue(all(abs(rendered[i] - point[i]) <= 2 for i in (0, 1)))

    def test_invalid_payload_outside_point_and_closed_target_do_nothing(self):
        panel = self.ui.catalog_panel
        token = self.ui.canvas_host.token('root')
        before = deepcopy(self.ui.document.graph)
        for payload in ({'version': 1, 'component_id': 'Identity', 'config': {'x': float('nan')}},
                        {'version': 1, 'component_id': 'ResNet18', 'config': {}},
                        {'version': 1, 'component_id': 'Identity', 'config': {}, 'extra': 1}):
            self.assertFalse(panel.place(payload, token))
        self.assertFalse(panel.place({'version': 1, 'component_id': 'Identity', 'config': {}}, token, screen_point=(-100, -100)))
        self.assertEqual(before, self.ui.document.graph)

    def test_queued_palette_callback_and_closed_modal_target_cannot_edit_new_focus(self):
        button = 'native_catalog_add_primitive.ReLU'
        callback = dpg.get_item_callback(button)
        token = dpg.get_item_user_data(button)
        key = self.ui.workbench_panel.open_call(['encode_a'])
        before = deepcopy(self.ui.document.graph)
        callback(button, None, token)
        self.assertEqual(before, self.ui.document.graph)
        self.ui.catalog_panel.place({'version': 1, 'component_id': 'fusion.weighted_sum', 'config': {}}, self.ui.canvas_host.token(key))
        self.ui.workbench_panel.close_view(key)
        self.invoke('native_catalog_policy_apply')
        self.assertEqual(before, self.ui.document.graph)

    def test_project_drag_creates_independent_instance_and_replacement_closes_catalog_modal(self):
        panel = self.ui.catalog_panel
        before = len(self.ui.document.graph['instances'])
        payload = {'version': 1, 'project_definition_id': 'encoder_chip', 'config': {}}
        self.assertTrue(panel.place(payload, self.ui.canvas_host.token('root')))
        first = self.ui.document.selected_instance['id']
        self.assertTrue(panel.place(payload, self.ui.canvas_host.token('root')))
        self.assertNotEqual(first, self.ui.document.selected_instance['id'])
        self.assertEqual(before + 2, len(self.ui.document.graph['instances']))
        self.invoke('native_catalog_config_primitive.Linear')
        self.ui.document.replace_graph(Document().graph); self.ui._reset_canvas_document()
        self.assertFalse(dpg.is_item_shown('native_catalog_config'))


if __name__ == '__main__':
    unittest.main()
