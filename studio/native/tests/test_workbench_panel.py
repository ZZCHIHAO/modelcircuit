"""B3 product controls exercise real application callbacks and DPG geometry."""
from copy import deepcopy
import tempfile
import unittest

import dearpygui.dearpygui as dpg
from native.document import Document
from native.paths import resource_root
from native.ui import NativeWorkbench
from native.catalog_panel import screen_rect


class WorkbenchPanelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.ui = NativeWorkbench(Document(), None, self.directory.name, resource_root())
        self.ui.build()

    def tearDown(self):
        self.ui.shutdown()
        self.directory.cleanup()

    def invoke(self, tag, data=None):
        return dpg.get_item_callback(tag)(tag, data, dpg.get_item_user_data(tag))

    def test_tabs_repeat_open_close_root_and_navigation_keep_model_and_editors(self):
        panel = self.ui.workbench_panel
        before = deepcopy(self.ui.document.graph)
        root_editor = self.ui.canvas_host.get('root').editor_tag
        key = panel.open_call(['encode_a'])
        self.assertEqual(panel.open_call(['encode_b']), key)
        self.assertEqual(self.ui.workbench.views[key].call_path, ['encode_b'])
        self.assertTrue(dpg.does_item_exist('native_tab_' + key))
        self.ui._back()
        self.assertEqual(self.ui.workbench.focused_view_id, 'root')
        self.assertIn(key, self.ui.workbench.views)
        panel.activate_view(key)
        self.invoke('native_view_close_primary')
        self.assertNotIn(key, self.ui.workbench.views)
        panel.close_view('root')
        self.assertIn('root', self.ui.workbench.views)
        self.assertEqual(root_editor, self.ui.canvas_host.get('root').editor_tag)
        self.assertEqual(self.ui.document.graph, before)

    def test_registered_layout_move_and_empty_pane_actions_are_real(self):
        panel = self.ui.workbench_panel
        key = panel.open_call(['encode_a'])
        self.invoke('native_layout_columns')
        self.invoke('native_view_move_primary')
        self.assertEqual(self.ui.workbench.views[key].pane, 'secondary')
        self.assertEqual(self.ui.workbench.focused_view_id, key)
        self.invoke('native_layout_rows')
        self.assertEqual(self.ui.workbench.focused_view_id, key)
        panel.close_view(key)
        self.assertTrue(dpg.is_item_shown('native_empty_secondary'))
        self.invoke('native_empty_open_secondary')
        self.assertEqual(self.ui.workbench.focused_pane, 'secondary')
        self.assertIsNotNone(self.ui.workbench.active_by_pane['secondary'])

    def test_secondary_can_move_back_and_generated_named_outputs_fit_default_editor(self):
        graph = deepcopy(self.ui.document.graph)
        definition = graph['definitions'][0]
        definition['output_mode'] = 'named'
        definition['graph']['outputs'] = [{'id': 'first', 'source': 'call:activate'}, {'id': 'second', 'source': 'call:project'}]
        self.ui.document.replace_graph(graph); self.ui._reset_canvas_document()
        panel = self.ui.workbench_panel
        key = panel.open_call(['encode_a'])
        for _ in range(3):
            dpg.render_dearpygui_frame()
        view = self.ui.canvas_host.get(key)
        height = dpg.get_item_rect_size(view.editor_tag)[1]
        for name in ('first', 'second'):
            item = view.ref_nodes['output:' + name]
            self.assertLessEqual(dpg.get_item_pos(item)[1] + dpg.get_item_rect_size(item)[1], height)
        self.invoke('native_view_move_primary')
        self.invoke('native_view_move_secondary')
        self.assertEqual(self.ui.workbench.views[key].pane, 'primary')

    def test_generated_output_nodes_do_not_overlap_calls_at_default_size(self):
        for _ in range(3):
            dpg.render_dearpygui_frame()
        view = self.ui.canvas_host.get('root')
        for ref, item in view.ref_nodes.items():
            if not ref.startswith('output:'):
                continue
            (x, y), (w, h) = screen_rect(item)
            for other_ref, other in view.ref_nodes.items():
                if other_ref.startswith('call:'):
                    (ox, oy), (ow, oh) = screen_rect(other)
                    self.assertTrue(x + w <= ox or ox + ow <= x or y + h <= oy or oy + oh <= y,
                                    (ref, other_ref))

    def test_default_root_nodes_fit_without_altering_saved_positions(self):
        before = deepcopy(self.ui.document.graph)
        for _ in range(3):
            dpg.render_dearpygui_frame()
        view = self.ui.canvas_host.get('root')
        width, height = dpg.get_item_rect_size(view.editor_tag)
        for ref, item in view.ref_nodes.items():
            x, y = dpg.get_item_pos(item)
            w, h = dpg.get_item_rect_size(item)
            self.assertLessEqual(x + w, width, ref)
            self.assertLessEqual(y + h, height, ref)
        self.assertEqual(before, self.ui.document.graph)

    def test_focus_location_validates_whole_reference_before_opening(self):
        panel = self.ui.workbench_panel
        state = deepcopy(self.ui.workbench.to_payload())
        self.assertFalse(panel.focus_location({'definition_id': 'encoder_chip', 'call_path': ['missing'], 'node_ref': 'call:project', 'field': None}))
        self.assertEqual(state, self.ui.workbench.to_payload())
        self.assertTrue(panel.focus_location({'definition_id': 'encoder_chip', 'call_path': ['encode_a'], 'node_ref': 'call:project', 'field': 'out_features'}))
        key = self.ui.workbench.focused_view_id
        self.assertEqual(self.ui.workbench.views[key].selected_refs, ['call:project'])

    def test_default_and_minimum_keep_search_scrollers_and_pane_controls_reachable(self):
        panel = self.ui.workbench_panel
        panel.open_call(['encode_a'])
        for width, height in ((1380, 860), (1100, 760)):
            dpg.set_viewport_width(width); dpg.set_viewport_height(height)
            for mode in ('single', 'columns', 'rows'):
                panel.set_layout(mode)
                self.ui.last_size = None
                self.ui._resize()
                for _ in range(3):
                    self.ui._resize()
                    dpg.render_dearpygui_frame()
                for tag in ('native_search', 'native_tree_scroll', 'native_catalog_scroll', 'native_layout_rows', 'native_view_picker_primary', 'native_right', 'native_run'):
                    (x, y), (w, h) = screen_rect(tag)
                    self.assertGreater(w, 20, tag)
                    self.assertGreater(h, 10, tag)
                    self.assertLessEqual(x + w, dpg.get_viewport_client_width() + 2, tag)
                    self.assertLessEqual(y + h, dpg.get_viewport_client_height() + 2, tag)
                tree_y = screen_rect('native_tree_scroll')[0][1]
                tree_h = dpg.get_item_rect_size('native_tree_scroll')[1]
                self.assertLessEqual(tree_y + tree_h, dpg.get_item_rect_min('native_search')[1])


if __name__ == '__main__':
    unittest.main()
