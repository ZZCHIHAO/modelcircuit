"""Independent review counterexamples delivered through registered controls."""
from copy import deepcopy
import tempfile
import unittest

import dearpygui.dearpygui as dpg
from native.document import Document
from native.paths import resource_root
from native.ui import NativeWorkbench


class B3ReviewFixTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.ui = NativeWorkbench(Document(), None, self.directory.name, resource_root())
        self.ui.build()

    def tearDown(self):
        self.ui.shutdown(); self.directory.cleanup()

    def saved(self, tag):
        callback, data = dpg.get_item_callback(tag), dpg.get_item_user_data(tag)
        return lambda: callback(tag, None, data)

    def frames(self):
        for _ in range(4):
            self.ui.tick(); dpg.render_dearpygui_frame()

    def quick_relu(self):
        children = dpg.get_item_children('native_library')
        def walk(item):
            for group in dpg.get_item_children(item).values():
                for child in group:
                    if dpg.get_item_configuration(child).get('label') == 'ReLU' and dpg.get_item_parent(child) != dpg.get_alias_id('native_palette'):
                        # The quick controls precede the results scroller.
                        if dpg.get_item_info(child)['type'].endswith('mvButton'):
                            yield child
                    yield from walk(child)
        return next(walk('native_library'))

    def test_quick_toolbar_rejects_events_from_other_focus_context_and_document(self):
        stale = self.saved(self.quick_relu())
        self.ui.workbench_panel.open_call(['encode_a'])
        before = deepcopy(self.ui.document.graph); stale()
        self.assertEqual(before, self.ui.document.graph)
        stale = self.saved(self.quick_relu())
        self.ui.workbench_panel.open_call(['encode_b'])
        before = deepcopy(self.ui.document.graph); stale()
        self.assertEqual(before, self.ui.document.graph)
        stale = self.saved(self.quick_relu())
        self.ui.document.replace_graph(Document().graph); self.ui._reset_canvas_document()
        self.ui.workbench_panel.open_call(['encode_a'])
        before = deepcopy(self.ui.document.graph); stale()
        self.assertEqual(before, self.ui.document.graph)
        before_count = len(self.ui.document.scope['calls'])
        self.saved(self.quick_relu())()
        self.assertEqual(before_count + 1, len(self.ui.document.scope['calls']))

    def test_config_apply_and_cancel_are_bound_to_distinct_request_not_latest_dialog(self):
        panel = self.ui.catalog_panel
        panel.open_config('primitive.Linear')
        stale_apply = self.saved('native_catalog_apply')
        cancel = next(c for group in dpg.get_item_children('native_catalog_config').values() for c in group if dpg.get_item_info(c)['type'].endswith('mvGroup'))
        cancel_button = next(c for group in dpg.get_item_children(cancel).values() for c in group if dpg.get_item_configuration(c).get('label') == '取消')
        stale_cancel = self.saved(cancel_button)
        panel.cancel_config(); panel.open_config('primitive.ReLU')
        before = deepcopy(self.ui.document.graph); stale_apply(); stale_cancel()
        self.assertEqual(before, self.ui.document.graph)
        self.assertTrue(dpg.is_item_shown('native_catalog_config'))
        stale_apply = self.saved('native_catalog_apply')
        self.ui.document.replace_graph(Document().graph); self.ui._reset_canvas_document()
        self.ui.workbench_panel.open_call(['encode_a']); panel.open_config('primitive.Linear')
        before = deepcopy(self.ui.document.graph); stale_apply()
        self.assertEqual(before, self.ui.document.graph)
        self.saved('native_catalog_apply')()
        self.assertEqual(self.ui.document.selected_instance['kind'], 'Linear')

    def test_policy_confirmation_and_cancel_cannot_consume_a_later_policy_request(self):
        panel = self.ui.catalog_panel
        def request(alpha):
            panel.place({'version': 1, 'component_id': 'fusion.weighted_sum', 'config': {'alpha': alpha}}, self.ui.canvas_host.token('root'))
        request(.25)
        stale_apply, stale_cancel = self.saved('native_catalog_policy_apply'), self.saved('native_catalog_policy_cancel')
        panel.cancel_policy(); request(1.25)
        before = deepcopy(self.ui.document.graph); stale_apply(); stale_cancel()
        self.assertEqual(before, self.ui.document.graph)
        self.assertTrue(dpg.is_item_shown('native_catalog_policy'))
        stale_apply = self.saved('native_catalog_policy_apply')
        self.ui.document.replace_graph(Document().graph); self.ui._reset_canvas_document(); request(-.5)
        before = deepcopy(self.ui.document.graph); stale_apply()
        self.assertEqual(before, self.ui.document.graph)
        current_apply = self.saved('native_catalog_policy_apply')
        current_apply(); after = deepcopy(self.ui.document.graph); current_apply()
        self.assertEqual(after, self.ui.document.graph)
        self.assertEqual(self.ui.document.selected_instance['params']['alpha'], -.5)

    def test_close_and_move_cannot_act_on_another_new_active_view(self):
        graph = deepcopy(self.ui.document.graph)
        other = deepcopy(graph['definitions'][0]); other['id'] = 'review_other_chip'; graph['definitions'].append(other)
        self.ui.document.replace_graph(graph); self.ui._reset_canvas_document()
        panel = self.ui.workbench_panel
        first = panel.open_call(['encode_a'])
        old_close, old_move = self.saved('native_view_close_primary'), self.saved('native_view_move_primary')
        second = panel.open_definition('review_other_chip')
        old_close(); old_move()
        self.assertIn(second, self.ui.workbench.views)
        self.assertEqual(self.ui.workbench.views[second].pane, 'primary')
        self.assertEqual(self.ui.workbench.layout, 'single')
        panel.activate_view(first); stale = self.saved('native_view_close_primary')
        self.ui.document.replace_graph(graph); self.ui._reset_canvas_document()
        panel.open_call(['encode_a']); stale()
        self.assertIn(first, self.ui.workbench.views)

    def test_fresh_close_move_rebind_after_same_view_parameter_redraw(self):
        key = self.ui.workbench_panel.open_call(['encode_a'])
        self.ui.canvas_host.apply_selection(key, ['call:project'], intentional=True)
        self.ui._focus_document(key); self.ui._render_inspector()
        old_close, old_move = self.saved('native_view_close_primary'), self.saved('native_view_move_primary')
        def edit(value):
            widget = self.ui.param_widgets['out_features']
            dpg.get_item_callback(widget)(widget, value, dpg.get_item_user_data(widget))
        edit('9')
        before = deepcopy(self.ui.workbench.to_payload())
        old_close(); old_move()
        self.assertEqual(before, self.ui.workbench.to_payload())
        self.saved('native_view_move_primary')()
        self.assertEqual(self.ui.workbench.views[key].pane, 'secondary')
        edit('10')
        self.saved('native_view_close_secondary')()
        self.assertNotIn(key, self.ui.workbench.views)

    def test_offscreen_source_and_target_keep_hittable_visible_wire_fragments(self):
        for offscreen in ('input:a', 'call:encode_a'):
            with self.subTest(offscreen=offscreen):
                self.ui.document.replace_graph(Document().graph); self.ui._reset_canvas_document(); self.frames()
                view = self.ui.canvas_host.get('root')
                link = next(link for link, target in view.link_targets.items() if target == ('call', 'encode_a', 'x'))
                position = (-180, 100) if offscreen == 'input:a' else (1700, 83)
                dpg.set_item_pos(view.ref_nodes[offscreen], position)
                self.frames()
                self.assertFalse(dpg.get_item_state(view.ref_nodes[offscreen])['visible'])
                segments = view.edge_overlay.segments.get(link)
                self.assertTrue(segments)
                a, b = max(segments, key=lambda segment: abs(segment[0][0] - segment[1][0]) + abs(segment[0][1] - segment[1][1]))
                point = [(a[i] + b[i]) / 2 for i in (0, 1)]
                self.assertEqual(self.ui._select_orthogonal_wire(self.ui.canvas_host.token('root'), point), link)
                self.ui._delete_selected()
                call = next(c for c in self.ui.document.graph['calls'] if c['id'] == 'encode_a')
                self.assertNotIn('x', call['inputs'])


if __name__ == '__main__':
    unittest.main()
