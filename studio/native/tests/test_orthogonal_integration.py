"""Real node-port geometry and wire callbacks, without desktop input."""
from copy import deepcopy
import tempfile
import unittest

import dearpygui.dearpygui as dpg
from native.document import Document
from native.paths import resource_root
from native.ui import NativeWorkbench


class OrthogonalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.ui = NativeWorkbench(Document(), None, self.directory.name, resource_root())
        self.ui.build()

    def tearDown(self):
        self.ui.shutdown(); self.directory.cleanup()

    def frames(self):
        for _ in range(4):
            self.ui.tick(); dpg.render_dearpygui_frame()

    def test_completed_wires_are_orthogonal_measured_at_actual_side_pins(self):
        self.frames()
        view = self.ui.canvas_host.get('root')
        self.assertTrue(view.edge_overlay.segments)
        for segments in view.edge_overlay.segments.values():
            for a, b in segments:
                self.assertTrue(a[0] == b[0] or a[1] == b[1])
        for link in view.link_targets:
            self.assertFalse(dpg.get_item_configuration(link)['show'], 'Hidden native curves must not receive invisible hits')
        for attribute, source in view.source_attributes.items():
            x, y = self.ui._socket_position(attribute, output=True)
            node = dpg.get_item_info(attribute)['parent']
            state = dpg.get_item_state(node)
            expected = state['rect_min'][0] + state['rect_size'][0] - (state['rect_size'][0] - (state['rect_max'][0] - state['rect_min'][0])) / 2
            self.assertAlmostEqual(x, expected)

    def test_visible_wire_hit_delete_uses_real_link_owner_and_clears_input(self):
        self.frames()
        view = self.ui.canvas_host.get('root')
        link, segments = next(iter(view.edge_overlay.segments.items()))
        a, b = max(segments, key=lambda segment: abs(segment[0][0] - segment[1][0]) + abs(segment[0][1] - segment[1][1]))
        point = [(a[i] + b[i]) / 2 for i in (0, 1)]
        token = self.ui.canvas_host.token('root')
        self.ui.canvas_host.apply_selection('root', ['call:classify'], intentional=True)
        self.ui._render_inspector()
        self.assertEqual(self.ui._select_orthogonal_wire(token, point), link)
        self.assertEqual(self.ui.workbench.views['root'].selected_refs, [])
        self.assertEqual(self.ui.param_widgets, {})
        kind, identifier, port = view.link_targets[link]
        self.ui._delete_selected()
        if kind == 'call':
            call = next(c for c in self.ui.document.graph['calls'] if c['id'] == identifier)
            self.assertNotIn(port, call['inputs'])
        self.assertFalse(self.ui._select_orthogonal_wire(token, point), 'Previous render token cannot delete a later wire')

    def test_actual_modal_hides_foreground_wires_and_blocks_wire_selection(self):
        self.frames()
        view = self.ui.canvas_host.get('root')
        token = self.ui.canvas_host.token('root')
        link, segments = next(iter(view.edge_overlay.segments.items()))
        a, b = segments[0]
        point = [(a[i] + b[i]) / 2 for i in (0, 1)]
        before = deepcopy(self.ui.document.graph)
        self.ui.catalog_panel.open_config('primitive.Linear')
        self.frames()
        self.assertEqual(view.edge_overlay.segments, {})
        self.assertFalse(self.ui._select_orthogonal_wire(token, point))
        self.assertEqual(before, self.ui.document.graph)
        self.ui.catalog_panel.cancel_config()
        self.frames()
        self.assertTrue(view.edge_overlay.segments)

    def test_hidden_closed_and_replaced_views_dispose_or_clear_their_wires(self):
        key = self.ui.workbench_panel.open_call(['encode_a'])
        self.frames()
        view = self.ui.canvas_host.get(key)
        self.assertTrue(view.edge_overlay.segments)
        self.ui.workbench_panel.activate_view('root')
        self.assertEqual(view.edge_overlay.segments, {})
        self.ui.workbench_panel.activate_view(key); self.frames()
        drawlist = view.edge_overlay.drawlist
        self.ui.workbench_panel.close_view(key)
        self.assertFalse(dpg.does_item_exist(drawlist))

    def test_empty_canvas_and_explicit_clear_remove_wire_selection(self):
        self.frames()
        view = self.ui.canvas_host.get('root')
        link, segments = next(iter(view.edge_overlay.segments.items()))
        a, b = max(segments, key=lambda segment: abs(segment[0][0] - segment[1][0]) + abs(segment[0][1] - segment[1][1]))
        point = [(a[i] + b[i]) / 2 for i in (0, 1)]
        token = self.ui.canvas_host.token('root')
        self.ui._select_orthogonal_wire(token, point)
        self.ui._clear_canvas_selection()
        self.assertEqual(view.selected_links, set())
        self.ui._select_orthogonal_wire(token, point)
        self.ui._select_orthogonal_wire(token, [-100, -100])
        self.assertEqual(view.selected_links, set())

    def test_selecting_node_after_wire_deletes_the_node_and_preserves_prior_wire(self):
        self.frames()
        view = self.ui.canvas_host.get('root')
        link, segments = next(iter(view.edge_overlay.segments.items()))
        a, b = max(segments, key=lambda segment: abs(segment[0][0] - segment[1][0]) + abs(segment[0][1] - segment[1][1]))
        self.ui._select_orthogonal_wire(self.ui.canvas_host.token('root'), [(a[i] + b[i]) / 2 for i in (0, 1)])
        self.ui.canvas_host.apply_selection('root', ['call:classify'], intentional=True)
        self.ui._delete_selected()
        self.assertNotIn('classify', [call['id'] for call in self.ui.document.graph['calls']])
        first = next(call for call in self.ui.document.graph['calls'] if call['id'] == 'encode_a')
        self.assertEqual(first['inputs'], {'x': 'input:a'})

    def test_multi_port_chip_has_only_declared_left_and_right_pins_and_real_probe_buttons(self):
        graph = deepcopy(self.ui.document.graph)
        definition = graph['definitions'][0]
        definition['output_mode'] = 'named'
        definition['graph']['inputs'].append({'id': 'extra', 'shape': [None, 4], 'dtype': 'float32'})
        definition['graph']['outputs'] = [{'id': 'features', 'source': 'call:activate'}, {'id': 'projected', 'source': 'call:project'}]
        self.ui.document.replace_graph(graph); self.ui._reset_canvas_document(); self.frames()
        view = self.ui.canvas_host.get('root')
        node = view.ref_nodes['call:encode_a']
        inputs = {value[2]: key for key, value in view.target_attributes.items() if value[:2] == ('call', 'encode_a')}
        outputs = {value.rsplit(':', 1)[1]: key for key, value in view.source_attributes.items() if value.startswith('call:encode_a:')}
        self.assertEqual(set(inputs), {'x', 'extra'})
        self.assertEqual(set(outputs), {'features', 'projected'})
        attrs = [a for groups in dpg.get_item_children(node).values() for a in groups]
        pin_attrs = [a for a in attrs if dpg.get_item_configuration(a).get('attribute_type') != dpg.mvNode_Attr_Static]
        self.assertEqual(set(pin_attrs), set(inputs.values()) | set(outputs.values()))
        for attribute in inputs.values():
            self.assertLess(self.ui._socket_position(attribute, output=False)[0], self.ui._socket_position(next(iter(outputs.values())), output=True)[0])
        output_attribute = outputs['projected']
        button = next(c for children in dpg.get_item_children(output_attribute).values() for c in children)
        dpg.get_item_callback(button)(button, None, dpg.get_item_user_data(button))
        self.assertEqual(self.ui._probe_source(), 'call:encode_a:projected')


if __name__ == '__main__':
    unittest.main()
