"""B2 uses real DPG items and registered callbacks, never desktop gestures."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dearpygui.dearpygui as dpg

from native.document import Document
from native.paths import resource_root
from native.ui import NativeWorkbench


def native_id(item):
    return dpg.get_alias_id(item) if isinstance(item, str) else item


class CanvasViewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.ui = NativeWorkbench(Document(), None, self.directory.name, resource_root())
        self.ui.build()

    def tearDown(self):
        self.ui.shutdown()
        self.directory.cleanup()

    def sync(self):
        self.ui.canvas_host.sync(self.ui.canvas_parent_by_pane)

    def chip(self, path='encode_a', pane='primary'):
        key = self.ui.workbench.open_definition('encoder_chip', [path], pane=pane)
        self.sync()
        return key

    def test_switching_keeps_native_editor_and_node_identity(self):
        root = self.ui.canvas_host.get('root')
        before = native_id(root.editor_tag), dict(root.ref_nodes)
        key = self.chip()
        self.ui.workbench.activate('root'); self.sync()
        self.assertEqual((native_id(root.editor_tag), root.ref_nodes), before)
        self.assertIn('call:encode_a', root.ref_nodes)
        self.assertFalse(dpg.is_item_shown(self.ui.canvas_host.get(key).editor_tag))
        self.assertEqual(native_id('native_editor'), before[0])

    def test_selection_is_logical_per_view_and_empty_getter_preserves_it(self):
        host = self.ui.canvas_host
        host.apply_selection('root', ['call:encode_a'], intentional=True)
        key = self.chip()
        host.apply_selection(key, ['call:project'], intentional=True)
        host.apply_selection(key, [], intentional=False)
        self.assertEqual(self.ui.workbench.views[key].selected_refs, ['call:project'])
        self.assertEqual(self.ui.workbench.views['root'].selected_refs, ['call:encode_a'])
        self.assertEqual(dpg.get_selected_nodes(host.get(key).editor_tag), [])
        host.apply_selection(key, [], intentional=True)
        self.assertEqual(self.ui.workbench.views[key].selected_refs, [])

    def test_tokens_reject_generation_context_document_and_closed_view(self):
        host = self.ui.canvas_host
        key = self.chip()
        token = host.token(key)
        self.chip('encode_b')
        self.assertFalse(host.accept(token))
        token = host.token(key)
        host.refresh_scope('encoder_chip'); self.sync()
        self.assertFalse(host.accept(token))
        token = host.token(key)
        self.ui.workbench.close(key); self.sync()
        self.assertFalse(host.accept(token))
        self.assertEqual(self.chip('encode_b'), key)
        self.assertFalse(host.accept(token), 'Closing/reopening the same definition must not revive its old token')
        root_token = host.token('root')
        host.reset_document()
        self.sync()
        self.assertFalse(host.accept(root_token))

    def test_columns_rows_and_close_do_not_edit_or_cancel(self):
        before = deepcopy(self.ui.document.graph)
        running_snapshot = {'id': 'existing-worker-snapshot', 'status': 'running'}
        self.ui.run_record = running_snapshot
        key = self.chip()
        self.ui.workbench.set_layout('columns')
        self.ui.workbench.move_to_pane(key, 'secondary'); self.sync()
        ids = {name: native_id(view.editor_tag) for name, view in self.ui.canvas_host.views.items()}
        for mode in ('rows', 'columns', 'single'):
            self.ui.workbench.set_layout(mode); self.sync()
            self.assertEqual({name: native_id(view.editor_tag) for name, view in self.ui.canvas_host.views.items()}, ids)
        self.ui.workbench.close(key); self.sync()
        self.assertEqual(self.ui.document.graph, before)
        self.assertIs(self.ui.run_record, running_snapshot)

    def invoke(self, item, data=None, callback=None, user_data=None):
        callback = callback or dpg.get_item_callback(item)
        if user_data is None:
            user_data = dpg.get_item_user_data(item)
        return callback(item, data, user_data)

    def children(self, parent):
        result = []
        for group in dpg.get_item_children(parent).values():
            for item in group:
                result.append(item)
                result.extend(self.children(item))
        return result

    def button(self, parent, label):
        return next(item for item in self.children(parent) if dpg.get_item_configuration(item).get('label') == label)

    def two_panes(self):
        graph = deepcopy(self.ui.document.graph)
        other = deepcopy(graph['definitions'][0]); other['id'] = 'other_chip'
        graph['definitions'].append(other)
        graph['instances'].append({'id': 'other', 'kind': 'chip', 'definition_id': 'other_chip', 'params': {}})
        graph['calls'][1]['instance_id'] = 'other'
        self.ui.document.replace_graph(graph); self.ui._reset_canvas_document()
        self.ui.workbench.set_layout('columns')
        left = self.chip()
        right = self.ui.workbench.open_definition('other_chip', ['encode_b'], pane='secondary')
        self.sync()
        return left, right

    def test_same_local_ids_have_isolated_ports_moves_selection_and_links(self):
        left, right = self.two_panes()
        host = self.ui.canvas_host
        a, b = host.get(left), host.get(right)
        self.assertNotEqual(a.ref_nodes['call:project'], b.ref_nodes['call:project'])
        host.apply_selection(left, ['call:project'], intentional=True)
        host.apply_selection(right, ['call:activate'], intentional=True)
        dpg.set_item_pos(a.ref_nodes['call:project'], (139, 177))
        self.ui._commit_canvas_positions(left)
        definitions = self.ui.document.graph['definitions']
        self.assertNotEqual(definitions[0]['graph']['calls'][0]['position'], definitions[1]['graph']['calls'][0]['position'])
        self.assertEqual(self.ui.workbench.views[left].selected_refs, ['call:project'])
        self.assertEqual(self.ui.workbench.views[right].selected_refs, ['call:activate'])
        source = next(item for item, ref in a.source_attributes.items() if ref == 'input:x')
        target = next(item for item, ref in a.target_attributes.items() if ref == ('call', 'activate', 'x'))
        foreign = next(item for item, ref in b.target_attributes.items() if ref == ('call', 'activate', 'x'))
        callback = dpg.get_item_callback(a.editor_tag)
        callback(a.editor_tag, [source, foreign], host.token(left))
        self.assertEqual(definitions[0]['graph']['calls'][1]['inputs']['x'], 'call:project')
        callback(a.editor_tag, [source, target], host.token(left))
        self.assertEqual(self.ui.document.graph['definitions'][0]['graph']['calls'][1]['inputs']['x'], 'input:x')
        self.assertEqual(self.ui.document.graph['definitions'][1]['graph']['calls'][1]['inputs']['x'], 'call:project')
        # Delinking must use this editor's native link ID.
        a, b = host.get(left), host.get(right)
        foreign_link = next(iter(b.link_targets))
        delink = dpg.get_item_configuration(a.editor_tag)['delink_callback']
        before = deepcopy(self.ui.document.graph)
        delink(a.editor_tag, foreign_link, host.token(left))
        self.assertEqual(self.ui.document.graph, before)

    def test_parameter_freeze_and_modal_callbacks_reject_old_call_context(self):
        key = self.chip()
        self.ui.canvas_host.apply_selection(key, ['call:project'], intentional=True)
        self.ui._render_inspector()
        widget = self.ui.param_widgets['out_features']
        callback, token = dpg.get_item_callback(widget), dpg.get_item_user_data(widget)
        freeze = self.button('native_inspector', '冻结此实例参数')
        freeze_callback, freeze_token = dpg.get_item_callback(freeze), dpg.get_item_user_data(freeze)
        self.ui._open_chip_tools()
        add = self.button('native_chip_tools', '新增输入')
        add_callback, add_token = dpg.get_item_callback(add), dpg.get_item_user_data(add)
        dpg.set_value('native_new_input_id', 'queued_input')
        before = deepcopy(self.ui.document.graph)
        self.chip('encode_b')
        callback(widget, '19', token)
        freeze_callback(freeze, True, freeze_token)
        add_callback(add, None, add_token)
        self.assertEqual(self.ui.document.graph, before)
        self.assertGreaterEqual(sum('过期画布' in message[2] for message in self.ui.messages), 3)

    def test_live_registered_parameter_and_interface_callbacks_edit_their_scope(self):
        left, right = self.two_panes()
        self.ui.workbench.activate(left); self.sync()
        self.ui.canvas_host.apply_selection(left, ['call:project'], intentional=True)
        self.ui._render_inspector()
        self.invoke(self.ui.param_widgets['out_features'], '12')
        self.assertEqual(self.ui.document.graph['definitions'][0]['graph']['instances'][0]['params']['out_features'], 12)
        self.assertEqual(self.ui.document.graph['definitions'][1]['graph']['instances'][0]['params']['out_features'], 8)
        self.ui._open_chip_tools()
        dpg.set_value('native_new_input_id', 'left_only')
        self.invoke(self.button('native_chip_tools', '新增输入'))
        self.assertIn('left_only', [item['id'] for item in self.ui.document.graph['definitions'][0]['graph']['inputs']])
        self.assertNotIn('left_only', [item['id'] for item in self.ui.document.graph['definitions'][1]['graph']['inputs']])

    def test_delete_is_focused_visible_and_inactive_input_only(self):
        left, right = self.two_panes()
        host = self.ui.canvas_host
        host.apply_selection(left, ['call:activate'], intentional=True)
        host.apply_selection(right, ['call:activate'], intentional=True)
        self.ui.workbench.activate(left); self.sync()
        before = deepcopy(self.ui.document.graph)
        with patch.object(dpg, 'is_item_hovered', return_value=True), patch.object(self.ui, '_canvas_input_active', return_value=True):
            self.invoke('native_delete_handler')
        self.assertEqual(self.ui.document.graph, before)
        queued_delete = dpg.get_item_callback('native_delete_handler')
        queued_token = dpg.get_item_user_data('native_delete_handler')
        self.ui.workbench.activate(right); self.sync()
        with patch.object(dpg, 'is_item_hovered', return_value=True), patch.object(self.ui, '_canvas_input_active', return_value=False):
            queued_delete('native_delete_handler', None, queued_token)
        self.assertEqual(self.ui.document.graph, before)
        self.ui.workbench.activate(left); self.sync()
        with patch.object(dpg, 'is_item_hovered', return_value=True), patch.object(self.ui, '_canvas_input_active', return_value=False):
            self.invoke('native_delete_handler')
        self.assertNotIn('activate', [item['id'] for item in self.ui.document.graph['definitions'][0]['graph']['calls']])
        self.assertIn('activate', [item['id'] for item in self.ui.document.graph['definitions'][1]['graph']['calls']])

    def test_explicit_replacement_closes_modals_and_invalidates_queued_widgets(self):
        key = self.chip()
        self.ui.canvas_host.apply_selection(key, ['call:project'], intentional=True)
        self.ui._render_inspector(); self.ui._open_chip_tools()
        widget = self.ui.param_widgets['out_features']
        callback, token = dpg.get_item_callback(widget), dpg.get_item_user_data(widget)
        self.ui.document.replace_graph(Document().graph)
        self.ui._reset_canvas_document()
        self.assertFalse(dpg.is_item_shown('native_chip_tools'))
        before = deepcopy(self.ui.document.graph)
        callback(widget, '99', token)
        self.assertEqual(self.ui.document.graph, before)
        self.assertEqual(list(self.ui.workbench.views), ['root'])

    def test_three_nested_scopes_refresh_transitively_and_preserve_unrelated_nodes(self):
        graph = deepcopy(self.ui.document.graph)
        leaf = graph['definitions'][0]
        leaf['public_parameters'] = []; leaf['bindings'] = {}
        graph['instances'][2]['params']['in_features'] = 12
        for identifier, child in (('middle_chip', 'encoder_chip'), ('outer_chip', 'middle_chip')):
            graph['definitions'].append({
                'id': identifier, 'version': '1.0.0', 'public_parameters': [], 'bindings': {},
                'graph': {'inputs': [{'id': 'x', 'shape': [None, 4], 'dtype': 'float32'}],
                          'instances': [{'id': 'nested', 'kind': 'chip', 'definition_id': child, 'params': {}}],
                          'calls': [{'id': 'inside', 'instance_id': 'nested', 'inputs': {'x': 'input:x'}}],
                          'outputs': [{'id': 'out', 'source': 'call:inside'}]}})
        unrelated = deepcopy(leaf); unrelated['id'] = 'unrelated_chip'; graph['definitions'].append(unrelated)
        graph['instances'][0]['definition_id'] = 'outer_chip'; graph['instances'][0]['params'] = {}
        self.ui.document.replace_graph(graph); self.ui._reset_canvas_document()
        host = self.ui.canvas_host
        outer = self.ui.workbench.open_definition('outer_chip', ['encode_a'])
        middle = self.ui.workbench.open_definition('middle_chip', ['encode_a', 'inside'])
        inner = self.ui.workbench.open_definition('encoder_chip', ['encode_a', 'inside', 'inside'])
        other = self.ui.workbench.open_definition('unrelated_chip', [])
        self.sync()
        before = {key: (native_id(view.editor_tag), dict(view.ref_nodes), view.generation) for key, view in host.views.items()}
        self.ui.workbench.activate(inner); self.sync()
        self.ui._set_param('dense', 'out_features', app_data='12')
        for key in ('root', outer, middle, inner):
            self.assertGreater(host.get(key).generation, before[key][2])
            self.assertEqual(native_id(host.get(key).editor_tag), before[key][0])
        self.assertEqual(dict(host.get(other).ref_nodes), before[other][1])
        self.assertEqual(host.get(other).generation, before[other][2])
        report = self.ui.document.validation()
        self.assertTrue(report['valid'], report['errors'])
        self.assertEqual(report['shapes']['call:encode_a'], [None, 12])
        host.refresh_scope('encoder_chip'); self.sync()
        self.assertEqual(dict(host.get(other).ref_nodes), before[other][1])

    def test_input_state_uses_installed_public_dpg_api(self):
        self.assertFalse(self.ui._canvas_input_active())

    def test_imported_operator_uses_core_ports_without_adding_catalog_entries(self):
        from native.ui import KINDS
        graph = deepcopy(self.ui.document.graph)
        graph['instances'][1]['kind'] = 'Multiply'
        self.ui.document.replace_graph(graph); self.ui._reset_canvas_document()
        ports = set(self.ui.canvas_host.get('root').target_attributes.values())
        self.assertIn(('call', 'fuse', 'a'), ports)
        self.assertIn(('call', 'fuse', 'b'), ports)
        self.assertNotIn(('call', 'fuse', 'x'), ports)
        self.assertNotIn('Multiply', KINDS)

    def test_named_socket_generation_is_protected_without_changing_root_on_navigation(self):
        graph = deepcopy(self.ui.document.graph)
        graph['definitions'][0]['output_mode'] = 'named'
        graph['definitions'][0]['graph']['outputs'].append({'id': 'encoded', 'source': 'call:project'})
        self.ui.document.replace_graph(graph); self.ui._reset_canvas_document()
        # The root named sockets belong to root; this inner definition has two
        # primitive outputs only, so use the root call button for queued tokens.
        root = self.ui.canvas_host.get('root')
        socket = self.button(root.ref_nodes['call:encode_a'], 'encoded')
        callback, old_token = dpg.get_item_callback(socket), dpg.get_item_user_data(socket)
        self.invoke(socket)
        self.assertEqual(self.ui._probe_source(), 'call:encode_a:encoded')
        node_before = root.ref_nodes['call:encode_a']
        self.chip('encode_b')
        self.ui.workbench.activate('root'); self.sync()
        self.assertEqual(root.ref_nodes['call:encode_a'], node_before)
        self.assertTrue(self.ui.canvas_host.accept(old_token))
        # An explicit redraw invalidates a queued socket callback independently
        # of call-context and document replacement.
        self.ui.canvas_host.refresh_scope(None); self.sync()
        before = deepcopy(self.ui.document.graph)
        callback(socket, None, old_token)
        self.assertEqual(self.ui.document.graph, before)
        self.assertFalse(self.ui.canvas_host.accept(old_token))

    def test_registered_navigation_preserves_views_and_focuses_root_selection(self):
        key = self.chip()
        self.ui._refresh(graph=False)
        button = self.button('native_tree', '检查实例参数')
        # Choose the encoder's first parameter button, a genuine registered
        # root-tree callback while the focused document is inside its chip.
        self.invoke(button)
        self.assertEqual(self.ui.workbench.focused_view_id, 'root')
        self.assertIsNone(self.ui.document.scope_id)
        self.assertEqual(self.ui.document.selected, 'call:encode_a')
        self.assertIn(key, self.ui.canvas_host.views)
        self.assertEqual(self.ui.selected_refs, ['call:encode_a'])

    def test_legacy_document_selection_is_adopted_before_a_full_refresh(self):
        key = self.chip()
        self.ui.document.select('output:out')
        self.ui._refresh()
        self.assertEqual(self.ui.workbench.views[key].selected_refs, ['output:out'])
        self.assertEqual(self.ui.document.selected, 'output:out')
        self.assertEqual(self.ui._probe_source(), 'call:encode_a/call:activate')

    def test_real_frames_hide_view_containers_and_keep_correct_node_geometry(self):
        from native.render_acceptance import _frames
        root = self.ui.canvas_host.get('root')
        key = self.chip()
        chip = self.ui.canvas_host.get(key)
        _frames(self.ui, 4)
        self.assertFalse(dpg.is_item_shown(root.container_tag))
        self.assertTrue(dpg.is_item_shown(chip.container_tag))
        # A genuinely rendered chip output is compact; rendering root nodes in
        # the same ImNodes context had produced a 300px-wide wrong rectangle.
        width, height = dpg.get_item_rect_size(chip.ref_nodes['output:out'])
        self.assertGreater(width, 60); self.assertLess(width, 220)
        self.assertGreater(height, 40); self.assertLess(height, 160)
        editor_id, nodes = native_id(chip.editor_tag), dict(chip.ref_nodes)
        self.ui.workbench.activate('root'); self.sync(); _frames(self.ui, 4)
        self.assertTrue(dpg.is_item_shown(root.container_tag))
        self.assertFalse(dpg.is_item_shown(chip.container_tag))
        self.ui.workbench.activate(key); self.sync(); _frames(self.ui, 4)
        self.assertEqual(native_id(chip.editor_tag), editor_id)
        self.assertEqual(chip.ref_nodes, nodes)

    def test_shutdown_disposes_host_even_when_query_cleanup_raises(self):
        self.chip()
        with patch.object(self.ui.queries, 'close', side_effect=RuntimeError('query cleanup failure')):
            with self.assertRaisesRegex(RuntimeError, 'query cleanup failure'):
                self.ui.shutdown()
        self.assertEqual(self.ui.canvas_host.views, {})
        self.assertFalse(self.ui._context_created)

    def test_navigation_callbacks_created_after_a_command_have_current_generation(self):
        self.ui._add('ReLU', None)
        self.invoke(self.button('native_tree', '进入 encode_a'))
        self.assertEqual(self.ui.document.scope_id, 'encoder_chip')

    def test_sync_only_moves_a_container_when_its_pane_changes(self):
        key = self.chip()
        real_move = dpg.move_item
        with patch.object(dpg, 'move_item', wraps=real_move) as moves:
            self.sync(); self.sync()
            self.assertEqual(moves.call_count, 0)
            self.ui.workbench.set_layout('columns')
            self.ui.workbench.move_to_pane(key, 'secondary'); self.sync()
            self.assertEqual(moves.call_count, 1)
            self.sync()
            self.assertEqual(moves.call_count, 1)

    def test_shared_palette_after_focus_change_targets_the_current_view(self):
        left, right = self.two_panes()
        self.ui.workbench.activate(right); self.sync()
        self.ui._set_param('dense', 'out_features', app_data='9')
        counts = [len(item['graph']['instances']) for item in self.ui.document.graph['definitions']]
        self.ui.workbench.activate(left); self.sync()
        self.invoke(self.button('native_palette', 'ReLU'))
        self.assertEqual([len(item['graph']['instances']) for item in self.ui.document.graph['definitions']], [counts[0] + 1, counts[1]])

    def test_registered_undo_removing_the_focused_definition_refreshes_root_panels(self):
        before = deepcopy(self.ui.document.graph)
        ref = self.ui.document.add_template('Review transient', 'review_transient')
        self.ui._refresh()
        self.ui._enter(ref.split(':', 1)[1])
        key = self.ui.workbench.focused_view_id
        self.ui.canvas_host.apply_selection(key, ['call:project'], intentional=True)
        self.ui._render_inspector(); self.ui._render_probe()
        self.assertTrue(self.ui.param_widgets)
        messages_before = len(self.ui.messages)
        self.invoke('native_undo')
        restored = deepcopy(self.ui.document.graph)
        restored['revision'] = before['revision']
        self.assertEqual(restored, before)
        self.assertEqual(self.ui.workbench.focused_view_id, 'root')
        self.assertNotIn(key, self.ui.canvas_host.views)
        self.assertIsNone(self.ui._bound_canvas_id)
        self.assertIsNone(self.ui.document.scope_id)
        self.assertIsNone(self.ui.document.selected)
        self.assertEqual(self.ui.selected_refs, [])
        root = self.ui.canvas_host.get('root')
        self.assertIs(self.ui.node_refs, root.node_refs)
        self.assertIs(self.ui.ref_nodes, root.ref_nodes)
        self.assertEqual(native_id('native_editor'), native_id(root.editor_tag))
        self.assertEqual(self.ui.param_widgets, {})
        self.assertIsNone(self.ui._probe_source())
        self.assertTrue(any('请选择一个调用' in str(dpg.get_value(item)) for item in self.children('native_probe')))
        self.assertIn('4 个调用', dpg.get_value('native_status'))
        self.assertNotIn('操作失败', dpg.get_value('native_status'))
        self.assertFalse([item for item in self.ui.messages[messages_before:] if item[1] == '错误'])

    def test_delayed_other_pane_parameter_callback_restores_focused_shared_controls(self):
        left, right = self.two_panes()
        self.ui.workbench.activate(left); self.sync()
        self.ui.canvas_host.apply_selection(left, ['call:project'], intentional=True)
        self.ui._render_inspector()
        widget = self.ui.param_widgets['out_features']
        delayed_callback = dpg.get_item_callback(widget)
        delayed_token = dpg.get_item_user_data(widget)
        self.ui.workbench.activate(right); self.sync()
        self.ui.canvas_host.apply_selection(right, ['call:project'], intentional=True)
        self.ui._render_inspector()
        counts = [len(item['graph']['instances']) for item in self.ui.document.graph['definitions']]
        delayed_callback(widget, '12', delayed_token)
        self.assertEqual(self.ui.document.graph['definitions'][0]['graph']['instances'][0]['params']['out_features'], 12)
        self.assertEqual(self.ui.workbench.focused_view_id, right)
        self.assertEqual(self.ui.document.scope_id, 'other_chip')
        self.assertIsNone(self.ui._bound_canvas_id)
        displayed_widget = self.ui.param_widgets['out_features']
        displayed_value = dpg.get_value(displayed_widget)
        self.invoke(displayed_widget, '10')
        self.assertEqual(self.ui.document.graph['definitions'][0]['graph']['instances'][0]['params']['out_features'], 12)
        self.assertEqual(self.ui.document.graph['definitions'][1]['graph']['instances'][0]['params']['out_features'], 10)
        self.invoke(self.button('native_palette', 'ReLU'))
        self.assertEqual([len(item['graph']['instances']) for item in self.ui.document.graph['definitions']], [counts[0], counts[1] + 1])
        self.assertEqual(displayed_value, '8')

    def test_neutral_other_pane_position_poll_preserves_unsubmitted_focused_input(self):
        left, right = self.two_panes()
        self.ui.workbench.activate(right); self.sync()
        self.ui.canvas_host.apply_selection(right, ['call:project'], intentional=True)
        self.ui._render_inspector()
        widget = self.ui.param_widgets['out_features']
        original_id = native_id(widget)
        dpg.set_value(widget, '17')  # Unsubmitted text; no Enter callback.
        before = deepcopy(self.ui.document.graph)
        for _ in range(3):
            self.ui._commit_canvas_positions(left)
            self.assertEqual(native_id(self.ui.param_widgets['out_features']), original_id)
            self.assertEqual(dpg.get_value(widget), '17')
            with self.ui._canvas_context(left):
                self.assertEqual(self.ui.document.scope_id, 'encoder_chip')
            self.sync()
            self.ui.last_poll = 0
            self.ui.tick()
            self.assertEqual(native_id(self.ui.param_widgets['out_features']), original_id)
            self.assertEqual(dpg.get_value(widget), '17')
        self.assertEqual(self.ui.document.graph, before)
        self.assertEqual(self.ui.document.graph['definitions'][1]['graph']['instances'][0]['params']['out_features'], 8)


if __name__ == '__main__':
    unittest.main()
