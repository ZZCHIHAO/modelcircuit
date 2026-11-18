"""Known diagnostic and measured tensor locations use actual graph scopes."""
from copy import deepcopy
import tempfile
import unittest

import dearpygui.dearpygui as dpg

from native.document import Document
from native.paths import resource_root
from native.ui import NativeWorkbench


def deep_graph():
    graph = deepcopy(Document().graph)
    graph['definitions'].append({'id': 'outer_chip', 'version': '1.0.0', 'public_parameters': [], 'bindings': {},
        'graph': {'inputs': [{'id': 'x', 'shape': [None, 4], 'dtype': 'float32'}],
                  'instances': [{'id': 'nested', 'kind': 'chip', 'definition_id': 'encoder_chip', 'params': {}}],
                  'calls': [{'id': 'inside', 'instance_id': 'nested', 'inputs': {'x': 'input:x'}}],
                  'outputs': [{'id': 'out', 'source': 'call:inside'}]}})
    graph['instances'][0]['definition_id'] = 'outer_chip'
    graph['instances'][0]['params'] = {}
    return graph


class WorkbenchLocationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.ui = NativeWorkbench(Document(deep_graph()), None, self.directory.name, resource_root())
        self.ui.build()

    def tearDown(self):
        self.ui.shutdown()
        self.directory.cleanup()

    def invoke(self, tag, data=None):
        return dpg.get_item_callback(tag)(tag, data, dpg.get_item_user_data(tag))

    def test_known_probe_callback_opens_full_path_without_training_or_fake_measurement(self):
        source = 'call:encode_a/call:inside/call:project'
        result = {'probes': {source: {'shape': [2, 8], 'mean': 1.25, 'values': [1.25]}}}
        self.ui.result = result
        self.ui._render_probe()
        self.invoke('native_probe_sources', source)
        key = self.ui.workbench.focused_view_id
        self.assertEqual(self.ui.workbench.views[key].target.definition_id, 'encoder_chip')
        self.assertEqual(self.ui.workbench.views[key].call_path, ['encode_a', 'inside'])
        self.assertEqual(self.ui.workbench.views[key].selected_refs, ['call:project'])
        self.assertEqual(self.ui._probe_source(), source)
        self.assertIs(self.ui.result, result)
        self.assertIsNone(self.ui.run_record)
        missing = 'call:encode_b/call:inside/call:activate'
        self.assertTrue(self.ui._focus_probe_source(missing))
        self.assertNotIn(missing, result['probes'])
        self.assertTrue(any(dpg.get_value(item) == '尚未采集' for item in dpg.get_item_children('native_probe')[1]
                            if dpg.get_item_type(item) == 'mvAppItemType::mvText'))

    def test_probe_sources_validate_path_and_named_port_before_navigation(self):
        before = self.ui.workbench.to_payload()
        for source in ('call:encode_a/call:missing/call:project', 'call:encode_b/call:inside/call:project:missing',
                       'input:a/../call:project', 'call:encode_b/call:project/extra', 'call:missing', ['call:encode_a']):
            with self.subTest(source=source):
                self.assertFalse(self.ui._focus_probe_source(source))
                self.assertEqual(self.ui.workbench.to_payload(), before)

    def test_structure_error_has_registered_button_for_unique_definition_node(self):
        # The real validator reports a unique local call ID but no path. It is
        # safe to open the definition without inventing a concrete invocation.
        self.ui.document.graph['definitions'][0]['graph']['calls'][0]['inputs'] = {}
        self.ui._validate()
        self.assertFalse(self.ui.report['valid'])
        self.invoke('native_error_location_0')
        key = self.ui.workbench.focused_view_id
        self.assertEqual(self.ui.workbench.views[key].target.definition_id, 'encoder_chip')
        self.assertEqual(self.ui.workbench.views[key].call_path, [])
        self.assertEqual(self.ui.workbench.views[key].selected_refs, ['call:project'])
        self.assertIsNone(self.ui._probe_source())

    def test_ambiguous_diagnostic_does_not_guess_same_name_in_another_scope(self):
        graph = deepcopy(self.ui.document.graph)
        duplicate = deepcopy(graph['definitions'][0]); duplicate['id'] = 'other_encoder'
        graph['definitions'].append(duplicate)
        self.ui.document.replace_graph(graph); self.ui._reset_canvas_document()
        self.assertIsNone(self.ui._issue_location({'node_id': 'project', 'field': 'inputs', 'message': 'invalid'}))
        self.assertFalse(self.ui.workbench_panel.focus_location({'definition_id': 'encoder_chip', 'call_path': ['encode_a'],
                                                               'node_ref': 'call:project', 'field': None}))

    def test_queued_location_callback_cannot_retarget_same_named_replacement(self):
        location = {'definition_id': 'encoder_chip', 'call_path': ['encode_b'], 'node_ref': 'call:project', 'field': 'out_features'}
        callback = self.ui.workbench_panel.location_callback(location)
        self.ui.document.replace_graph(deepcopy(self.ui.document.graph)); self.ui._reset_canvas_document()
        self.assertFalse(callback(None, None, None))
        self.assertEqual(self.ui.workbench.focused_view_id, 'root')

    def test_historical_error_button_stays_invalid_after_console_is_redrawn(self):
        location = {'definition_id': 'encoder_chip', 'call_path': ['encode_b', 'inside'],
                    'node_ref': 'call:project', 'field': None}
        self.ui.log('old error', '错误', location=location)
        self.ui.document.replace_graph(deepcopy(self.ui.document.graph)); self.ui._reset_canvas_document()
        self.ui.log('new project')
        self.assertFalse(self.invoke('native_error_location_0'))
        self.assertEqual(self.ui.workbench.focused_view_id, 'root')

    def test_interface_rename_and_deleted_definition_undo_reconcile_saved_references(self):
        key = self.ui.workbench_panel.open_call(['encode_b', 'inside'])
        self.ui.canvas_host.apply_selection(key, ['output:out'], intentional=True)
        self.ui.document.rename_output('out', 'renamed')
        self.ui._refresh()
        self.assertEqual(self.ui.workbench.views[key].selected_refs, ['output:renamed'])
        self.assertTrue(self.ui.document.undo())
        self.ui._refresh()
        self.assertIn(key, self.ui.workbench.views)
        graph = deepcopy(self.ui.document.graph)
        graph['instances'] = [item for item in graph['instances'] if item.get('definition_id') != 'encoder_chip']
        removed = {call['id'] for call in graph['calls'] if call['instance_id'] not in {item['id'] for item in graph['instances']}}
        graph['calls'] = [call for call in graph['calls'] if call['id'] not in removed]
        graph['definitions'] = [item for item in graph['definitions'] if item['id'] != 'encoder_chip']
        self.ui.document.graph = graph
        self.ui.workbench.reconcile(self.ui.document)
        self.assertNotIn(key, self.ui.workbench.views)


if __name__ == '__main__':
    unittest.main()
