from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from native.attention_probe_panel import matching_capture, normalize_probe_requests
from native.document import Document
from native.workspace_state import read_workspace_full, save_workspace, synthetic_config
from native.workbench_state import WorkbenchState
from test_tensor_component_insert import chip


REQUEST = {'source': 'call:encoder/call:softmax', 'sample_index': 0,
           'head_index': 1, 'query_range': [0, 3], 'key_range': [1, 4], 'statistics': ['mean']}


class AttentionProbePanelTests(unittest.TestCase):
    def test_workspace_roundtrip_only_serializes_request(self):
        doc = Document()
        doc.add_component(chip(), apply_policy=True)
        doc.attention_probe_requests = [REQUEST]
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'workspace.json'
            views = WorkbenchState().to_payload()
            save_workspace(path, doc, synthetic_config(), workbench=views)
            loaded, config, workbench = read_workspace_full(path)
        self.assertEqual(loaded.attention_probe_requests, [REQUEST])
        self.assertEqual(workbench, views)
        self.assertEqual(config, synthetic_config())

    def test_capture_requires_actual_matching_selection(self):
        capture = {'selection': {key: REQUEST[key] for key in ('sample_index', 'head_index', 'query_range', 'key_range')},
                   'values': [[0.1]]}
        self.assertIs(matching_capture(capture, REQUEST), capture)
        self.assertIsNone(matching_capture(capture, {**REQUEST, 'head_index': 2}))
        self.assertIsNone(matching_capture(None, REQUEST))

    def test_rejects_unbounded_request_and_duplicate_source(self):
        with self.assertRaisesRegex(ValueError, '64'):
            normalize_probe_requests([{**REQUEST, 'key_range': [0, 65]}])
        with self.assertRaisesRegex(ValueError, 'distinct'):
            normalize_probe_requests([REQUEST, REQUEST])

    def test_frozen_request_owns_truncated_and_omitted_results(self):
        truncated = {'selection': {'query_range': [0, 2], 'key_range': [1, 3]}, 'truncated': True}
        omitted = {'omitted': True, 'reason': 'Capture byte budget exhausted'}
        self.assertIs(matching_capture(truncated, REQUEST, REQUEST), truncated)
        self.assertIs(matching_capture(omitted, REQUEST, REQUEST), omitted)
        self.assertIsNone(matching_capture(truncated, {**REQUEST, 'head_index': 2}, REQUEST))

    def test_registered_callback_saves_request_and_displays_actual_slice(self):
        import dearpygui.dearpygui as dpg
        from native.paths import resource_root
        from native.ui import NativeWorkbench

        doc = Document()
        call = doc.add_component(chip(), apply_policy=True)
        with TemporaryDirectory() as directory:
            ui = NativeWorkbench(doc, None, directory, resource_root())
            ui.build()
            try:
                ui._focus_probe_source(call + ':y')
                tag = 'native_attention_apply'
                self.assertTrue(dpg.does_item_exist(tag))
                dpg.set_value('native_attention_head', 1)
                dpg.set_value('native_attention_query_range', '[0, 2]')
                dpg.set_value('native_attention_key_range', '[0, 2]')
                dpg.get_item_callback(tag)(tag, None, dpg.get_item_user_data(tag))
                request = doc.attention_probe_requests[0]
                self.assertEqual(request['head_index'], 1)
                self.assertEqual(ui.attention_probe_panel.run_requests(), [request])
                ui.run_config = {'attention_probe_requests': [request]}
                ui.result = {'probes': {request['source']: {'selection': request, 'shape': [1, 2, 2, 2],
                                                            'dtype': 'float32', 'values': [[.1, .9], [.3, .7]]}}}
                ui.result_key = ui._experiment_identity()
                ui._render_probe()
                self.assertTrue(any(dpg.get_item_type(item).endswith('mvHeatSeries') for item in dpg.get_all_items()))
            finally:
                ui.shutdown()
