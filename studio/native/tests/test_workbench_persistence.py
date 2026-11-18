"""B4 storage compatibility, recovery and real native save callbacks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dearpygui.dearpygui as dpg

from native.document import Document
from native.paths import create_runtime_paths, resource_root
from native.workbench_state import WorkbenchState
from native import workspace_state as storage
from native import main as startup
from native.ui import NativeWorkbench


def split_state(document):
    state = WorkbenchState()
    state.set_layout('columns')
    key = state.open_definition('encoder_chip', ['encode_b'], pane='secondary')
    state.views[key].selected_refs = ['call:project']
    state.views[key].probe_port = 'out'
    return state


class WorkbenchStorageTests(unittest.TestCase):
    def test_v2_and_plain_graph_keep_two_return_api_and_full_default(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'workspace.json'
            document = Document()
            for legacy in (False, True):
                if legacy:
                    document.save_json(path)
                else:
                    storage.save_workspace(path, document, storage.synthetic_config())
                full = storage.read_workspace_full(path)
                self.assertEqual(len(storage.read_workspace(path)), 2)
                self.assertEqual(full[0].graph, document.graph)
                self.assertIsNone(full[2])

    def test_v3_roundtrip_preserves_order_context_probe_and_model_identity(self):
        from studio_core.experiment_spec import checkpoint_signature, default_classification_config
        document = Document()
        experiment = storage.synthetic_config()
        training = default_classification_config('csv')
        identity = (document.semantic_key(), checkpoint_signature(document.graph, training, 'samples'))
        state = split_state(document)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'workspace.json'
            storage.save_workspace(path, document, experiment, workbench=state.to_payload())
            payload = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(payload['workspace_version'], 3)
            self.assertEqual(payload['model'], document.graph)
            loaded, settings, raw = storage.read_workspace_full(path)
            self.assertEqual(raw, state.to_payload())
            self.assertEqual(settings, experiment)
            self.assertEqual(len(storage.read_workspace(path)), 2)
            self.assertEqual(identity, (loaded.semantic_key(), checkpoint_signature(loaded.graph, training, 'samples')))

    def test_damaged_views_are_returned_raw_so_ui_can_report_the_repair(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'workspace.json'
            payload = {'workspace_version': 3, 'model': Document().graph,
                       'experiment': storage.synthetic_config(), 'workbench': {'views': 'damaged'}}
            path.write_text(json.dumps(payload), encoding='utf-8')
            document, experiment, raw = storage.read_workspace_full(path)
            self.assertEqual(raw, {'views': 'damaged'})
            restored = WorkbenchState.from_payload(raw, document)
            self.assertEqual(list(restored.views), ['root'])
            self.assertTrue(restored.warnings)
            self.assertEqual(experiment, storage.synthetic_config())

    def test_save_rejects_nonfinite_or_unbounded_view_storage_atomically(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'workspace.json'
            path.write_text('original', encoding='utf-8')
            valid = WorkbenchState().to_payload()
            cases = []
            for field, value in [('native_handle', 12), ('source', 'exec(code)'), ('path', 'C:/secret.py')]:
                candidate = deepcopy(valid); candidate['views'][0][field] = value; cases.append(candidate)
            candidate = deepcopy(valid); candidate['views'][0]['probe_port'] = float('nan'); cases.append(candidate)
            candidate = deepcopy(valid); candidate['layout'] = []; cases.append(candidate)
            for raw in cases:
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    storage.save_workspace(path, Document(), storage.synthetic_config(), workbench=raw)
                self.assertEqual(path.read_text(encoding='utf-8'), 'original')

    def test_startup_backs_up_unknown_and_broken_json_and_protects_autosave(self):
        for damaged in ('{"workspace_version":99}', '{broken'):
            with self.subTest(damaged=damaged), tempfile.TemporaryDirectory() as folder:
                paths = create_runtime_paths(folder)
                source = paths['workspace']; source.write_text(damaged, encoding='utf-8')
                Document().save_json(paths['draft'])
                with self.assertLogs(level='ERROR'):
                    loaded = startup.load_saved_workspace_full(paths)
                with self.assertLogs(level='ERROR'):
                    self.assertEqual(len(startup.load_saved_workspace(paths)), 2)
                self.assertIsNone(loaded[2])
                self.assertIn(source.resolve(), loaded[0].workspace_protected_paths)
                backups = list(Path(folder).glob('workspace-invalid-*.json'))
                self.assertTrue(backups)
                self.assertTrue(all(item.read_text(encoding='utf-8') == damaged for item in backups))
                ui = NativeWorkbench(*loaded[:1], None, folder, resource_root(), experiment=loaded[1], workbench_payload=loaded[2])
                ui.build()
                try:
                    ui._save_draft()
                finally:
                    ui.shutdown()
                self.assertEqual(source.read_text(encoding='utf-8'), damaged)

    def test_unreadable_workspace_backup_failure_still_preserves_original(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = create_runtime_paths(folder)
            paths['workspace'].write_text('{broken', encoding='utf-8')
            with self.assertLogs(level='ERROR'), patch('native.main.shutil.copy2', side_effect=PermissionError('read only')):
                loaded, _, _ = startup.load_saved_workspace_full(paths)
            self.assertTrue(loaded.workspace_load_warnings)
            self.assertIn(paths['workspace'].resolve(), loaded.workspace_protected_paths)
            self.assertEqual(paths['workspace'].read_text(encoding='utf-8'), '{broken')

    def test_v3_null_view_reports_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'workspace.json'
            raw = {'workspace_version': 3, 'model': Document().graph,
                   'experiment': storage.synthetic_config(), 'workbench': None}
            path.write_text(json.dumps(raw), encoding='utf-8')
            document, _, payload = storage.read_workspace_full(path)
            self.assertIsNone(payload)
            self.assertTrue(document.workspace_load_warnings)

    def test_missing_v3_workbench_salvages_model_but_unknown_envelope_field_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'workspace.json'
            model = Document().graph
            raw = {'workspace_version': 3, 'model': model, 'experiment': storage.synthetic_config()}
            path.write_text(json.dumps(raw), encoding='utf-8')
            document, experiment, view = storage.read_workspace_full(path)
            self.assertEqual(document.graph, model)
            self.assertEqual(experiment, storage.synthetic_config())
            self.assertIsNone(view)
            self.assertTrue(document.workspace_load_warnings)
            raw['source_code'] = 'untrusted'
            path.write_text(json.dumps(raw), encoding='utf-8')
            with self.assertRaises(ValueError):
                storage.read_workspace_full(path)

    def test_oversize_save_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'workspace.json'
            path.write_text('original', encoding='utf-8')
            config = {'profile': 'classification-v2', 'data': {'kind': 'csv', 'path': 'x' * (8 * 1024 * 1024)}}
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, '8 MiB'):
                storage.save_workspace(path, Document(), config, workbench=WorkbenchState().to_payload())
            self.assertEqual(path.read_bytes(), original)

    def test_pretty_serialization_cannot_write_a_workspace_over_the_read_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'workspace.json'
            path.write_text('original', encoding='utf-8')
            document = Document()
            config = {'profile': 'classification-v2', 'data': {'kind': 'csv', 'path': ''}}
            raw = storage.workspace_payload(document, config, workbench=WorkbenchState().to_payload())
            overhead = len(json.dumps(raw, ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode('utf-8'))
            config['data']['path'] = 'x' * (8 * 1024 * 1024 - overhead - 1)
            with self.assertRaisesRegex(ValueError, '8 MiB'):
                storage.save_workspace(path, document, config, workbench=WorkbenchState().to_payload())
            self.assertEqual(path.read_text(encoding='utf-8'), 'original')

    def test_excessive_json_nesting_is_a_recoverable_failed_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'workspace.json'
            path.write_text('[' * 20000 + '0' + ']' * 20000, encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Invalid workspace JSON'):
                storage.read_workspace_full(path)

    def test_parsed_deeply_nested_view_recovers_without_copying_valid_model_away(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = create_runtime_paths(folder)
            model = Document().graph; model['revision'] = 23
            model['name'] = 'Recovered model, not fallback'
            experiment = {**storage.synthetic_config(), 'lr': 0.01}
            raw = json.dumps({'workspace_version': 3, 'model': model, 'experiment': experiment})[:-1]
            raw += ',"workbench":' + '[' * 600 + '0' + ']' * 600 + '}'
            self.assertEqual(json.loads(raw)['model'], model)
            paths['workspace'].write_text(raw, encoding='utf-8')
            document, settings, view = storage.read_workspace_full(paths['workspace'])
            self.assertEqual(document.graph, model)
            self.assertEqual(settings, experiment)
            restored = WorkbenchState.from_payload(view, document)
            self.assertEqual(list(restored.views), ['root'])
            self.assertTrue(restored.warnings)
            startup_document, startup_settings, startup_view = startup.load_saved_workspace_full(paths)
            self.assertEqual(startup_document.graph, model)
            self.assertEqual(startup_settings, experiment)
            self.assertTrue(WorkbenchState.from_payload(startup_view, startup_document).warnings)

    def test_overflow_exponents_anywhere_in_raw_view_are_failed_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'workspace.json'
            prefix = json.dumps({'workspace_version': 3, 'model': Document().graph,
                                 'experiment': storage.synthetic_config()})[:-1]
            for view in ('{"layout":1e309}', '{"views":[[[-1e309]]]}',
                         '{"layout":1e99999,"layout":"single"}'):
                with self.subTest(view=view):
                    path.write_text(prefix + ',"workbench":' + view + '}', encoding='utf-8')
                    with self.assertRaisesRegex(ValueError, 'finite'):
                        storage.read_workspace_full(path)

    def test_finite_exponent_in_damaged_view_still_uses_view_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'workspace.json'
            prefix = json.dumps({'workspace_version': 3, 'model': Document().graph,
                                 'experiment': storage.synthetic_config()})[:-1]
            path.write_text(prefix + ',"workbench":{"layout":1e308}}', encoding='utf-8')
            document, _, view = storage.read_workspace_full(path)
            state = WorkbenchState.from_payload(view, document)
            self.assertEqual(list(state.views), ['root'])
            self.assertTrue(state.warnings)


class NativePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.ui = NativeWorkbench(Document(), None, self.directory.name, resource_root())
        self.ui.build()

    def tearDown(self):
        self.ui.shutdown()
        self.directory.cleanup()

    def split(self):
        self.ui.workbench_panel.set_layout('rows')
        key = self.ui.workbench_panel.open_call(['encode_b'], pane='secondary')
        self.ui.canvas_host.apply_selection(key, ['call:project'], intentional=True)
        self.ui._focus_document(key)
        self.ui._select_probe_output('project', app_data='out')
        return key

    def test_invalid_form_auto_and_exit_save_keep_v3_layout_and_last_experiment(self):
        key = self.split()
        expected = deepcopy(self.ui.experiment)
        self.ui.experiment_mode = 'csv'
        dpg.set_value('native_csv_features', '{invalid json')
        self.ui._save_draft()
        path = Path(self.directory.name) / 'workspace.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(payload['workspace_version'], 3)
        self.assertEqual(payload['experiment'], expected)
        self.assertEqual(payload['workbench']['layout'], 'rows')
        self.assertEqual(payload['workbench']['active_by_pane']['secondary'], key)
        self.ui.shutdown()
        self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['workspace_version'], 3)

    def test_explicit_save_and_reopen_restore_context_and_invalidate_old_tokens(self):
        key = self.split()
        old_token = self.ui.canvas_host.token(key)
        expected = self.ui.workbench.to_payload()
        target = Path(self.directory.name) / 'explicit.json'
        self.ui._json_saved({'file_path_name': str(target)})
        self.ui.workbench_panel.set_layout('single')
        self.ui._json_opened({'file_path_name': str(target)})
        self.assertEqual(self.ui.workbench.to_payload(), expected)
        self.assertFalse(self.ui.canvas_host.accept(old_token))
        self.assertEqual(self.ui._probe_source(), 'call:encode_b/call:project')

    def test_failed_open_and_permission_error_keep_source_draft_and_measurements(self):
        self.split()
        target = Path(self.directory.name) / 'bad.json'; target.write_text('{bad', encoding='utf-8')
        self.ui.python_code = '# unsaved source'; self.ui.python_dirty = True
        result = {'probes': {'call:encode_a': {'mean': 1}}}; self.ui.result = result
        before = deepcopy(self.ui.document.graph)
        with self.assertLogs(level='ERROR'), self.assertRaises(ValueError):
            self.ui._json_opened({'file_path_name': str(target)})
        self.assertEqual(self.ui.document.graph, before)
        backups = list(target.parent.glob('bad-invalid-*.json'))
        self.assertTrue(backups)
        self.assertEqual(backups[0].read_text(encoding='utf-8'), '{bad')
        with patch('native.workspace_state.replace_with_retry', side_effect=PermissionError('locked')):
            with self.assertRaises(PermissionError):
                self.ui._save_draft()
        self.assertEqual(self.ui.python_code, '# unsaved source')
        self.assertTrue(self.ui.python_dirty)
        self.assertIs(self.ui.result, result)
        self.assertEqual(self.ui.draft_signature, '')

    def test_damaged_and_deleted_references_reset_or_sanitize_with_warnings(self):
        key = self.split()
        raw = self.ui.workbench.to_payload()
        raw['views'][1]['selected_refs'] = ['call:gone']
        raw['views'][1]['call_path'] = ['gone']
        restored = WorkbenchState.from_payload(raw, self.ui.document)
        self.assertEqual(restored.views[key].selected_refs, [])
        self.assertEqual(restored.views[key].call_path, [])
        self.assertIsNone(restored.views[key].probe_port)
        self.assertTrue(restored.warnings)
        self.ui._restore_workbench({'layout': []})
        self.assertEqual(list(self.ui.workbench.views), ['root'])
        self.assertTrue(any('workbench' in message[2].lower() for message in self.ui.messages))

    def test_python_and_starter_replacement_drop_old_views_and_tokens(self):
        key = self.split()
        token = self.ui.canvas_host.token(key)
        result = {'metrics': {'loss': 0.1}}; self.ui.result = result
        dpg.set_value('native_python_code', self.ui.document.export_python())
        self.ui._import_python()
        self.assertFalse(self.ui.canvas_host.accept(token))
        self.assertEqual(list(self.ui.workbench.views), ['root'])
        self.assertIs(self.ui.result, result)
        key = self.split(); token = self.ui.canvas_host.token(key)
        self.ui.pending_starter = deepcopy(Document().graph)
        self.ui._replace_starter()
        self.assertFalse(self.ui.canvas_host.accept(token))
        self.assertEqual(list(self.ui.workbench.views), ['root'])

    def test_named_probe_choice_is_live_after_explicit_reopen(self):
        graph = deepcopy(self.ui.document.graph)
        definition = graph['definitions'][0]
        definition['output_mode'] = 'named'
        definition['graph']['outputs'].append({'id': 'projected', 'source': 'call:project'})
        for call in graph['calls']:
            call['inputs'] = {port: source + ':out' if source in ('call:encode_a', 'call:encode_b') else source
                              for port, source in call['inputs'].items()}
        self.ui.document.replace_graph(graph); self.ui._reset_canvas_document()
        self.ui.canvas_host.apply_selection('root', ['call:encode_b'], intentional=True)
        self.ui._select_probe_output('encode_b', app_data='projected')
        target = Path(self.directory.name) / 'named.json'
        self.ui._json_saved({'file_path_name': str(target)})
        self.ui._json_opened({'file_path_name': str(target)})
        self.assertEqual(self.ui.workbench.views['root'].probe_port, 'projected')
        self.assertEqual(self.ui._probe_source(), 'call:encode_b:projected')

    def test_overflow_view_open_backs_up_and_protects_current_workspace_source(self):
        source = Path(self.directory.name) / 'workspace.json'
        model = deepcopy(self.ui.document.graph); model['revision'] = 23
        raw = json.dumps({'workspace_version': 3, 'model': model,
                          'experiment': storage.synthetic_config()})[:-1] + ',"workbench":{"layout":1e309}}'
        source.write_text(raw, encoding='utf-8')
        before = deepcopy(self.ui.document.graph)
        with patch('native.main.logging.exception'), self.assertRaises(ValueError):
            self.ui._json_opened({'file_path_name': str(source)})
        self.assertEqual(self.ui.document.graph, before)
        self.assertIn(source.resolve(), self.ui._protected_save_paths)
        backups = list(source.parent.glob('workspace-invalid-*.json'))
        self.assertTrue(backups)
        self.assertEqual(backups[0].read_text(encoding='utf-8'), raw)
        self.ui._save_draft(); self.ui.shutdown()
        self.assertEqual(source.read_text(encoding='utf-8'), raw)

    def test_any_remaining_read_exception_is_backed_up_before_autosave(self):
        source = Path(self.directory.name) / 'workspace.json'
        model = deepcopy(self.ui.document.graph); model['revision'] = 23
        raw = json.dumps({'workspace_version': 3, 'model': model,
                          'experiment': storage.synthetic_config(), 'workbench': WorkbenchState().to_payload()})
        source.write_text(raw, encoding='utf-8')
        self.ui.python_code = '# still unsaved'; self.ui.python_dirty = True
        result = {'probes': {'input:a': {'shape': [2, 4]}}}; self.ui.result = result
        with patch('native.ui.read_workspace_full', side_effect=RecursionError('unanticipated read failure')):
            with patch('native.main.logging.exception'), self.assertRaises(RecursionError):
                self.ui._json_opened({'file_path_name': str(source)})
        self.assertIn(source.resolve(), self.ui._protected_save_paths)
        backups = list(source.parent.glob('workspace-invalid-*.json'))
        self.assertTrue(backups)
        self.assertEqual(backups[0].read_text(encoding='utf-8'), raw)
        self.ui._save_draft(); self.ui.shutdown()
        self.assertEqual(source.read_text(encoding='utf-8'), raw)
        self.assertEqual(self.ui.python_code, '# still unsaved')
        self.assertTrue(self.ui.python_dirty)
        self.assertIs(self.ui.result, result)

    def test_salvaged_deep_view_preserves_source_until_explicit_manual_save(self):
        source = Path(self.directory.name) / 'workspace.json'
        model = deepcopy(self.ui.document.graph); model['revision'] = 23
        model['name'] = 'Recovered architecture'
        experiment = {**storage.synthetic_config(), 'lr': 0.01}
        raw = json.dumps({'workspace_version': 3, 'model': model, 'experiment': experiment})[:-1]
        raw += ',"workbench":' + '[' * 600 + '0' + ']' * 600 + '}'
        source.write_text(raw, encoding='utf-8')
        self.ui._json_opened({'file_path_name': str(source)})
        self.assertEqual(self.ui.document.graph['name'], 'Recovered architecture')
        self.assertEqual(self.ui.experiment['lr'], 0.01)
        self.assertEqual(list(self.ui.workbench.views), ['root'])
        self.assertIn(source.resolve(), self.ui._protected_save_paths)
        self.assertTrue(any('自动保存' in message[2] for message in self.ui.messages))
        self.ui._save_draft()
        self.assertEqual(source.read_text(encoding='utf-8'), raw)
        recovered_draft, _ = storage.read_workspace(source.with_name('draft.json'))
        self.assertEqual(recovered_draft.graph['name'], 'Recovered architecture')
        self.ui._json_saved({'file_path_name': str(source)})
        self.assertNotIn(source.resolve(), self.ui._protected_save_paths)
        repaired = json.loads(source.read_text(encoding='utf-8'))
        self.assertEqual(repaired['workbench']['views'][0]['view_id'], 'root')
        self.ui.workbench_panel.set_layout('rows')
        self.ui._save_draft(); self.ui.shutdown()
        self.assertEqual(json.loads(source.read_text(encoding='utf-8'))['workbench']['layout'], 'rows')

    def test_startup_damaged_view_protection_survives_loader_and_exit(self):
        paths = create_runtime_paths(Path(self.directory.name) / 'recovered-startup')
        model = Document().graph; model['revision'] = 23
        model['name'] = 'Startup recovery model'
        raw = json.dumps({'workspace_version': 3, 'model': model,
                          'experiment': storage.synthetic_config(), 'workbench': {'views': 'damaged'}})
        paths['workspace'].write_text(raw, encoding='utf-8')
        document, experiment, payload = startup.load_saved_workspace_full(paths)
        self.assertEqual(document.graph['revision'], 23)
        self.assertIn(paths['workspace'].resolve(), document.workspace_protected_paths)
        self.assertTrue(document.workspace_load_warnings)
        self.ui.shutdown()
        self.ui = NativeWorkbench(document, None, paths['data'], resource_root(),
                                  experiment=experiment, workbench_payload=payload)
        self.ui.build()
        self.assertEqual(self.ui.document.graph['name'], 'Startup recovery model')
        self.assertEqual(list(self.ui.workbench.views), ['root'])
        self.ui._save_draft(); self.ui.shutdown()
        self.assertEqual(paths['workspace'].read_text(encoding='utf-8'), raw)


if __name__ == '__main__':
    unittest.main()
