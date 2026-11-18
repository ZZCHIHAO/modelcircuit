"""S4 diagnostics fail closed, preserve reports, and release owned resources."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


def acceptance():
    try:
        return importlib.import_module('native.s4_acceptance')
    except ModuleNotFoundError as error:
        raise AssertionError('Production S4 diagnostic is missing') from error


class S4AcceptanceTests(unittest.TestCase):
    def test_python_metadata_free_contract_checks_execution_and_detects_parameter_changes(self):
        from copy import deepcopy
        from studio_core.python_io import import_python
        module = acceptance()
        document = module.build_s4_document()
        clean = '\n'.join(line for line in document.export_python().splitlines() if not line.startswith('STUDIO_VISUAL_METADATA = '))
        restored = import_python(clean)
        self.assertEqual(module._execution_contract(restored), module._execution_contract(document.graph))
        changed = deepcopy(restored)
        primitive = next(instance for definition in changed['definitions'] for instance in definition['graph']['instances'] if instance['kind'] == 'BatchNorm2d')
        primitive['params']['eps'] = .01
        self.assertNotEqual(module._execution_contract(changed), module._execution_contract(document.graph))

    def test_fixture_cleanup_retries_transient_windows_lock(self):
        module = acceptance()
        class LockedOnce:
            def __init__(self):
                self.attempts = 0
            def cleanup(self):
                self.attempts += 1
                if self.attempts == 1:
                    raise PermissionError('temporary scanner lock')
        fixture = LockedOnce()
        module._cleanup_fixture(fixture)
        self.assertEqual(fixture.attempts, 2)

    def test_resnet_fixture_graph_is_hardware_bounded_without_gui_or_torch_import(self):
        script = ("import sys; from native.s4_acceptance import build_s4_document,check_catalog; "
                  "d=build_s4_document(); r=d.validation(); assert r['valid'],r; "
                  "assert r['parameter_count']>20000000; assert d.graph['resource_profile']=='hardware-v5'; "
                  "assert d.graph['resource_batch_bound']==2; assert len(check_catalog()['model_ids'])==14; "
                  "assert 'torch' not in sys.modules; assert not any(x.startswith('dearpygui') for x in sys.modules)")
        completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_invalid_device_fails_before_submission_and_writes_failure_report(self):
        from server import RunManager
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manager = RunManager(directory / 'runs')
            try:
                for device in ('cuda', 'cuda:-1', 'cpu:0'):
                    with self.assertRaisesRegex(ValueError, 'cpu|cuda'):
                        acceptance().check_s4_runtime(manager, directory, device=device)
                    report = json.loads((directory / 's4-acceptance.json').read_text(encoding='utf-8'))
                    self.assertEqual(report['status'], 'failed')
                    self.assertEqual(report['device'], device)
                    self.assertFalse(manager.runs)
            finally:
                manager.close()

    def test_worker_failure_removes_fixture_and_restores_thread_count(self):
        import torch
        from server import RunManager
        module = acceptance()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manager = RunManager(directory / 'runs')
            previous = torch.get_num_threads()
            try:
                with patch('native.s4_acceptance._wait', side_effect=RuntimeError('worker failed deliberately')):
                    with self.assertRaisesRegex(RuntimeError, 'worker failed deliberately'):
                        module.check_s4_runtime(manager, directory)
                report = json.loads((directory / 's4-acceptance.json').read_text(encoding='utf-8'))
                self.assertEqual(report['status'], 'failed')
                self.assertTrue(report['fixtures_cleaned'])
                self.assertFalse(Path(report['fixture_directory']).exists())
                self.assertEqual(torch.get_num_threads(), previous)
                self.assertFalse(manager.closed, 'The caller owns RunManager shutdown')
            finally:
                manager.close()

    def test_cli_dispatch_failure_still_closes_workers_and_reports_device(self):
        from native.main import main
        acceptance()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = directory / 'report.json'
            with patch('native.s4_acceptance.check_s4_runtime', side_effect=ValueError('selected cuda:999 unavailable')):
                self.assertEqual(main(['--data-dir', str(directory / 'data'), '--s4-check', str(report),
                                       '--check-device', 'cuda:999']), 1)
            result = json.loads(report.read_text(encoding='utf-8'))
            self.assertEqual(result['status'], 'failed')
            self.assertTrue(result['workers_stopped'])
            self.assertIn('cuda:999', result['error'])

    def test_s4_modes_are_mutually_exclusive_with_old_diagnostics(self):
        from native.main import main
        with self.assertRaises(SystemExit) as caught:
            main(['--s4-check', 'a.json', '--s3-check', 'b.json'])
        self.assertEqual(caught.exception.code, 2)

    def test_render_build_failure_shuts_down_native_context_and_fixtures(self):
        try:
            module = importlib.import_module('native.s4_render_acceptance')
        except ModuleNotFoundError as error:
            self.fail('Production S4 render diagnostic is missing: ' + str(error))
        from native.paths import resource_root
        from native.ui import NativeWorkbench
        observed = []
        original_shutdown = NativeWorkbench.shutdown
        def shutdown(ui):
            observed.append(Path(ui.data_directory))
            original_shutdown(ui)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(NativeWorkbench, 'build', side_effect=RuntimeError('native build failed')), patch.object(NativeWorkbench, 'shutdown', shutdown):
                with self.assertRaisesRegex(RuntimeError, 'native build failed'):
                    module.check_s4_render(None, directory / 'data', resource_root(), directory / 'out')
            self.assertEqual(len(observed), 1)
            self.assertFalse(observed[0].exists())
            report = json.loads((directory / 'out' / 's4-render-acceptance.json').read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'failed')
            self.assertTrue(report['fixtures_cleaned'])


if __name__ == '__main__':
    unittest.main()
