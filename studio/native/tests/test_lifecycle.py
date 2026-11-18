import json
import multiprocessing
import tempfile
import time
import unittest
from pathlib import Path

from native.main import close_workers, main
from native.paths import create_runtime_paths
from native.document import Document
from native.workspace_state import save_workspace


def sleeping_child():
    time.sleep(30)


class LifecycleTests(unittest.TestCase):
    def test_saved_workspace_precedes_legacy_draft_and_restores_experiment(self):
        from native.main import load_saved_workspace
        with tempfile.TemporaryDirectory() as temporary:
            paths=create_runtime_paths(temporary)
            document=Document()
            config={'profile':'classification-v2','data':{'path':'chosen.csv'},'device':'cpu'}
            save_workspace(Path(temporary)/'workspace.json',document,config)
            legacy=Document();legacy.graph['name']='legacy graph'
            legacy.save_json(paths['draft'])
            restored,experiment=load_saved_workspace(paths)
            self.assertEqual(restored.graph,document.graph)
            self.assertEqual(experiment,config)

    def test_invalid_workspace_is_preserved_and_legacy_draft_can_be_recovered(self):
        from native.main import load_saved_workspace
        with tempfile.TemporaryDirectory() as temporary:
            paths=create_runtime_paths(temporary)
            invalid=Path(temporary)/'workspace.json';invalid.write_text('{bad',encoding='utf-8')
            document=Document();document.graph['name']='legacy recovery';document.save_json(paths['draft'])
            with self.assertLogs(level='ERROR'):
                restored,experiment=load_saved_workspace(paths)
            self.assertEqual(restored.graph['name'],'legacy recovery')
            self.assertEqual(experiment['steps'],20)
            self.assertEqual(len(list(Path(temporary).glob('workspace-invalid-*.json'))),1)
            self.assertEqual(invalid.read_text(),'{bad')

    def test_cleanup_stops_owned_process_even_if_persistence_close_fails(self):
        process = multiprocessing.get_context('spawn').Process(target=sleeping_child)
        process.start()
        class BrokenManager:
            runs = {'test': {'_process': process}}
            def close(self):
                raise OSError('simulated disk failure')
        try:
            with self.assertRaisesRegex(OSError, 'simulated disk failure'):
                close_workers(BrokenManager())
            self.assertFalse(process.is_alive(), 'Cleanup must still terminate the owned child')
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)

    def test_initialization_failure_produces_a_diagnostic_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            blocked = directory / 'not-a-directory'
            blocked.write_text('existing data')
            report = directory / 'report.json'
            with self.assertLogs(level='ERROR'):
                self.assertEqual(main(['--data-dir', str(blocked), '--self-test', str(report)]), 1)
            self.assertEqual(json.loads(report.read_text(encoding='utf-8'))['status'], 'failed')
            self.assertEqual(blocked.read_text(), 'existing data')


if __name__ == '__main__':
    unittest.main()
