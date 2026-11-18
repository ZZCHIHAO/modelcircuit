"""S3 headless acceptance traverses the same native model/data/worker APIs."""
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def acceptance():
    try:
        return importlib.import_module('native.s3_acceptance')
    except ModuleNotFoundError as exc:
        raise AssertionError('Native S3 acceptance is not implemented') from exc


class S3AcceptanceTests(unittest.TestCase):
    def test_full_native_s3_no_edit_python_roundtrip_preserves_training_identity(self):
        from studio_core.classification_data import inspect_dataset
        from studio_core.experiment_spec import checkpoint_signature, default_classification_config, model_digest
        from studio_core.python_io import import_python
        with tempfile.TemporaryDirectory() as temporary:
            fixture = acceptance().create_paired_fixture(temporary)
            manifest = inspect_dataset(fixture['data_spec'])
            document, _ = acceptance().build_s3_document(manifest)
            restored = import_python(document.export_python())
            self.assertEqual(model_digest(document.graph), model_digest(restored))
            config = default_classification_config('paired_csv')
            config['data'] = fixture['data_spec']
            self.assertEqual(checkpoint_signature(document.graph, config, manifest['dataset_digest']), checkpoint_signature(restored, config, manifest['dataset_digest']))

    def test_fixture_and_document_helpers_need_no_gui_or_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = "import sys; from native.s3_acceptance import create_paired_fixture,build_s3_document; from studio_core.classification_data import inspect_dataset; f=create_paired_fixture(sys.argv[1]); d,m=build_s3_document(inspect_dataset(f['data_spec'])); assert d.validation()['valid']; assert len(d.output_ports(m['call_id']))==2; assert 'torch' not in sys.modules; assert not any(k.startswith('dearpygui') for k in sys.modules)"
            completed = subprocess.run([sys.executable, '-c', script, temporary], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_actual_manager_runs_paired_chip_resume_python_and_storage_acceptance(self):
        from server import RunManager
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); manager = RunManager(directory / 'runs')
            try:
                report = acceptance().check_s3_runtime(manager, directory / 'acceptance', device='cpu')
                self.assertEqual(report['status'], 'passed')
                self.assertEqual(report['kind'], 'native-s3-runtime')
                self.assertTrue(report['test_fixtures'])
                self.assertEqual(report['device'], 'cpu')
                self.assertFalse(report['gpu_execution']['executed'])
                self.assertEqual(report['training']['completed_epochs'], 3)
                self.assertTrue(report['resume']['bit_exact'])
                self.assertTrue(report['checkpoint_inference']['predictions_equal'])
                self.assertEqual(report['checkpoint_inference']['count'], 6)
                self.assertTrue(report['sharing']['same_instance'])
                self.assertEqual(len(report['sharing']['call_ids']), 2)
                self.assertEqual(report['python_round_trip']['profile'], 'ports-pytorch-v3')
                self.assertTrue(report['python_round_trip']['metadata_free_close'])
                self.assertTrue(report['python_round_trip']['standalone_close'])
                self.assertTrue(report['storage']['workspace_equal'])
                self.assertTrue(report['storage']['chip_reimport_reused'])
                self.assertTrue(report['storage']['fresh_library_forward_close'])
                self.assertTrue(all(Path(path).is_file() for path in report['artifacts'].values()))
                self.assertEqual(json.loads(Path(report['report_path']).read_text(encoding='utf-8')), report)
                self.assertEqual(len(manager.runs), 3)
                self.assertFalse(manager.closed, 'Caller retains responsibility for closing workers')
                json.dumps(report, allow_nan=False)
            finally:
                manager.close()


if __name__ == '__main__': unittest.main()
