"""Project runs can be relocated, audited and compared on locked data."""
import json
from pathlib import Path
import tempfile
import unittest

from studio_core.project_runner import compare_project_runs, read_project_run, train_project, predict_project


class ProjectRunTests(unittest.TestCase):
    def test_compares_only_equal_evaluation_protocols(self):
        origin = {'operation': 'train', 'status': 'succeeded', 'profile': 'classification-v2',
                  'dataset_digest': 'a' * 64, 'evaluation': 'classification-v2-test-v1',
                  'metrics': {'test': {'accuracy': .25, 'count': 4}}}
        latest = {**origin, 'metrics': {'test': {'accuracy': .75, 'count': 4}}}
        self.assertEqual(compare_project_runs(origin, latest)['metrics']['accuracy']['delta'], .5)
        for changed in ({'dataset_digest': 'b' * 64}, {'evaluation': 'other'}, {'operation': 'predict'}):
            self.assertFalse(compare_project_runs(origin, {**latest, **changed})['comparable'])

    def test_report_rejects_absolute_and_outside_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / 'output' / 'project-run.json'
            report.parent.mkdir()
            outside = root / 'not-a-result.json'
            outside.write_text('{}', encoding='utf-8')
            base = {'format': 'modelcircuit-project-run', 'version': 1, 'run_id': 'x',
                    'status': 'succeeded', 'operation': 'train'}
            for name in ('../not-a-result.json', str(outside)):
                report.write_text(json.dumps({**base, 'artifacts': {'escape': name}}), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Artifact path'):
                    read_project_run(report)

    def test_report_verification_filters_changed_metrics_and_missing_segmentation_mask(self):
        import hashlib

        def digest(value):
            return hashlib.sha256(value).hexdigest()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inference = root / 'inference.json'
            inference.write_bytes(b'{}')
            metrics = root / 'metrics.json'
            metrics.write_bytes(b'{"wrong":true}')
            report = root / 'project-run.json'
            report.write_text(json.dumps({'format': 'modelcircuit-project-run', 'version': 1, 'run_id': 'x',
                'status': 'succeeded', 'operation': 'train', 'dataset_digest': 'a' * 64,
                'profile': 'segmentation-v1', 'evaluation': 'segmentation-v1-test-v1',
                'metrics': {'test': {'count': 1, 'foreground_dice': .5}},
                'artifacts': {'metrics': metrics.name, 'inference': inference.name},
                'artifact_sha256': {'metrics': digest(b'{}'), 'inference': digest(b'{}')},
                'output_sha256': {'masks/000001.nii.gz': digest(b'fake')}}), encoding='utf-8')
            checked = read_project_run(report, verify=True)
            self.assertEqual(checked['changed_artifacts'], ['metrics'])
            self.assertEqual(checked['missing_outputs'], ['masks/000001.nii.gz'])
            self.assertEqual(checked['artifacts'], {})
            self.assertFalse(compare_project_runs(checked, checked)['comparable'])

    def test_different_training_seeds_can_use_same_fixed_test_protocol(self):
        from studio_core.classification_data import inspect_dataset
        from studio_core.project_examples import create_example_project
        from studio_core.project_runner import evaluation_dataset_digest
        from studio_core.research_project import bind_project, portable_dataset, read_project

        with tempfile.TemporaryDirectory() as tmp:
            path, data, _ = create_example_project(Path(tmp) / 'source')
            project = read_project(path)
            config, original = bind_project(project, data)
            other = inspect_dataset(config['data'], seed=config['seed'] + 1)
            a, b = portable_dataset(original), portable_dataset(other)
            self.assertNotEqual(a['digest'], b['digest'])
            self.assertEqual(evaluation_dataset_digest(a), evaluation_dataset_digest(b))
            base = {'operation': 'train', 'status': 'succeeded', 'profile': 'classification-v2',
                    'evaluation': 'classification-v2-test-v1', 'metrics': {'test': {'accuracy': .5, 'count': 4}}}
            rows = [{**base, 'dataset_digest': lock['digest'], 'evaluation_dataset_digest': evaluation_dataset_digest(lock)}
                    for lock in (a, b)]
            self.assertTrue(compare_project_runs(*rows)['comparable'])

    def test_unverified_regression_report_cannot_be_compared(self):
        origin = {'operation': 'train', 'status': 'succeeded', 'profile': 'regression-v1',
                  'dataset_digest': 'a' * 64, 'evaluation': 'table-csv-regression-v1-heldout-v1',
                  'metrics': {'test': {'mae': 2.0, 'rmse': 2.5, 'r2': None, 'count': 3}}}
        latest = {**origin, 'metrics': {'test': {'mae': 1.5, 'rmse': 2.0, 'r2': .5, 'count': 3}}}
        self.assertFalse(compare_project_runs(origin, latest)['comparable'])

    def test_reference_example_runs_in_new_paths_and_predicts_with_registered_weights(self):
        from shutil import copytree
        import torch
        from studio_core.project_examples import create_example_project

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, old_data, _ = create_example_project(root / 'source')
            copytree(root / 'source', root / 'copy')
            data = root / 'copy' / Path(old_data).relative_to(root / 'source')
            copied = root / 'copy' / source.relative_to(root / 'source')
            first = train_project(source, old_data, root / 'first')
            second = train_project(copied, data, root / 'second')
            original = read_project_run(first['artifacts']['project_run'])
            repeated = read_project_run(second['artifacts']['project_run'])
            self.assertEqual(original['dataset_digest'], repeated['dataset_digest'])
            self.assertEqual(original['model_digest'], repeated['model_digest'])
            self.assertEqual(original['metrics'], repeated['metrics'])
            self.assertTrue(compare_project_runs(original, repeated)['comparable'])
            sample = next(data.rglob('*.png'))
            result = predict_project(copied, sample, root / 'inference', weights=repeated['path'])
            inference = read_project_run(result['artifacts']['project_run'])
            self.assertEqual(inference['operation'], 'predict')
            self.assertEqual(inference['cases'], 1)
            self.assertTrue(inference['artifacts']['predictions'].is_file())
            self.assertEqual(inference['checkpoint_sha256'], repeated['artifact_sha256']['best_checkpoint'])
            with self.assertRaisesRegex(ValueError, 'new run directory'):
                train_project(copied, data, root / 'second')

    def test_invalid_binding_records_a_failed_run(self):
        from studio_core.project_examples import create_example_project

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, _, _ = create_example_project(root / 'source')
            with self.assertRaises(ValueError):
                train_project(project, root / 'missing', root / 'failed')
            report = read_project_run(root / 'failed')
            self.assertEqual(report['status'], 'failed')
            self.assertTrue(report['error'])
            self.assertEqual(set(report['artifacts']), {'progress_log'})
