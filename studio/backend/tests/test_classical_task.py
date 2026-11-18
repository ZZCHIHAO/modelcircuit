"""CPU SVM execution and isolated fit/validation/test coverage."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest

from server import RunManager
from studio_core.classical_task import classical_contract, classical_graph_contract, contains_classical, run_classical
from studio_core.classification_data import inspect_dataset


def make_graph(kind):
    return {'schema_version': 1, 'resource_profile': 'classification-v2', 'id': 'svm_example', 'revision': 1,
            'inputs': [{'id': 'x', 'shape': [None, 2], 'dtype': 'float32'}],
            'instances': [{'id': 'svm', 'kind': kind, 'params': {'C': 1.1, 'kernel': 'linear'}}],
            'calls': [{'id': 'svm_call', 'instance_id': 'svm', 'inputs': {'x': 'input:x'}}],
            'outputs': [{'id': 'prediction', 'source': 'call:svm_call:out'}], 'definitions': []}


def fixture(root, kind):
    config = {'profile': 'classification-v2' if kind == 'SVC' else 'regression-v1',
              'data': {'kind': 'csv' if kind == 'SVC' else 'regression_csv', 'path': str(root / (kind + '.csv')),
                       'target_column': 'target', 'feature_columns': ['one', 'two'], 'split_column': 'split'}}
    with Path(config['data']['path']).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['one', 'two', 'target', 'split'])
        for index in range(36):
            first = index / 10 if index % 2 else -index / 10 - 0.1
            second = (index % 7) / 5
            target = ('positive' if first > 0 else 'negative') if kind == 'SVC' else 2 * first + second
            writer.writerow([first, second, target, 'train' if index < 24 else 'val' if index < 30 else 'test'])
    return config


class ClassicalContractTests(unittest.TestCase):
    def test_rejects_unsupported_device_task_and_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            graph, config = make_graph('SVC'), fixture(Path(directory), 'SVC')
            self.assertEqual(classical_graph_contract(graph)['kind'], 'SVC')
            self.assertEqual(classical_contract(graph, config)['kind'], 'SVC')
            for kwargs in ({'device': 'cuda:0'}, {'resume_checkpoint': 'old.pt'}, {'profile': 'regression-v1'}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    classical_contract(graph, dict(config, **kwargs))
            nested = dict(graph, instances=[], definitions=[{'id': 'chip', 'graph': graph}])
            self.assertTrue(contains_classical(nested))
            with self.assertRaisesRegex(ValueError, 'one root'):
                classical_contract(nested, config)


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'scikit-learn runtime is not installed')
class ClassicalExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_fit_evaluation_and_artifacts_match_direct_sklearn(self):
        import numpy as np
        from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, root_mean_squared_error
        from sklearn.svm import SVC, SVR

        for kind, constructor in [('SVC', SVC), ('SVR', SVR)]:
            with self.subTest(kind=kind):
                config = fixture(self.root, kind)
                events = []
                result = run_classical(make_graph(kind), config, self.root / kind, progress=events.append)
                self.assertEqual(result['algorithm'], kind)
                self.assertEqual(result['execution'], 'classical')
                self.assertEqual(result['history'], [])
                self.assertEqual(result.get('parameters', []), [])
                self.assertEqual([event['phase'] for event in events], ['inspect', 'fit', 'train', 'val', 'test', 'complete'])
                self.assertEqual(set(result['artifacts']), {'manifest', 'run_spec', 'metrics', 'predictions'})
                paths = {key: Path(path) for key, path in result['artifacts'].items()}
                manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))
                summary = json.loads(paths['metrics'].read_text(encoding='utf-8'))
                self.assertEqual(summary['metrics'], result['metrics'])
                with paths['predictions'].open(newline='', encoding='utf-8') as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 6)
                mean, std = manifest['preprocessing']['mean'], manifest['preprocessing']['std']
                def xy(split):
                    rows = [record for record in manifest['records'] if record['split'] == split]
                    return (np.asarray((np.asarray([record['features'] for record in rows]) - mean) / std, dtype=np.float32),
                            np.asarray([record['label'] for record in rows]))
                estimator = constructor(C=1.1, kernel='linear').fit(*xy('train'))
                test_x, test_y = xy('test')
                expected = estimator.predict(test_x)
                np.testing.assert_allclose([float(row['prediction']) for row in rows], expected, rtol=0, atol=1e-12)
                actual = result['metrics']['test']
                self.assertEqual(actual['count'], len(test_y))
                if kind == 'SVC':
                    self.assertEqual(actual['accuracy'], accuracy_score(test_y, expected))
                else:
                    self.assertAlmostEqual(actual['mae'], mean_absolute_error(test_y, expected))
                    self.assertAlmostEqual(actual['rmse'], root_mean_squared_error(test_y, expected))
                    self.assertAlmostEqual(actual['r2'], r2_score(test_y, expected))

    def test_dataset_drift_and_cancellation(self):
        config = fixture(self.root, 'SVR')
        graph = make_graph('SVR')
        manifest = inspect_dataset(config['data'])
        path = Path(config['data']['path'])
        path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Dataset digest changed'):
            run_classical(graph, dict(config, expected_dataset_digest=manifest['dataset_digest']), self.root / 'drift')
        with self.assertRaises(InterruptedError):
            run_classical(graph, config, self.root / 'cancel', cancelled=lambda: True)
        self.assertFalse((self.root / 'cancel').exists())

    def test_coordinator_executes_classical_worker(self):
        config = fixture(self.root, 'SVC')
        manager = RunManager(self.root / 'runs')
        self.addCleanup(manager.close)
        submitted = manager.submit(make_graph('SVC'), config)
        self.assertEqual(submitted['execution'], 'classical')
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            record = manager.get(submitted['run_id'])
            if record['status'] in ('succeeded', 'failed', 'cancelled'):
                break
            time.sleep(0.04)
        self.assertEqual(record['status'], 'succeeded', record['error'])
        self.assertEqual(record['result']['metrics']['test']['accuracy'], 1.0)
