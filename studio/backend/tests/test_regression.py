import csv
import math
import tempfile
import unittest
from pathlib import Path

from studio_core.classification_data import inspect_dataset
from studio_core.classification_runtime import run_supervised
from studio_core.experiment_spec import checkpoint_signature, validate_supervised_config
from native.workspace_state import starter_graph


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'samples.csv'
        self.rows = [(i, i * 2 - 7, i // 2, 'train' if i < 12 else 'val' if i < 16 else 'test') for i in range(20)]
        self.write()
        self.spec = {'kind': 'regression_csv', 'path': str(self.path), 'target_column': 'target',
                     'feature_columns': ['x'], 'group_column': 'group', 'split_column': 'split'}
        self.config = {'profile': 'regression-v1', 'data': self.spec, 'epochs': 2,
                       'batch_size': 4, 'lr': 0.05, 'seed': 10}

    def write(self):
        with self.path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['x', 'target', 'group', 'split'])
            writer.writerows(self.rows)

    def test_frozen_partition_scaling_and_leakage(self):
        manifest = inspect_dataset(self.spec)
        self.assertEqual(manifest['counts'], {'train': 12, 'val': 4, 'test': 4})
        self.assertEqual(manifest['preprocessing']['mean'], [5.5])
        self.assertEqual(manifest['records'][0]['label'], -7.0)
        automatic = dict(self.spec, split_column=None)
        first = inspect_dataset(automatic, seed=4)
        self.assertEqual(first, inspect_dataset(automatic, seed=4))
        groups = {}
        for record in first['records']:
            self.assertEqual(groups.setdefault(record['group'], record['split']), record['split'])
        self.rows[-1] = (19, 4, 0, 'test')
        self.write()
        with self.assertRaisesRegex(ValueError, 'Cross-split group'):
            inspect_dataset(self.spec)
        self.rows[-1] = (19, 'inf', 9, 'test')
        self.write()
        with self.assertRaisesRegex(ValueError, 'numeric target'):
            inspect_dataset(self.spec)

    def test_fit_resume_and_independent_metrics(self):
        import torch
        torch.set_num_threads(1)
        manifest = inspect_dataset(self.spec, seed=10)
        graph = starter_graph(manifest)
        events = []
        result = run_supervised(graph, self.config, self.root / 'full', progress=events.append)
        self.assertEqual(result['adapter']['id'], 'regression')
        self.assertNotIn('accuracy', result['metrics']['test'])
        self.assertIn('val_rmse', next(e for e in events if e['phase'] == 'epoch'))
        with Path(result['artifacts']['predictions']).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        target = [float(row['target']) for row in rows]
        pred = [float(row['prediction']) for row in rows]
        errors = [p - t for p, t in zip(pred, target)]
        squared = sum(e*e for e in errors)
        mean = sum(target) / len(target)
        metrics = result['metrics']['test']
        self.assertAlmostEqual(metrics['mae'], sum(map(abs, errors)) / len(errors))
        self.assertAlmostEqual(metrics['rmse'], math.sqrt(squared / len(errors)))
        self.assertAlmostEqual(metrics['r2'], 1 - squared / sum((t-mean)**2 for t in target))
        one = run_supervised(graph, dict(self.config, epochs=1), self.root / 'one')
        continued = run_supervised(graph, dict(self.config, resume_checkpoint=one['artifacts']['last_checkpoint']), self.root / 'continued')
        self.assertEqual(continued['history'], result['history'])
        self.assertEqual(continued['metrics'], result['metrics'])
        wrong = dict(graph, outputs=[dict(graph['outputs'][0], id='logits')])
        with self.assertRaisesRegex(ValueError, 'named prediction'):
            run_supervised(wrong, self.config, self.root / 'wrong')
        valid = validate_supervised_config(self.config)
        self.assertNotEqual(checkpoint_signature(graph, valid, 'same'),
                            checkpoint_signature(graph, dict(valid, profile='classification-v2'), 'same'))

    def test_constant_test_target_has_undefined_r2(self):
        import torch
        torch.set_num_threads(1)
        self.rows = [(x, 4 if split == 'test' else y, g, split) for x, y, g, split in self.rows]
        self.write()
        manifest = inspect_dataset(self.spec)
        result = run_supervised(starter_graph(manifest), self.config, self.root / 'constant')
        self.assertIsNone(result['metrics']['test']['r2'])
