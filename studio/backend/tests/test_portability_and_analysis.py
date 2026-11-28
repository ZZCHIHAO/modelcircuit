from copy import deepcopy
import csv
from pathlib import Path
import shutil
import tempfile
import unittest

from studio_core.classification_data import inspect_dataset
from studio_core.debug_runtime import run_debug
from studio_core.project_runner import train_project
from studio_core.research_project import bind_project, export_project, read_project
from native.workspace_state import starter_graph


class PortableNumericTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fixture(self, regression=False):
        path = self.root / ('regression.csv' if regression else 'classification.csv')
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['x', 'y', 'label', 'split'])
            for i in range(12):
                writer.writerow([i, i + 20, i / 2 if regression else str(i % 2),
                                 ('train', 'val', 'test')[i // 4]])
        config = {'profile': 'regression-v1' if regression else 'classification-v2',
                  'data': {'kind': 'regression_csv' if regression else 'csv', 'path': str(path),
                           'target_column': 'label', 'feature_columns': ['x', 'y'],
                           'split_column': 'split', 'standardize': False},
                  'batch_size': 2, 'epochs': 1, 'seed': 9}
        manifest = inspect_dataset(config['data'], config['seed'])
        return path, config, manifest, starter_graph(manifest)

    def test_numeric_projects_reopen_after_a_move_without_the_original_csv(self):
        for regression in (False, True):
            with self.subTest(regression=regression):
                source, config, manifest, graph = self.fixture(regression)
                key = source.stem
                initial = export_project(self.root / key, graph, config, include_data=True)
                relocated = self.root / (key + '-relocated')
                shutil.move(initial.parent, relocated)
                source.unlink()
                project = read_project(relocated)
                self.assertEqual(project['graph']['inputs'][0]['shape'], [None, 2])
                bound, restored = bind_project(project, None)
                self.assertEqual(restored['counts'], {'train': 4, 'val': 4, 'test': 4})
                self.assertNotEqual(restored['dataset_digest'], manifest['dataset_digest'])
                self.assertTrue(Path(bound['data']['path']).is_relative_to(relocated))
                result = train_project(relocated, None, self.root / (key + '-run'), torch_threads=1)
                self.assertEqual(result['status'], 'succeeded')
                Path(bound['data']['path']).write_text('x,y,label,split\n', encoding='utf-8')
                with self.assertRaises(ValueError):
                    bind_project(project, None)

    def test_fixed_csv_rows_keep_requested_order_and_are_comparable_between_graphs(self):
        from studio_core.debug_spec import compare_debug_results

        _, config, manifest, graph = self.fixture()
        expected = [manifest['records'][i]['id'] for i in (3, 1)]
        options = {'mode': 'step', 'experiment': config, 'sample_ids': expected, 'batch_size': 2,
                   'seed': 117, 'sources': ['input:x']}
        first = run_debug(graph, options)
        self.assertEqual(first['sample_ids'], expected)
        self.assertEqual(first['probes']['input:x']['activation']['values'], [3.0, 23.0, 1.0, 21.0])
        self.assertEqual(first['optimizer_steps'], 1)
        changed = deepcopy(graph)
        changed['instances'].append({'id': 'gain', 'kind': 'Scale', 'params': {'factor': 2.0}})
        changed['calls'].append({'id': 'gain', 'instance_id': 'gain',
                                 'inputs': {'x': changed['outputs'][0]['source']}})
        changed['outputs'][0]['source'] = 'call:gain:out'
        second = run_debug(changed, options)
        comparison = compare_debug_results(first, second)
        self.assertTrue(comparison['comparable'])
        self.assertFalse(comparison['same_model'])
        self.assertNotAlmostEqual(comparison['loss_delta'], 0.0, places=6)
        shifted = run_debug(graph, {**options, 'sample_ids': [manifest['records'][i]['id'] for i in (2, 1)]})
        self.assertFalse(compare_debug_results(first, shifted)['comparable'])
        with self.assertRaises(ValueError):
            run_debug(graph, {**options, 'sample_ids': [manifest['records'][4]['id']]})

    def test_unlabeled_pca_and_kmeans_match_independent_fit_and_travel_together(self):
        import json
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from threadpoolctl import threadpool_limits
        from studio_core.unsupervised_task import run_unsupervised, unsupervised_graph

        rows = [[-4.0, 0.0], [-3.0, 1.0], [-5.0, 0.5], [4.0, 0.0], [3.0, -1.0], [5.0, -0.5]]
        source = self.root / 'points.csv'
        with source.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['u', 'v'])
            writer.writerows(rows)
        config = {'profile': 'unsupervised-v1', 'seed': 3,
                  'data': {'kind': 'unsupervised_csv', 'path': str(source), 'feature_columns': ['u', 'v'],
                           'standardize': False}}
        manifest = inspect_dataset(config['data'], config['seed'])
        for algorithm in ('PCA', 'KMeans', 'PCA+KMeans'):
            with self.subTest(algorithm=algorithm):
                graph = unsupervised_graph(manifest, algorithm)
                fitted = run_unsupervised(graph, config, self.root / algorithm)
                if 'PCA' in algorithm:
                    with threadpool_limits(limits=1):
                        pca = PCA(n_components=2, svd_solver='full').fit(rows)
                    np.testing.assert_allclose(fitted['analysis']['explained_variance_ratio'], pca.explained_variance_ratio_)
                if 'KMeans' in algorithm:
                    features = pca.transform(rows) if 'PCA' in algorithm else rows
                    with threadpool_limits(limits=1):
                        own = KMeans(n_clusters=2, n_init=10, random_state=3).fit(features)
                    self.assertAlmostEqual(fitted['metrics']['fit']['inertia'], own.inertia_)
                    self.assertEqual(sorted(fitted['analysis']['cluster_counts']), [3, 3])
        project = export_project(self.root / 'portable', graph, config, include_data=True)
        relocated = self.root / 'new-device'
        shutil.move(project.parent, relocated)
        source.unlink()
        decoded = read_project(relocated)
        self.assertIsNone(decoded['graph_error'])
        self.assertEqual([item['kind'] for item in decoded['graph']['instances']], ['PCA', 'KMeans'])
        run = train_project(relocated, None, self.root / 'replayed')
        report = json.loads(Path(run['artifacts']['project_run']).read_text(encoding='utf-8'))
        self.assertEqual(report['metrics']['fit']['count'], 6)
        self.assertEqual((self.root / 'replayed' / 'projection.csv').read_bytes(),
                         Path(fitted['artifacts']['projection']).read_bytes())
