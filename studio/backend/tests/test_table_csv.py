"""Table CSV fold isolation, metrics and classical interchange."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import time
import unittest

from studio_core.classical_task import run_classical
from studio_core.component_catalog import build_component, list_components
from studio_core.python_io import export_python, import_python
from studio_core.schema import validate_graph


def graph_for(kind):
    return {
        'schema_version': 1, 'id': 'forest', 'revision': 1,
        'inputs': [{'id': 'x', 'shape': [None, 2], 'dtype': 'float32'}],
        'instances': [{'id': 'estimator', 'kind': kind, 'params': {'n_estimators': 7, 'max_depth': 4}}],
        'calls': [{'id': 'apply_estimator', 'instance_id': 'estimator', 'inputs': {'x': 'input:x'}}],
        'outputs': [{'id': 'prediction', 'source': 'call:apply_estimator:out'}], 'definitions': [],
    }


class TableCSVTests(unittest.TestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.root = Path(root.name)

    def fixture(self, task='classification', split_column='partition', *, leak=False, rare_patients=()):
        path = self.root / (task + '.csv')
        rows = {}
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample_id', 'patient_id', 'partition', 'label', 'score', 'site'])
            for patient in range(8):
                split = 'train' if patient < 4 else 'val' if patient < 6 else 'test'
                for visit in range(2):
                    case_id = f'case{patient}-{visit}'
                    value = patient + (900 if split == 'test' else 0)
                    score = None if visit else value
                    label = ('healthy' if patient % 2 == 0 else 'condition') if task == 'classification' else 1 + 2 * value + visit
                    if patient in rare_patients:
                        label = 'rare'
                    group = 'patient0' if leak and patient == 6 and visit == 0 else f'patient{patient}'
                    category = f'site{patient}'
                    writer.writerow([case_id, group, split, label, score, category])
                    rows[case_id] = {'group': group, 'split': split, 'score': score, 'site': category, 'label': label}
        data = {'kind': 'table_csv', 'task': task, 'path': str(path), 'split_column': split_column,
                'numeric_columns': ['score'], 'categorical_columns': ['site'], 'folds': 3}
        return data, rows

    def test_inspection_freezes_explicit_partitions_and_grouped_folds(self):
        from studio_core.table_data import inspect_table_dataset, table_defaults, validate_table_spec

        data, raw = self.fixture()
        draft = table_defaults()
        self.assertEqual(validate_table_spec(draft, editable=True), draft)
        with self.assertRaises(ValueError):
            validate_table_spec(draft)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            validate_table_spec(dict(data, categorical_columns=['patient_id']))
        manifest = inspect_table_dataset(data, seed=27)
        self.assertEqual(manifest['counts'], {'train': 8, 'val': 4, 'test': 4})
        self.assertEqual(manifest['input_shape'], [None, 2])
        self.assertEqual(manifest['class_names'], ['condition', 'healthy'])
        self.assertEqual({row['id'] for row in manifest['records']}, set(raw))
        self.assertEqual(len(manifest['fold_plan']), 3)
        covered = []
        development = {case_id for case_id, row in raw.items() if row['split'] != 'test'}
        for fold in manifest['fold_plan']:
            train, validation = set(fold['train_ids']), set(fold['val_ids'])
            self.assertEqual(train | validation, development)
            self.assertTrue(train.isdisjoint(validation))
            self.assertTrue({raw[i]['group'] for i in train}.isdisjoint(raw[i]['group'] for i in validation))
            covered.extend(validation)
        self.assertCountEqual(covered, development)

    def test_automatic_holdout_repeats_after_moving_csv(self):
        from studio_core.table_data import inspect_table_dataset

        data, raw = self.fixture(split_column=None)
        first = inspect_table_dataset(data, seed=27)
        copy_path = self.root / 'moved.csv'
        shutil.copyfile(data['path'], copy_path)
        moved = inspect_table_dataset(dict(data, path=str(copy_path)), seed=27)
        self.assertEqual(first['dataset_digest'], moved['dataset_digest'])
        self.assertEqual(first['fold_plan'], moved['fold_plan'])
        self.assertEqual(first['counts'], {'train': 12, 'val': 0, 'test': 4})
        by_group = {}
        for row in moved['records']:
            self.assertEqual(row['group'], raw[row['id']]['group'])
            self.assertEqual(by_group.setdefault(row['group'], row['split']), row['split'])

    def test_explicit_partitions_reject_patient_leakage(self):
        from studio_core.table_data import inspect_table_dataset

        data, _ = self.fixture(leak=True)
        with self.assertRaisesRegex(ValueError, 'patient0.*case6-0'):
            inspect_table_dataset(data, seed=27)

    def test_automatic_holdout_reserves_two_patients_per_class_for_fold_training(self):
        from studio_core.table_data import inspect_table_dataset

        for rare_patients in ((3, 4), (3, 4, 5)):
            with self.subTest(rare_patients=rare_patients):
                data, raw = self.fixture(split_column=None, rare_patients=rare_patients)
                manifest = inspect_table_dataset(data, seed=27)
                test_groups = {row['group'] for row in manifest['records'] if row['split'] == 'test'}
                self.assertEqual(len(test_groups), 2)
                rare_groups = {f'patient{patient}' for patient in rare_patients}
                self.assertEqual(len(rare_groups & test_groups), len(rare_patients) - 2)
                self.assertEqual(inspect_table_dataset(data, seed=27)['fold_plan'], manifest['fold_plan'])
                missing_validation_folds = []
                for fold in manifest['fold_plan']:
                    self.assertEqual({raw[i]['label'] for i in fold['train_ids']}, {'condition', 'healthy', 'rare'})
                    self.assertTrue(test_groups.isdisjoint(raw[i]['group'] for i in fold['train_ids'] + fold['val_ids']))
                    if 'rare' not in {raw[i]['label'] for i in fold['val_ids']}:
                        missing_validation_folds.append(fold['fold'])
                self.assertEqual(len(missing_validation_folds), 1)
                self.assertEqual([warning['fold'] for warning in manifest['warnings']
                                  if warning['split'] == 'val' and 'rare' in warning['classes']], missing_validation_folds)

    def test_explicit_development_rejects_a_class_in_only_one_patient(self):
        from studio_core.table_data import inspect_table_dataset

        data, _ = self.fixture(rare_patients=(0, 6))
        with self.assertRaisesRegex(ValueError, 'two development patients.*rare'):
            inspect_table_dataset(data, seed=27)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'scikit-learn runtime is not installed')
    def test_fold_artifact_reports_missing_validation_classes(self):
        data, _ = self.fixture(split_column=None, rare_patients=(3, 4, 5))
        config = {'profile': 'classification-v2', 'data': data, 'seed': 27}
        result = run_classical(graph_for('RandomForestClassifier'), config, self.root / 'sparse-run')
        report = json.loads(Path(result['artifacts']['cv_folds']).read_text(encoding='utf-8'))
        self.assertEqual(report['split_policy']['training_class_coverage'], 'required')
        warnings = [warning for warning in report['warnings'] if warning['split'] == 'val' and 'rare' in warning['classes']]
        self.assertEqual(len(warnings), 1)
        self.assertIn(warnings[0]['fold'], {fold['fold'] for fold in report['folds']})
        self.assertEqual(result['metrics']['test']['count'], 4)

    def test_forest_catalog_and_python_constructor_edits(self):
        ids = {item.id for item in list_components('随机森林', 'classical')}
        self.assertEqual(ids, {'primitive.RandomForestClassifier', 'primitive.RandomForestRegressor'})
        for kind in ('RandomForestClassifier', 'RandomForestRegressor'):
            with self.subTest(kind=kind):
                entry = build_component(kind, {'n_estimators': 12}).entry
                self.assertEqual(entry['params'], {'n_estimators': 12, 'max_depth': None, 'min_samples_leaf': 1})
                self.assertEqual(build_component(kind, {}).entry['params'],
                                 {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1})
                with self.assertRaises(ValueError):
                    build_component(kind, {'min_samples_leaf': False})
                graph = graph_for(kind)
                validate_graph(graph)
                python = export_python(graph)
                updated = import_python(python.replace('n_estimators=7', 'n_estimators=13'))
                self.assertEqual(updated['instances'][0]['params']['n_estimators'], 13)
                self.assertEqual(updated['instances'][0]['params']['max_depth'], 4)
                self.assertEqual(import_python(export_python(updated)), updated)
                with self.assertRaises(ValueError):
                    import_python(python.replace('n_estimators=7', 'n_estimators=int(7)'))

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'scikit-learn runtime is not installed')
    def test_forest_runs_fit_each_fold_only_on_its_training_patients(self):
        for task, profile, kind in [('classification', 'classification-v2', 'RandomForestClassifier'),
                                    ('regression', 'regression-v1', 'RandomForestRegressor')]:
            with self.subTest(task=task):
                data, raw = self.fixture(task)
                config = {'profile': profile, 'data': data, 'seed': 27}
                result = run_classical(graph_for(kind), config, self.root / task)
                folds = json.loads(Path(result['artifacts']['cv_folds']).read_text(encoding='utf-8'))
                self.assertEqual(len(folds['folds']), 3)
                self.assertEqual(folds['final_preprocessing']['numeric_medians'], {'score': 2.5})
                self.assertEqual(folds['final_preprocessing']['categorical_vocabulary'],
                                 {'site': ['site0', 'site1', 'site2', 'site3', 'site4', 'site5']})
                self.assertEqual(result['metrics']['cv']['count'], 12)
                for fold in folds['folds']:
                    training = [raw[case_id] for case_id in fold['train_ids']]
                    validation = [raw[case_id] for case_id in fold['val_ids']]
                    train_scores = [row['score'] for row in training if row['score'] is not None]
                    prep = fold['preprocessing']
                    median = statistics.median(train_scores)
                    self.assertEqual(prep['numeric_medians'], {'score': median})
                    self.assertAlmostEqual(prep['numeric_center']['score'], (sum(train_scores) + median * len(train_scores)) / len(training))
                    expected_sites = sorted({row['site'] for row in training})
                    self.assertEqual(prep['categorical_vocabulary'], {'site': expected_sites})
                    self.assertTrue(set(expected_sites).isdisjoint(row['site'] for row in validation))
                    self.assertEqual(fold['metrics']['count'], len(validation))
                with Path(result['artifacts']['predictions']).open(newline='', encoding='utf-8') as stream:
                    predictions = list(csv.DictReader(stream))
                self.assertCountEqual([row['sample_id'] for row in predictions],
                                      ['case6-0', 'case6-1', 'case7-0', 'case7-1'])
                self.assertEqual(result['metrics']['test']['count'], 4)
                if task == 'classification':
                    self.assertEqual(result['metrics']['test']['accuracy'],
                                     sum(row['label_name'] == row['prediction_name'] for row in predictions) / 4)
                else:
                    self.assertAlmostEqual(result['metrics']['test']['mae'],
                                           sum(abs(float(row['residual'])) for row in predictions) / 4)
                spec = json.loads(Path(result['artifacts']['run_spec']).read_text(encoding='utf-8'))
                self.assertEqual(spec['effective_estimator_params']['random_state'], 27)

    def test_table_submit_rejects_nonclassical_graph_before_starting_worker(self):
        from server import RunManager

        data, _ = self.fixture()
        graph = graph_for('Identity')
        graph['instances'][0]['params'] = {}
        manager = RunManager(self.root / 'runs')
        self.addCleanup(manager.close)
        with self.assertRaisesRegex(ValueError, 'table_csv requires a classical estimator'):
            manager.submit(graph, {'profile': 'classification-v2', 'data': data, 'seed': 27})
        self.assertEqual(manager.runs, {})

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'scikit-learn runtime is not installed')
    def test_managed_worker_fits_categorical_only_train_and_test_csv(self):
        from server import RunManager

        data, _ = self.fixture()
        path = Path(data['path'])
        path.write_text(path.read_text(encoding='utf-8').replace(',val,', ',train,').replace(',site0', ','), encoding='utf-8')
        data.update(numeric_columns=[], standardize=False)
        graph = graph_for('RandomForestClassifier')
        graph['inputs'][0]['shape'] = [None, 1]
        manager = RunManager(self.root / 'runs')
        self.addCleanup(manager.close)
        submitted = manager.submit(graph, {'profile': 'classification-v2', 'data': data, 'seed': 27})
        self.assertEqual(submitted['execution'], 'classical')
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            run = manager.get(submitted['run_id'])
            if run['status'] in ('succeeded', 'failed', 'cancelled'):
                break
            time.sleep(0.04)
        self.assertEqual(run['status'], 'succeeded', run['error'])
        result = run['result']
        self.assertEqual(result['dataset']['counts'], {'train': 12, 'val': 0, 'test': 4})
        cv_folds = json.loads(Path(result['artifacts']['cv_folds']).read_text(encoding='utf-8'))
        self.assertIn(None, cv_folds['final_preprocessing']['categorical_vocabulary']['site'])
        self.assertEqual(cv_folds['final_preprocessing']['numeric_medians'], {})
