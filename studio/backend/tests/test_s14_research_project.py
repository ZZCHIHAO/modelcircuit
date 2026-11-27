"""Independent Python replay for moved S14 research projects."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from studio_core.project_runner import compare_project_runs, read_project_run, train_project
from studio_core.research_project import bind_project, export_project, read_project
from test_table_csv import graph_for


class S14ResearchProjectTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_table_projects_retrain_the_same_folds_and_test_rows_after_a_move(self):
        for task, profile, estimator in (
            ('classification', 'classification-v2', 'RandomForestClassifier'),
            ('regression', 'regression-v1', 'RandomForestRegressor'),
        ):
            with self.subTest(task=task):
                root = self.root / task
                root.mkdir()
                source_csv = root / 'data.csv'
                with source_csv.open('w', newline='', encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['sample_id', 'patient_id', 'partition', 'label', 'score', 'site'])
                    for patient in range(8):
                        for visit in range(2):
                            target = f'class_{patient % 2}' if task == 'classification' else patient * 2 + visit
                            writer.writerow([f'case_{patient}-{visit}', f'patient_{patient}',
                                             'train' if patient < 6 else 'test', target,
                                             None if visit else patient, f'synthetic_private_site_{patient}'])
                config = {'profile': profile, 'data': {'kind': 'table_csv', 'task': task,
                          'path': str(source_csv), 'split_column': 'partition', 'folds': 3,
                          'numeric_columns': ['score'], 'categorical_columns': ['site']}}
                graph = graph_for(estimator)
                project_path = export_project(root / 'project', graph, config)
                move = root / 'moved'
                project_copy = shutil.copytree(project_path.parent, move / 'project')
                data_copy = Path(shutil.copyfile(source_csv, move / 'clinical.csv'))
                lock_text = (project_copy / 'dataset.json').read_text(encoding='utf-8')
                self.assertNotIn('synthetic_private_site_', lock_text)
                self.assertNotIn(str(root), lock_text)
                self.assertEqual(len(json.loads(lock_text)['fold_plan']), 3)
                original = train_project(project_path, source_csv, root / 'original-run')
                cli = subprocess.run(
                    [sys.executable, '-I', 'run.py', 'train', '--data', str(data_copy), '--out', 'repeated-run'],
                    cwd=project_copy, capture_output=True, text=True, timeout=90,
                )
                self.assertEqual(cli.returncode, 0, cli.stderr)
                first = read_project_run(original['artifacts']['project_run'], verify=True)
                second = read_project_run(project_copy / 'repeated-run', verify=True)
                self.assertEqual(first['metrics'], second['metrics'])
                compared = compare_project_runs(first, second)
                self.assertTrue(compared['comparable'])
                if task == 'regression':
                    self.assertEqual(set(compared['metrics']), {'mae', 'rmse', 'r2'})
                for artifact in ('predictions', 'cv_folds'):
                    self.assertEqual(first['artifacts'][artifact].read_bytes(), second['artifacts'][artifact].read_bytes())
                with second['artifacts']['predictions'].open(newline='', encoding='utf-8') as stream:
                    self.assertEqual({row['sample_id'] for row in csv.DictReader(stream)},
                                     {'case_6-0', 'case_6-1', 'case_7-0', 'case_7-1'})
                project = read_project(project_copy)
                data_copy.write_text(data_copy.read_text(encoding='utf-8').replace('synthetic_private_site_7', 'changed_site'),
                                     encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'project lock'):
                    bind_project(project, data_copy)

    def test_table_export_requires_retraining_instead_of_loading_serialized_estimators(self):
        config = {'profile': 'regression-v1', 'data': {'kind': 'table_csv', 'task': 'regression',
                  'path': str(self.root / 'data.csv'), 'numeric_columns': ['score']}}
        with self.assertRaisesRegex(ValueError, 'retrain'):
            export_project(self.root / 'project', graph_for('RandomForestRegressor'), config,
                           checkpoint=self.root / 'estimator.pkl')

    def test_multimodal_projects_rebind_independent_sources_and_compare_input_subsets(self):
        import numpy as np
        from PIL import Image
        import torch
        from studio_core.classification_data import inspect_dataset
        from studio_core.multimodal_templates import multimodal_graph

        torch.set_num_threads(1)
        images = self.root / 'images'
        images.mkdir()
        source_csv = images / 'cohort.csv'
        with source_csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['case_id', 'patient_id', 'split', 'label', 'image_path', 'score'])
            for index, split in enumerate(('train', 'train', 'val', 'val', 'test', 'test')):
                name = f'case_{index}.png'
                pixels = np.random.default_rng(index).integers(0, 256, (8, 8), dtype=np.uint8)
                Image.fromarray(pixels).save(images / name)
                writer.writerow([f'case_{index}', f'patient_{index}', split, index % 2, name, 45678 + index])
        data = {'kind': 'multimodal_csv', 'path': str(source_csv), 'source_root': str(images),
                'split_column': 'split', 'modalities': ['image', 'tabular'], 'feature_columns': ['score'],
                'image_size': [8, 8], 'image_channels': 1}
        config = {'profile': 'classification-v2', 'data': data, 'epochs': 1, 'batch_size': 2, 'seed': 7}
        manifest = inspect_dataset(data, seed=7)
        fusion_mapping = {'image': 'raster', 'tabular': 'clinical'}
        fusion = export_project(self.root / 'fusion', multimodal_graph(manifest, fusion_mapping),
                                {**config, 'input_ports': fusion_mapping})
        move = self.root / 'move'
        copied_project = shutil.copytree(fusion.parent, move / 'fusion')
        copied_images = shutil.copytree(images, move / 'images')
        copied_csv = Path(shutil.copyfile(source_csv, move / 'cases.csv'))
        project = read_project(copied_project)
        rebound, _ = bind_project(project, copied_csv, source_root=copied_images)
        self.assertEqual(rebound['input_ports'], fusion_mapping)
        origin = train_project(fusion, source_csv, self.root / 'original', source_root=images)
        cli = subprocess.run([sys.executable, '-I', 'run.py', 'train', '--data', str(copied_csv), '--out', 'repeated',
                              '--source-root', str(copied_images)], cwd=copied_project,
                             capture_output=True, text=True, timeout=90)
        self.assertEqual(cli.returncode, 0, cli.stderr)
        first = read_project_run(origin['artifacts']['project_run'], verify=True)
        repeat = read_project_run(copied_project / 'repeated', verify=True)
        self.assertEqual(first['artifacts']['predictions'].read_bytes(), repeat['artifacts']['predictions'].read_bytes())
        self.assertNotEqual(first['local_dataset_digest'], repeat['local_dataset_digest'])
        self.assertEqual(project['dataset']['records'][0]['sources']['image'][0]['relative_path'], 'case_0.png')
        self.assertTrue(all('features' not in row for row in project['dataset']['records']))

        weighted = export_project(self.root / 'weighted', project['graph'], rebound,
                                  checkpoint=first['artifacts']['run_spec'])
        weights = read_project(weighted)['weights']
        state = torch.load(weights['checkpoint'], map_location='cpu', weights_only=True)
        self.assertEqual(weights['training_dataset_digest'], first['local_dataset_digest'])
        self.assertEqual(state['input_ports'], fusion_mapping)
        self.assertEqual(state['input_axes']['image'], ['batch', 'channel', 'height', 'width'])
        self.assertNotIn('optimizer_state', state)

        single_mapping = {'image': 'raster'}
        single_graph = multimodal_graph(manifest, single_mapping)
        single_config = {**config, 'input_ports': single_mapping}
        single_project = export_project(self.root / 'single', single_graph, single_config)
        single_result = train_project(single_project, source_csv, self.root / 'single-run', source_root=images)
        single = read_project_run(single_result['artifacts']['project_run'], verify=True)
        self.assertTrue(compare_project_runs(single, first)['comparable'])

        copied_csv.write_text(copied_csv.read_text(encoding='utf-8').replace('patient_5', 'patient_5_different'),
                              encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'project lock'):
            bind_project(project, copied_csv, source_root=copied_images)
        altered = export_project(self.root / 'altered', single_graph, {**single_config, 'data': {
            **data, 'path': str(copied_csv), 'source_root': str(copied_images)}})
        other = train_project(altered, copied_csv, self.root / 'altered-run', source_root=copied_images)
        other_path = Path(other['artifacts']['project_run'])
        self.assertFalse(compare_project_runs(first, read_project_run(other_path, verify=True))['comparable'])
        raw = json.loads(other_path.read_text(encoding='utf-8'))
        raw['evaluation_dataset_digest'] = first['evaluation_dataset_digest']
        raw['dataset_digest'] = first['dataset_digest']
        other_path.write_text(json.dumps(raw), encoding='utf-8')
        self.assertFalse(compare_project_runs(first, read_project_run(other_path, verify=True))['comparable'])

        single_path = single['path']
        self.assertFalse(compare_project_runs(first, read_project_run(single_path))['comparable'])
        single_raw = json.loads(single_path.read_text(encoding='utf-8'))
        predictions = single['artifacts']['predictions']
        edited = predictions.read_bytes().replace(b'case_5', b'case_9')
        predictions.write_bytes(edited)
        prediction_hash = hashlib.sha256(edited).hexdigest()
        single_raw['artifact_sha256']['predictions'] = prediction_hash
        proof_path = single['artifacts']['heldout']
        proof = json.loads(proof_path.read_text(encoding='utf-8'))
        proof['artifact_sha256']['predictions'] = prediction_hash
        proof_bytes = json.dumps(proof).encode('utf-8')
        proof_path.write_bytes(proof_bytes)
        single_raw['artifact_sha256']['heldout'] = hashlib.sha256(proof_bytes).hexdigest()
        single_path.write_text(json.dumps(single_raw), encoding='utf-8')
        rejected = read_project_run(single_path, verify=True)
        self.assertIn('predictions', rejected['artifacts'])
        self.assertIn('heldout', rejected['changed_artifacts'])
        self.assertFalse(compare_project_runs(first, rejected)['comparable'])
