"""Portability contracts for exported research projects."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from studio_core.classification_data import inspect_dataset
from studio_core.experiment_spec import default_classification_config, model_digest
from studio_core.research_project import bind_project, export_project, portable_dataset, read_project
from studio_core.segmentation_templates import segmentation_graph
from test_segmentation import config_for


def classification_fixture(root):
    root.mkdir()
    for label, name in enumerate(('blue', 'green')):
        (root / name).mkdir()
        for index in range(4):
            pixels = np.random.default_rng(label * 4 + index).integers(0, 256, (8, 8), dtype=np.uint8)
            Image.fromarray(pixels).save(root / name / f'{index}.png')
    config = default_classification_config('image_folder')
    config.update(epochs=1, batch_size=2)
    config['data'].update(path=str(root), image_size=[8, 8], channels=1)
    graph = {
        'schema_version': 1, 'id': 'portable_classification', 'revision': 1, 'resource_profile': 'classification-v2',
        'inputs': [{'id': 'x', 'shape': [None, 1, 8, 8], 'dtype': 'float32'}],
        'instances': [
            {'id': 'flatten', 'kind': 'Flatten', 'params': {}},
            {'id': 'head', 'kind': 'Linear', 'params': {'in_features': 64, 'out_features': 2}},
        ],
        'calls': [
            {'id': 'features', 'instance_id': 'flatten', 'inputs': {'x': 'input:x'}},
            {'id': 'scores', 'instance_id': 'head', 'inputs': {'x': 'call:features'}, 'position': {'x': 40, 'y': 80}},
        ],
        'outputs': [{'id': 'logits', 'source': 'call:scores'}], 'definitions': [],
    }
    return graph, config


class ResearchProjectTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_image_project_relocates_without_copying_data_or_reusing_local_digest(self):
        source = self.root / 'private_image_folder'
        graph, config = classification_fixture(source)
        original = inspect_dataset(config['data'], config['seed'])
        config.update(expected_dataset_digest=original['dataset_digest'], resume_checkpoint=str(self.root / 'private.pt'))
        project_file = export_project(self.root / 'experiment', graph, config, workbench={'layout': 'single'})
        destination = project_file.parent.rename(self.root / 'moved_experiment')
        moved = source.rename(self.root / 'new_location')

        project = read_project(destination)
        self.assertEqual(project['path'], destination / 'project.json')
        self.assertEqual(read_project(project['path'])['project_id'], project['project_id'])
        self.assertEqual(project['workbench'], {'layout': 'single'})
        self.assertEqual(project['initial_model_digest'], project['model_digest'])
        self.assertEqual(project['model_digest'], model_digest(project['graph']))
        self.assertEqual(project['config']['data']['path'], '')
        self.assertIsNone(project['weights'])
        self.assertFalse(list(destination.rglob('*.png')))
        self.assertTrue((destination / 'runtime' / 'studio_core' / 'classification_runtime.py').is_file())
        self.assertTrue((destination / 'requirements.txt').is_file())
        for name in ('project.json', 'dataset.json'):
            self.assertNotIn(source.name, (destination / name).read_text(encoding='utf-8'))
            self.assertNotIn('private.pt', (destination / name).read_text(encoding='utf-8'))

        bound, local_manifest = bind_project(project, moved)
        self.assertNotEqual(local_manifest['dataset_digest'], original['dataset_digest'])
        self.assertEqual(bound['expected_dataset_digest'], local_manifest['dataset_digest'])
        self.assertIsNone(bound['resume_checkpoint'])
        self.assertEqual(portable_dataset(local_manifest), project['dataset'])
        with self.assertRaises(InterruptedError):
            bind_project(project, moved, cancelled=lambda: True)
        self.assertEqual([(r['relative_path'], r['split']) for r in original['records']],
                         [(r['relative_path'], r['split']) for r in local_manifest['records']])

        # Preserve decoded pixels and names while changing the actual source bytes.
        path = moved / 'blue' / '0.png'
        with Image.open(path) as image:
            image.load()
            image.save(path, compress_level=0)
        with self.assertRaisesRegex(ValueError, 'Dataset.*(changed|match)'):
            bind_project(project, moved)

    def test_segmentation_project_resolves_pairs_relative_to_an_explicit_root_and_checks_order(self):
        source = self.root / 'volumes'
        source.mkdir()
        config, _ = config_for(source, volume=True)
        manifest = inspect_dataset(config['data'], config['seed'])
        project = read_project(export_project(self.root / 'experiment', segmentation_graph(manifest), config))
        self.assertEqual(project['config']['data']['source_root'], '')
        relocated = source.rename(self.root / 'relocated_volumes')
        selection = (relocated / 'pairs.csv').rename(self.root / 'data-selection.csv')
        bound, effective = bind_project(project, selection, source_root=relocated)
        self.assertEqual(bound['data']['source_root'], str(relocated))
        self.assertEqual(effective['input_shape'], [None, 1, 4, 9, 8])
        self.assertEqual(portable_dataset(effective)['digest'], project['dataset']['digest'])

        with selection.open(encoding='utf-8', newline='') as stream:
            header, *rows = csv.reader(stream)
        with selection.open('w', encoding='utf-8', newline='') as stream:
            csv.writer(stream).writerows([header, rows[1], rows[0], *rows[2:]])
        with self.assertRaisesRegex(ValueError, 'Dataset.*(changed|match)'):
            bind_project(project, selection, source_root=relocated)

    def test_absolute_segmentation_pairs_rebind_by_directory_after_project_and_data_move(self):
        for explicit in (False, True):
            with self.subTest(explicit_split=explicit):
                source = self.root / f'absolute-{explicit}'
                source.mkdir()
                config, _ = config_for(source, volume=True)
                pair_list = Path(config['data']['path'])
                with pair_list.open(encoding='utf-8', newline='') as stream:
                    _, *rows = csv.reader(stream)
                columns = ['case', 'patient', 'scan', 'label'] + (['partition'] if explicit else [])
                with pair_list.open('w', encoding='utf-8', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(columns)
                    for sample, patient, image, mask, split in rows:
                        writer.writerow([sample, patient, str(source / image), str(source / mask)] + ([split] if explicit else []))
                config['data'].update(id_column='case', group_column='patient', image_column='scan', mask_column='label',
                                      split_column='partition' if explicit else None)
                initial = inspect_dataset(config['data'], config['seed'])
                project_path = export_project(self.root / f'project-{explicit}', segmentation_graph(initial), config)
                moved_project = project_path.parent.rename(self.root / f'moved-project-{explicit}')
                moved_data = source.rename(self.root / f'moved-volumes-{explicit}')
                project = read_project(moved_project)
                self.assertEqual(project['paired_csv'], moved_project / 'dataset-pairs.csv')
                with project['paired_csv'].open(encoding='utf-8', newline='') as stream:
                    header, *relative_rows = csv.reader(stream)
                self.assertEqual(header, columns)
                self.assertEqual([row[:4] for row in relative_rows], [row[:4] for row in rows])
                with self.assertRaisesRegex(ValueError, 'inside the selected source root'):
                    bind_project(project, moved_data / 'pairs.csv')
                bound, rebound = bind_project(project, moved_data)
                self.assertEqual(bound['data']['path'], str(project['paired_csv']))
                self.assertEqual(bound['data']['source_root'], str(moved_data))
                self.assertEqual(portable_dataset(initial), portable_dataset(rebound))
                self.assertNotEqual(initial['dataset_digest'], rebound['dataset_digest'])

                with project['paired_csv'].open('w', encoding='utf-8', newline='') as stream:
                    csv.writer(stream).writerows([header, *reversed(relative_rows)])
                with self.assertRaisesRegex(ValueError, 'Dataset.*match'):
                    bind_project(project, moved_data)

                metadata = project['path']
                payload = json.loads(metadata.read_text(encoding='utf-8'))
                payload['paired_csv'] = '../dataset-pairs.csv'
                metadata.write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'paired CSV path reference'):
                    read_project(moved_project)

    def test_code_edits_drive_canvas_and_incompatible_task_keeps_parsed_graph(self):
        graph, config = classification_fixture(self.root / 'data')
        project_path = export_project(self.root / 'experiment', graph, config)
        model_path = project_path.with_name('model.py')
        original = model_path.read_text(encoding='utf-8')
        self.assertIn('out_features=2', original)
        changed = original.replace('out_features=2', 'out_features=3')
        model_path.write_text(changed, encoding='utf-8')
        project = read_project(project_path)
        self.assertEqual(project['graph']['instances'][-1]['params']['out_features'], 3)
        self.assertIn('logits', project['graph_error'])
        self.assertNotEqual(project['initial_model_digest'], project['model_digest'])
        self.assertEqual(model_path.read_text(encoding='utf-8'), changed)
        with self.assertRaisesRegex(ValueError, 'logits'):
            bind_project(project, self.root / 'data')
        model_path.write_text(original + '\nprint("unsafe top-level statement")\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'model.py.*line'):
            read_project(project_path)

    def test_source_reconnection_preserves_shared_module_and_discards_stale_call_layout(self):
        graph, config = classification_fixture(self.root / 'data')
        graph['instances'] += [
            {'id': 'shared', 'kind': 'Linear', 'params': {'in_features': 64, 'out_features': 64}},
            {'id': 'join', 'kind': 'Add', 'params': {}},
        ]
        graph['calls'][1:1] = [
            {'id': 'first', 'instance_id': 'shared', 'inputs': {'x': 'call:features'}, 'position': {'x': 1, 'y': 2}},
            {'id': 'second', 'instance_id': 'shared', 'inputs': {'x': 'call:features'}, 'position': {'x': 3, 'y': 4}},
            {'id': 'join', 'instance_id': 'join', 'inputs': {'a': 'call:first', 'b': 'call:second'}},
        ]
        graph['calls'][-1]['inputs']['x'] = 'call:join'
        model = export_project(self.root / 'experiment', graph, config).with_name('model.py')
        code = model.read_text(encoding='utf-8')
        previous = "call_second = self.components['shared'](input=call_features)"
        self.assertIn(previous, code)
        model.write_text(code.replace(previous, previous.replace('call_features', 'call_first')), encoding='utf-8')
        project = read_project(model.parent)
        calls = {call['id']: call for call in project['graph']['calls']}
        self.assertIsNone(project['graph_error'])
        self.assertEqual(calls['first']['instance_id'], calls['second']['instance_id'])
        self.assertEqual(calls['second']['inputs'], {'x': 'call:first'})
        self.assertEqual(calls['first']['position'], {'x': 1, 'y': 2})
        self.assertNotIn('position', calls['second'])

    def test_classification_weights_reexport_after_dataset_rebind_requires_saved_lock(self):
        import torch
        from studio_core.classification_runtime import run_supervised
        torch.set_num_threads(1)
        source = self.root / 'source_images'
        graph, config = classification_fixture(source)
        run = self.root / 'training_run'
        run_supervised(graph, config, run)
        moved = source.rename(self.root / 'rebound_images')
        config['data']['path'] = str(moved)

        project = read_project(export_project(self.root / 'portable', graph, config, checkpoint=run / 'run_spec.json'))
        checkpoint = torch.load(project['weights']['checkpoint'], weights_only=True, map_location='cpu')
        self.assertEqual(checkpoint['version'], 2)
        self.assertEqual(checkpoint['class_names'], ['blue', 'green'])
        self.assertNotIn('best', checkpoint)

        history = run / 'manifest.json'
        contents = json.loads(history.read_text(encoding='utf-8'))
        contents['records'].reverse()
        history.write_text(json.dumps(contents), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, '[Dd]ataset.*identity'):
            export_project(self.root / 'invalid', graph, config, checkpoint=run / 'run_spec.json')
        self.assertFalse((self.root / 'invalid').exists())

    def test_selects_first_matching_checkpoint_and_surfaces_its_missing_weights(self):
        import torch
        from studio_core.classification_runtime import run_supervised
        torch.set_num_threads(1)
        graph, config = classification_fixture(self.root / 'wanted_images')
        _, changed_data = classification_fixture(self.root / 'changed_images')
        with next(Path(changed_data['data']['path']).rglob('*.png')).open('ab') as stream:
            stream.write(b'changed image bytes')
        different_graph = deepcopy(graph)
        different_graph['instances'].append({'id': 'relu', 'kind': 'ReLU', 'params': {}})
        different_graph['calls'].append({'id': 'relu', 'instance_id': 'relu', 'inputs': {'x': 'call:scores'}})
        different_graph['outputs'][0]['source'] = 'call:relu'

        data_run, model_run, intended_run = [self.root / name for name in ('other_data', 'other_model', 'intended_run')]
        run_supervised(graph, changed_data, data_run)
        run_supervised(different_graph, config, model_run)
        run_supervised(graph, config, intended_run)
        candidates = tuple(run / 'run_spec.json' for run in (data_run, model_run, intended_run))
        path = export_project(self.root / 'project', graph, config, checkpoint=candidates)
        intended = json.loads(candidates[-1].read_text(encoding='utf-8'))
        self.assertEqual(read_project(path)['weights']['checkpoint_signature'], intended['checkpoint_signature'])

        # A compatible run with a missing weight is a broken export, even if a later run matches.
        incomplete = self.root / 'incomplete'
        incomplete.mkdir()
        missing = incomplete / 'run_spec.json'
        shutil.copyfile(candidates[-1], missing)
        with self.assertRaisesRegex(ValueError, 'Could not read the saved checkpoint'):
            export_project(self.root / 'rejected', graph, config, checkpoint=(*candidates[:-1], missing, candidates[-1]))
        self.assertFalse((self.root / 'rejected').exists())

    def test_export_preserves_existing_destination_and_read_rejects_invalid_reference(self):
        graph, config = classification_fixture(self.root / 'data')
        destination = self.root / 'experiment'
        destination.mkdir()
        existing = destination / 'precious.txt'
        existing.write_text('keep', encoding='utf-8')
        with self.assertRaisesRegex((FileExistsError, ValueError), 'exist'):
            export_project(destination, graph, config)
        self.assertEqual(list(destination.iterdir()), [existing])
        self.assertEqual(existing.read_text(encoding='utf-8'), 'keep')
        project_file = export_project(self.root / 'other', graph, config)
        payload = json.loads(project_file.read_text(encoding='utf-8'))
        shutil.copyfile(project_file.with_name('model.py'), self.root / 'model.py')
        payload['model'] = '../model.py'
        project_file.write_text(json.dumps(payload), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'model.*(path|reference)'):
            read_project(project_file)

    def test_segmentation_weights_keep_original_inference_identity_and_strip_training_state(self):
        import torch
        from studio_core.classification_runtime import run_supervised
        from studio_core.segmentation_inference import _load_model
        torch.set_num_threads(1)
        source = self.root / 'private_pairs'
        source.mkdir()
        config, _ = config_for(source)
        manifest = inspect_dataset(config['data'], config['seed'])
        graph = segmentation_graph(manifest)
        run = self.root / 'training_run'
        run_supervised(graph, config, run)
        exported = export_project(self.root / 'experiment', graph, config, checkpoint=run / 'run_spec.json')
        project = read_project(exported)
        weights = project['weights']
        self.assertEqual(weights['checkpoint'].parent, exported.parent / 'weights')
        saved = json.loads(weights['run_spec'].read_text(encoding='utf-8'))
        self.assertNotIn(source.name, json.dumps(saved))
        state = torch.load(weights['checkpoint'], weights_only=True, map_location='cpu')
        self.assertNotIn('optimizer_state', state)
        self.assertNotIn('rng_state', state)
        self.assertEqual(state['checkpoint_signature'], saved['checkpoint_signature'])
        self.assertEqual(state['dataset_digest'], manifest['dataset_digest'])
        _, _, _, _, provenance, _ = _load_model({'run_spec': str(weights['run_spec']), 'checkpoint': '', 'device': 'cpu'}, lambda: None)
        self.assertEqual(provenance['checkpoint_signature'], state['checkpoint_signature'])
        with weights['checkpoint'].open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'checkpoint.*digest'):
            read_project(exported)


if __name__ == '__main__':
    unittest.main()
