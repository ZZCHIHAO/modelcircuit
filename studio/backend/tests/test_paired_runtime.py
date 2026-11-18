import csv
import json
import tempfile
import unittest
from pathlib import Path

from paired_fixture import paired_fixture, paired_graph
from studio_core.classification_runtime import run_classification


class PairedRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch; torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.spec, self.rows = paired_fixture(self.root)
        self.config = {'profile': 'classification-v2', 'data': self.spec, 'epochs': 3, 'batch_size': 5, 'lr': 0.02, 'optimizer': 'adamw', 'seed': 77, 'device': 'cpu'}

    def run_training(self, name, **updates):
        return run_classification(paired_graph(), dict(self.config, **updates), self.root / name)

    def test_both_named_modalities_train_and_have_same_actual_probe_ids(self):
        result = self.run_training('run')
        self.assertEqual(result['data_source'], 'paired_csv')
        self.assertEqual(result['global_step'], 9)
        self.assertEqual(result['metrics']['test']['count'], 6)
        self.assertEqual(result['probes']['input:image']['shape'], [5, 3, 8, 8])
        self.assertEqual(result['probes']['input:clinical']['shape'], [5, 2])
        self.assertEqual(result['probe_context']['input_ports'], {'image': 'image', 'tabular': 'clinical'})
        self.assertEqual(result['probe_context']['sample_ids'], [row['sample_id'] for row in self.rows[12:17]])
        self.assertTrue(any('conv.weight' in item['name'] and item['grad_norm'] > 0 for item in result['parameters']))
        self.assertTrue(any('clinical_dense.weight' in item['name'] and item['grad_norm'] > 0 for item in result['parameters']))
        json.dumps(result, allow_nan=False)

    def test_cpu_completed_epoch_resume_is_exact_with_joint_shuffle_and_dropout(self):
        full = self.run_training('full')
        first = self.run_training('first', epochs=1)
        restored = self.run_training('restored', resume_checkpoint=first['artifacts']['last_checkpoint'])
        self.assertEqual(full['history'], restored['history'])
        self.assertEqual(full['metrics'], restored['metrics'])
        load = lambda result: self.torch.load(result['artifacts']['last_checkpoint'], weights_only=True, map_location='cpu')
        expected, actual = load(full), load(restored)
        self.assertEqual(actual['input_shapes'], {'image': [None, 3, 8, 8], 'tabular': [None, 2]})
        for name, value in expected['model_state'].items(): self.assertTrue(self.torch.equal(value, actual['model_state'][name]), name)
        self.assertTrue(self.torch.equal(expected['shuffle_state'], actual['shuffle_state']))

    def test_best_checkpoint_independent_inference_and_custom_port_mapping(self):
        from studio_core.classification_data import build_datasets
        from studio_core.runtime import build_model
        graph = paired_graph('scan', 'values')
        result = run_classification(graph, dict(self.config, input_ports={'image': 'scan', 'tabular': 'values'}), self.root / 'mapped')
        checkpoint = self.torch.load(result['artifacts']['best_checkpoint'], weights_only=True, map_location='cpu')
        model = build_model(graph, capture=False); model.load_state_dict(checkpoint['model_state']); model.eval()
        manifest = json.loads(Path(result['artifacts']['manifest']).read_text(encoding='utf-8'))
        dataset = build_datasets(manifest)['test']
        with Path(result['artifacts']['predictions']).open(encoding='utf-8', newline='') as stream: predictions = list(csv.DictReader(stream))
        with self.torch.no_grad():
            for index in range(len(dataset)):
                inputs, label, identifier = dataset[index]
                logits = model(scan=inputs['image'].unsqueeze(0), values=inputs['tabular'].unsqueeze(0))['logits']
                self.assertEqual(int(predictions[index]['prediction']), int(logits.argmax(1)))
                self.assertEqual(predictions[index]['sample_id'], identifier)

    def test_mapping_shape_and_changed_input_identity_fail(self):
        with self.assertRaises(ValueError): self.run_training('duplicate', input_ports={'image': 'image', 'tabular': 'image'})
        graph = paired_graph(); graph['inputs'][0]['shape'] = [None, 3, 10, 10]
        with self.assertRaisesRegex(ValueError, 'shape'): run_classification(graph, self.config, self.root / 'shape')
        first = self.run_training('first', epochs=1)
        with self.assertRaisesRegex(ValueError, 'checkpoint.*identity'):
            run_classification(paired_graph('scan', 'values'), dict(self.config, input_ports={'image': 'scan', 'tabular': 'values'}, resume_checkpoint=first['artifacts']['last_checkpoint']), self.root / 'changed_mapping')
        changed = dict(self.spec, feature_columns=['constant', 'f'])
        with self.assertRaisesRegex(ValueError, 'checkpoint.*identity'):
            self.run_training('changed_features', data=changed, resume_checkpoint=first['artifacts']['last_checkpoint'])

    def test_cancelled_paired_epoch_retains_committed_checkpoint_for_exact_replay(self):
        full = self.run_training('full')
        stop = [False]
        def progress(event):
            if event['phase'] == 'train' and event['epoch'] == 2 and event['batch'] == 1: stop[0] = True
        with self.assertRaises(InterruptedError):
            run_classification(paired_graph(), self.config, self.root / 'cancelled', progress=progress, cancelled=lambda: stop[0])
        checkpoint = self.root / 'cancelled' / 'last.pt'
        self.assertEqual(self.torch.load(checkpoint, weights_only=True, map_location='cpu')['completed_epochs'], 1)
        restored = self.run_training('restored', resume_checkpoint=str(checkpoint))
        self.assertEqual(full['history'], restored['history'])


if __name__ == '__main__': unittest.main()
