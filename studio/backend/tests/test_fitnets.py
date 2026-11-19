"""FitNets checks using a trainable tabular graph and a frozen BN teacher."""
import csv
from pathlib import Path
import tempfile
import unittest

import torch

from studio_core.fitnets import fitnets_teacher_graph, run_fitnets, validate_fitnets_graph
from studio_core.runtime import build_model


def graph_fixture():
    def layer(identifier, kind, width_in=None, width_out=None):
        params = {'in_features': width_in, 'out_features': width_out} if kind == 'Linear' else {}
        return {'id': identifier, 'kind': kind, 'params': params}

    def definition(identifier, width, instances, outputs):
        return {'id': identifier, 'graph': {
            'inputs': [{'id': 'x', 'dtype': 'float32', 'shape': [None, width]}],
            'instances': instances,
            'calls': [{'id': item['id'], 'instance_id': item['id'], 'inputs': {'x': 'call:' + instances[i-1]['id'] if i else 'input:x'}}
                      for i, item in enumerate(instances)],
            'outputs': [{'id': port, 'source': 'call:' + item} for port, item in outputs.items()],
        }}
    definitions = [definition('Teacher', 2, [layer('early', 'Linear', 2, 5), layer('head', 'Linear', 5, 2)],
                              {'hint': 'early', 'logits': 'head'}),
                   definition('Student', 2, [layer('early', 'Linear', 2, 3), layer('head', 'Linear', 3, 2)],
                              {'guided': 'early', 'logits': 'head'}),
                   definition('Projector', 3, [layer('align', 'Linear', 3, 5)], {'out': 'align'})]
    return {'schema_version': 1, 'resource_profile': 'classification-v2', 'id': 'fitnet_test', 'revision': 1,
            'inputs': [{'id': 'x', 'shape': [None, 2], 'dtype': 'float32'}], 'definitions': definitions,
            'instances': [{'id': name.lower(), 'kind': 'chip', 'definition_id': name, 'params': {}}
                          for name in ('Teacher','Student','Projector')],
            'calls': [{'id': 'apply_' + name, 'instance_id': name, 'inputs': {'x': source}} for name, source in (
                ('teacher', 'input:x'), ('student', 'input:x'), ('projector', 'call:apply_student:guided'))],
            'outputs': [{'id': name, 'source': 'call:' + source} for name, source in (
                ('logits', 'apply_student:logits'), ('teacher_logits', 'apply_teacher:logits'), ('teacher_hint', 'apply_teacher:hint'),
                ('student_guided', 'apply_student:guided'), ('student_projection', 'apply_projector'))]}


class FitNetsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.graph = graph_fixture()
        tabular = self.root/'data.csv'
        with tabular.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(('a', 'b', 'label', 'split'))
            for start, split in ((0, 'train'), (8, 'val'), (16, 'test')):
                for i in range(start, start + 8):
                    writer.writerow((i/3, (i%3)*0.25, 'positive' if i%2 else 'negative', split))
        self.weights = self.root/'teacher.pt'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(9)
            teacher = build_model(fitnets_teacher_graph(self.graph), capture=False)
        torch.save({'model_state': teacher.state_dict(), 'class_names': ['negative','positive']}, self.weights)
        self.config = {'profile':'fitnets-v1','device':'cpu','epochs':2,'batch_size':4,'seed':42,'lr':0.03,
                       'data':{'kind':'csv','path':str(tabular),'target_column':'label','feature_columns':['a','b'], 'split_column':'split'},
                       'fitnets':{'teacher_checkpoint':str(self.weights),'hint_epochs':2,'temperature':2.,'distill_weight':0.5}}

    def test_phase_updates_and_student_export(self):
        report = run_fitnets(self.graph, self.config, self.root/'fitnets')
        self.assertEqual(report['phase_summary']['teacher_unchanged'], True)
        self.assertEqual(report['phase_summary']['hint_changed_instances']['student'], ['early'])
        self.assertEqual(report['phase_summary']['distill_changed_instances']['projector'], [])
        self.assertIn('head', report['phase_summary']['distill_changed_instances']['student'])
        self.assertEqual(report['metrics']['test']['count'], 8)
        self.assertEqual(report['global_step'], 8)
        student = torch.load(report['artifacts']['student_checkpoint'], weights_only=True)
        model = build_model(student['graph'], device='cpu', capture=False)
        model.load_state_dict(student['model_state'], strict=True)
        self.assertEqual(set(model(x=torch.zeros(1,2))), {'logits'})

    def test_topology_rejects_wrong_projection_connection(self):
        self.graph['outputs'][-1]['source'] = 'call:apply_teacher:hint'
        with self.assertRaisesRegex(ValueError, 'student_projection'):
            validate_fitnets_graph(self.graph)

    def test_teacher_class_order_rejected(self):
        value = torch.load(self.weights, weights_only=True)
        value['class_names'].reverse()
        torch.save(value,self.weights)
        with self.assertRaisesRegex(ValueError,'class order'):
            run_fitnets(self.graph,self.config,self.root/'fitnets')
