import unittest

from studio_core.classification_task import ClassificationTask
from studio_core.experiment_spec import checkpoint_signature, model_digest
from studio_core.multimodal_templates import multimodal_graph
from studio_core.python_io import export_python, import_python
from studio_core.schema import prepare_graph


class MultimodalGraphContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            'kind': 'multimodal_csv', 'class_names': ['control', 'case'],
            'input_shapes': {'image': [None, 1, 8, 8], 'volume': [None, 2, 4, 8, 8], 'tabular': [None, 2]},
            'input_axes': {'image': ['batch', 'channel', 'height', 'width'],
                           'volume': ['batch', 'channel', 'depth', 'height', 'width'],
                           'tabular': ['batch', 'feature']},
        }

    def test_fusion_and_subset_graphs_bind_named_axes_and_round_trip(self):
        for mapping in ({'volume': 'scan', 'tabular': 'clinical'}, {'image': 'image', 'volume': 'scan'},
                        {'tabular': 'clinical'}, {'image': 'view', 'volume': 'scan', 'tabular': 'clinical'}):
            with self.subTest(mapping=mapping):
                config = {'profile': 'classification-v2', 'data': {'kind': 'multimodal_csv'},
                          'input_ports': mapping, 'optimizer': 'adamw', 'lr': .001, 'batch_size': 2, 'seed': 4}
                graph = multimodal_graph(self.manifest, mapping)
                resolved, shapes, _ = prepare_graph(graph)
                ClassificationTask(config).validate_contract(resolved, shapes, self.manifest)
                self.assertEqual(model_digest(import_python(export_python(graph))), model_digest(graph))
                renamed = {**config, 'input_ports': {key: port + '_new' for key, port in mapping.items()}}
                self.assertNotEqual(checkpoint_signature(graph, config, 'a' * 64),
                                    checkpoint_signature(graph, renamed, 'a' * 64))

    def test_axis_error_with_equal_dimension_lengths_is_rejected(self):
        mapping = {'volume': 'scan'}
        graph = multimodal_graph(self.manifest, mapping)
        resolved, shapes, _ = prepare_graph(graph)
        self.manifest['input_axes']['volume'][-2:] = ['width', 'height']
        task = ClassificationTask({'data': {'kind': 'multimodal_csv'}, 'input_ports': mapping})
        with self.assertRaisesRegex(ValueError, 'axes'):
            task.validate_contract(resolved, shapes, self.manifest)
