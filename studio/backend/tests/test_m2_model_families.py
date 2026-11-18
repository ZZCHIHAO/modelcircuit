"""Hardware-limited standard ResNet34/50 and VGG19 model acceptance."""
import unittest
import hashlib
import json

import torch

from test_backbone_templates import graph_for, without_visual
from studio_core.model_registry import list_registered_models, build_registered_model
from studio_core.schema import prepare_graph
from studio_core.python_io import export_python, import_python
from studio_core.runtime import build_model
from model_family_reference import make_reference
from model_state_mapping import copy_platform_state
from model_acceptance import run_model_acceptance


M2_IDS = ('backbone.resnet18', 'model.resnet18_classifier',
          'backbone.resnet34', 'model.resnet34_classifier',
          'backbone.resnet50', 'model.resnet50_classifier',
          'backbone.vgg11', 'backbone.vgg13', 'backbone.vgg16', 'backbone.vgg19',
          'backbone.squeezenet1_0', 'model.squeezenet1_0_classifier',
          'backbone.squeezenet1_1', 'model.squeezenet1_1_classifier')

# Frozen by M1 before hardware-v5 changes; independent golden hashes copied
# from output/native-s4-models-workspace/m1/model-inventory.json.
LEGACY_CLOSURE_HASHES = {
    'backbone.resnet18': 'acf4d352284d4d6ab1aaacabdb5e1a2a7ea02da4e40df59db4e520723ec39abe',
    'model.resnet18_classifier': '9f539b3a2d16724f06cde14cb0d7a9f1e8f3d639159d87c0a6bf17fbfa3a5265',
    'backbone.vgg11': '71b57c5f9cdd2cd2a213d908602b1d740857aed8acbca33afeb2cde02ca7bac3',
    'backbone.vgg13': 'ef61c0ce40b1709596770740ec3c4cc46c475c48e5bb0bc5889d831ae02dd757',
    'backbone.vgg16': '79d07f1e23a1dd44125d1296a27756dfdc3faa059e17bed7aafc908ec2d4402c',
    'backbone.squeezenet1_0': 'b757a3727d3666fd1b48664257dbfd025f1719eb4f422d2a4a633cc374df2487',
    'model.squeezenet1_0_classifier': '07f0c7a3b4f1be4e482c7193d6e587bbaa15f60e6430e5b6ec00cfa58da5bcaf',
    'backbone.squeezenet1_1': '020c353beeb05563a29021a15e819a46c4cb511b30b1ae6bc2ff62ac2a4e7e29',
    'model.squeezenet1_1_classifier': '51e6f3507ddcce2515bc3cb8214fd9340ee42fd3553081f9e6ae7ef10e78ae0a',
}


class HardwareModelFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_registry_has_fourteen_real_hardware_models(self):
        from studio_core.component_catalog import list_components
        specs = list_registered_models()
        self.assertEqual(tuple(spec.id for spec in specs), M2_IDS)
        self.assertEqual(len(list_components()), 36)
        for spec in specs:
            with self.subTest(model=spec.id):
                self.assertEqual(spec.config_defaults['resource_profile'], 'hardware-v5')
                self.assertEqual(spec.resources['resource_profile'], 'hardware-v5')
                for key in ('max_parameters', 'max_tensor_elements', 'max_execution_elements'):
                    self.assertIsNone(spec.resources[key])
                self.assertEqual(spec.config_fields['resource_profile']['items'], ['hardware-v5', 'vision-v4'])

    def test_new_standard_topologies_counts_and_bottleneck_stride(self):
        specs = {spec.id for spec in list_registered_models()}
        self.assertIn('model.resnet50_classifier', specs, 'ResNet50 factory has not been registered')
        for identifier, count, features, definitions in (
                ('backbone.resnet34', 21_284_672, 512, 25),
                ('model.resnet34_classifier', 21_797_672, 512, 25),
                ('backbone.resnet50', 23_508_032, 2048, 26),
                ('model.resnet50_classifier', 25_557_032, 2048, 26),
                ('backbone.vgg19', 20_024_384, 512, 6)):
            with self.subTest(identifier=identifier):
                bundle = build_registered_model(identifier, {})
                graph = graph_for(bundle)
                resolved, shapes, actual = prepare_graph(graph)
                self.assertEqual(actual, count)
                self.assertEqual(len(bundle.definitions), definitions)
                self.assertEqual(bundle.requirements, {'operator_set': 'vision-ops-v4', 'resource_profile': 'hardware-v5', 'resource_batch_bound': 2})
                self.assertEqual(shapes['call:entry:' + ('logits' if identifier.startswith('model.') else 'features')],
                                 [None, 1000 if identifier.startswith('model.') else features])
                if 'resnet' in identifier:
                    root = resolved['instances'][0]['_graph']
                    for stage, blocks in enumerate((3, 4, 6, 3), 1):
                        stage_graph = root['instances'][stage]['_graph']
                        self.assertEqual([node['id'] for node in stage_graph['instances']], ['block' + str(i) for i in range(blocks)])
                    if 'resnet50' in identifier:
                        block = root['instances'][2]['_graph']['instances'][0]['_graph']
                        nodes = {node['id']: node for node in block['instances']}
                        self.assertEqual(nodes['conv1']['params']['kernel_size'], 1)
                        self.assertEqual(nodes['conv1']['params']['stride'], 1)
                        self.assertEqual(nodes['conv2']['params']['kernel_size'], 3)
                        self.assertEqual(nodes['conv2']['params']['stride'], 2)
                        self.assertEqual(nodes['conv3']['params']['out_channels'], 512)

    def test_every_hardware_default_static_python_and_no_metadata_roundtrip(self):
        self.assertEqual(len(list_registered_models()), 14)
        for spec in list_registered_models():
            with self.subTest(model=spec.id):
                graph = graph_for(build_registered_model(spec.id, {}))
                count = prepare_graph(graph)[2]
                for code in (export_python(graph), without_visual(export_python(graph))):
                    restored = import_python(code)
                    self.assertEqual(restored['resource_profile'], 'hardware-v5')
                    self.assertEqual(restored['resource_batch_bound'], 2)
                    self.assertEqual(prepare_graph(restored)[2], count)

    def test_legacy_v4_is_explicit_exact_and_rejects_new_oversized_models(self):
        spec = list_registered_models()[0]
        self.assertIn('resource_profile', spec.config_defaults)
        bundle = build_registered_model('model.resnet18_classifier', {'resource_profile': 'vision-v4'})
        self.assertEqual(bundle.requirements['resource_profile'], 'vision-v4')
        self.assertEqual(prepare_graph(graph_for(bundle))[2], 11_689_512)
        for identifier in ('backbone.resnet34', 'model.resnet34_classifier', 'backbone.resnet50',
                           'model.resnet50_classifier', 'backbone.vgg19'):
            with self.subTest(identifier=identifier), self.assertRaisesRegex(ValueError, 'parameter budget'):
                build_registered_model(identifier, {'resource_profile': 'vision-v4'})
        with self.assertRaises(ValueError):
            build_registered_model('backbone.resnet18', {'resource_profile': 'invented'})

    def test_new_large_families_train_gradients_sgd_state_and_standalone_python(self):
        self.assertEqual(len(list_registered_models()), 14)
        for identifier, count, bn in (('model.resnet34_classifier', 21_797_672, 36),
                                      ('model.resnet50_classifier', 25_557_032, 53),
                                      ('backbone.vgg19', 20_024_384, 0)):
            with self.subTest(identifier=identifier):
                result = run_model_acceptance(identifier, device='cpu', num_classes=1000)
                self.assertTrue(result['passed'])
                self.assertEqual(result['parameter_count'], count)
                self.assertEqual(result['bn_counters'], [1] * bn)
                self.assertEqual(result['resource_profile'], 'hardware-v5')

    def test_resnet50_and_vgg19_224_eval_match_independent_standard_features(self):
        self.assertEqual(len(list_registered_models()), 14)
        for identifier in ('model.resnet50_classifier', 'backbone.vgg19'):
            with self.subTest(identifier=identifier):
                ports = ['features', 'stage1', 'stage2', 'stage3', 'stage4'] if 'resnet' in identifier else ['features', 'featuremap']
                graph = graph_for(build_registered_model(identifier, {'feature_outputs': ports}))
                model = build_model(graph, capture=False).eval()
                reference = make_reference(identifier, num_classes=1000 if identifier.startswith('model.') else None).eval()
                copy_platform_state(reference, model, identifier)
                x = torch.randn(2, 3, 224, 224)
                with torch.no_grad():
                    actual, expected = model(x=x), reference(x)
                for port, value in actual.items():
                    torch.testing.assert_close(value, expected[port], rtol=2e-5, atol=2e-6)
                self.assertEqual(list(actual['features'].shape), [2, 2048 if 'resnet50' in identifier else 512])

    def test_hardware_policy_does_not_reject_two_standard_resnet34_instances(self):
        self.assertIn('backbone.resnet34', [spec.id for spec in list_registered_models()])
        graph = graph_for(build_registered_model('backbone.resnet34', {'input_shape': [None, 3, 32, 32]}))
        graph['instances'].append({**graph['instances'][0], 'id': 'other'})
        self.assertEqual(prepare_graph(graph)[2], 42_569_344)

    def test_explicit_legacy_closures_match_frozen_m1_bytes(self):
        for identifier, expected in LEGACY_CLOSURE_HASHES.items():
            with self.subTest(identifier=identifier):
                bundle = build_registered_model(identifier, {'resource_profile': 'vision-v4'})
                graph = graph_for(bundle)
                # M1's inventory fixture root ID is deliberately stable.
                graph['id'] = 'inventory'
                serialized = json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
                self.assertEqual(hashlib.sha256(serialized).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main()
