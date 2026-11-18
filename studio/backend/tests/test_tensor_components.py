import unittest

from studio_core.component_catalog import build_component, list_components
from studio_core.model_registry import list_registered_models


class TensorComponentsTest(unittest.TestCase):
    def test_explicit_tensor_catalog_preserves_legacy_catalog(self):
        legacy = [spec.id for spec in list_components()]
        current = [spec.id for spec in list_components(include_tensor=True)]
        self.assertEqual(current[:len(legacy)], legacy)
        self.assertEqual(len(list_registered_models()), 14)
        self.assertEqual(current[len(legacy):], ['tensor.attention', 'tensor.encoder', 'vit.small_classifier'])

    def test_configured_attention_ports_and_requirements(self):
        bundle = build_component('tensor.attention', {'masked': True, 'return_attention': True})
        graph = bundle.definitions[0]['graph']
        self.assertEqual(graph['inputs'][-1]['dtype'], 'bool')
        self.assertEqual(graph['outputs'][-1]['id'], 'attention')
        self.assertEqual(bundle.requirements['capability_profile'], 'tensor-v1')
        with self.assertRaisesRegex(ValueError, 'num_heads'):
            build_component('tensor.encoder', {'hidden_dim': 10, 'num_heads': 3})
        with self.assertRaisesRegex(ValueError, 'patch_size'):
            build_component('vit.small_classifier', {'patch_size': 7})
