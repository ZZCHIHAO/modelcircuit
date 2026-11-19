"""Usable catalog contracts: actual entries, independent defaults and strict JSON."""
import importlib
import subprocess
import sys
import unittest

from studio_core.operators import operator_kinds, operator_defaults


def catalog():
    try:
        return importlib.import_module('studio_core.component_catalog')
    except ModuleNotFoundError as error:
        raise AssertionError('Usable component catalog is missing') from error


class CatalogTests(unittest.TestCase):
    def test_every_registry_operator_and_five_factories_build(self):
        module = catalog()
        specs = module.list_components()
        self.assertEqual(len(specs), 41)
        self.assertEqual([item.id for item in specs[:22]], ['primitive.' + kind for kind in operator_kinds()])
        self.assertEqual([item.id for item in specs[22:27]], ['head.mlp', 'fusion.concat_projection',
            'fusion.weighted_sum', 'fusion.gated', 'block.residual_basic'])
        for item in specs:
            with self.subTest(component=item.id):
                bundle = module.build_component(item.id, item.config_defaults)
                self.assertIsInstance(bundle, module.ComponentBundle)
                self.assertTrue(item.input_ports)
                self.assertTrue(item.output_ports)
                self.assertEqual(set(item.capabilities), {'expandable', 'forward', 'train', 'python_roundtrip', 'weights'})
                if item.id.startswith('primitive.'):
                    kind = item.id.removeprefix('primitive.')
                    self.assertEqual(bundle.entry['kind'], kind)
                    self.assertEqual(bundle.requirements, {'operator_set': 'vision-ops-v4'})
                    self.assertEqual(bundle.definitions, [])
                    for key, value in operator_defaults(kind).items():
                        self.assertEqual(item.config_defaults[key], value)
                else:
                    self.assertEqual(bundle.entry['kind'], 'chip')
                    self.assertEqual(bundle.requirements['resource_profile'], 'hardware-v5' if item.category in ('backbone', 'model') else 'vision-v4')

    def test_chinese_english_search_category_and_stable_order(self):
        module = catalog()
        for query, identifier in [('逐元素相加', 'primitive.Add'), ('沿轴拼接', 'primitive.Concat'), ('bAtChNoRm', 'primitive.BatchNorm2d'), ('weighted', 'fusion.weighted_sum')]:
            self.assertIn(identifier, [item.id for item in module.list_components(query)])
        self.assertEqual({item.id for item in module.list_components(category='normalization')}, {'primitive.BatchNorm2d', 'primitive.LayerNorm'})
        self.assertEqual(module.list_components(category='future'), [])
        self.assertEqual(module.list_components('not_a_component'), [])

    def test_svm_requires_its_own_library_and_uses_academic_aliases(self):
        module = catalog()
        expected = {'primitive.SVC', 'primitive.SVR'}
        self.assertEqual({item.id for item in module.list_components(category='classical')}, expected)
        self.assertEqual({item.id for item in module.list_components('支持向量机')}, expected)
        self.assertEqual({item.id for item in module.list_components('svm')}, expected)

    def test_defaults_and_bundles_are_independent(self):
        module = catalog()
        specs = module.list_components()
        pool = next(item for item in specs if item.id == 'primitive.AdaptiveAvgPool2d')
        pool.config_defaults['output_size'][0] = 999
        pool.capabilities['forward'] = False
        fresh = next(item for item in module.list_components() if item.id == pool.id)
        self.assertEqual(fresh.config_defaults['output_size'], [1, 1])
        self.assertTrue(fresh.capabilities['forward'])
        first = module.build_component('block.residual_basic', {})
        first.definitions[0]['graph']['inputs'][0]['shape'][1] = 99
        second = module.build_component('block.residual_basic', {})
        self.assertEqual(second.definitions[0]['graph']['inputs'][0]['shape'][1], 8)

    def test_config_rejects_unknown_nonfinite_and_invalid_values(self):
        module = catalog()
        for identifier, config in [('Linear', {'extra': 1}), ('Scale', {'factor': float('inf')}), ('Linear', {'in_features': True}), ('Dropout', {'p': -1}), ('head.mlp', {'hidden_features': 0}), ('fusion.weighted_sum', {'alpha': float('nan')}), ('block.residual_basic', {'input_shape': [None, 2, 8, 8]}), ('Scale', {'factor': object()})]:
            with self.subTest(identifier=identifier, config=config), self.assertRaises(ValueError):
                module.build_component(identifier, config)
        with self.assertRaises(ValueError): module.build_component('ResNet18', {})
        with self.assertRaises(ValueError): module.build_component('Linear', [])

    def test_import_and_build_are_torch_and_gui_free(self):
        catalog()
        script = "from studio_core.component_catalog import *; import sys; [build_component(s.id,s.config_defaults) for s in list_components()]; assert not any(k == 'torch' or k.startswith('dearpygui') for k in sys.modules)"
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_optional_constructor_parameters_are_editable_in_descriptors(self):
        module = catalog()
        batchnorm = next(item for item in module.list_components() if item.id == 'primitive.BatchNorm2d')
        self.assertEqual(batchnorm.config_defaults, {'num_features': 8, 'eps': 1e-5,
            'momentum': .1, 'affine': True, 'track_running_stats': True})
        bundle = module.build_component('BatchNorm2d', {'affine': False, 'momentum': None})
        self.assertFalse(bundle.entry['params']['affine'])
        self.assertIsNone(bundle.entry['params']['momentum'])

    def test_canonical_primitive_ids_build_internal_kind_without_duplicate_alias_rows(self):
        module = catalog()
        descriptors = module.list_components()
        self.assertEqual(len(descriptors), len({item.id for item in descriptors}))
        for kind in operator_kinds():
            canonical = 'primitive.' + kind
            with self.subTest(component=canonical):
                matches = module.list_components(canonical)
                self.assertEqual([item.id for item in matches], [canonical])
                self.assertFalse(any(item.id == kind for item in descriptors))
                bundle = module.build_component(canonical, {})
                self.assertEqual(bundle.entry['kind'], kind)
                self.assertEqual(bundle, module.build_component(kind, {}))
        with self.assertRaises(ValueError): module.build_component('primitive.ResNet18', {})


if __name__ == '__main__': unittest.main()
