"""Real model closures, strict descriptors and independent numerical oracles."""
import ast
import copy
import importlib.util
import json
import subprocess
import sys
import unittest

import torch

from component_fixture import bundle_graph
from studio_core.schema import prepare_graph, validate_graph
from studio_core.runtime import build_model
from studio_core.python_io import export_python, import_python
from model_family_reference import make_reference
from model_state_mapping import canonical_parameters, canonical_state_dict, copy_platform_state


POLICY = {'resource_profile': 'hardware-v5', 'resource_batch_bound': 2}
MODEL_IDS = ('backbone.resnet18', 'model.resnet18_classifier',
             'backbone.resnet34', 'model.resnet34_classifier', 'backbone.resnet50', 'model.resnet50_classifier',
             'backbone.vgg11', 'backbone.vgg13', 'backbone.vgg16', 'backbone.vgg19',
             'backbone.squeezenet1_0', 'model.squeezenet1_0_classifier',
             'backbone.squeezenet1_1', 'model.squeezenet1_1_classifier')


def modules():
    assert importlib.util.find_spec('studio_core.backbone_templates') is not None, 'Model graph factories are missing'
    assert importlib.util.find_spec('studio_core.model_registry') is not None, 'Model registry is missing'
    from studio_core import backbone_templates, model_registry
    return backbone_templates, model_registry


def graph_for(bundle):
    definition = next(item for item in bundle.definitions if item['id'] == bundle.entry['definition_id'])
    return bundle_graph(bundle, {port['id']: port['shape'] for port in definition['graph']['inputs']},
                        {key: value for key, value in bundle.requirements.items() if key != 'operator_set'})


def without_visual(source):
    tree = ast.parse(source)
    tree.body = [item for item in tree.body if not (isinstance(item, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == 'STUDIO_VISUAL_METADATA' for target in item.targets))]
    assert not any(isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and
        target.id == 'STUDIO_VISUAL_METADATA' for target in item.targets) for item in tree.body)
    return ast.unparse(tree)


class BackboneTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_registered_catalog_is_real_complete_fresh_and_json_only(self):
        _, registry = modules()
        from studio_core.component_catalog import list_components, build_component
        specs = registry.list_registered_models()
        self.assertEqual(tuple(item.id for item in specs), MODEL_IDS)
        self.assertEqual([item.id for item in list_components()[-len(MODEL_IDS):]], list(MODEL_IDS))
        for spec in specs:
            with self.subTest(model=spec.id):
                self.assertIn(spec.category, ('backbone', 'model'))
                self.assertEqual(spec.family, 'resnet' if 'resnet' in spec.id else 'vgg' if 'vgg' in spec.id else 'squeezenet')
                self.assertEqual(spec.initialization['weights'], 'none')
                self.assertIn('platform', spec.initialization['policy'])
                original = copy.deepcopy(spec.config_defaults)
                bundle = build_component(spec.id, {})
                self.assertEqual(bundle, registry.build_registered_model(spec.id, {}))
                self.assertEqual(bundle.requirements, {'operator_set': 'vision-ops-v4', **POLICY})
                self.assertEqual(spec.config_defaults, original)
                self.assertTrue(validate_graph(graph_for(bundle))['valid'])
                json.dumps(bundle.__dict__, allow_nan=False)
        specs[0].config_defaults['input_shape'][1] = 99
        specs[0].capabilities['train'] = False
        fresh = registry.list_registered_models()[0]
        self.assertEqual(fresh.config_defaults['input_shape'], [None, 3, 224, 224])
        self.assertTrue(fresh.capabilities['train'])
        script = ('from studio_core.model_registry import *; import sys; '
                  '[build_registered_model(s.id,{}) for s in list_registered_models()]; '
                  'assert not any(k == "torch" or k.startswith("dearpygui") for k in sys.modules)')
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_resnet_exact_hierarchy_standard_count_and_stage_shapes(self):
        factories, _ = modules()
        bundle = factories.make_resnet18([None, 3, 224, 224], num_classes=1000,
            feature_outputs=('features', 'stage1', 'stage2', 'stage3', 'stage4'))
        graph = graph_for(bundle)
        resolved, shapes, count = prepare_graph(graph)
        self.assertEqual(count, 11_689_512)
        self.assertEqual(len(bundle.definitions), 17)
        root = next(d for d in bundle.definitions if d['id'] == bundle.entry['definition_id'])['graph']
        self.assertEqual([i['id'] for i in root['instances']][:5], ['stem', 'stage1', 'stage2', 'stage3', 'stage4'])
        self.assertEqual(shapes['call:entry:features'], [None, 512])
        for stage, channels, side in ((1, 64, 56), (2, 128, 28), (3, 256, 14), (4, 512, 7)):
            self.assertEqual(shapes['call:entry:stage' + str(stage)], [None, channels, side, side])
            stage_graph = resolved['instances'][0]['_graph']['instances'][stage]['_graph']
            self.assertEqual([i['id'] for i in stage_graph['instances']], ['block0', 'block1'])
        self.assertLessEqual(resolved['_resource_ledger']['execution_elements'], 134_217_728)

    def test_variants_budget_counts_standard_conv_and_fire_topology(self):
        _, registry = modules()
        counts = {'backbone.resnet18': 11_176_512, 'model.resnet18_classifier': 11_689_512,
                  'backbone.vgg11': 9_220_480, 'backbone.vgg13': 9_404_992, 'backbone.vgg16': 14_714_688,
                  'backbone.squeezenet1_0': 735_424, 'model.squeezenet1_0_classifier': 1_248_424,
                  'backbone.squeezenet1_1': 722_496, 'model.squeezenet1_1_classifier': 1_235_496}
        for identifier, count in counts.items():
            with self.subTest(identifier=identifier):
                bundle = registry.build_registered_model(identifier, {})
                resolved, shapes, actual = prepare_graph(graph_for(bundle))
                self.assertEqual(actual, count)
                self.assertLessEqual(resolved['_resource_ledger']['execution_elements'], 134_217_728)
                self.assertLessEqual(len(bundle.definitions), 32)
                self.assertEqual(shapes['call:entry:' + ('logits' if identifier.startswith('model.') else 'features')],
                                 [None, 1000 if identifier.startswith('model.') else 512])
                if 'squeezenet' in identifier:
                    fires = [d for d in bundle.definitions if 'Fire' in d['id']]
                    self.assertEqual(len(fires), 8)
                    for fire in fires:
                        kinds = [i['kind'] for i in fire['graph']['instances']]
                        self.assertEqual(kinds.count('Conv2d'), 3)
                        self.assertEqual(kinds.count('Concat'), 1)
                    if identifier.startswith('model.'):
                        classifier = next(d for d in bundle.definitions if d['id'].endswith('Classifier'))
                        self.assertEqual([i['kind'] for i in classifier['graph']['instances']],
                            ['Dropout', 'Conv2d', 'ReLU', 'AdaptiveAvgPool2d', 'Flatten'])
        self.assertFalse(any('alexnet' in item.id for item in registry.list_registered_models()))

    def test_strict_finite_config_and_explicit_resource_rejection(self):
        factories, registry = modules()
        invalid = [[], {'weights': 'DEFAULT'}, {'unknown': 1}, {'input_shape': (None, 3, 32, 32)},
                   {'input_shape': [None, 2, 32, 32]}, {'input_shape': [True, 3, 32, 32]},
                   {'input_shape': [None, 3, float('inf'), 32]}, {'input_shape': [None, 3, 0, 32]},
                   {'resource_batch_bound': True}, {'resource_batch_bound': 65},
                   {'input_shape': [3, 3, 32, 32]}, {'feature_outputs': ['missing']},
                   {'feature_outputs': ['features', 'features']}, {'feature_outputs': []},
                   {'num_classes': 5}]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                registry.build_registered_model('backbone.resnet18', config)
        for identifier, config in [('model.resnet18_classifier', {'num_classes': True}),
                ('model.resnet18_classifier', {'num_classes': 100_000, 'resource_profile': 'vision-v4'}),
                ('model.squeezenet1_0_classifier', {'dropout': float('nan')}),
                ('model.squeezenet1_0_classifier', {'dropout': 10 ** 400}),
                ('model.squeezenet1_0_classifier', {'dropout': -(10 ** 400)}),
                ('model.squeezenet1_1_classifier', {'dropout': True}),
                ('model.squeezenet1_1_classifier', {'dropout': -0.1}),
                ('backbone.vgg11', {'input_shape': [None, 3, 16, 16]}),
                ('backbone.vgg16', {'resource_batch_bound': 64, 'resource_profile': 'vision-v4'}),
                ('missing', {}), (True, {})]:
            with self.subTest(identifier=identifier, config=config), self.assertRaises(ValueError):
                registry.build_registered_model(identifier, config)
        gray = factories.make_resnet18([None, 1, 32, 32])
        self.assertEqual(prepare_graph(graph_for(gray))[2], 11_170_240)
        with self.assertRaises(ValueError): factories.make_resnet18([None, 3, 224, 224], resource_batch_bound=64, resource_profile='vision-v4')

    def test_native_registry_extension_uses_existing_catalog_and_runtime(self):
        _, registry = modules()
        from studio_core.component_catalog import list_components, build_component
        from studio_core.module_templates import make_module
        spec = registry.list_registered_models()[0]
        spec.id = 'backbone.example_mlp'
        spec.family, spec.architecture = 'example', 'native_mlp'
        spec.label, spec.description = 'Native extension example', 'Trusted JSON graph factory'
        spec.config_defaults = {'in_features': 3, 'hidden_features': 4, 'num_classes': 2, 'resource_batch_bound': 2}
        spec.required_config = ('in_features', 'hidden_features', 'num_classes')
        spec.config_fields = {key: {'type': 'integer'} for key in spec.config_defaults}
        spec.output_ports = ('logits',)
        spec.inputs = {'x': {'shape': [None, 3], 'dtype': 'float32'}}
        spec.outputs = {'logits': {'shape': [None, 2], 'dtype': 'float32'}}
        spec.resources = {'operator_set': 'vision-ops-v4', 'resource_profile': 'vision-v4', 'resource_batch_bound': 2,
                          'max_parameters': 20_000_000, 'max_tensor_elements': 16_777_216, 'max_execution_elements': 134_217_728}
        extension = registry.DEFAULT_MODEL_REGISTRY.with_model(spec, lambda config: make_module('head.mlp', config))
        self.assertEqual(len(registry.list_registered_models()), 14)
        self.assertEqual(len(registry.list_registered_models(registry=extension)), 15)
        self.assertIn(spec.id, [item.id for item in list_components(registry=extension)])
        bundle = build_component(spec.id, {}, registry=extension)
        model = build_model(graph_for(bundle), capture=False)
        self.assertEqual(list(model(x=torch.ones(2, 3))['logits'].shape), [2, 2])
        with self.assertRaises(ValueError): extension.with_model(spec, lambda _: bundle)
        with self.assertRaises(ValueError): build_component(spec.id, {})
        with self.assertRaises(ValueError): build_component(spec.id, {'weights': None}, registry=extension)

    def test_registry_rejects_misdescribed_native_factories_and_id_collisions(self):
        _, registry = modules()
        from studio_core.module_templates import make_module
        spec = registry.list_registered_models()[0]
        for identifier in ('head.mlp', 'primitive.Identity', 'resnet18', 'backbone.bad-id', 'model.bad_category'):
            bad = copy.deepcopy(spec)
            bad.id = identifier
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                registry.ModelRegistry().with_model(bad, lambda _: None)
        bad = copy.deepcopy(spec)
        bad.config_fields['input_shape'] = {'default': object()}
        with self.assertRaises(ValueError): registry.ModelRegistry().with_model(bad, lambda _: None)
        for change in ('ports', 'inputs', 'batch', 'mutated_config', 'capabilities'):
            bad = copy.deepcopy(spec)
            bad.id = 'backbone.example_mismatch'
            bad.config_defaults = {'in_features': 3, 'hidden_features': 4, 'num_classes': 2, 'resource_batch_bound': 2}
            bad.config_fields = {key: {'type': 'integer'} for key in bad.config_defaults}
            bad.required_config = ('in_features', 'hidden_features', 'num_classes')
            bad.input_ports, bad.output_ports = ('x',), ('logits',)
            bad.inputs = {'x': {'shape': [None, 3], 'dtype': 'float32'}}
            bad.outputs = {'logits': {'shape': [None, 2], 'dtype': 'float32'}}
            bad.resources = {'operator_set': 'vision-ops-v4', 'resource_profile': 'vision-v4', 'resource_batch_bound': 2,
                             'max_parameters': 20_000_000, 'max_tensor_elements': 16_777_216, 'max_execution_elements': 134_217_728}
            if change == 'ports':
                bad.output_ports = ('features',)
                bad.outputs = {'features': {'shape': [None, 2], 'dtype': 'float32'}}
            if change == 'inputs': bad.inputs['x']['shape'] = [None, 4]
            if change == 'capabilities': bad.capabilities['weights'] = 'downloaded'
            def factory(config):
                if change == 'mutated_config': config['resource_batch_bound'] = 1
                bundle = make_module('head.mlp', {'in_features': 3, 'hidden_features': 4, 'num_classes': 2,
                    'resource_batch_bound': 1 if change == 'batch' else config['resource_batch_bound']})
                return bundle
            with self.subTest(change=change), self.assertRaises(ValueError):
                registry.ModelRegistry().with_model(bad, factory).build(bad.id, {})
        for identifier in ('model.resnet18_classifier', 'model.squeezenet1_0_classifier'):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                registry.build_registered_model(identifier, {'num_classes': None, 'feature_outputs': ['features']})

    def test_registry_rejects_malformed_descriptor_shapes_types_dtype_and_limits(self):
        _, registry = modules()
        original = registry.list_registered_models()[0]
        cases = [('config_defaults', []), ('required_config', None), ('required_config', ('input_shape', 'input_shape')),
                 ('input_ports', (['x'],)), ('input_ports', ('x', 'x')),
                 ('output_ports', ('features', 1)), ('config_fields', []),
                 ('inputs', {'x': {}}), ('inputs', {'x': None})]
        for field, value in cases:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                bad = copy.deepcopy(original)
                setattr(bad, field, value)
                registry.ModelRegistry().with_model(bad, lambda _: None)
        for port_type, port, field, value in (
                ('outputs', 'features', 'shape', None), ('outputs', 'features', 'shape', []),
                ('outputs', 'features', 'shape', [None, False]), ('outputs', 'features', 'shape', [None, 0]),
                ('outputs', 'features', 'shape', [None, '']), ('outputs', 'features', 'dtype', 'string'),
                ('inputs', 'x', 'shape', [None, None, 32, 32]), ('inputs', 'x', 'dtype', 'float64')):
            with self.subTest(port_type=port_type, field=field, value=value), self.assertRaises(ValueError):
                bad = copy.deepcopy(original)
                getattr(bad, port_type)[port][field] = value
                registry.ModelRegistry().with_model(bad, lambda _: None)
        for field, value in (('max_parameters', 100_000_000), ('resource_batch_bound', True), ('resource_profile', 'classification-v2')):
            with self.subTest(resource=field), self.assertRaises(ValueError):
                bad = copy.deepcopy(original)
                bad.resources[field] = value
                registry.ModelRegistry().with_model(bad, lambda _: None)

    def test_resnet224_batch2_eval_matches_all_independent_stages(self):
        factories, _ = modules()
        bundle = factories.make_resnet18([None, 3, 224, 224], num_classes=1000,
            feature_outputs=('features', 'stage1', 'stage2', 'stage3', 'stage4'))
        model = build_model(graph_for(bundle), capture=False).eval()
        reference = make_reference('resnet18', num_classes=1000).eval()
        copy_platform_state(reference, model, 'model.resnet18_classifier')
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11_689_512)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            actual, expected = model(x=x), reference(x)
        for port, tensor in actual.items():
            torch.testing.assert_close(tensor, expected[port], rtol=2e-5, atol=2e-6)
        with self.assertRaisesRegex(ValueError, 'resource_batch_bound'):
            model(x=torch.zeros(3, 3, 224, 224))

    def test_representative_families_train_gradients_sgd_and_all_state(self):
        _, registry = modules()
        cases = ('model.resnet18_classifier', 'backbone.vgg11',
                 'model.squeezenet1_0_classifier', 'model.squeezenet1_1_classifier')
        for identifier in cases:
            with self.subTest(identifier=identifier):
                config = {'input_shape': [None, 3, 32, 32]}
                classes = 7 if identifier.startswith('model.') else None
                if classes is not None: config['num_classes'] = classes
                graph = graph_for(registry.build_registered_model(identifier, config))
                model = build_model(graph, capture=False).train()
                reference = make_reference(identifier, num_classes=classes).train()
                copy_platform_state(reference, model, identifier)
                params = canonical_parameters(model, identifier)
                reference_params = dict(reference.named_parameters())
                self.assertEqual(set(params), set(reference_params))
                x = torch.randn(2, 3, 32, 32, requires_grad=True)
                independent = x.detach().clone().requires_grad_()
                torch.manual_seed(728)
                actual = model(x=x)
                torch.manual_seed(728)
                expected = reference(independent)
                for port, tensor in actual.items():
                    torch.testing.assert_close(tensor, expected[port], rtol=2e-5, atol=2e-6)
                sum(tensor.square().mean() for tensor in actual.values()).backward()
                sum(expected[port].square().mean() for port in actual).backward()
                torch.testing.assert_close(x.grad, independent.grad, rtol=2e-5, atol=2e-6)
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(torch.count_nonzero(x.grad).item(), 0)
                for name, parameter in params.items():
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(parameter.grad, reference_params[name].grad, rtol=2e-5, atol=2e-6, msg=name)
                before = {name: parameter.detach().clone() for name, parameter in params.items()}
                torch.optim.SGD(model.parameters(), lr=.03).step()
                torch.optim.SGD(reference.parameters(), lr=.03).step()
                self.assertTrue(any(not torch.equal(parameter, before[name]) for name, parameter in params.items()))
                state = canonical_state_dict(model, identifier)
                self.assertEqual(set(state), set(reference.state_dict()))
                for name, tensor in state.items():
                    torch.testing.assert_close(tensor, reference.state_dict()[name], rtol=2e-5, atol=2e-6, msg=name)
                if 'resnet' in identifier:
                    tracked = [tensor.item() for name, tensor in state.items() if name.endswith('num_batches_tracked')]
                    self.assertEqual(tracked, [1] * 20)
                replay = build_model(graph, capture=False)
                replay.load_state_dict(model.state_dict(), strict=True)
                model.eval(); reference.eval(); replay.eval()
                with torch.no_grad():
                    after = model(x=x.detach())
                    replayed = replay(x=x.detach())
                    reference_after = reference(x.detach())
                for port, tensor in after.items():
                    torch.testing.assert_close(tensor, replayed[port], rtol=0, atol=0)
                    torch.testing.assert_close(tensor, reference_after[port], rtol=2e-5, atol=2e-6)

    def test_python_with_and_without_metadata_strict_state_and_gradients(self):
        _, registry = modules()
        for identifier in ('model.resnet18_classifier', 'backbone.vgg11', 'model.squeezenet1_1_classifier'):
            with self.subTest(identifier=identifier):
                config = {'input_shape': [None, 3, 32, 32]}
                if identifier.startswith('model.'): config['num_classes'] = 5
                graph = graph_for(registry.build_registered_model(identifier, config))
                model = build_model(graph, capture=False).train()
                source = export_python(graph)
                for metadata, code in ((True, source), (False, without_visual(source))):
                    with self.subTest(metadata=metadata):
                        restored = import_python(code)
                        self.assertEqual(restored['resource_profile'], 'hardware-v5')
                        self.assertEqual(restored['resource_batch_bound'], 2)
                        roundtrip = build_model(restored, capture=False).train()
                        roundtrip.load_state_dict(model.state_dict(), strict=True)
                        namespace = {}
                        exec(code, namespace)
                        standalone = namespace['Model']().train()
                        standalone.load_state_dict(model.state_dict(), strict=True)
                        x = torch.randn(2, 3, 32, 32)
                        outputs, inputs = [], []
                        for candidate in (model, roundtrip, standalone):
                            candidate.zero_grad(set_to_none=True)
                            sample = x.clone().requires_grad_()
                            torch.manual_seed(932)
                            output = candidate(x=sample)
                            sum(value.square().mean() for value in output.values()).backward()
                            outputs.append(output); inputs.append(sample)
                        for output in outputs[1:]:
                            self.assertEqual(set(output), set(outputs[0]))
                            for port in output:
                                torch.testing.assert_close(output[port], outputs[0][port], rtol=2e-5, atol=2e-6)
                        for sample in inputs[1:]:
                            torch.testing.assert_close(sample.grad, inputs[0].grad, rtol=2e-5, atol=2e-6)
                        expected_params = dict(model.named_parameters())
                        for candidate in (roundtrip, standalone):
                            self.assertEqual(set(dict(candidate.named_parameters())), set(expected_params))
                            for name, parameter in candidate.named_parameters():
                                torch.testing.assert_close(parameter.grad, expected_params[name].grad, rtol=2e-5, atol=2e-6, msg=name)
                        for candidate in (model, roundtrip, standalone):
                            torch.optim.SGD(candidate.parameters(), lr=.03).step()
                        for candidate in (roundtrip, standalone):
                            actual_state, expected_state = candidate.state_dict(), model.state_dict()
                            self.assertEqual(set(actual_state), set(expected_state))
                            for name, tensor in actual_state.items():
                                torch.testing.assert_close(tensor, expected_state[name], rtol=2e-5, atol=2e-6, msg=name)
                        with self.assertRaisesRegex(ValueError, 'resource_batch_bound'):
                            standalone(x=torch.zeros(3, 3, 32, 32))

    def test_all_variants_python_no_metadata_topology_policy_and_count(self):
        _, registry = modules()
        for identifier in MODEL_IDS:
            with self.subTest(identifier=identifier):
                graph = graph_for(registry.build_registered_model(identifier, {}))
                restored = import_python(without_visual(export_python(graph)))
                self.assertEqual(prepare_graph(restored)[2], prepare_graph(graph)[2])
                self.assertEqual([port['id'] for port in restored['outputs']], [port['id'] for port in graph['outputs']])
                self.assertEqual(restored['resource_batch_bound'], 2)

    def test_vgg_and_squeeze224_batch2_eval_match_independent_standard_features(self):
        _, registry = modules()
        for identifier in ('backbone.vgg11', 'backbone.vgg13', 'backbone.vgg16',
                           'model.squeezenet1_0_classifier', 'model.squeezenet1_1_classifier'):
            with self.subTest(identifier=identifier):
                bundle = registry.build_registered_model(identifier, {'feature_outputs': ['featuremap', 'features']})
                model = build_model(graph_for(bundle), capture=False).eval()
                reference = make_reference(identifier, num_classes=1000 if identifier.startswith('model.') else None).eval()
                copy_platform_state(reference, model, identifier)
                x = torch.randn(2, 3, 224, 224)
                with torch.no_grad():
                    actual, expected = model(x=x), reference(x)
                for port, tensor in actual.items():
                    torch.testing.assert_close(tensor, expected[port], rtol=2e-5, atol=2e-6)
                self.assertEqual(list(actual['featuremap'].shape), [2, 512, 13, 13] if 'squeezenet' in identifier else [2, 512, 7, 7])

    def test_current_python_source_classifier_edit_wins_over_stale_metadata(self):
        _, registry = modules()
        for identifier in ('model.resnet18_classifier', 'model.squeezenet1_1_classifier'):
            graph = graph_for(registry.build_registered_model(identifier, {'input_shape': [None, 3, 32, 32], 'num_classes': 5}))
            source = export_python(graph)
            for metadata, code in ((True, source), (False, without_visual(source))):
                with self.subTest(identifier=identifier, metadata=metadata):
                    tree = ast.parse(code)
                    changed = 0
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('Linear', 'Conv2d'):
                            for keyword in node.keywords:
                                if keyword.arg in ('out_features', 'out_channels') and isinstance(keyword.value, ast.Constant) and keyword.value.value == 5:
                                    keyword.value = ast.Constant(7)
                                    changed += 1
                    self.assertEqual(changed, 1)
                    edited = ast.unparse(ast.fix_missing_locations(tree))
                    restored = import_python(edited)
                    self.assertEqual(prepare_graph(restored)[1]['call:entry:logits'], [None, 7])
                    self.assertEqual(prepare_graph(restored)[2] - prepare_graph(graph)[2], 1026)
                    model = build_model(restored, capture=False).eval()
                    namespace = {}; exec(edited, namespace)
                    standalone = namespace['Model']().eval()
                    standalone.load_state_dict(model.state_dict(), strict=True)
                    x = torch.randn(2, 3, 32, 32)
                    with torch.no_grad():
                        torch.testing.assert_close(model(x=x)['logits'], standalone(x=x)['logits'])

    def test_fire_instances_and_reused_calls_preserve_weight_identity(self):
        _, registry = modules()
        bundle = registry.build_registered_model('backbone.squeezenet1_1', {'input_shape': [None, 3, 32, 32]})
        for shared in (True, False):
            with self.subTest(shared=shared):
                graph = graph_for(bundle)
                graph['inputs'].append({'id': 'other', 'shape': [None, 3, 32, 32], 'dtype': 'float32'})
                if not shared:
                    graph['instances'].append({**copy.deepcopy(bundle.entry), 'id': 'other'})
                graph['calls'].append({'id': 'other', 'instance_id': 'entry' if shared else 'other', 'inputs': {'x': 'input:other'}})
                graph['outputs'].append({'id': 'other', 'source': 'call:other:features'})
                model = build_model(graph, capture=False)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 722_496 * (1 if shared else 2))
                first = model.components['entry']
                state_first = dict(first.named_parameters())
                # Repeated Fire shapes still have independent parameters.
                layers = first.components['feature_layers'].components
                self.assertNotEqual(layers['fire11'].components['squeeze'].weight.data_ptr(),
                                    layers['fire12'].components['squeeze'].weight.data_ptr())
                if not shared:
                    second = model.components['other']
                    second.load_state_dict(first.state_dict(), strict=True)
                    state_second = dict(second.named_parameters())
                    for name, parameter in state_first.items():
                        self.assertNotEqual(parameter.data_ptr(), state_second[name].data_ptr())
                x, other = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
                output = model(x=x, other=other)
                output['features'].square().mean().backward()
                if not shared:
                    self.assertTrue(all(p.grad is None for p in model.components['other'].parameters()))
                self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in state_first.values()))
                with self.assertRaisesRegex(ValueError, 'parameter budget'):
                    oversized = graph_for(registry.build_registered_model('backbone.resnet18', {'input_shape': [None, 3, 32, 32], 'resource_profile': 'vision-v4'}))
                    oversized['instances'].append({**copy.deepcopy(oversized['instances'][0]), 'id': 'other'})
                    prepare_graph(oversized)


if __name__ == '__main__':
    unittest.main()
