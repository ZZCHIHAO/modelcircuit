"""Factory contracts measured against independent Torch formulas and modules."""
import ast
import copy
import importlib.util
import json
import subprocess
import sys
import unittest
from dataclasses import asdict

import torch
from torch import nn

from component_fixture import bundle_graph, bundle_model
from studio_core.python_io import export_python, import_python
from studio_core.runtime import build_model
from studio_core.schema import validate_graph


POLICY = {'resource_profile': 'vision-v4', 'resource_batch_bound': 2}
CASES = [
    ('head.mlp', {'in_features': 3, 'hidden_features': 4, 'num_classes': 2}, {'x': [None, 3]}, ['logits']),
    ('fusion.concat_projection', {'a_features': 2, 'b_features': 3, 'out_features': 4},
     {'a': [None, 2], 'b': [None, 3]}, ['features']),
    ('fusion.weighted_sum', {'features': 3}, {'a': [None, 3], 'b': [None, 3]}, ['features']),
    ('fusion.gated', {'features': 3}, {'a': [None, 3], 'b': [None, 3]}, ['features', 'gate']),
    ('block.residual_basic', {'input_shape': [None, 2, 5, 5], 'in_channels': 2, 'out_channels': 2},
     {'x': [None, 2, 5, 5]}, ['features']),
]


def without_visual(source):
    tree = ast.parse(source)
    tree.body = [item for item in tree.body if not (isinstance(item, ast.Assign) and
        any(isinstance(target, ast.Name) and target.id == 'STUDIO_VISUAL_METADATA' for target in item.targets))]
    return ast.unparse(tree)


class ReferenceGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 3)

    def forward(self, a, b):
        gate = torch.sigmoid(self.linear(torch.cat([a, b], dim=1)))
        return {'features': gate * a + (1 - gate) * b, 'gate': gate}


class ReferenceBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outgoing)
        if incoming != outgoing or stride != 1:
            self.shortcut_conv = nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(outgoing)

    def forward(self, x):
        residual = self.shortcut_bn(self.shortcut_conv(x)) if hasattr(self, 'shortcut_conv') else x
        y = self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(x)))))
        return torch.relu(y + residual)


class ModuleTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make(self, identifier, config):
        self.assertIsNotNone(importlib.util.find_spec('studio_core.module_templates'),
                             'A3 make_module factory is not implemented')
        from studio_core.module_templates import make_module
        return make_module(identifier, config)

    def graph(self, bundle, shapes):
        policy = {key: value for key, value in bundle.requirements.items() if key != 'operator_set'}
        return bundle_graph(bundle, shapes, policy)

    def test_weighted_sum_alpha_changes_complement_consistently(self):
        a, b = torch.tensor([[1., 3.]]), torch.tensor([[5., 7.]])
        for alpha in (0., 0.25, 1., 1.25, -0.5):
            with self.subTest(alpha=alpha):
                bundle = self.make('fusion.weighted_sum', {'features': 2, 'alpha': alpha})
                model = bundle_model(bundle, {'a': [None, 2], 'b': [None, 2]}, POLICY)
                torch.testing.assert_close(model(a=a, b=b)['features'], a + alpha * (b - a))

    def test_weighted_override_gradients_and_no_stale_complement(self):
        bundle = self.make('fusion.weighted_sum', {'features': 2, 'alpha': 0.25})
        public = bundle.definitions[0]['public_parameters']
        self.assertEqual(public, [{'id': 'alpha', 'type': 'float', 'default': 0.25}])
        self.assertEqual(len(bundle.definitions[0]['bindings']['alpha']), 1)
        for alpha in (-2., 0., 0.75, 2.):
            bundle.entry['params'] = {'alpha': alpha}
            model = bundle_model(bundle, {'a': [None, 2], 'b': [None, 2]}, POLICY)
            a, b = torch.randn(2, 2, requires_grad=True), torch.randn(2, 2, requires_grad=True)
            torch.testing.assert_close(model(a=a, b=b)['features'], a + alpha * (b - a))
            model(a=a, b=b)['features'].sum().backward()
            torch.testing.assert_close(a.grad, torch.full_like(a, 1 - alpha))
            torch.testing.assert_close(b.grad, torch.full_like(b, alpha))
            self.assertEqual(list(model.parameters()), [])

    def test_mlp_and_concat_match_independent_native_modules_and_gradients(self):
        for identifier, config, shapes, ports in CASES[:2]:
            with self.subTest(identifier=identifier):
                model = build_model(self.graph(self.make(identifier, config), shapes))
                chip = model.components['entry'].components
                if identifier == 'head.mlp':
                    reference = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
                    mapping = [(chip['linear1'], reference[0]), (chip['linear2'], reference[2])]
                else:
                    reference = nn.Sequential(nn.Linear(5, 4), nn.ReLU())
                    mapping = [(chip['linear'], reference[0])]
                for actual, expected in mapping:
                    expected.load_state_dict(actual.state_dict(), strict=True)
                inputs = {key: torch.randn(2, shape[1], requires_grad=True) for key, shape in shapes.items()}
                independent = {key: value.detach().clone().requires_grad_() for key, value in inputs.items()}
                expected = reference(independent['x'] if identifier == 'head.mlp' else
                                     torch.cat([independent['a'], independent['b']], dim=1))
                actual = model(**inputs)[ports[0]]
                torch.testing.assert_close(actual, expected)
                actual.square().sum().backward(); expected.square().sum().backward()
                for key in inputs:
                    torch.testing.assert_close(inputs[key].grad, independent[key].grad)
                for actual_module, expected_module in mapping:
                    for actual_param, expected_param in zip(actual_module.parameters(), expected_module.parameters()):
                        torch.testing.assert_close(actual_param.grad, expected_param.grad)

    def test_gated_outputs_gradients_and_one_parameter_update_match_reference(self):
        bundle = self.make('fusion.gated', {'features': 3})
        model = bundle_model(bundle, {'a': [None, 3], 'b': [None, 3]}, POLICY)
        reference = ReferenceGate()
        reference.linear.load_state_dict(model.components['entry'].components['gate_linear'].state_dict())
        inputs = {key: torch.randn(2, 3, requires_grad=True) for key in ('a', 'b')}
        independent = {key: value.detach().clone().requires_grad_() for key, value in inputs.items()}
        actual, expected = model(**inputs), reference(**independent)
        self.assertEqual(set(actual), {'features', 'gate'})
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key])
        sum(value.square().sum() for value in actual.values()).backward()
        sum(value.square().sum() for value in expected.values()).backward()
        for key in inputs:
            torch.testing.assert_close(inputs[key].grad, independent[key].grad)
        before = reference.linear.weight.detach().clone()
        for actual_param, expected_param in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual_param.grad, expected_param.grad)
        torch.optim.SGD(model.parameters(), lr=0.05).step()
        torch.optim.SGD(reference.parameters(), lr=0.05).step()
        self.assertFalse(torch.equal(before, reference.linear.weight))
        for actual_param, expected_param in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual_param, expected_param)

    def test_basic_block_identity_and_projection_forward_gradient_update_buffers(self):
        for incoming, outgoing, stride in ((2, 2, 1), (2, 3, 2), (2, 3, 1)):
            with self.subTest(incoming=incoming, outgoing=outgoing, stride=stride):
                bundle = self.make('block.residual_basic', {'input_shape': [None, incoming, 5, 5],
                    'in_channels': incoming, 'out_channels': outgoing, 'stride': stride})
                model = bundle_model(bundle, {'x': [None, incoming, 5, 5]}, POLICY)
                chip = model.components['entry'].components
                reference = ReferenceBlock(incoming, outgoing, stride)
                for name, component in reference.named_children():
                    component.load_state_dict(chip[name].state_dict(), strict=True)
                x = torch.randn(2, incoming, 5, 5, requires_grad=True)
                independent = x.detach().clone().requires_grad_()
                actual, expected = model(x=x)['features'], reference(independent)
                self.assertEqual(list(actual.shape), [2, outgoing, 3 if stride == 2 else 5, 3 if stride == 2 else 5])
                torch.testing.assert_close(actual, expected)
                actual.square().mean().backward(); expected.square().mean().backward()
                torch.testing.assert_close(x.grad, independent.grad)
                for name, component in reference.named_children():
                    for actual_param, expected_param in zip(chip[name].parameters(), component.parameters()):
                        torch.testing.assert_close(actual_param.grad, expected_param.grad)
                torch.optim.SGD(model.parameters(), lr=0.03).step()
                torch.optim.SGD(reference.parameters(), lr=0.03).step()
                for name, component in reference.named_children():
                    for key, expected_value in component.state_dict().items():
                        torch.testing.assert_close(chip[name].state_dict()[key], expected_value)
                model.eval(); reference.eval()
                torch.testing.assert_close(model(x=x.detach())['features'], reference(x.detach()))

    def test_shared_and_independent_instances_keep_distinct_parameter_and_bn_state(self):
        bundle = self.make('block.residual_basic', CASES[-1][1])
        for shared in (True, False):
            graph = self.graph(bundle, CASES[-1][2])
            graph['inputs'].append({'id': 'other', 'shape': [None, 2, 5, 5], 'dtype': 'float32'})
            if not shared:
                graph['instances'].append({**copy.deepcopy(graph['instances'][0]), 'id': 'other'})
            graph['calls'].append({'id': 'other', 'instance_id': 'entry' if shared else 'other', 'inputs': {'x': 'input:other'}})
            graph['outputs'].append({'id': 'other', 'source': 'call:other:features'})
            model = build_model(graph)
            first = model.components['entry'].components
            if not shared:
                second = model.components['other'].components
                model.components['other'].load_state_dict(model.components['entry'].state_dict())
                self.assertNotEqual(first['conv1'].weight.data_ptr(), second['conv1'].weight.data_ptr())
            x, y = torch.randn(2, 2, 5, 5), torch.randn(2, 2, 5, 5) + 5
            outputs = model(x=x, other=y)
            self.assertEqual(first['bn1'].num_batches_tracked.item(), 2 if shared else 1)
            outputs['features'].square().mean().backward()
            before = first['conv1'].weight.detach().clone()
            other_before = None if shared else second['conv1'].weight.detach().clone()
            torch.optim.SGD(model.parameters(), lr=0.01).step()
            self.assertFalse(torch.equal(first['conv1'].weight, before))
            if not shared:
                self.assertIsNone(second['conv1'].weight.grad)
                torch.testing.assert_close(second['conv1'].weight, other_before)
                self.assertEqual(second['bn1'].num_batches_tracked.item(), 1)
                self.assertFalse(torch.equal(first['bn1'].running_mean, second['bn1'].running_mean))

    def test_all_factories_named_ports_fresh_json_version_layout_and_explicit_policy(self):
        for identifier, config, shapes, ports in CASES:
            with self.subTest(identifier=identifier):
                before = copy.deepcopy(config)
                bundle = self.make(identifier, config)
                self.assertEqual(bundle.requirements, {'operator_set': 'vision-ops-v4', **POLICY})
                self.assertEqual(config, before)
                self.assertEqual(json.loads(json.dumps(asdict(bundle), allow_nan=False)), asdict(bundle))
                self.assertTrue(validate_graph(self.graph(bundle, shapes))['valid'])
                self.assertEqual([output['id'] for output in bundle.definitions[0]['graph']['outputs']], ports)
                self.assertTrue(all(definition['version'] == '1.0.0' for definition in bundle.definitions))
                calls = bundle.definitions[0]['graph']['calls']
                self.assertEqual(len({tuple(call['position'].values()) for call in calls}), len(calls))
                unchanged = copy.deepcopy(asdict(bundle))
                bundle.entry['params']['injected'] = 99
                bundle.definitions[0]['graph']['instances'].clear()
                bundle.requirements['resource_batch_bound'] = 64
                self.assertEqual(asdict(self.make(identifier, config)), unchanged)

    def test_shared_and_independent_python_instances_keep_calls_gradients_and_buffers(self):
        bundle = self.make('block.residual_basic', CASES[-1][1])
        for shared in (True, False):
            graph = self.graph(bundle, CASES[-1][2])
            graph['inputs'].append({'id': 'other', 'shape': [None, 2, 5, 5], 'dtype': 'float32'})
            if not shared:
                graph['instances'].append({**copy.deepcopy(graph['instances'][0]), 'id': 'other'})
            graph['calls'].append({'id': 'other', 'instance_id': 'entry' if shared else 'other', 'inputs': {'x': 'input:other'}})
            graph['outputs'].append({'id': 'other', 'source': 'call:other:features'})
            initial = build_model(graph)
            state = copy.deepcopy(initial.state_dict())
            candidates = [initial]
            for source in (export_python(graph), without_visual(export_python(graph))):
                candidates.append(build_model(import_python(source)))
                namespace = {}; exec(source, namespace)
                candidates.append(namespace['Model']())
            for candidate in candidates:
                with self.subTest(shared=shared, model=type(candidate).__name__):
                    candidate.load_state_dict(state, strict=True)
                    candidate.train()
                    first = ReferenceBlock(2, 2, 1)
                    second = first if shared else ReferenceBlock(2, 2, 1)
                    for entry_name, reference in (('entry', first), ('entry' if shared else 'other', second)):
                        for name, component in reference.named_children():
                            component.load_state_dict(candidate.components[entry_name].components[name].state_dict(), strict=True)
                    a, b = torch.randn(2, 2, 5, 5, requires_grad=True), torch.randn(2, 2, 5, 5, requires_grad=True)
                    ref_a, ref_b = a.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
                    expected = {'features': first(ref_a), 'other': second(ref_b)}
                    actual = candidate(x=a, other=b)
                    for key in expected:
                        torch.testing.assert_close(actual[key], expected[key])
                    sum(value.square().mean() for value in actual.values()).backward()
                    sum(value.square().mean() for value in expected.values()).backward()
                    torch.testing.assert_close(a.grad, ref_a.grad)
                    torch.testing.assert_close(b.grad, ref_b.grad)
                    references = [('entry', first)] if shared else [('entry', first), ('other', second)]
                    for entry_name, reference in references:
                        chip = candidate.components[entry_name].components
                        self.assertEqual(chip['bn1'].num_batches_tracked.item(), 2 if shared else 1)
                        for name, component in reference.named_children():
                            for key, value in component.state_dict().items():
                                torch.testing.assert_close(chip[name].state_dict()[key], value)
                            for actual_param, expected_param in zip(chip[name].parameters(), component.parameters()):
                                torch.testing.assert_close(actual_param.grad, expected_param.grad)

    def test_closure_bundle_versions_and_unrelated_or_recursive_definitions_rejected(self):
        from native.chip_library import validate_bundle
        for identifier, config, shapes, ports in CASES:
            bundle = self.make(identifier, config)
            payload = {'chip_format_version': 2, 'definition_id': bundle.entry['definition_id'],
                       'definitions': bundle.definitions, 'requirements': bundle.requirements}
            validate_bundle(payload)
            for version in ('1', '1.0', '', None, 1):
                wrong = copy.deepcopy(payload); wrong['definitions'][0]['version'] = version
                with self.subTest(identifier=identifier, version=version), self.assertRaises(ValueError):
                    validate_bundle(wrong)
            wrong = copy.deepcopy(payload)
            extra = copy.deepcopy(wrong['definitions'][0]); extra['id'] = 'Unrelated'
            wrong['definitions'].append(extra)
            with self.assertRaisesRegex(ValueError, 'unrelated'):
                validate_bundle(wrong)
            wrong = copy.deepcopy(payload)
            wrong['definitions'][0]['graph']['instances'].append({'id': 'reentry', 'kind': 'chip',
                'definition_id': payload['definition_id'], 'params': {}})
            with self.assertRaisesRegex(ValueError, 'Recursive'):
                validate_bundle(wrong)

    def test_required_unknown_type_dimension_and_batch_config_rejected(self):
        for identifier, config, shapes, ports in CASES:
            optional = {'stride'} if identifier == 'block.residual_basic' else set()
            for key in config.keys() - optional:
                changed = dict(config); changed.pop(key)
                with self.subTest(identifier=identifier, missing=key), self.assertRaises(ValueError):
                    self.make(identifier, changed)
            for bad in (None, [], {'unknown': 3}, {**config, 'unknown': 1}):
                with self.subTest(identifier=identifier, bad=bad), self.assertRaises(ValueError):
                    self.make(identifier, bad)
            for key in config.keys() - {'input_shape'}:
                for bad in (True, 0, -1, 1.5, '2', float('inf'), float('nan')):
                    with self.subTest(identifier=identifier, key=key, bad=bad), self.assertRaises(ValueError):
                        self.make(identifier, {**config, key: bad})
            for bad in (True, 0, 65, 2.0, None):
                with self.subTest(identifier=identifier, bound=bad), self.assertRaises(ValueError):
                    self.make(identifier, {**config, 'resource_batch_bound': bad})
        for identifier in ('not.registered', None, 2, []):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                self.make(identifier, {})

    def test_alpha_rejects_nonfinite_bool_and_non_json_numbers_in_all_paths(self):
        for bad in (True, False, float('nan'), float('inf'), -float('inf'), '0.5', None, 10 ** 500):
            with self.subTest(bad=type(bad).__name__), self.assertRaises(ValueError):
                self.make('fusion.weighted_sum', {'features': 2, 'alpha': bad})
            bundle = self.make('fusion.weighted_sum', {'features': 2})
            bundle.entry['params'] = {'alpha': bad}
            self.assertFalse(validate_graph(self.graph(bundle, {'a': [None, 2], 'b': [None, 2]}))['valid'])

    def test_block_rejects_invalid_shape_channel_stride_and_resource_overflow(self):
        base = CASES[-1][1]
        for shape in ([None, 3, 5, 5], [None, 2, None, 5], [None, 2, 0, 5], [3, 2, 5, 5],
                      [True, 2, 5, 5], [None, 2, True, 5], [None, 2, 5], (None, 2, 5, 5)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.make('block.residual_basic', {**base, 'input_shape': shape})
        for stride in (0, 3, True, 1.0):
            with self.subTest(stride=stride), self.assertRaises(ValueError):
                self.make('block.residual_basic', {**base, 'stride': stride})
        with self.assertRaises(ValueError):
            self.make('head.mlp', {'in_features': 20_000_000, 'hidden_features': 2, 'num_classes': 2})

    def test_batch_bound_endpoints_and_concrete_block_batch(self):
        for bound in (1, 64):
            bundle = self.make('fusion.weighted_sum', {'features': 2, 'resource_batch_bound': bound})
            model = build_model(self.graph(bundle, {'a': [None, 2], 'b': [None, 2]}))
            model(a=torch.ones(bound, 2), b=torch.zeros(bound, 2))
            with self.assertRaisesRegex(ValueError, 'resource_batch_bound'):
                model(a=torch.ones(bound + 1, 2), b=torch.zeros(bound + 1, 2))
        bundle = self.make('block.residual_basic', {**CASES[-1][1], 'input_shape': [2, 2, 5, 5]})
        self.assertEqual(bundle.definitions[0]['graph']['inputs'][0]['shape'], [2, 2, 5, 5])

    def test_python_roundtrip_with_and_without_metadata_preserves_real_state_and_ports(self):
        for identifier, config, shapes, ports in CASES:
            bundle = self.make(identifier, config)
            graph = self.graph(bundle, shapes)
            original = build_model(graph)
            original.train()
            inputs = {name: torch.randn(*[2 if axis is None else axis for axis in shape]) for name, shape in shapes.items()}
            original(**inputs)  # Populate BN buffers before exact state copying.
            original.eval()
            for source in (export_python(graph), without_visual(export_python(graph))):
                with self.subTest(identifier=identifier, metadata='STUDIO_VISUAL_METADATA' in source):
                    restored = import_python(source)
                    self.assertEqual({key: restored[key] for key in POLICY}, POLICY)
                    roundtrip = build_model(restored)
                    roundtrip.load_state_dict(original.state_dict(), strict=True)
                    namespace = {}; exec(source, namespace)
                    standalone = namespace['Model']()
                    standalone.load_state_dict(original.state_dict(), strict=True)
                    roundtrip.eval(); standalone.eval()
                    expected = original(**inputs)
                    for candidate in (roundtrip, standalone):
                        actual = candidate(**inputs)
                        self.assertEqual(list(actual), ports)
                        for name in ports:
                            torch.testing.assert_close(actual[name], expected[name])
                        for key, value in original.state_dict().items():
                            torch.testing.assert_close(candidate.state_dict()[key], value)

    def test_python_alpha_source_override_extrapolates_and_rejects_invalid_values(self):
        bundle = self.make('fusion.weighted_sum', {'features': 2})
        source = without_visual(export_python(self.graph(bundle, {'a': [None, 2], 'b': [None, 2]})))
        for alpha in (-1.25, 1.25):
            tree = ast.parse(source)
            chip = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Chip_WeightedSum')
            constructor = next(node for node in chip.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
            constructor.args.defaults[0] = ast.Constant(alpha)
            changed = ast.unparse(ast.fix_missing_locations(tree))
            model = build_model(import_python(changed))
            a, b = torch.randn(2, 2), torch.randn(2, 2)
            torch.testing.assert_close(model(a=a, b=b)['features'], a + alpha * (b - a))
        for bad in (True, float('inf'), float('nan')):
            tree = ast.parse(source)
            chip = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Chip_WeightedSum')
            constructor = next(node for node in chip.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
            constructor.args.defaults[0] = ast.Constant(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                import_python(ast.unparse(ast.fix_missing_locations(tree)))

    def test_invalid_interfaces_and_closure_are_rejected_by_real_consumers(self):
        bundle = self.make('fusion.gated', {'features': 3})
        graph = self.graph(bundle, {'a': [None, 3], 'b': [None, 3]})
        changes = []
        wrong = copy.deepcopy(graph); wrong['calls'][0]['inputs'].pop('b'); changes.append(wrong)
        wrong = copy.deepcopy(graph); wrong['outputs'][0]['source'] = 'call:entry:missing'; changes.append(wrong)
        wrong = copy.deepcopy(graph); wrong['definitions'] = []; changes.append(wrong)
        wrong = copy.deepcopy(graph); wrong['definitions'][0]['graph']['instances'][0] = {
            'id': 'recursive', 'kind': 'chip', 'definition_id': bundle.entry['definition_id'], 'params': {}}
        changes.append(wrong)
        wrong = copy.deepcopy(graph); wrong['inputs'][1]['shape'] = [None, 4]; changes.append(wrong)
        for invalid in changes:
            self.assertFalse(validate_graph(invalid)['valid'])
        with self.assertRaises(ValueError):
            bundle_model(bundle, {'a': [None, 3]}, POLICY)

    def test_factory_import_and_execution_stay_torch_and_gui_free(self):
        self.make('fusion.weighted_sum', {'features': 2})
        script = '''import sys
from studio_core.module_templates import make_module
make_module('block.residual_basic', {'input_shape': [None, 2, 5, 5], 'in_channels': 2, 'out_channels': 3, 'stride': 2})
assert not any(name == 'torch' or name.startswith(('torch.', 'dearpygui', 'PySide', 'PyQt')) for name in sys.modules)
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
