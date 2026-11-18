"""Static Python v4, standalone operator/state semantics and policy provenance."""
import ast
import copy
import unittest

from studio_core.python_io import export_python, import_python, python_profile
from test_vision_policy import identity_graph


def without_visual(code):
    tree = ast.parse(code)
    tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign) and
        any(isinstance(target, ast.Name) and target.id == 'STUDIO_VISUAL_METADATA' for target in node.targets))]
    assert not any(isinstance(node, ast.Name) and node.id == 'STUDIO_VISUAL_METADATA' for node in ast.walk(tree))
    return ast.unparse(tree)


def operator_graph(affine=True, track=True, vision=True):
    graph = identity_graph(2, [None, 3, 2, 2])
    graph['instances'] = [{'id': 'bn', 'kind': 'BatchNorm2d', 'params': {'num_features': 3, 'affine': affine, 'track_running_stats': track}, 'frozen': True},
        {'id': 'identity', 'kind': 'Identity', 'params': {}}, {'id': 'sigmoid', 'kind': 'Sigmoid', 'params': {}},
        {'id': 'multiply', 'kind': 'Multiply', 'params': {}}, {'id': 'scale', 'kind': 'Scale', 'params': {'factor': 2}},
        {'id': 'inverse', 'kind': 'OneMinus', 'params': {}}]
    graph['calls'] = [{'id': 'bn0', 'instance_id': 'bn', 'inputs': {'x': 'input:x'}},
        {'id': 'bn1', 'instance_id': 'bn', 'inputs': {'x': 'call:bn0'}},
        {'id': 'identity', 'instance_id': 'identity', 'inputs': {'x': 'call:bn1'}},
        {'id': 'sigmoid', 'instance_id': 'sigmoid', 'inputs': {'x': 'call:identity'}},
        {'id': 'multiply', 'instance_id': 'multiply', 'inputs': {'a': 'call:sigmoid', 'b': 'call:identity'}},
        {'id': 'scale', 'instance_id': 'scale', 'inputs': {'x': 'call:multiply'}},
        {'id': 'inverse', 'instance_id': 'inverse', 'inputs': {'x': 'call:scale'}}]
    graph['outputs'] = [{'id': 'out', 'source': 'call:inverse'}]
    if not vision:
        graph.pop('resource_profile')
        graph.pop('resource_batch_bound')
    return graph


class VisionPythonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def test_policy_survives_no_metadata_and_current_source_wins(self):
        graph = identity_graph()
        self.assertEqual(python_profile(graph), 'vision-pytorch-v4')
        code = export_python(graph)
        self.assertIn('MLFORGE_EXECUTION_POLICY', code)
        for source in (code, without_visual(code)):
            restored = import_python(source)
            self.assertEqual(restored['resource_profile'], 'vision-v4')
            self.assertEqual(restored['resource_batch_bound'], 2)
        edited = code.replace("'resource_batch_bound': 2", "'resource_batch_bound': 3")
        self.assertEqual(import_python(edited)['resource_batch_bound'], 3)
        namespace = {}
        exec(without_visual(code), namespace)
        model = namespace['Model']()
        model(x=self.torch.ones(1, 3, 8, 8))
        with self.assertRaisesRegex(ValueError, 'resource_batch_bound'):
            model(x=self.torch.ones(3, 3, 8, 8))

    def test_legacy_profiles_keep_new_helper_names_as_public_parameters(self):
        from studio_core.schema import validate_graph
        for profile in ('s1-pytorch-v1', 'classification-pytorch-v2', 'ports-pytorch-v3'):
            for name in ('Scale', 'Multiply', 'OneMinus', 'BatchNorm2d', '_check_execution_policy', 'MLFORGE_EXECUTION_POLICY'):
                graph = identity_graph()
                graph.pop('resource_profile'); graph.pop('resource_batch_bound')
                graph['inputs'] = [{'id': '_check_execution_policy', 'shape': [None, 3], 'dtype': 'float32'}]
                definition = {'id': 'block', 'public_parameters': [{'id': name, 'default': 1}], 'bindings': {}, 'graph': {
                    'inputs': [{'id': 'x', 'shape': [None, 3], 'dtype': 'float32'}],
                    'instances': [{'id': 'relu', 'kind': 'ReLU', 'params': {}}],
                    'calls': [{'id': 'relu', 'instance_id': 'relu', 'inputs': {'x': 'input:x'}}],
                    'outputs': [{'id': 'out', 'source': 'call:relu'}]}}
                if profile == 'classification-pytorch-v2': graph['resource_profile'] = 'classification-v2'
                if profile == 'ports-pytorch-v3': definition['public_parameters'][0]['type'] = 'int'
                graph['definitions'] = [definition]
                graph['instances'] = [{'id': 'block', 'kind': 'chip', 'definition_id': 'block', 'params': {}}]
                graph['calls'] = [{'id': 'block', 'instance_id': 'block', 'inputs': {'x': 'input:_check_execution_policy'}}]
                graph['outputs'] = [{'id': 'out', 'source': 'call:block'}]
                with self.subTest(profile=profile, name=name):
                    self.assertTrue(validate_graph(graph)['valid'])
                    self.assertEqual(python_profile(graph), profile)
                    source = export_python(graph)
                    restored = import_python(source)
                    self.assertEqual(restored['definitions'][0]['public_parameters'][0]['id'], name)
                    namespace = {}; exec(source, namespace)
                    self.assertEqual(tuple(namespace['Model']()(self.torch.ones(2, 3))['out'].shape), (2, 3))
                    if name == '_check_execution_policy':
                        graph['instances'].append({'id': 'unused_identity', 'kind': 'Identity', 'params': {}})
                        source = export_python(graph)
                        self.assertEqual(python_profile(graph), 'vision-pytorch-v4')
                        self.assertEqual(import_python(source)['definitions'][0]['public_parameters'][0]['id'], name)

    def test_legacy_handwritten_class_named_scale_is_a_chip(self):
        source = '''import torch
from torch import nn
class Scale(nn.Module):
    def __init__(self, width=1):
        super().__init__()
        self.components = nn.ModuleDict({'relu': nn.ReLU()})
    def forward(self, x: 'float32[?,3]'):
        out = self.components['relu'](x)
        return out
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.components = nn.ModuleDict({'block': Scale()})
    def forward(self, x: 'float32[?,3]'):
        out = self.components['block'](x)
        return {'out': out}
'''
        restored = import_python(source)
        self.assertEqual(restored['definitions'][0]['id'], 'Scale')
        self.assertEqual(restored['instances'][0]['kind'], 'chip')

    def test_vision_source_requires_exact_root_guard_and_helper(self):
        source = export_python(identity_graph())
        for metadata in (True, False):
            code = source if metadata else without_visual(source)
            for edit in ('missing_guard', 'missing_helper', 'moved_guard', 'replaced_guard'):
                tree = ast.parse(code)
                model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Model')
                forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
                if edit == 'missing_guard': forward.body.pop(0)
                elif edit == 'missing_helper': tree.body = [node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name == '_check_execution_policy')]
                elif edit == 'moved_guard': forward.body.insert(1, forward.body.pop(0))
                else: forward.body[0].value.args = []
                changed = ast.unparse(tree)
                with self.subTest(metadata=metadata, edit=edit), self.assertRaisesRegex(ValueError, 'guard|helper'):
                    import_python(changed)
                if edit == 'missing_guard':
                    namespace = {}; exec(changed, namespace)
                    self.assertEqual(namespace['Model']()(self.torch.ones(3, 3, 8, 8))['out'].shape[0], 3)
        valid = source.replace("'resource_batch_bound': 2", "'resource_batch_bound': 3")
        self.assertEqual(import_python(valid)['resource_batch_bound'], 3)
        namespace = {}; exec(valid, namespace)
        self.assertEqual(namespace['Model']()(self.torch.ones(3, 3, 8, 8))['out'].shape[0], 3)
        for policy in ({'resource_profile': None}, {'resource_profile': 'classification-v2'}):
            changed = source.replace("{'resource_profile': 'vision-v4', 'resource_batch_bound': 2}", repr(policy))
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, 'guard|helper'): import_python(changed)

    def test_v4_helpers_still_reserve_names_and_reject_internal_guards(self):
        graph = identity_graph()
        scope = {key: copy.deepcopy(graph[key]) for key in ('inputs', 'instances', 'calls', 'outputs')}
        graph['definitions'] = [{'id': 'block', 'public_parameters': [{'id': 'Scale', 'default': 1}], 'bindings': {}, 'graph': scope}]
        with self.assertRaisesRegex(ValueError, 'reserved'): export_python(graph)
        graph['definitions'][0]['public_parameters'] = []
        source = export_python(graph)
        tree = ast.parse(source)
        chip = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Chip_block')
        next(node for node in chip.body if isinstance(node, ast.FunctionDef) and node.name == 'forward').body.insert(0, ast.parse('_check_execution_policy(x)').body[0])
        with self.assertRaisesRegex(ValueError, 'guard|helper'): import_python(ast.unparse(tree))

    def test_active_policy_and_guard_globals_cannot_be_rebound_by_chip_classes(self):
        graph = identity_graph()
        scope = {key: copy.deepcopy(graph[key]) for key in ('inputs', 'instances', 'calls', 'outputs')}
        graph['definitions'] = [{'id': 'block', 'public_parameters': [], 'bindings': {}, 'graph': scope}]
        graph['instances'] = [{'id': 'block', 'kind': 'chip', 'definition_id': 'block', 'params': {}}]
        graph['calls'] = [{'id': 'block', 'instance_id': 'block', 'inputs': {'x': 'input:x'}}]
        graph['outputs'] = [{'id': 'out', 'source': 'call:block'}]
        source = export_python(graph)
        for symbol in ('_check_execution_policy', 'MLFORGE_EXECUTION_POLICY'):
            changed = source.replace('Chip_block', symbol)
            for code in (changed, without_visual(changed)):
                with self.subTest(symbol=symbol), self.assertRaisesRegex(ValueError, 'conflict|binding|symbol'):
                    import_python(code)
            namespace = {}; exec(changed, namespace)
            with self.assertRaises(TypeError): namespace['Model']()(self.torch.ones(2, 3, 8, 8))

    def test_active_global_bindings_cannot_be_overwritten_by_imports(self):
        source = export_python(identity_graph())
        for symbol in ('_check_execution_policy', 'MLFORGE_EXECUTION_POLICY', 'Scale', 'Multiply', 'OneMinus', 'BatchNorm2d'):
            changed = source + '\nimport torch as ' + symbol + '\n'
            with self.subTest(symbol=symbol), self.assertRaisesRegex(ValueError, 'conflict|binding|symbol'):
                import_python(changed)

    def test_inactive_policy_symbol_names_remain_ordinary_legacy_chip_classes(self):
        for symbol in ('_check_execution_policy', 'MLFORGE_EXECUTION_POLICY'):
            graph = identity_graph()
            graph.pop('resource_profile'); graph.pop('resource_batch_bound')
            graph['instances'][0]['kind'] = 'ReLU'
            scope = {key: copy.deepcopy(graph[key]) for key in ('inputs', 'instances', 'calls', 'outputs')}
            graph['definitions'] = [{'id': 'block', 'public_parameters': [], 'bindings': {}, 'graph': scope}]
            graph['instances'] = [{'id': 'block', 'kind': 'chip', 'definition_id': 'block', 'params': {}}]
            graph['calls'] = [{'id': 'block', 'instance_id': 'block', 'inputs': {'x': 'input:x'}}]
            graph['outputs'] = [{'id': 'out', 'source': 'call:block'}]
            source = export_python(graph).replace('Chip_block', symbol)
            with self.subTest(symbol=symbol):
                restored = import_python(source)
                self.assertEqual(restored['definitions'][0]['id'], symbol)
                namespace = {}; exec(source, namespace)
                self.assertEqual(namespace['Model']()(self.torch.ones(2, 3, 8, 8))['out'].shape[0], 2)

    def test_active_root_guard_rejects_local_call_result_and_alias_bindings(self):
        source = export_python(identity_graph(2, [None, 3]))
        direct = source.replace('call_identity', '_check_execution_policy')
        tree = ast.parse(source)
        root = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Model')
        forward = next(node for node in root.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
        forward.body.insert(-1, ast.parse('_check_execution_policy = call_identity').body[0])
        alias = ast.unparse(tree)
        for changed in (direct, alias):
            for code in (changed, without_visual(changed)):
                with self.subTest(alias=changed == alias), self.assertRaisesRegex(ValueError, 'guard.*local|local.*guard|shadow'):
                    import_python(code)
            namespace = {}; exec(changed, namespace)
            with self.assertRaises(UnboundLocalError): namespace['Model']()(self.torch.ones(2, 3))

    def test_inactive_guard_name_is_allowed_for_local_results(self):
        for vision_ops in (False, True):
            graph = identity_graph(2, [None, 3])
            graph.pop('resource_profile'); graph.pop('resource_batch_bound')
            if not vision_ops: graph['instances'][0]['kind'] = 'ReLU'
            source = export_python(graph).replace('call_identity', '_check_execution_policy')
            for code in (source, without_visual(source)):
                restored = import_python(code)
                self.assertEqual(restored['calls'][0]['id'], '_check_execution_policy')
                namespace = {}; exec(code, namespace)
                self.assertEqual(tuple(namespace['Model']()(self.torch.ones(2, 3))['out'].shape), (2, 3))

    def test_v4_absent_and_explicit_null_policy_keep_noedit_checkpoint_identity(self):
        from studio_core.experiment_spec import model_digest, checkpoint_signature, default_classification_config
        from studio_core.schema import validate_graph
        config = default_classification_config()
        for fixture in (identity_graph(2, [None, 3]), operator_graph()):
            absent = copy.deepcopy(fixture)
            absent.pop('resource_profile'); absent.pop('resource_batch_bound')
            explicit = dict(absent, resource_profile=None)
            digest = model_digest(absent)
            signature = checkpoint_signature(absent, config, 'a'*64)
            for graph in (absent, explicit):
                self.assertTrue(validate_graph(graph)['valid'])
                self.assertEqual(python_profile(graph), 'vision-pytorch-v4')
                with self.subTest(explicit_null='resource_profile' in graph, kind=graph['instances'][0]['kind']):
                    self.assertEqual(model_digest(graph), digest)
                    self.assertEqual(checkpoint_signature(graph, config, 'a'*64), signature)
                    source = export_python(graph)
                    for code in (source, without_visual(source)):
                        restored = import_python(code)
                        restored['id'] = graph['id']
                        self.assertEqual(model_digest(restored), model_digest(graph))
                        self.assertEqual(checkpoint_signature(restored, config, 'a'*64), checkpoint_signature(graph, config, 'a'*64))

    def test_legacy_null_representation_is_not_recanonicalized(self):
        from studio_core.experiment_spec import model_digest
        legacy = identity_graph(2, [None, 3])
        legacy['instances'][0]['kind'] = 'ReLU'
        legacy.pop('resource_profile'); legacy.pop('resource_batch_bound')
        self.assertEqual(python_profile(legacy), 's1-pytorch-v1')
        self.assertNotEqual(model_digest(legacy), model_digest(dict(legacy, resource_profile=None)))

    def test_policy_rejects_dynamic_duplicate_unknown_or_malformed_literals(self):
        code = export_python(identity_graph())
        for replacement in ("dict(resource_profile='vision-v4', resource_batch_bound=2)",
            "{'resource_profile': 'vision-v4', 'resource_batch_bound': True}",
            "{'resource_profile': 'vision-v4', 'resource_batch_bound': 2, 'unknown': 1}",
            "{'resource_profile': 'vision-v4', 'resource_batch_bound': 2, 'resource_batch_bound': 3}",
            "{'resource_profile': 'classification-v2', 'resource_batch_bound': 2}"):
            tree = ast.parse(code)
            declaration = next(node for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'MLFORGE_EXECUTION_POLICY')
            declaration.value = ast.parse(replacement, mode='eval').body
            with self.subTest(value=replacement), self.assertRaises(ValueError): import_python(ast.unparse(tree))
        with self.assertRaises(ValueError): import_python(code + "\nMLFORGE_EXECUTION_POLICY = {'resource_profile': 'vision-v4', 'resource_batch_bound': 2}\n")

    def test_six_operators_standalone_state_sharing_and_bn_modes(self):
        from studio_core.runtime import build_model
        for affine in (False, True):
            for track in (False, True):
                graph = operator_graph(affine, track)
                code = export_python(graph)
                for source in (code, without_visual(code)):
                    restored = import_python(source)
                    self.assertEqual([i['kind'] for i in restored['instances']], [i['kind'] for i in graph['instances']])
                    namespace = {}
                    exec(source, namespace)
                    independent = namespace['Model']()
                    actual = build_model(graph, capture=False)
                    self.assertEqual(list(independent.state_dict()), list(actual.state_dict()))
                    independent.load_state_dict(actual.state_dict(), strict=True)
                    self.assertEqual([p.requires_grad for p in independent.parameters()], [p.requires_grad for p in actual.parameters()])
                    for train in (True, False):
                        independent.train(train); actual.train(train)
                        x = self.torch.arange(24, dtype=self.torch.float32).reshape(2, 3, 2, 2)
                        self.assertTrue(self.torch.equal(independent(x)['out'], actual(x=x)['out']))
                        for name, value in actual.state_dict().items(): self.assertTrue(self.torch.equal(value, independent.state_dict()[name]), name)
                    if track: self.assertEqual(actual.components['bn'].num_batches_tracked.item(), 2)
                    else: self.assertEqual(list(actual.state_dict()), ['components.bn.weight', 'components.bn.bias'] if affine else [])

    def test_new_operators_do_not_raise_legacy_resource_profile(self):
        graph = operator_graph(vision=False)
        self.assertEqual(python_profile(graph), 'vision-pytorch-v4')
        code = export_python(graph)
        self.assertIn("MLFORGE_EXECUTION_POLICY = {'resource_profile': None}", code)
        for source in (code, without_visual(code)):
            self.assertNotIn('resource_profile', import_python(source))

    def test_v4_preserves_none_classification_and_vision_policy_with_cnn(self):
        for profile in (None, 'classification-v2', 'vision-v4'):
            graph = operator_graph()
            if profile is None: graph.pop('resource_profile')
            else: graph['resource_profile'] = profile
            if profile != 'vision-v4': graph.pop('resource_batch_bound')
            graph['instances'].insert(0, {'id': 'conv', 'kind': 'Conv2d', 'params': {'in_channels': 3, 'out_channels': 3, 'kernel_size': 1}})
            graph['calls'].insert(0, {'id': 'conv', 'instance_id': 'conv', 'inputs': {'x': 'input:x'}})
            graph['calls'][1]['inputs']['x'] = 'call:conv'
            for source in (export_python(graph), without_visual(export_python(graph))):
                restored = import_python(source)
                self.assertEqual(restored.get('resource_profile'), profile)
                self.assertEqual(restored.get('resource_batch_bound'), 2 if profile == 'vision-v4' else None)

    def test_no_edit_python_model_and_checkpoint_identity(self):
        from studio_core.experiment_spec import model_digest, checkpoint_signature, default_classification_config
        graph = operator_graph()
        source = export_python(graph)
        for code in (source, without_visual(source)):
            restored = import_python(code)
            # Removing visual metadata intentionally removes the document ID;
            # compare execution identity under the same document identity.
            restored['id'] = graph['id']
            self.assertEqual(model_digest(restored), model_digest(graph))
            self.assertEqual(checkpoint_signature(restored, default_classification_config(), 'a'*64), checkpoint_signature(graph, default_classification_config(), 'a'*64))

    def test_named_typed_v4_and_dynamic_selector_rejection(self):
        graph = identity_graph()
        child = operator_graph()
        scope = {key: child[key] for key in ('inputs', 'instances', 'calls', 'outputs')}
        scope['outputs'].append({'id': 'alias', 'source': 'call:inverse:out'})
        definition = {'id': 'block', 'output_mode': 'named', 'public_parameters': [{'id': 'gain', 'type': 'float', 'default': 2}], 'bindings': {'gain': ['scale.factor']}, 'graph': scope}
        graph['inputs'] = copy.deepcopy(scope['inputs'])
        graph['instances'] = [{'id': 'block', 'kind': 'chip', 'definition_id': 'block', 'params': {'gain': 3}}]
        graph['calls'] = [{'id': 'block', 'instance_id': 'block', 'inputs': {'x': 'input:x'}}]
        graph['outputs'] = [{'id': 'out', 'source': 'call:block:out'}, {'id': 'alias', 'source': 'call:block:alias'}]
        graph['definitions'] = [definition]
        code = export_python(graph)
        restored = import_python(without_visual(code))
        self.assertEqual(restored['definitions'][0]['public_parameters'][0]['type'], 'float')
        self.assertEqual(restored['instances'][0]['params'], {'gain': 3})
        self.assertEqual(restored['definitions'][0]['bindings'], {'gain': ['scale.factor']})
        self.assertEqual(restored['definitions'][0]['output_mode'], 'named')
        from studio_core.runtime import build_model
        namespace = {}; exec(without_visual(code), namespace)
        independent = namespace['Model']()
        actual = build_model(graph, capture=False)
        independent.load_state_dict(actual.state_dict(), strict=True)
        x = self.torch.randn(2, 3, 2, 2)
        actual_outputs = actual(x=x); independent_outputs = independent(x)
        for port in actual_outputs: self.assertTrue(self.torch.equal(actual_outputs[port], independent_outputs[port]))
        with self.assertRaises(ValueError): import_python(code.replace("call_block['alias']", "call_block[next(iter(call_block))]"))

    def test_modified_helpers_and_semantics_are_not_hidden_by_metadata(self):
        graph = operator_graph()
        code = export_python(graph)
        restored = import_python(code.replace('factor=2', 'factor=3'))
        self.assertEqual(next(i for i in restored['instances'] if i['id'] == 'scale')['params']['factor'], 3)
        for before, after in [('return a * b', 'return a + b'), ('return 1 - input', 'return 2 - input'), ('return self.factor * input', 'return input')]:
            self.assertIn(before, code)
            with self.assertRaisesRegex(ValueError, 'helper'): import_python(code.replace(before, after))

    def test_runtime_guards_match_platform_and_scale_large_integer(self):
        graph = operator_graph()
        code = export_python(graph)
        namespace = {}; exec(code, namespace)
        model = namespace['Model']()
        with self.assertRaises(ValueError): model.components['multiply'](self.torch.ones(2, 3), self.torch.ones(1, 3))
        bn = model.components['bn']
        before = bn.num_batches_tracked.clone()
        for shape in ((1, 3, 1, 1), (2, 4, 2, 2), (2, 3)):
            with self.assertRaises(ValueError): bn(self.torch.ones(shape))
        self.assertTrue(self.torch.equal(before, bn.num_batches_tracked))
        huge = identity_graph()
        huge['instances'][0].update(kind='Scale', params={'factor': 10**1000})
        with self.assertRaises(ValueError): export_python(huge)


if __name__ == '__main__': unittest.main()
