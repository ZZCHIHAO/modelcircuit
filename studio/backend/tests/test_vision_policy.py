"""Explicit vision policy and independent compilation/execution accounting."""
import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from studio_core.schema import prepare_graph, validate_graph, resource_limits, ResourceLimits


def identity_graph(bound=2, shape=None):
    return {'schema_version': 1, 'id': 'policy', 'revision': 1,
            'resource_profile': 'vision-v4', 'resource_batch_bound': bound,
            'inputs': [{'id': 'x', 'shape': shape or [None, 3, 8, 8], 'dtype': 'float32'}],
            'instances': [{'id': 'identity', 'kind': 'Identity', 'params': {}}],
            'calls': [{'id': 'identity', 'instance_id': 'identity', 'inputs': {'x': 'input:x'}}],
            'outputs': [{'id': 'out', 'source': 'call:identity'}], 'definitions': []}


def nested_graph(width=100, depth=3, calls=1):
    graph = identity_graph(2, [None, width])
    definitions = []
    scope = {key: copy.deepcopy(graph[key]) for key in ('inputs', 'instances', 'calls', 'outputs')}
    for index in range(depth):
        identifier = 'level' + str(index)
        definitions.append({'id': identifier, 'public_parameters': [], 'bindings': {}, 'graph': scope})
        scope = {'inputs': copy.deepcopy(graph['inputs']),
                 'instances': [{'id': 'child', 'kind': 'chip', 'definition_id': identifier, 'params': {}}],
                 'calls': [{'id': 'c' + str(n), 'instance_id': 'child', 'inputs': {'x': 'input:x'}} for n in range(calls)],
                 'outputs': [{'id': 'out', 'source': 'call:c0'}]}
    graph.update(scope, definitions=definitions)
    return graph


class VisionPolicyTests(unittest.TestCase):
    def test_future_api_contract_and_224_stem_estimates(self):
        from studio_core.schema import bounded_elements
        self.assertEqual(bounded_elements([None, 64, 112, 112], resource_limits('vision-v4', batch_bound=2)), 1_605_632)
        self.assertEqual(bounded_elements([None, 64, 112, 112], resource_limits('classification-v2')), 51_380_224)
        self.assertEqual((resource_limits('vision-v4', batch_bound=2).max_parameters,
                          resource_limits('vision-v4', batch_bound=2).max_tensor_elements,
                          resource_limits('vision-v4', batch_bound=2).max_inferred_elements), (20_000_000, 16_777_216, 134_217_728))

    def test_explicit_bounds_and_legacy_contract(self):
        from studio_core.schema import bounded_elements, graph_resource_limits
        self.assertEqual(ResourceLimits(1, 2, 3).batch_bound, 64)
        self.assertFalse(ResourceLimits(1, 2, 3).enforce_batch_bound)
        for bound in (1, 64):
            limits = resource_limits('vision-v4', batch_bound=bound)
            self.assertEqual(bounded_elements([None, 3, 4], limits), bound * 12)
            self.assertTrue(limits.enforce_batch_bound)
            self.assertTrue(validate_graph(identity_graph(bound))['valid'])
        for value in (0, 65, True, 2.0, None):
            graph = identity_graph(value)
            self.assertFalse(validate_graph(graph)['valid'], value)
        graph = identity_graph()
        del graph['resource_batch_bound']
        self.assertEqual(validate_graph(graph)['errors'][0]['field'], 'resource_batch_bound')
        for profile in (None, 'classification-v2'):
            graph = identity_graph()
            graph['resource_profile'] = profile
            self.assertEqual(validate_graph(graph)['errors'][0]['field'], 'resource_batch_bound')
            with self.assertRaises(ValueError): resource_limits(profile, batch_bound=2)
        self.assertEqual(graph_resource_limits(identity_graph()).batch_bound, 2)
        self.assertFalse(validate_graph(identity_graph(shape=[None, None, 8, 8]))['valid'])
        for shape in ([2, None], [False, 2], [1.0, 2], [], [1]*9):
            with self.assertRaises(ValueError): bounded_elements(shape, resource_limits('vision-v4', batch_bound=2))

    def test_compilation_and_each_execution_path_are_independent(self):
        graph = nested_graph(width=1_000_000, depth=4)
        resolved, _, _ = prepare_graph(graph)
        ledger = resolved['_resource_ledger']
        self.assertEqual(ledger['execution_elements'], 20_000_000)
        self.assertEqual(ledger['definition_execution_elements'], {'level0': 4_000_000, 'level1': 8_000_000, 'level2': 12_000_000, 'level3': 16_000_000})
        self.assertEqual(ledger['compile_tensor_elements'], 228_000_000)
        self.assertLessEqual(ledger['inference_nodes'], 16384)
        graph['resource_profile'] = 'classification-v2'
        del graph['resource_batch_bound']
        for scope in [graph] + [d['graph'] for d in graph['definitions']]: scope['inputs'][0]['shape'][0] = 2
        self.assertFalse(validate_graph(graph)['valid'])

    def test_multi_calls_are_not_merged_and_alias_outputs_count_once(self):
        graph = nested_graph(width=10, depth=1, calls=2)
        graph['definitions'][0]['graph']['outputs'].append({'id': 'alias', 'source': 'call:identity:out'})
        graph['outputs'][0]['source'] = 'call:c0:out'
        resolved, _, _ = prepare_graph(graph)
        self.assertEqual(resolved['_resource_ledger']['execution_elements'], 140)
        graph = nested_graph(width=1_000_000, depth=3, calls=3)
        self.assertFalse(validate_graph(graph)['valid'])

    def test_unused_definition_execution_is_bounded(self):
        graph = identity_graph(2, [None, 1])
        scope = identity_graph(2, [None, 8_000_000])
        scope['calls'] *= 0
        scope['calls'] = [{'id': 'c' + str(n), 'instance_id': 'identity', 'inputs': {'x': 'input:x'}} for n in range(9)]
        scope['outputs'][0]['source'] = 'call:c0'
        graph['definitions'] = [{'id': 'unused', 'public_parameters': [], 'bindings': {}, 'graph': scope}]
        self.assertFalse(validate_graph(graph)['valid'])

    def test_execution_counts_call_origins_even_when_identity_returns_same_tensor(self):
        import torch
        from studio_core.runtime import build_model, _ExecutionBudget
        graph = nested_graph(width=10, depth=1, calls=1)
        child = graph['definitions'][0]['graph']
        child['calls'].append({'id': 'second', 'instance_id': 'identity', 'inputs': {'x': 'input:x'}})
        child['outputs'].append({'id': 'other', 'source': 'call:second'})
        graph['outputs'][0]['source'] = 'call:c0:out'
        resolved, _, _ = prepare_graph(graph)
        model = build_model(graph, capture=False)
        budget = _ExecutionBudget(model.limits)
        model._execute({'x': torch.ones(2, 10)}, budget)
        self.assertEqual(resolved['_resource_ledger']['execution_elements'], 120)
        self.assertEqual(budget.total, 120)

    def test_compile_work_node_limits_still_apply(self):
        self.assertFalse(validate_graph(nested_graph(width=1, depth=6, calls=3))['valid'])

    def test_nested_same_source_aliases_match_runtime_ledger(self):
        import torch
        from studio_core.runtime import build_model, _ExecutionBudget
        graph = nested_graph(width=10, depth=2)
        graph['definitions'][0]['graph']['outputs'].append({'id': 'alias', 'source': 'call:identity:out'})
        outer = graph['definitions'][1]['graph']
        outer['outputs'] = [{'id': 'out', 'source': 'call:c0:out'}, {'id': 'alias', 'source': 'call:c0:alias'}]
        graph['outputs'][0]['source'] = 'call:c0:out'
        resolved, _, _ = prepare_graph(graph)
        model = build_model(graph, capture=False)
        budget = _ExecutionBudget(model.limits)
        model._execute({'x': torch.ones(2, 10)}, budget)
        self.assertEqual(resolved['_resource_ledger']['execution_elements'], 120)
        self.assertEqual(budget.total, 120)

    def test_runtime_bound_and_old_batch65(self):
        import torch
        from studio_core.runtime import build_model
        graph = identity_graph(2, [None, 3])
        model = build_model(graph, capture=False)
        self.assertEqual(list(model(x=torch.ones(1, 3))['out'].shape), [1, 3])
        with self.assertRaisesRegex(ValueError, 'resource_batch_bound'):
            model(x=torch.ones(3, 3))
        for profile in (None, 'classification-v2'):
            legacy = copy.deepcopy(graph)
            legacy.pop('resource_batch_bound')
            legacy['resource_profile'] = profile
            self.assertEqual(build_model(legacy)(x=torch.ones(65, 3))['out'].shape[0], 65)

    def test_config_batch_rejected_before_allocation_or_data_scan(self):
        from studio_core.classification_runtime import run_classification
        from studio_core.experiment_spec import default_classification_config
        config = default_classification_config()
        config['batch_size'] = 3
        config['data'].update(path='not-read.csv', feature_columns=['x'])
        with patch('studio_core.runtime.build_model') as build, patch('studio_core.classification_runtime.inspect_dataset') as scan:
            with self.assertRaisesRegex(ValueError, 'resource_batch_bound'):
                run_classification(identity_graph(2), config, 'unused')
            build.assert_not_called()
            scan.assert_not_called()
        from studio_core.runtime import run_experiment
        with patch('studio_core.runtime.build_model') as build:
            with self.assertRaisesRegex(ValueError, 'resource_batch_bound'):
                run_experiment(identity_graph(2), {'batch_size': 3})
            build.assert_not_called()

    def test_policy_changes_model_and_checkpoint_identity(self):
        from studio_core.experiment_spec import model_digest, checkpoint_signature, default_classification_config
        graph = identity_graph(2)
        changed = identity_graph(3)
        self.assertNotEqual(model_digest(graph), model_digest(changed))
        self.assertNotEqual(checkpoint_signature(graph, default_classification_config(), 'a'*64), checkpoint_signature(changed, default_classification_config(), 'a'*64))

    def test_bn_cpu_checkpoint_epoch_resume_and_strict_buffers(self):
        import torch
        from PIL import Image
        from studio_core.classification_runtime import run_classification
        from studio_core.experiment_spec import default_classification_config
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split, samples in (('train', 4), ('val', 2), ('test', 2)):
                for label in range(2):
                    for index in range(samples):
                        path = root / 'images' / split / str(label) / (str(index) + '.png')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.new('RGB', (3, 3), (label * 90 + index * 10, index * 20 + len(split), label * 70)).save(path)
            graph = identity_graph(2, [None, 3, 8, 8])
            graph['instances'] = [{'id': 'bn', 'kind': 'BatchNorm2d', 'params': {'num_features': 3}},
                {'id': 'drop', 'kind': 'Dropout', 'params': {'p': .25}},
                {'id': 'pool', 'kind': 'AdaptiveAvgPool2d', 'params': {'output_size': [1, 1]}},
                {'id': 'flat', 'kind': 'Flatten', 'params': {}},
                {'id': 'head', 'kind': 'Linear', 'params': {'in_features': 3, 'out_features': 2}}]
            graph['calls'] = [{'id': 'bn0', 'instance_id': 'bn', 'inputs': {'x': 'input:x'}},
                {'id': 'bn1', 'instance_id': 'bn', 'inputs': {'x': 'call:bn0'}},
                {'id': 'drop', 'instance_id': 'drop', 'inputs': {'x': 'call:bn1'}},
                {'id': 'pool', 'instance_id': 'pool', 'inputs': {'x': 'call:drop'}},
                {'id': 'flat', 'instance_id': 'flat', 'inputs': {'x': 'call:pool'}},
                {'id': 'head', 'instance_id': 'head', 'inputs': {'x': 'call:flat'}}]
            graph['outputs'] = [{'id': 'logits', 'source': 'call:head'}]
            config = default_classification_config('image_folder')
            config.update(epochs=2, batch_size=2, seed=77, device='cpu')
            config['data'].update(path=str(root / 'images'), split_mode='directories', image_size=[8, 8])
            full = run_classification(graph, config, root / 'full')
            partial = run_classification(graph, dict(config, epochs=1), root / 'partial')
            resumed = run_classification(graph, dict(config, resume_checkpoint=partial['artifacts']['last_checkpoint']), root / 'resumed')
            load = lambda result: torch.load(result['artifacts']['last_checkpoint'], weights_only=True, map_location='cpu')
            expected, actual = load(full), load(resumed)
            self.assertEqual(full['history'], resumed['history'])
            self.assertEqual(full['metrics'], resumed['metrics'])
            self.assertEqual(expected['model_state']['components.bn.num_batches_tracked'].item(), 16)
            for key, value in expected['model_state'].items(): self.assertTrue(torch.equal(value, actual['model_state'][key]), key)
            changed = copy.deepcopy(graph); changed['resource_batch_bound'] = 3
            with self.assertRaisesRegex(ValueError, 'identity'):
                run_classification(changed, dict(config, resume_checkpoint=partial['artifacts']['last_checkpoint']), root / 'different')
            broken = load(partial)
            del broken['model_state']['components.bn.running_mean']
            torch.save(broken, root / 'missing-buffer.pt')
            with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                run_classification(graph, dict(config, resume_checkpoint=str(root / 'missing-buffer.pt')), root / 'broken')


if __name__ == '__main__': unittest.main()
