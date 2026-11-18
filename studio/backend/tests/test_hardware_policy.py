"""Uncapped static v5 policy, strict source bridge, and pre-allocation admission."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studio_core.schema import prepare_graph, resource_limits, validate_graph
from test_vision_policy import identity_graph, nested_graph


def hardware_graph(width=2, outputs=2):
    graph = identity_graph(2, [None, width])
    graph.update(resource_profile='hardware-v5')
    graph['instances'] = [{'id': 'dense', 'kind': 'Linear', 'params': {'in_features': width, 'out_features': outputs}}]
    graph['calls'] = [{'id': 'dense', 'instance_id': 'dense', 'inputs': {'x': 'input:x'}}]
    graph['outputs'] = [{'id': 'logits', 'source': 'call:dense'}]
    return graph


def snapshot(free=10**12, device='cpu', host_free=None):
    return {'device': device, 'available_bytes': free, 'total_bytes': free,
            'host_available_bytes': free if host_free is None else host_free,
            'host_total_bytes': free, 'source': 'test', 'queried_at': 'test'}


class HardwarePolicyTests(unittest.TestCase):
    def test_resume_minimum_excludes_optional_training_but_projection_keeps_full_batch(self):
        import csv
        from studio_core.classification_runtime import run_classification
        from studio_core.experiment_spec import default_classification_config
        from studio_core.runtime import _GraphModule
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data = root / 'resume-data.csv'
            with data.open('w', newline='') as stream:
                writer = csv.writer(stream); writer.writerow(['x', 'y', 'label', 'split'])
                for index in range(84):
                    writer.writerow([index / 84, index + 0.1, str(index % 2), 'train' if index < 80 else 'val' if index < 82 else 'test'])
            config = default_classification_config()
            config['data'].update(path=str(data), feature_columns=['x', 'y'], split_column='split')
            config.update(batch_size=64, epochs=1, optimizer='sgd')
            graph = hardware_graph(); graph['resource_batch_bound'] = 64
            with patch('studio_core.devices.memory_snapshot', return_value=snapshot()):
                initial = run_classification(graph, config, root / 'initial')
            for epochs, expected_batches in ((1, {2}), (2, {2, 16, 64})):
                batches = []; execute = _GraphModule._execute
                def observe(module, inputs, budget):
                    batches.extend(value.shape[0] for value in inputs.values())
                    return execute(module, inputs, budget)
                resumed_config = dict(config, epochs=epochs, resume_checkpoint=initial['artifacts']['last_checkpoint'])
                with self.subTest(epochs=epochs), \
                     patch('studio_core.devices.memory_snapshot', return_value=snapshot(800)) as query, \
                     patch.object(_GraphModule, '_execute', observe):
                    result = run_classification(graph, resumed_config, root / ('resume-' + str(epochs)))
                    self.assertEqual(query.call_count, 1)
                    self.assertEqual(set(batches), expected_batches)
                    estimate = result['memory_report']['estimate']
                    self.assertEqual(estimate['batch_size'], 64)
                    self.assertEqual(estimate['minimum_batch_size'], 2)
                    self.assertEqual(estimate['input_bytes'], 512)
                    self.assertEqual(estimate['minimum_input_bytes'], 16)
                    self.assertEqual(estimate['minimum_device_bytes'], 712)
                    self.assertEqual(result['completed_epochs'], epochs)
                    self.assertTrue(result['warnings'])

    def test_short_partition_real_run_keeps_configured_batch_and_uses_batch_two(self):
        import csv
        import torch
        from studio_core.classification_runtime import run_classification
        from studio_core.experiment_spec import default_classification_config
        from studio_core.runtime import _GraphModule
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data = root / 'data.csv'
            with data.open('w', newline='') as stream:
                writer = csv.writer(stream); writer.writerow(['x', 'y', 'label', 'split'])
                for index, split in enumerate(('train', 'train', 'val', 'val', 'test', 'test')):
                    writer.writerow([index, index + 0.1, str(index % 2), split])
            config = default_classification_config()
            config['data'].update(path=str(data), feature_columns=['x', 'y'], split_column='split')
            config.update(batch_size=64, epochs=1, optimizer='sgd')
            graph = hardware_graph(); graph['resource_batch_bound'] = 64
            batches = []; execute = _GraphModule._execute
            def observe(module, inputs, budget):
                batches.extend(value.shape[0] for value in inputs.values())
                return execute(module, inputs, budget)
            with patch('studio_core.devices.memory_snapshot', return_value=snapshot(100)), \
                 patch.object(_GraphModule, '_execute', observe):
                result = run_classification(graph, config, root / 'run')
            self.assertEqual(result['status'], 'succeeded')
            self.assertEqual(set(batches), {2})
            self.assertEqual(result['memory_report']['estimate']['minimum_device_bytes'], 88)
            self.assertTrue(result['warnings'])
            self.assertEqual(json.loads(Path(result['artifacts']['run_spec']).read_text())['config']['batch_size'], 64)
            checkpoint = torch.load(result['artifacts']['last_checkpoint'], weights_only=True, map_location='cpu')
            self.assertEqual(checkpoint['config']['batch_size'], 64)

    def test_short_classification_partitions_use_attainable_cpu_cuda_minimum(self):
        import torch
        from studio_core.classification_runtime import run_classification
        from studio_core.experiment_spec import default_classification_config
        manifest = {'kind': 'csv', 'dataset_digest': 'test', 'input_shape': [None, 2],
                    'class_names': ['a', 'b'], 'counts': {'train': 2, 'val': 2, 'test': 2}}
        graph = hardware_graph(); graph['resource_batch_bound'] = 64
        config = default_classification_config()
        config['data'].update(path='unused.csv', feature_columns=['x', 'y'])
        config.update(batch_size=64, epochs=1, optimizer='sgd')
        for device, free, host_free, expected_minimum, expected_host in (
                ('cpu', 100, 100, 88, 0), ('cuda:3', 100, 100, 40, 72)):
            admitted = []
            from studio_core.memory_admission import admit_memory
            def remember(*args, **kwargs):
                report = admit_memory(*args, **kwargs)
                admitted.append(report)
                return report
            with self.subTest(device=device), \
                 patch('studio_core.classification_runtime.inspect_dataset', return_value=manifest), \
                 patch('studio_core.devices.resolve_device', return_value=torch.device(device)), \
                 patch('studio_core.devices.memory_snapshot', return_value=snapshot(free, device, host_free)), \
                 patch('studio_core.memory_admission.admit_memory', side_effect=remember), \
                 patch('studio_core.classification_runtime.build_datasets', side_effect=InterruptedError('dataset boundary reached')):
                with self.assertRaisesRegex(InterruptedError, 'dataset boundary reached'):
                    run_classification(graph, dict(config, device=device), 'unused-output')
                estimate = admitted[0]['estimate']
                self.assertEqual(estimate['batch_size'], 2)
                self.assertEqual(estimate['input_bytes'], 16)
                self.assertEqual(estimate['minimum_device_bytes'], expected_minimum)
                self.assertEqual(estimate['minimum_host_bytes'], expected_host)
                self.assertTrue(admitted[0]['warnings'])
        self.assertEqual(config['batch_size'], 64)

    def test_tensor_and_execution_caps_are_absent_only_in_v5(self):
        for width, calls in ((20_000_000, 1), (8_000_000, 9)):
            graph = identity_graph(2, [None, width])
            graph['calls'] = [{'id': 'c' + str(n), 'instance_id': 'identity', 'inputs': {'x': 'input:x'}} for n in range(calls)]
            graph['outputs'][0]['source'] = 'call:c0'
            self.assertFalse(validate_graph(graph)['valid'])
            graph['resource_profile'] = 'hardware-v5'
            self.assertTrue(validate_graph(graph)['valid'])
        from studio_core.runtime import _check_runtime_budget
        _check_runtime_budget(10**30, 10**40, resource_limits('hardware-v5', batch_bound=2))

    def test_hardware_observations_do_not_modify_graph_or_any_identity(self):
        from studio_core.runtime import build_model
        from studio_core.experiment_spec import model_digest, experiment_signature
        graph = hardware_graph()
        original = copy.deepcopy(graph)
        identities = []
        for free in (10**8, 10**12):
            with patch('studio_core.devices.memory_snapshot', return_value=snapshot(free)):
                model = build_model(graph)
            identities.append((model_digest(graph), experiment_signature(graph, {'device': 'cpu'})))
            self.assertEqual(model.memory_report['snapshot']['available_bytes'], free)
        self.assertEqual(identities[0], identities[1])
        self.assertEqual(graph, original)

    def test_classification_admits_once_before_dataset_model_or_resume_load(self):
        from studio_core.classification_runtime import run_classification
        from studio_core.experiment_spec import default_classification_config
        manifest = {'kind': 'csv', 'dataset_digest': 'test', 'input_shape': [None, 2], 'class_names': ['a', 'b'], 'counts': {'train': 2, 'val': 2, 'test': 2}}
        config = default_classification_config()
        config['data'].update(path='unused.csv', feature_columns=['x', 'y'])
        config.update(batch_size=2, epochs=1, resume_checkpoint='unused.pt')
        with patch('studio_core.classification_runtime.inspect_dataset', return_value=manifest), \
             patch('studio_core.devices.memory_snapshot', return_value=snapshot(1)), \
             patch('studio_core.classification_runtime.build_datasets', side_effect=AssertionError('allocated dataset')), \
             patch('studio_core.runtime.make_operator', side_effect=AssertionError('allocated operator')), \
             patch('torch.load', side_effect=AssertionError('allocated checkpoint')):
            with self.assertRaisesRegex(ValueError, 'minimum.*available'):
                run_classification(hardware_graph(), config, 'unused-output')

    def test_classification_tiny_run_reports_estimate_without_second_query(self):
        import csv
        from studio_core.classification_runtime import run_classification
        from studio_core.experiment_spec import default_classification_config
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data.csv'
            with data.open('w', newline='') as stream:
                writer = csv.writer(stream); writer.writerow(['x', 'y', 'label', 'split'])
                for index in range(12): writer.writerow([index / 12, index % 3, str(index % 2), 'train' if index < 6 else 'val' if index < 10 else 'test'])
            config = default_classification_config()
            config['data'].update(path=str(data), feature_columns=['x', 'y'], split_column='split')
            config.update(batch_size=2, epochs=1)
            with patch('studio_core.devices.memory_snapshot', return_value=snapshot()) as query:
                result = run_classification(hardware_graph(), config, root / 'run')
            self.assertEqual(query.call_count, 1)
            self.assertEqual(result['status'], 'succeeded')
            self.assertGreater(result['memory_report']['estimate']['host_checkpoint_bytes'], 0)
            self.assertEqual(json.loads(Path(result['artifacts']['memory_report']).read_text()), result['memory_report'])
            self.assertEqual(json.loads(Path(result['artifacts']['run_spec']).read_text())['memory_report'], result['memory_report'])

    def test_oom_from_synthetic_allocation_has_a_readable_diagnostic(self):
        from studio_core.runtime import run_experiment
        with patch('studio_core.devices.memory_snapshot', return_value=snapshot()), \
             patch('studio_core.runtime.make_operator', side_effect=MemoryError()):
            with self.assertRaisesRegex(ValueError, 'Hardware memory allocation failed'):
                run_experiment(hardware_graph(), {'steps': 1, 'batch_size': 2})

    def test_uncapped_static_accepts_large_model_without_querying_hardware(self):
        graph = hardware_graph(100_000, 100_000)
        with patch('studio_core.devices.memory_snapshot', side_effect=AssertionError('static queried hardware')):
            limits = resource_limits('hardware-v5', batch_bound=2)
            self.assertIsNone(limits.max_parameters)
            self.assertIsNone(limits.max_tensor_elements)
            self.assertIsNone(limits.max_inferred_elements)
            self.assertTrue(validate_graph(graph)['valid'])
            self.assertEqual(prepare_graph(graph)[2], 10_000_100_000)
        for profile in ('vision-v4', 'classification-v2', None):
            legacy = copy.deepcopy(graph)
            legacy['resource_profile'] = profile
            if profile != 'vision-v4': legacy.pop('resource_batch_bound')
            self.assertFalse(validate_graph(legacy)['valid'])

    def test_v5_retains_shape_batch_and_compile_work_contracts(self):
        graph = hardware_graph()
        for bound in (None, 0, True, 65):
            graph['resource_batch_bound'] = bound
            self.assertFalse(validate_graph(graph)['valid'])
        graph = nested_graph(width=1, depth=6, calls=3)
        graph['resource_profile'] = 'hardware-v5'
        self.assertFalse(validate_graph(graph)['valid'])

    def test_python_v5_roundtrip_source_wins_and_guard_remains_exact(self):
        from studio_core.python_io import export_python, import_python, python_profile
        from studio_core.experiment_spec import model_digest
        graph = hardware_graph(5000, 5000)
        with patch('studio_core.devices.memory_snapshot', side_effect=AssertionError('export queried hardware')):
            source = export_python(graph)
        self.assertEqual(python_profile(graph), 'vision-pytorch-v5')
        self.assertIn("'resource_profile': 'hardware-v5'", source)
        self.assertEqual(model_digest(import_python(source)), model_digest(graph))
        stripped = '\n'.join(line for line in source.splitlines() if not line.startswith('STUDIO_VISUAL_METADATA'))
        restored = import_python(stripped)
        self.assertEqual(restored['resource_profile'], 'hardware-v5')
        edited = import_python(source.replace('out_features=5000', 'out_features=5001'))
        self.assertEqual(edited['instances'][0]['params']['out_features'], 5001)
        for broken in (source.replace('input.ndim < 1', 'input.ndim < 0'),
                       source.replace('        _check_execution_policy(x)\n', ''),
                       source.replace('        _check_execution_policy(x)', '        _check_execution_policy = None\n        _check_execution_policy(x)')):
            with self.assertRaises(ValueError): import_python(broken)

    def test_insufficient_cpu_rejects_before_operator_or_synthetic_input(self):
        import torch
        from studio_core.runtime import build_model, run_experiment
        graph = hardware_graph()
        with patch('studio_core.devices.memory_snapshot', return_value=snapshot(1)), \
             patch('studio_core.runtime.make_operator', side_effect=AssertionError('allocated operator')), \
             patch.object(torch, 'randn', side_effect=AssertionError('allocated synthetic input')):
            with self.assertRaisesRegex(ValueError, 'minimum.*available'): build_model(graph)
            with self.assertRaisesRegex(ValueError, 'minimum.*available'):
                run_experiment(graph, {'steps': 1, 'batch_size': 2})
        with patch('studio_core.devices.memory_snapshot', side_effect=OSError('unavailable counter')), \
             patch('studio_core.runtime.make_operator', side_effect=AssertionError('allocated operator')):
            with self.assertRaisesRegex(ValueError, 'Cannot query.*memory'): build_model(graph)

    def test_worker_minimum_includes_inputs_even_for_parameterless_graph(self):
        import torch
        from studio_core.runtime import run_experiment
        graph = hardware_graph(20_000_000, 2)
        # The unused huge input is still constructed by the synthetic worker;
        # logits read a tiny second input. Admission precedes trainability checks.
        graph['inputs'].append({'id': 'small', 'shape': [None, 2], 'dtype': 'float32'})
        graph['instances'], graph['calls'] = [], []
        graph['outputs'][0]['source'] = 'input:small'
        with patch('studio_core.devices.memory_snapshot', return_value=snapshot(10**6)), \
             patch('studio_core.runtime.make_operator', side_effect=AssertionError('allocated operator')), \
             patch.object(torch, 'randn', side_effect=AssertionError('allocated input')):
            with self.assertRaisesRegex(ValueError, 'minimum.*available'):
                run_experiment(graph, {'steps': 1, 'batch_size': 2})

    def test_cuda_and_cpu_staging_insufficiency_reject_before_cpu_construction(self):
        from studio_core.runtime import build_model
        for measured in (snapshot(1, 'cuda:3'), snapshot(10**12, 'cuda:3', host_free=1)):
            with patch('studio_core.devices.memory_snapshot', return_value=measured), \
                 patch('studio_core.runtime.make_operator', side_effect=AssertionError('allocated operator')):
                with self.assertRaisesRegex(ValueError, 'minimum.*available'): build_model(hardware_graph(), device='cuda:3')

    def test_small_synthetic_training_returns_runtime_memory_report(self):
        from studio_core.runtime import run_experiment
        from studio_core.experiment_spec import model_digest
        graph = hardware_graph()
        before = model_digest(graph)
        with patch('studio_core.devices.memory_snapshot', return_value=snapshot()):
            result = run_experiment(graph, {'steps': 1, 'batch_size': 2})
        self.assertEqual(result['status'], 'succeeded')
        self.assertIn('memory_report', result)
        self.assertEqual(before, model_digest(graph))
        json.dumps(result['memory_report'], allow_nan=False)


class HardwareMemoryTests(unittest.TestCase):
    def test_separate_mandatory_batch_preserves_projection_and_cuda_host_floor(self):
        from studio_core.memory_admission import estimate_memory, admit_memory
        graph = hardware_graph(); graph['resource_batch_bound'] = 64
        resolved = prepare_graph(graph)[0]
        options = {'batch_size': 64, 'minimum_batch_size': 2, 'training': True,
                   'workflow': 'classification', 'resume': True, 'host_dataset_bytes': 672}
        with patch('studio_core.devices.memory_snapshot', return_value=snapshot(50, 'cuda:3', 700)):
            report = admit_memory(resolved, 'cuda:3', **options)
        estimate = report['estimate']
        self.assertEqual((estimate['input_bytes'], estimate['minimum_input_bytes']), (512, 16))
        self.assertEqual((estimate['minimum_device_bytes'], estimate['minimum_host_bytes']), (40, 696))
        self.assertGreater(estimate['projected_device_bytes'], 50)
        self.assertTrue(report['warnings'])
        for measured in (snapshot(39, 'cuda:3', 700), snapshot(50, 'cuda:3', 695)):
            with patch('studio_core.devices.memory_snapshot', return_value=measured):
                with self.assertRaisesRegex(ValueError, 'minimum.*available'):
                    admit_memory(resolved, 'cuda:3', **options)
        default = estimate_memory(resolved, batch_size=64, workflow='classification')
        self.assertEqual(default['minimum_input_bytes'], default['input_bytes'])
        for minimum in (0, True, 65):
            with self.assertRaisesRegex(ValueError, 'minimum_batch_size'):
                estimate_memory(resolved, batch_size=64, minimum_batch_size=minimum)

    def test_frozen_parent_covers_all_nested_parameters_and_cuda_staging(self):
        from studio_core.memory_admission import estimate_memory
        child = hardware_graph()
        graph = nested_graph(width=2, depth=1)
        graph['resource_profile'] = 'hardware-v5'
        graph['definitions'][0]['graph'] = {key: child[key] for key in ('inputs', 'instances', 'calls', 'outputs')}
        graph['definitions'][0]['graph']['outputs'][0]['id'] = 'out'
        graph['instances'][0]['frozen'] = True
        report = estimate_memory(prepare_graph(graph)[0], device='cuda:3', batch_size=2, training=True, optimizer='adamw')
        self.assertEqual(report['parameter_bytes'], 24)
        self.assertEqual(report['trainable_parameter_bytes'], 0)
        self.assertEqual(report['gradient_bytes'], 0)
        self.assertEqual(report['optimizer_bytes'], 0)
        self.assertEqual(report['host_staging_bytes'], 24)

    def test_selected_cuda_memory_and_host_are_queried_without_fallback(self):
        from studio_core.devices import memory_snapshot
        with patch('studio_core.devices._host_memory_snapshot', return_value={'available_bytes': 100, 'total_bytes': 200, 'source': 'host'}), \
             patch('studio_core.devices.resolve_device', return_value=__import__('torch').device('cuda:2')), \
             patch('torch.cuda.mem_get_info', return_value=(50, 80)) as query:
            measured = memory_snapshot('cuda:2')
        self.assertEqual(measured['available_bytes'], 50)
        self.assertEqual(measured['host_available_bytes'], 100)
        self.assertEqual(str(query.call_args.args[0]), 'cuda:2')
        with patch('studio_core.devices.resolve_device', side_effect=ValueError('CUDA unavailable')):
            with self.assertRaisesRegex(ValueError, 'CUDA unavailable'): memory_snapshot('cuda:2')

    def test_real_host_memory_is_finite_positive_and_runtime_only(self):
        from studio_core.devices import memory_snapshot
        measured = memory_snapshot('cpu')
        self.assertGreater(measured['available_bytes'], 0)
        self.assertGreaterEqual(measured['total_bytes'], measured['available_bytes'])
        self.assertEqual(measured['host_available_bytes'], measured['available_bytes'])

    def test_shared_repeated_frozen_and_bn_buffers(self):
        from studio_core.memory_admission import estimate_memory
        graph = hardware_graph()
        graph['calls'].append({'id': 'again', 'instance_id': 'dense', 'inputs': {'x': 'input:x'}})
        shared = estimate_memory(prepare_graph(graph)[0], batch_size=2, training=True, optimizer='sgd', capture=False)
        self.assertEqual(shared['parameter_bytes'], 24)
        self.assertEqual(shared['gradient_bytes'], 24)
        self.assertEqual(shared['optimizer_bytes'], 0)
        graph['instances'].append(dict(copy.deepcopy(graph['instances'][0]), id='unused', frozen=True))
        unused = estimate_memory(prepare_graph(graph)[0], batch_size=2, training=True, optimizer='adamw', capture=True, workflow='classification', resume=True)
        self.assertEqual(unused['parameter_bytes'], 48)
        self.assertEqual(unused['trainable_parameter_bytes'], 24)
        self.assertGreater(unused['optimizer_bytes'], 24)
        self.assertGreater(unused['capture_bytes'], 0)
        self.assertGreater(unused['host_checkpoint_bytes'], 0)
        self.assertGreater(unused['host_measurement_bytes'], 0)
        self.assertEqual(unused['execution_elements'], shared['execution_elements'])
        bn = identity_graph(2, [None, 3, 2, 2]); bn['resource_profile'] = 'hardware-v5'
        bn['instances'][0] = {'id': 'identity', 'kind': 'BatchNorm2d', 'params': {'num_features': 3}}
        self.assertEqual(estimate_memory(prepare_graph(bn)[0], batch_size=2)['buffer_bytes'], 32)

    def test_execution_ledger_uses_called_root_not_unused_definitions(self):
        from studio_core.memory_admission import estimate_memory
        graph = nested_graph(width=10, depth=2, calls=2); graph['resource_profile'] = 'hardware-v5'
        resolved = prepare_graph(graph)[0]
        report = estimate_memory(resolved, batch_size=2)
        self.assertEqual(report['execution_elements'], resolved['_execution_elements'])
        unused = identity_graph(2, [None, 1000000])
        graph['definitions'].append({'id': 'unused', 'public_parameters': [], 'bindings': {}, 'graph': unused})
        self.assertEqual(estimate_memory(prepare_graph(graph)[0], batch_size=2)['execution_elements'], report['execution_elements'])

    def test_projection_is_warning_and_query_failure_is_clear(self):
        from studio_core.memory_admission import admit_memory
        graph = hardware_graph()
        with patch('studio_core.devices.memory_snapshot', return_value=snapshot(30)):
            report = admit_memory(prepare_graph(graph)[0], 'cpu', batch_size=2, training=True)
        self.assertTrue(report['warnings'])
        self.assertEqual(report['admission'], 'minimum_fits_projection_warning')
        with patch('studio_core.devices.memory_snapshot', side_effect=OSError('query failed')):
            with self.assertRaisesRegex(ValueError, 'Cannot query.*memory'): admit_memory(prepare_graph(graph)[0], 'cpu')


if __name__ == '__main__': unittest.main()
