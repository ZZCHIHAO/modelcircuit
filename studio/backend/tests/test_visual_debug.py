import importlib.util
import unittest


def shared_graph():
    return {
        'schema_version': 1, 'id': 'debug_example', 'name': '共享层与融合', 'revision': 1,
        'inputs': [{'id': name, 'shape': [None, 2], 'dtype': 'float32'} for name in ('a', 'b')],
        'instances': [
            {'id': 'shared', 'kind': 'Linear', 'params': {'in_features': 2, 'out_features': 2}},
            {'id': 'add', 'kind': 'Add', 'params': {}},
            {'id': 'head', 'kind': 'Linear', 'params': {'in_features': 2, 'out_features': 2}, 'frozen': True},
        ],
        'calls': [
            {'id': 'left', 'instance_id': 'shared', 'inputs': {'x': 'input:a'}},
            {'id': 'right', 'instance_id': 'shared', 'inputs': {'x': 'input:b'}},
            {'id': 'fuse', 'instance_id': 'add', 'inputs': {'a': 'call:left:out', 'b': 'call:right:out'}},
            {'id': 'predict', 'instance_id': 'head', 'inputs': {'x': 'call:fuse:out'}},
        ],
        'outputs': [{'id': 'logits', 'source': 'call:predict:out'}], 'definitions': [],
    }


class VisualDebugTests(unittest.TestCase):
    def runtime(self):
        self.assertIsNotNone(importlib.util.find_spec('studio_core.debug_runtime'), 'visual debug runtime is missing')
        from studio_core.debug_runtime import run_debug
        return run_debug

    def test_one_real_update_matches_independent_shared_pytorch_model(self):
        import torch
        run = self.runtime()
        result = run(shared_graph(), {'mode': 'step', 'seed': 7, 'batch_size': 2, 'lr': 0.05,
                                      'sources': ['input:a', 'input:b', 'call:left:out', 'call:right:out']})
        probes = result['probes']
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(7)
            shared, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2).requires_grad_(False)
        original = shared.weight.detach().clone()
        a, b = [torch.tensor(probes['input:' + key]['activation']['values']).reshape(2, 2) for key in ('a', 'b')]
        left, right = shared(a), shared(b)
        left.retain_grad()
        right.retain_grad()
        loss = torch.nn.functional.cross_entropy(head(left + right), torch.tensor([0, 1]))
        optimizer = torch.optim.SGD(shared.parameters(), lr=0.05)
        loss.backward()
        optimizer.step()
        self.assertEqual(result['optimizer_steps'], 1)
        self.assertAlmostEqual(result['loss'], loss.item(), places=6)
        for key, value in [('left', left), ('right', right)]:
            observed = probes[f'call:{key}:out']
            self.assertEqual(observed['instance'], 'shared')
            torch.testing.assert_close(torch.tensor(observed['activation']['values']), value.detach().flatten())
            torch.testing.assert_close(torch.tensor(observed['gradient']['values']), value.grad.flatten())
        measured = {item['parameter']: item for item in result['parameters']}
        weight = measured['components.shared.weight']
        torch.testing.assert_close(torch.tensor(weight['before']['values']), original.flatten())
        torch.testing.assert_close(torch.tensor(weight['after']['values']), shared.weight.detach().flatten())
        self.assertEqual(measured['components.head.weight']['update_norm'], 0)
        self.assertIsNone(measured['components.head.weight']['gradient'])
        self.assertEqual(len([key for key in measured if key.endswith('.weight')]), 2)

    def test_scan_uses_same_input_and_does_not_edit_graph(self):
        run = self.runtime()
        graph = shared_graph()
        graph['instances'].append({'id': 'scale', 'kind': 'Scale', 'params': {'factor': 1.0}})
        graph['calls'].append({'id': 'gain', 'instance_id': 'scale', 'inputs': {'x': 'call:predict:out'}})
        graph['outputs'][0]['source'] = 'call:gain:out'
        result = run(graph, {'mode': 'sweep', 'sweep': {'instance': 'scale', 'parameter': 'factor', 'values': [1.0, 2.0]}})
        first, second = result['sweep']
        self.assertEqual(first['value'], 1.0)
        self.assertEqual(second['value'], 2.0)
        self.assertAlmostEqual(first['outputs']['logits']['l2'] * 2, second['outputs']['logits']['l2'], places=6)
        self.assertEqual(graph['instances'][-1]['params']['factor'], 1.0)
        self.assertEqual(result['optimizer_steps'], 0)

    def test_forward_reports_the_requested_interval_without_updates(self):
        run = self.runtime()
        config = {'mode': 'forward', 'seed': 17, 'sources': ['call:left:out']}
        full = run(shared_graph(), config)['probes']['call:left:out']['activation']
        result = run(shared_graph(), {**config, 'probe_offset': 2})
        probe = result['probes']['call:left:out']
        self.assertEqual(probe['activation']['values'], full['values'][2:])
        self.assertEqual(probe['activation']['offset'], 2)
        self.assertIsNone(probe['gradient'])
        self.assertEqual(result['optimizer_steps'], 0)

    def test_cancelled_before_model_allocation(self):
        run = self.runtime()
        with self.assertRaises(InterruptedError):
            run(shared_graph(), {'mode': 'forward'}, cancelled=lambda: True)
