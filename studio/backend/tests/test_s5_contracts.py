import ast
import unittest
from unittest.mock import patch

import torch

from studio_core.memory_admission import estimate_memory
from studio_core.probe_capture import make_capture
from studio_core.python_io import export_python, import_python
from studio_core.schema import prepare_graph
from studio_core.task_adapters import get_task_adapter
from studio_core.transformer_templates import make_attention
from transformer_acceptance import bundle_graph


class S5ContractTests(unittest.TestCase):
    def test_capture_transfers_only_the_bounded_slice(self):
        request = {'source': 'call:weights:out', 'sample_index': 0, 'head_index': 1,
                   'query_range': [2, 100], 'key_range': [1, 101], 'statistics': ['mean']}
        value = torch.arange(2 * 110 * 110, dtype=torch.float32).reshape(1, 2, 110, 110)
        spec = {'dtype': 'float32', 'axes': ['batch', 'head', 'query', 'key']}
        original_to = torch.Tensor.to
        transferred = []

        def observe(tensor, *args, **kwargs):
            transferred.append(tuple(tensor.shape))
            return original_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, 'to', observe):
            small = make_capture({'requests': [request], 'max_bytes': 32}, {})
            small.observe(request['source'], value, spec)
            self.assertTrue(small.results[request['source']]['omitted'])
            self.assertEqual(transferred, [])
            capture = make_capture({'requests': [request]}, {})
            capture.observe('call:unrequested', value, spec)
            capture.observe(request['source'], value, spec)
        result = capture.results[request['source']]
        self.assertEqual(transferred, [(64, 64)])
        self.assertEqual(result['captured_bytes'], 64 * 64 * 4)
        self.assertTrue(result['truncated'])
        torch.testing.assert_close(torch.tensor(result['values']), value[0, 1, 2:66, 1:65])

    def test_static_import_rejects_changed_mask_helper(self):
        graph = bundle_graph(make_attention(sequence_length=5, hidden_dim=8, num_heads=2, causal=True))
        code = export_python(graph)
        tampered = code.replace('masked_fill(~keep, -torch.inf)', 'masked_fill(keep, -torch.inf)')
        self.assertNotEqual(code, tampered)
        with self.assertRaisesRegex(ValueError, 'modified MaskApply'):
            import_python(tampered)
        tree = ast.parse(code)
        root = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Model')
        forward = next(node for node in root.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
        forward.args.args[1].annotation = ast.Constant('float32[?,5,8]')
        with self.assertRaisesRegex(ValueError, 'tensor-v1 requires'):
            import_python(ast.unparse(tree))

    def test_typed_memory_counts_bool_buffer_and_persistent_parameters(self):
        graph = bundle_graph(make_attention(sequence_length=5, hidden_dim=8, num_heads=2, causal=True))
        resolved, _, count = prepare_graph(graph)
        estimate = estimate_memory(resolved, batch_size=2, training=True, capture=True)
        self.assertEqual(estimate['buffer_bytes'], 25)
        self.assertEqual(estimate['parameter_bytes'], count * 4)
        self.assertEqual(estimate['capture_bytes'], 1_048_576)

    def test_adapter_matches_classification_loss_and_weighted_summary(self):
        task = get_task_adapter('classification', 1)({'input_ports': None, 'input_port': 'x', 'data': {'kind': 'csv'}})
        logits = torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)
        batch = task.bind_batch((torch.zeros(2, 1), torch.tensor([0, 1]), ('a', 'b')), 'cpu')
        loss = task.loss({'logits': logits}, batch)
        torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits, batch['targets']))
        self.assertEqual(task.finalize(task.summarize({'logits': logits}, batch, loss)),
                         {'loss': float(loss.detach()), 'accuracy': 1., 'count': 2})
        with self.assertRaisesRegex(ValueError, 'Unknown task'):
            get_task_adapter('classification', 999)
