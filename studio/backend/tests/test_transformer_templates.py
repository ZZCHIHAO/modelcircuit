import ast
import unittest

import torch

from studio_core.experiment_spec import model_digest
from studio_core.python_io import export_python, import_python
from studio_core.runtime import build_model
from studio_core.schema import prepare_graph, validate_graph
from studio_core.transformer_templates import make_attention, make_encoder, make_small_vit
from transformer_acceptance import (
    bundle_graph, compare_reference, encoder_state_map, run_attention_case, run_encoder_case, run_vit_case,
)
from transformer_reference import EncoderReference


class TransformerTemplatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_attention_is_expandable_and_weights_are_the_softmax_output(self):
        settings = dict(sequence_length=5, hidden_dim=8, num_heads=2)
        default = bundle_graph(make_attention(**settings))
        self.assertEqual([port['id'] for port in default['outputs']], ['features'])
        graph = bundle_graph(make_attention(**settings, masked=True, return_attention=True))
        scope = graph['definitions'][0]['graph']
        self.assertEqual(graph['inputs'][1]['dtype'], 'bool')
        self.assertEqual(graph['inputs'][1]['shape'], [None, 5])
        self.assertEqual(graph['inputs'][1]['axes'], ['batch', 'key'])
        self.assertEqual([instance['kind'] for instance in scope['instances']].count('MatMul'), 2)
        name = next(instance['id'] for instance in scope['instances'] if instance['kind'] == 'Softmax')
        self.assertEqual(scope['outputs'][1], dict(scope['outputs'][1], source=f'call:{name}'))
        report = validate_graph(graph)
        self.assertEqual(report['tensor_specs']['call:entry:attention'], {
            'dtype': 'float32', 'batch_axis': 0, 'axes': ['batch', 'head', 'query', 'key'],
            'shape': [None, 2, 5, 5],
        })
        self.assertEqual(report['parameter_count'], 288)

    def test_attention_outputs_gradients_sgd_and_state(self):
        for tokens in (1, 5):
            for heads in (1, 2, 4):
                for mode in ('none', 'external', 'causal'):
                    for batch in (1, 2):
                        with self.subTest(tokens=tokens, heads=heads, mode=mode, batch=batch):
                            run_attention_case(tokens=tokens, heads=heads, mode=mode, batch=batch)

    def test_attention_rejects_invisible_query_rows_at_mask_call(self):
        model = build_model(bundle_graph(make_attention(sequence_length=3, hidden_dim=8, num_heads=2, masked=True)),
                            capture=False)
        inputs = {'x': torch.randn(2, 3, 8), 'keep': torch.tensor([[True, True, True], [False, False, False]])}
        with self.assertRaisesRegex((ValueError, RuntimeError), r'(?i)(entry.*mask|entry.*softmax)'):
            model(**inputs)

    def test_attention_dropout_keeps_the_exposed_probabilities_normalized(self):
        run_attention_case(tokens=5, heads=2, mode='external', batch=2, dropout=0.35)

    def test_encoder_compares_train_and_eval_with_dropout(self):
        run_encoder_case()

    def test_encoder_definition_edit_does_not_also_edit_block_input(self):
        bundle = make_encoder(sequence_length=5, hidden_dim=8, num_heads=2, mlp_dim=11)
        encoder, block, _ = bundle.definitions
        encoder['graph']['inputs'][0]['axes'][1] = 'sequence'
        self.assertEqual(block['graph']['inputs'][0]['axes'][1], 'token')

    def test_vit_defaults_match_independent_reference(self):
        # 3,136 patch + 4,224 class/position + 2*33,472 blocks + 128 norm + 130 classifier.
        graph = bundle_graph(make_small_vit())
        report = validate_graph(graph)
        self.assertEqual(report['parameter_count'], 74_562)
        self.assertEqual(report['tensor_specs']['call:entry:logits'], {
            'shape': [None, 2], 'dtype': 'float32', 'axes': ['batch', 'class'], 'batch_axis': 0,
        })
        resolved, _, _ = prepare_graph(graph)
        self.assertEqual(resolved['instances'][0]['_graph']['_tensor_specs']['call:tokens']['shape'], [None, 65, 64])
        model = build_model(graph, capture=False)
        self.assertEqual(tuple(model.state_dict()['components.entry.components.pos_token.value'].shape), (1, 65, 64))
        self.assertTrue(torch.equal(model.state_dict()['components.entry.components.cls_token.value'], torch.zeros(1, 1, 64)))
        run_vit_case()

    def test_vit_nondefault_geometry_classes_and_dropout(self):
        run_vit_case(settings=dict(image_size=8, patch_size=2, channels=1, hidden_dim=8,
                                   num_heads=2, mlp_dim=12, depth=1, num_classes=3,
                                   dropout=0.2, attention_dropout=0.15))

    def test_vit_python_roundtrip_preserves_training_and_source_edits(self):
        graph = bundle_graph(make_small_vit())
        code = export_python(graph)
        tree = ast.parse(code)
        tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'STUDIO_VISUAL_METADATA' for target in node.targets))]
        plain = ast.unparse(tree)
        baseline = build_model(graph, capture=False)
        namespace = {}
        exec(compile(plain, '<generated-vit-export>', 'exec'), namespace)
        standalone = namespace['Model']()
        with_metadata, without_metadata = (import_python(source) for source in (code, plain))
        for restored in (with_metadata, without_metadata):
            self.assertEqual(model_digest(restored), model_digest(graph))
        implementations = [baseline, standalone, *(build_model(restored, capture=False) for restored in
                                                    (with_metadata, without_metadata))]
        initial = {name: value.detach().clone() for name, value in baseline.state_dict().items()}
        inputs = torch.randn(2, 3, 32, 32)
        results = []
        for model in implementations:
            model.load_state_dict(initial, strict=True)
            images = inputs.clone().requires_grad_(True)
            logits = model(x=images)['logits']
            torch.nn.functional.cross_entropy(logits, torch.tensor([1, 0])).backward()
            torch.optim.SGD(model.parameters(), lr=.01).step()
            results.append((logits.detach(), images.grad, {key: value.grad for key, value in model.named_parameters()},
                            model.state_dict()))
        for result in results[1:]:
            torch.testing.assert_close(result, results[0], atol=2e-6, rtol=1e-4)

        edited_tree = ast.parse(code)
        classifiers = [value for node in ast.walk(edited_tree) if isinstance(node, ast.Dict)
                       for key, value in zip(node.keys, node.values)
                       if isinstance(key, ast.Constant) and key.value == 'classifier' and isinstance(value, ast.Call)]
        self.assertEqual(len(classifiers), 1)
        width = next(keyword for keyword in classifiers[0].keywords if keyword.arg == 'out_features')
        width.value = ast.Constant(3)
        edited = import_python(ast.unparse(edited_tree))
        self.assertNotEqual(model_digest(edited), model_digest(graph))
        self.assertEqual(validate_graph(edited)['shapes']['call:entry:logits'], [None, 3])
        self.assertEqual(tuple(build_model(edited, capture=False).state_dict()[
            'components.entry.components.classifier.weight'].shape), (3, 64))

    def test_encoder_blocks_can_share_one_instance_without_duplicate_state(self):
        graph = bundle_graph(make_encoder(sequence_length=5, hidden_dim=8, num_heads=2, mlp_dim=11, depth=2))
        scope = next(item['graph'] for item in graph['definitions'] if item['id'] == graph['instances'][0]['definition_id'])
        separate = validate_graph(graph)['parameter_count']
        block_calls = [call for call in scope['calls'] if call['id'].startswith('block')]
        self.assertNotEqual(block_calls[0]['instance_id'], block_calls[1]['instance_id'])
        separate_instance = block_calls[1]['instance_id']
        block_calls[1]['instance_id'] = block_calls[0]['instance_id']
        scope['instances'] = [item for item in scope['instances'] if item['id'] != separate_instance]
        shared = validate_graph(graph)['parameter_count']
        self.assertEqual(shared, 531)
        self.assertEqual(separate - shared, 515)
        reference = _TiedEncoderReference(8, 2, 11, 1)
        compare_reference(graph, reference, encoder_state_map(1), torch.randn(2, 5, 8))

    def test_factory_configuration_checks_before_weight_creation(self):
        invalid = (
            (make_attention, dict(sequence_length=5, hidden_dim=8, num_heads=3)),
            (make_attention, dict(sequence_length=5, hidden_dim=8, num_heads=2, dropout=float('nan'))),
            (make_attention, dict(sequence_length=5, hidden_dim=8, num_heads=2, masked=True, causal=True)),
            (make_attention, dict(sequence_length=5, hidden_dim=8, num_heads=2, return_attention=1)),
            (make_encoder, dict(sequence_length=5, hidden_dim=8, num_heads=2, mlp_dim=0)),
            (make_small_vit, dict(image_size=[32, 24])),
            (make_small_vit, dict(image_size=32, patch_size=5)),
            (make_small_vit, dict(depth=0)),
            (make_small_vit, dict(num_classes=1)),
            (make_small_vit, dict(resource_batch_bound=65)),
        )
        for factory, kwargs in invalid:
            with self.subTest(factory=factory, kwargs=kwargs), self.assertRaises(ValueError):
                factory(**kwargs)

    def test_reference_mapping_fails_on_missing_or_wrong_shape_state(self):
        graph = bundle_graph(make_encoder(sequence_length=5, hidden_dim=8, num_heads=2, mlp_dim=11))
        reference = EncoderReference(8, 2, 11, 1)
        mapping = encoder_state_map(1)
        x = torch.randn(2, 5, 8)
        with self.assertRaisesRegex(AssertionError, 'Graph state keys differ'):
            compare_reference(graph, reference, dict(list(mapping.items())[1:]), x)
        reference.norm.weight = torch.nn.Parameter(torch.ones(9))
        with self.assertRaisesRegex(AssertionError, 'State shape/dtype differs'):
            compare_reference(graph, reference, mapping, x)


class _TiedEncoderReference(EncoderReference):
    def forward(self, x):
        block = self.blocks[0]
        return {'features': self.norm(block(block(x)))}
