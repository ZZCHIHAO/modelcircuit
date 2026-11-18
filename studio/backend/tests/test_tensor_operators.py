import unittest

import torch

from studio_core.runtime import build_model
from studio_core.schema import validate_graph


def spec(shape, axes, dtype='float32', batch_axis=0):
    return dict(shape=shape, axes=axes, dtype=dtype, batch_axis=batch_axis)


def graph(inputs, instances, calls, source):
    return dict(id='tensor_case', revision=1, schema_version=2, capability_profile='tensor-v1',
                resource_profile='hardware-v5', resource_batch_bound=2,
                inputs=[dict(id=name, **typed) for name, typed in inputs.items()],
                instances=instances, calls=calls, outputs=[dict(id='out', source=source)])


def operation(kind, params, inputs):
    return graph(inputs, [dict(id='op', kind=kind, params=params)],
                 [dict(id='apply_op', instance_id='op', inputs={name: 'input:' + name for name in inputs})],
                 'call:apply_op')


class TensorOperatorTests(unittest.TestCase):
    def assert_output_spec(self, document, output):
        report = validate_graph(document)
        self.assertTrue(report['valid'], report['errors'])
        self.assertEqual(report['tensor_specs'][document['outputs'][0]['source']], output)

    def assert_graph_error(self, document, pattern):
        report = validate_graph(document)
        self.assertFalse(report['valid'], report)
        self.assertRegex(report['errors'][0]['message'], pattern)

    def test_axis_operators_and_noncontiguous_reshape(self):
        x = spec([None, 3, 2], ['batch', 'token', 'feature'])
        actual = torch.arange(12, dtype=torch.float32).view(2, 2, 3).transpose(1, 2)
        for kind, params, output, expected in (
            ('Reshape', dict(shape=['B', 6], axes=['batch', 'flat']),
             spec([None, 6], ['batch', 'flat']), torch.tensor([[0., 3., 1., 4., 2., 5.], [6., 9., 7., 10., 8., 11.]])),
            ('Permute', dict(dims=[0, 2, 1]), spec([None, 2, 3], ['batch', 'feature', 'token']), actual.transpose(1, 2)),
            ('AxisRename', dict(axes=['batch', 'query', 'feature']), spec([None, 3, 2], ['batch', 'query', 'feature']), actual),
            ('Select', dict(dim=1, index=-1), spec([None, 2], ['batch', 'feature']), actual[:, 2]),
            ('Softmax', dict(dim=-2), x, torch.softmax(actual, -2)),
        ):
            with self.subTest(kind=kind):
                document = operation(kind, params, {'x': x})
                self.assert_output_spec(document, output)
                torch.testing.assert_close(build_model(document, capture=False)(x=actual)['out'], expected)

    def test_axis_rules_reject_invalid_dimensions(self):
        x = spec([None, 3, 2], ['batch', 'token', 'feature'])
        for kind, params, pattern in (
            ('Reshape', dict(shape=['B', 5], axes=['batch', 'flat']), 'element count'),
            ('Reshape', dict(shape=[6], axes=['flat']), 'batch'),
            ('Reshape', dict(shape=['B', -1], axes=['batch', 'flat']), 'positive integer'),
            ('Permute', dict(dims=[0, 1, 1]), 'permutation'),
            ('Permute', dict(dims=[1, 0, 2]), 'batch'),
            ('AxisRename', dict(axes=['sample', 'time', 'feature']), 'batch'),
            ('Softmax', dict(dim=-3), 'batch'),
            ('Select', dict(dim=1, index=3), 'index'),
            ('Select', dict(dim=0, index=0), 'batch'),
        ):
            with self.subTest(kind=kind, params=params):
                self.assert_graph_error(operation(kind, params, {'x': x}), pattern)

    def test_broadcast_sums_gradients_and_requires_explicit_batched_like(self):
        params = dict(shape=['B', 3, 2], axes=['batch', 'token', 'feature'])
        ports = dict(x=spec([1, 2], ['singleton', 'feature'], batch_axis=None),
                     like=spec([None, 7], ['batch', 'unrelated']))
        document = operation('BroadcastTo', params, ports)
        self.assert_output_spec(document, spec([None, 3, 2], ['batch', 'token', 'feature']))
        x = torch.tensor([[2., 5.]], requires_grad=True)
        result = build_model(document, capture=False)(x=x, like=torch.zeros(2, 7))['out']
        torch.testing.assert_close(result, torch.tensor([[[2., 5.]] * 3] * 2))
        self.assertEqual(result.untyped_storage().data_ptr(), x.untyped_storage().data_ptr())
        result.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 6.))
        self.assert_graph_error(operation('BroadcastTo', params, {**ports, 'like': spec([2, 7], ['time', 'feature'], batch_axis=None)}), 'like.*batch')

    def test_matmul_rejects_prefix_broadcast_and_axis_collisions(self):
        a = spec([None, 2, 3, 4], ['batch', 'head', 'query', 'feature'])
        b = spec([None, 2, 4, 5], ['batch', 'head', 'feature', 'key'])
        document = operation('MatMul', {}, dict(a=a, b=b))
        self.assert_output_spec(document, spec([None, 2, 3, 5], ['batch', 'head', 'query', 'key']))
        x, y = torch.randn(2, 2, 3, 4), torch.randn(2, 2, 4, 5)
        torch.testing.assert_close(build_model(document, capture=False)(a=x, b=y)['out'], torch.einsum('bhqd,bhdk->bhqk', x, y))
        self.assert_graph_error(operation('MatMul', {}, dict(a=a, b={**b, 'shape': [None, 1, 4, 5]})), 'prefix')
        self.assert_graph_error(operation('MatMul', {}, dict(a={**a, 'dtype': 'bool'}, b=b)), 'float32')
        self.assert_graph_error(operation('MatMul', {}, dict(a=a, b={**b, 'axes': ['batch', 'head', 'feature', 'query']})), 'Axes')

    def test_parameter_sources_share_each_instance_and_freeze(self):
        params = dict(spec=spec([1, 3], ['token', 'feature'], batch_axis=None), init=dict(kind='zeros'))
        document = graph({}, [dict(id='shared', kind='ParameterTensor', params=params),
                              dict(id='copy_tensor', kind='ParameterTensor', params=params, frozen=True)],
                         [dict(id='first', instance_id='shared', inputs={}),
                          dict(id='repeat', instance_id='shared', inputs={}),
                          dict(id='independent', instance_id='copy_tensor', inputs={})], 'call:first')
        document['outputs'] += [dict(id='repeat', source='call:repeat'), dict(id='independent', source='call:independent')]
        self.assertEqual(validate_graph(document)['parameter_count'], 6)
        model = build_model(document, capture=False)
        values = model()
        self.assertIs(values['repeat'], values['out'])
        self.assertIsNot(values['independent'], values['out'])
        self.assertFalse(values['independent'].requires_grad)
        self.assertEqual(set(model.state_dict()), {'components.shared.value', 'components.copy_tensor.value'})
        torch.testing.assert_close(values['out'], torch.zeros(1, 3))
        (values['out'].sum() + values['repeat'].sum()).backward()
        torch.testing.assert_close(model.components['shared'].value.grad, torch.full((1, 3), 2.))
        torch.optim.SGD(model.parameters(), lr=0.125).step()
        reloaded = build_model(document, capture=False)
        reloaded.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(reloaded()['out'], torch.full((1, 3), -0.25))
        torch.testing.assert_close(reloaded()['independent'], torch.zeros(1, 3))
        document['instances'][0]['params']['init'] = dict(kind='normal', mean=2., std=0.)
        torch.testing.assert_close(build_model(document, capture=False)()['out'], torch.full((1, 3), 2.))

    def test_causal_mask_is_persistent_and_boolean(self):
        document = graph({}, [dict(id='mask', kind='CausalMask', params={'length': 3})],
                         [dict(id='causal', instance_id='mask', inputs={})], 'call:causal')
        self.assertEqual(validate_graph(document)['parameter_count'], 0)
        model = build_model(document, capture=False)
        self.assertEqual(set(model.state_dict()), {'components.mask.keep'})
        self.assertFalse(model()['out'].requires_grad)
        torch.testing.assert_close(model()['out'], torch.tensor([[[[True, False, False], [True, True, False], [True, True, True]]]]))
        document['instances'][0]['params']['length'] = True
        self.assert_graph_error(document, 'positive integer')

    def mask_graph(self):
        scores = spec([None, 2, 3], ['batch', 'query', 'key'])
        return graph({'scores': scores, 'keep': {**scores, 'dtype': 'bool'}},
                     [dict(id='mask', kind='MaskApply'), dict(id='normalize', kind='Softmax', params=dict(dim=-1))],
                     [dict(id='masked', instance_id='mask', inputs=dict(scores='input:scores', keep='input:keep')),
                      dict(id='probability', instance_id='normalize', inputs=dict(x='call:masked:out'))], 'call:probability')

    def test_mask_checks_every_row_and_never_accepts_additive_mask(self):
        document = self.mask_graph()
        model = build_model(document, capture=False)
        scores = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]])
        keep = torch.tensor([[[True, False, True], [False, True, False]]])
        expected = torch.tensor([[[0.11920292, 0., 0.88079708], [0., 1., 0.]]])
        torch.testing.assert_close(model(scores=scores, keep=keep)['out'], expected)
        with self.assertRaisesRegex(ValueError, r'call:masked.*all.masked'):
            model(scores=scores, keep=torch.zeros_like(keep))
        with self.assertRaisesRegex(ValueError, 'keep.*bool'):
            model(scores=scores, keep=keep.float())
        with self.assertRaisesRegex(ValueError, 'batch'):
            model(scores=scores, keep=keep.expand(2, -1, -1))
        scores[0, 0, 1] = torch.nan  # A masked key is still not a valid numeric input.
        with self.assertRaisesRegex(ValueError, 'input:scores.*non-finite'):
            model(scores=scores, keep=keep)
        document['inputs'][1]['dtype'] = 'float32'
        self.assert_graph_error(document, 'bool')

    def test_masked_score_only_flows_directly_to_last_axis_softmax(self):
        document = self.mask_graph()
        document['outputs'].append(dict(id='unsafe', source='call:masked'))
        self.assert_graph_error(document, 'masked.score|MaskApply')
        document = self.mask_graph()
        document['instances'][1]['params']['dim'] = 1
        self.assert_graph_error(document, 'Softmax')
        document = self.mask_graph()
        document['instances'][1] = dict(id='normalize', kind='Identity')
        self.assert_graph_error(document, 'Softmax')
        document = self.mask_graph()
        document['calls'].pop()
        document['outputs'][0]['source'] = 'input:scores'
        self.assert_graph_error(document, 'Softmax')

    def test_v2_linear_and_layernorm_kwargs_leave_v1_unchanged(self):
        x = spec([None, 3], ['batch', 'feature'])
        for kind, params, expected in (
            ('Linear', dict(in_features=3, out_features=2, bias=False), 6),
            ('LayerNorm', dict(normalized_shape=3, eps=1e-6, elementwise_affine=False), 0),
        ):
            with self.subTest(kind=kind):
                document = operation(kind, params, dict(x=x))
                self.assertEqual(validate_graph(document)['parameter_count'], expected)
                model = build_model(document, capture=False)
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), expected)
                if kind == 'LayerNorm':
                    values = torch.tensor([[1.001, 1.002, 1.003]])
                    torch.testing.assert_close(model(x=values)['out'], torch.nn.functional.layer_norm(values, (3,), eps=1e-6))
                document['schema_version'] = 1
                document.pop('capability_profile')
                self.assert_graph_error(document, 'unsupported parameters')
