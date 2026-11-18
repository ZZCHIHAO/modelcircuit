from copy import deepcopy
import unittest

from studio_core.experiment_spec import model_digest
from studio_core.schema import validate_graph
from studio_core.tensor_specs import normalize_tensor_spec, tensor_nbytes


def spec(shape, axes, dtype='float32', batch_axis=0):
    return dict(shape=shape, axes=axes, dtype=dtype, batch_axis=batch_axis)


def forwarding(port):
    return dict(inputs=[dict(id='x', **port)], instances=[], calls=[],
                outputs=[dict(id='out', source='input:x')])


def tensor_graph(port):
    return dict(id='tokens', revision=1, schema_version=2, capability_profile='tensor-v1',
                resource_profile='hardware-v5', resource_batch_bound=2, **forwarding(port))


class TensorSpecsTests(unittest.TestCase):
    def test_bytes_respect_dtype_and_actual_batch(self):
        self.assertEqual(tensor_nbytes(spec([None, 65, 64], ['batch', 'token', 'feature']), batch_size=2), 2*65*64*4)
        self.assertEqual(tensor_nbytes(spec([1, 1, 65, 65], ['mask_batch', 'head', 'query', 'key'], 'bool', None), batch_size=2), 65*65)
        with self.assertRaisesRegex(ValueError, 'batch'):
            tensor_nbytes(spec([2, 64], ['batch', 'feature']), batch_size=3)

    def test_reject_ambiguous_or_unsupported_specs(self):
        valid = spec([None, 65, 64], ['batch', 'token', 'feature'])
        for invalid in (dict(shape=[None,None,64]), dict(axes=['batch','token','token']), dict(dtype='int64'),
                        dict(dtype=[]), dict(axes=['batch','token']), dict(batch_axis=False), dict(batch_axis=None),
                        dict(shape=[None, True, 64]), dict(axes=['batch', 'token', 'axis name']),
                        dict(axes=['other','token','feature'])):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_tensor_spec({**valid, **invalid})

    def test_static_boolean_forward_is_valid_but_legacy_consumers_refuse(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from server import RunManager
        from studio_core.python_io import export_python
        from studio_core.runtime import build_model

        mask = spec([1, 1, 4, 4], ['mask_batch', 'head', 'query', 'key'], 'bool', None)
        graph = tensor_graph(mask)
        original = deepcopy(graph)
        report = validate_graph(graph)
        self.assertTrue(report['valid'], report)
        self.assertEqual(report['tensor_specs']['input:x'], mask)
        self.assertEqual(report['shapes']['input:x'], [1, 1, 4, 4])
        self.assertEqual(graph, original)
        with self.assertRaisesRegex(ValueError, 'tensor-v1'):
            export_python(graph)
        with self.assertRaisesRegex(ValueError, 'tensor-v1'):
            build_model(graph)
        with TemporaryDirectory() as temporary:
            manager = RunManager(Path(temporary) / 'runs')
            try:
                with self.assertRaisesRegex(ValueError, 'static validation only'):
                    manager.submit(graph, {})
                self.assertEqual(manager.runs, {})
            finally:
                manager.close()

    def test_nested_contract_propagates_boolean_type_and_checks_axes(self):
        mask = spec([1, 1, 4, 4], ['mask_batch', 'head', 'query', 'key'], 'bool', None)
        graph = tensor_graph(mask)
        graph['definitions'] = [dict(id='keep_mask', graph=forwarding(mask))]
        graph['instances'] = [dict(id='mask_chip', kind='chip', definition_id='keep_mask')]
        graph['calls'] = [dict(id='apply_mask', instance_id='mask_chip', inputs={'x': 'input:x'})]
        graph['outputs'][0]['source'] = 'call:apply_mask:out'
        report = validate_graph(graph)
        self.assertTrue(report['valid'], report)
        self.assertEqual(report['tensor_specs']['call:apply_mask:out'], mask)
        graph['inputs'][0]['axes'] = ['mask_batch', 'head', 'key', 'query']
        report = validate_graph(graph)
        self.assertFalse(report['valid'])
        self.assertEqual(report['errors'][0]['node_id'], 'apply_mask')

    def test_linear_inherits_token_axes_and_rejects_bool(self):
        graph = tensor_graph(spec([None, 5, 4], ['batch', 'token', 'feature']))
        graph['instances'] = [dict(id='projection', kind='Linear', params=dict(in_features=4, out_features=8))]
        graph['calls'] = [dict(id='project', instance_id='projection', inputs={'x': 'input:x'})]
        graph['outputs'][0]['source'] = 'call:project:out'
        report = validate_graph(graph)
        self.assertTrue(report['valid'], report)
        self.assertEqual(report['tensor_specs']['call:project:out'], spec([None, 5, 8], ['batch', 'token', 'feature']))
        graph['inputs'][0]['dtype'] = 'bool'
        report = validate_graph(graph)
        self.assertFalse(report['valid'])
        self.assertEqual(report['errors'][0]['node_id'], 'project')

    def test_profiles_and_semantic_identity(self):
        graph = tensor_graph(spec([None, 5, 4], ['batch', 'token', 'feature']))
        original_digest = model_digest(graph)
        graph['inputs'][0]['axes'][1] = 'time'
        self.assertNotEqual(model_digest(graph), original_digest)
        for changed in (dict(capability_profile='unknown'), dict(resource_profile='vision-v4'), dict(schema_version=1)):
            with self.subTest(changed=changed):
                self.assertFalse(validate_graph({**graph, **changed})['valid'])
        graph['inputs'][0]['shape'][0] = 3
        self.assertFalse(validate_graph(graph)['valid'])

    def test_linear_cannot_consume_the_batch_axis_as_features(self):
        graph = tensor_graph(spec([2], ['batch']))
        graph['instances'] = [dict(id='projection', kind='Linear', params=dict(in_features=2, out_features=4))]
        graph['calls'] = [dict(id='project', instance_id='projection', inputs={'x': 'input:x'})]
        graph['outputs'][0]['source'] = 'call:project'
        report = validate_graph(graph)
        self.assertFalse(report['valid'], report)
        self.assertEqual(report['errors'][0]['node_id'], 'project')

    def test_batchnorm_requires_an_actual_batch_axis(self):
        graph = tensor_graph(spec([2, 3, 4, 4], ['frame', 'channel', 'height', 'width'], batch_axis=None))
        graph['instances'] = [dict(id='norm', kind='BatchNorm2d', params={'num_features': 3})]
        graph['calls'] = [dict(id='apply_norm', instance_id='norm', inputs={'x': 'input:x'})]
        graph['outputs'][0]['source'] = 'call:apply_norm'
        report = validate_graph(graph)
        self.assertFalse(report['valid'], report)
        self.assertEqual(report['errors'][0]['node_id'], 'apply_norm')

    def test_legacy_graph_report_keeps_its_shape(self):
        graph = dict(id='old', revision=1, schema_version=1, **forwarding(dict(shape=[None, 4], dtype='float32')))
        report = validate_graph(graph)
        self.assertTrue(report['valid'], report)
        self.assertNotIn('tensor_specs', report)
        self.assertEqual(report['shapes']['input:x'], [None, 4])
