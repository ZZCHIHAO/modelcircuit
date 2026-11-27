"""Editable encoders and classifier for a checked cohort's named inputs."""

from .experiment_spec import multimodal_ports, validate_multimodal_bindings

_NAMES = {'image': '二维影像', 'volume': '三维体数据', 'tabular': '临床表格'}
_FEATURE_WIDTH = 8


def _add(graph, identifier, kind, inputs, params, x, y, **fields):
    graph['instances'].append({'id': identifier, 'kind': kind, 'params': params, **fields})
    graph['calls'].append({'id': identifier, 'instance_id': identifier, 'inputs': inputs, 'position': {'x': x, 'y': y}})
    return 'call:' + identifier


def _port(identifier, shape, axes, x, y, **fields):
    return {'id': identifier, 'dtype': 'float32', 'shape': list(shape), 'axes': list(axes),
            'batch_axis': 0, 'position': {'x': x, 'y': y}, **fields}


def _encoder(modality, shape, axes):
    graph = {'inputs': [_port('x', shape, axes, 0, 100)], 'instances': [], 'calls': [], 'outputs': []}
    if modality == 'tabular':
        feature = _add(graph, 'dense', 'Linear', {'x': 'input:x'},
                       {'in_features': shape[1], 'out_features': _FEATURE_WIDTH}, 250, 100)
        feature = _add(graph, 'activation', 'ReLU', {'x': feature}, {}, 495, 100)
    else:
        rank = len(shape) - 2
        feature = _add(graph, 'conv', f'Conv{rank}d', {'x': 'input:x'},
                       {'in_channels': shape[1], 'out_channels': _FEATURE_WIDTH,
                        'kernel_size': 3, 'padding': 1}, 240, 100)
        feature = _add(graph, 'activation', 'ReLU', {'x': feature}, {}, 475, 100)
        feature = _add(graph, 'pool', f'AdaptiveAvgPool{rank}d', {'x': feature}, {'output_size': [1] * rank}, 700, 100)
        feature = _add(graph, 'flatten', 'Reshape', {'x': feature},
                       {'shape': ['B', _FEATURE_WIDTH], 'axes': ['batch', 'feature']}, 950, 100)
    graph['outputs'].append({'id': 'features', 'source': feature, 'position': {'x': 1400, 'y': 100}})
    return {'id': 's14_' + modality + '_encoder', 'name': _NAMES[modality] + '编码器',
            'version': '1.0.0', 'output_mode': 'named', 'public_parameters': [], 'bindings': {}, 'graph': graph}


def multimodal_graph(manifest, mapping=None):
    if manifest.get('kind') != 'multimodal_csv':
        raise ValueError('Inspect a multimodal cohort before creating its classifier')
    shapes, axes = manifest['input_shapes'], manifest['input_axes']
    mapping = multimodal_ports({'modalities': shapes}) if mapping is None else mapping
    validate_multimodal_bindings(shapes, mapping)
    graph = {'id': 'cohort_classifier', 'name': '病例分类', 'schema_version': 2, 'capability_profile': 'tensor-v1',
             'resource_profile': 'hardware-v5', 'resource_batch_bound': 64, 'revision': 1,
             'inputs': [], 'instances': [], 'calls': [], 'outputs': [], 'definitions': []}
    encodings = []
    for index, modality in enumerate(shapes):
        if modality not in mapping:
            continue
        y = 85 + index * 158
        port = mapping[modality]
        graph['inputs'].append(_port(port, shapes[modality], axes[modality], 0, y, name=_NAMES[modality]))
        encoder = _encoder(modality, shapes[modality], axes[modality])
        graph['definitions'].append(encoder)
        encoding = _add(graph, 'encode_' + modality, 'chip', {'x': 'input:' + port}, {}, 210, y,
                        definition_id=encoder['id'], name=encoder['name'])
        encodings.append(encoding + ':features')
    fused, *rest = encodings
    x = 445
    for index, other in enumerate(rest, 1):
        fused = _add(graph, 'fusion_' + str(index), 'Concat', {'a': fused, 'b': other}, {'dim': 1}, x, 145)
        x += 198
    head = _add(graph, 'head', 'Linear', {'x': fused},
                {'in_features': _FEATURE_WIDTH * len(encodings), 'out_features': len(manifest['class_names'])}, x, 145)
    logits = _add(graph, 'class_axis', 'AxisRename', {'x': head}, {'axes': ['batch', 'class']}, x + 198, 145)
    graph['outputs'].append({'id': 'logits', 'source': logits, 'position': {'x': x + 380, 'y': 145}})
    return graph
