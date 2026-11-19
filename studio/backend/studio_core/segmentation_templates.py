"""Editable 2D/3D encoder-decoder starter with one skip connection."""
from .segmentation_task import spatial_axes


def segmentation_graph(manifest):
    if manifest['kind'] != 'segmentation_csv':
        raise ValueError('Inspect segmentation pairs before creating a segmentation starter')
    shape = manifest['input_shape']
    axes = spatial_axes(shape)
    rank = len(axes)
    convolution = f'Conv{rank}d'
    instances, calls = [], []

    def add(name, kind, x, y, inputs, **params):
        instances.append({'id': name, 'kind': kind, 'params': params})
        calls.append({'id': name, 'instance_id': name, 'inputs': inputs, 'position': {'x': x, 'y': y}})
        return 'call:' + name

    encoder = add('encoder', convolution, 250, 100, {'x': 'input:x'}, in_channels=shape[1], out_channels=8,
                  kernel_size=3, padding=1)
    skip = add('encoder_relu', 'ReLU', 500, 100, {'x': encoder})
    pool = add('pool', f'MaxPool{rank}d', 500, 320, {'x': skip}, kernel_size=2)
    middle = add('bottleneck', convolution, 750, 320, {'x': pool}, in_channels=8, out_channels=16, kernel_size=3, padding=1)
    middle = add('bottleneck_relu', 'ReLU', 1000, 320, {'x': middle})
    up = add('upsample', 'Upsample', 1250, 320, {'x': middle}, size=list(shape[2:]))
    join = add('skip_fusion', 'Concat', 1500, 100, {'a': skip, 'b': up}, dim=1)
    decoder = add('decoder', convolution, 1750, 100, {'x': join}, in_channels=24, out_channels=8, kernel_size=3, padding=1)
    decoder = add('decoder_relu', 'ReLU', 2000, 100, {'x': decoder})
    logits = add('mask_head', convolution, 2250, 100, {'x': decoder}, in_channels=8,
                 out_channels=len(manifest['mask_values']), kernel_size=1)
    logits = add('class_axis', 'AxisRename', 2500, 100, {'x': logits}, axes=['batch', 'class', *axes])
    return {'id': f'segmentation_{rank}d', 'schema_version': 2, 'capability_profile': 'tensor-v1',
            'resource_profile': 'hardware-v5', 'resource_batch_bound': 64, 'revision': 1,
            'inputs': [{'id': 'x', 'dtype': 'float32', 'shape': list(shape), 'axes': ['batch', 'channel', *axes],
                        'batch_axis': 0, 'position': {'x': 0, 'y': 100}}],
            'instances': instances, 'calls': calls, 'definitions': [],
            'outputs': [{'id': 'mask_logits', 'source': logits, 'position': {'x': 2750, 'y': 100}}]}
