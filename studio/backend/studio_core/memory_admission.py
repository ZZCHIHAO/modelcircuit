"""Pure float32 allocation estimates and runtime-only hardware admission.

The minimum covers persistent model storage and worker-owned inputs/datasets,
not a promise that an entire run fits.
Projected peaks deliberately include conservative autograd/workspace allowances;
exceeding a projection warns, while a provably impossible minimum rejects.
PyTorch's allocator remains authoritative. No device query occurs in estimation.
"""
import math

from .operator_rules import _parameter_count, infer_operator


def estimate_memory(resolved, *, batch_size=None, minimum_batch_size=None, training=False, optimizer='sgd',
                    capture=True, workflow='build', resume=False, device='cpu', host_dataset_bytes=0):
    """Estimate bytes from constructed instances and actual root call paths.

    Parameters are charged once per constructed instance, including unused ones.
    Execution elements follow runtime's per-call ledger (including chip inputs
    and parent outputs), never the definition-validation/compilation ledger.
    batch_size plans the largest possible execution. minimum_batch_size may
    identify a smaller guaranteed worker input allocation before resume progress
    is known; omitted values preserve the same-batch minimum.
    """
    from .schema import graph_resource_limits
    limits = graph_resource_limits(resolved)
    batch_size = limits.batch_bound if batch_size is None else batch_size
    if type(batch_size) is not int or not 1 <= batch_size <= limits.batch_bound:
        raise ValueError('batch_size must be between 1 and resource_batch_bound')
    minimum_batch_size = batch_size if minimum_batch_size is None else minimum_batch_size
    if type(minimum_batch_size) is not int or not 1 <= minimum_batch_size <= batch_size:
        raise ValueError('minimum_batch_size must be between 1 and projected batch_size')
    if optimizer not in ('sgd', 'adamw'):
        raise ValueError('Memory estimation supports SGD or AdamW')
    if type(host_dataset_bytes) is not int or host_dataset_bytes < 0:
        raise ValueError('host_dataset_bytes must be a nonnegative integer')
    parameter_bytes = trainable_bytes = buffer_bytes = largest_parameter_bytes = trainable_tensors = 0
    pending = [(resolved, False)]
    while pending:
        scope, inherited_frozen = pending.pop()
        for instance in scope['instances']:
            frozen = inherited_frozen or instance.get('frozen', False)
            kind, params = instance['kind'], instance.get('params', {})
            if kind == 'chip':
                pending.append((instance['_graph'], frozen))
                continue
            count = _parameter_count(kind, params)
            size = 4 * count
            parameter_bytes += size
            if kind == 'Linear': tensor_sizes = [params['in_features'] * params['out_features'] * 4, params['out_features'] * 4]
            elif kind == 'Conv2d':
                bias = params['out_channels'] * 4 if params['bias'] else 0
                tensor_sizes = [size - bias] + ([bias] if bias else [])
            elif kind in ('LayerNorm', 'BatchNorm2d') and count: tensor_sizes = [size // 2, size // 2]
            else: tensor_sizes = []
            largest_parameter_bytes = max([largest_parameter_bytes, *tensor_sizes])
            if not frozen:
                trainable_bytes += size
                trainable_tensors += len(tensor_sizes)
            if kind == 'BatchNorm2d' and params['track_running_stats']:
                buffer_bytes += params['num_features'] * 8 + 8  # two float32 vectors, int64 counter

    def execute(scope, input_shapes):
        shapes = {'input:' + key: list(shape) for key, shape in input_shapes.items()}
        origins = {source: ('input', source) for source in shapes}
        total = sum(math.prod(shape) for shape in input_shapes.values())
        instances = {instance['id']: instance for instance in scope['instances']}
        for call in scope['_ordered_calls']:
            instance = instances[call['instance_id']]
            arguments = {key: shapes[source] for key, source in call['inputs'].items()}
            if instance['kind'] == 'chip':
                outputs, aliases, child_total = execute(instance['_graph'], arguments)
                total += child_total
            else:
                shape, _ = infer_operator(instance['kind'], instance.get('params', {}), arguments)
                outputs, aliases = {'out': shape}, {'out': 'out'}
            charged = set()
            for port, shape in outputs.items():
                origin = ('call', call['id'], aliases[port])
                ref = 'call:' + call['id'] + ':' + port
                shapes[ref], origins[ref] = shape, origin
                if origin not in charged:
                    total += math.prod(shape)
                    charged.add(origin)
                if len(outputs) == 1:
                    shapes['call:' + call['id']], origins['call:' + call['id']] = shape, origin
        return ({port['id']: shapes[port['source']] for port in scope['outputs']},
                {port['id']: origins[port['source']] for port in scope['outputs']}, total)

    input_shapes = {port['id']: [batch_size if axis is None else axis for axis in port['shape']] for port in resolved['inputs']}
    _, _, execution_elements = execute(resolved, input_shapes)
    input_bytes = sum(math.prod(shape) * 4 for shape in input_shapes.values())
    minimum_input_bytes = sum(math.prod([minimum_batch_size if axis is None else axis for axis in port['shape']]) * 4
                              for port in resolved['inputs'])
    execution_bytes = execution_elements * 4
    gradient_bytes = trainable_bytes if training else 0
    optimizer_bytes = (trainable_bytes * 2 + trainable_tensors * 4) if training and optimizer == 'adamw' else 0
    activation_bytes = execution_bytes * (3 if training else 1)
    capture_bytes = execution_bytes if capture or workflow == 'classification' else 0
    temporary_bytes = max(input_bytes, largest_parameter_bytes, execution_bytes // 2)
    state_bytes = parameter_bytes + buffer_bytes
    # build_model constructs on CPU then moves to CUDA. Preserve its established
    # seeded initialization semantics and include that host staging explicitly.
    host_staging_bytes = state_bytes if str(device).startswith('cuda:') else 0
    host_checkpoint_bytes = host_measurement_bytes = 0
    device_measurement_bytes = 0
    if workflow == 'classification':
        # Initial state; previous/current epoch snapshots; best snapshot; final
        # restricted reload's model/best trees; optional loaded resume trees.
        host_checkpoint_bytes = state_bytes * (6 + (3 if resume else 0)) + optimizer_bytes * (4 + (2 if resume else 0))
        # CPU transfer plus float64 current, initial and difference temporaries;
        # gradient double-norm can coexist on the selected compute device.
        host_measurement_bytes = largest_parameter_bytes * 7 + execution_bytes * 2
        if str(device).startswith('cuda:'):
            device_measurement_bytes = largest_parameter_bytes * 2  # float64 gradient norm temporary
    elif workflow == 'synthetic':
        host_checkpoint_bytes = parameter_bytes  # originals for update norms
        host_measurement_bytes = largest_parameter_bytes
    device_projected = state_bytes + gradient_bytes + optimizer_bytes + activation_bytes + capture_bytes + temporary_bytes + device_measurement_bytes
    host_extra = host_staging_bytes + host_checkpoint_bytes + host_measurement_bytes + host_dataset_bytes * 12
    if str(device) == 'cpu': device_projected += host_extra
    worker_inputs = minimum_input_bytes if workflow in ('synthetic', 'classification') else 0
    minimum_device = state_bytes + worker_inputs
    if str(device) == 'cpu': minimum_device += host_dataset_bytes
    minimum_host = host_dataset_bytes + max(state_bytes, worker_inputs) if str(device).startswith('cuda:') else 0
    return {'parameter_bytes': parameter_bytes, 'trainable_parameter_bytes': trainable_bytes,
            'buffer_bytes': buffer_bytes, 'state_bytes': state_bytes, 'gradient_bytes': gradient_bytes,
            'optimizer_bytes': optimizer_bytes, 'input_bytes': input_bytes, 'minimum_input_bytes': minimum_input_bytes,
            'execution_elements': execution_elements, 'activation_bytes': activation_bytes,
            'capture_bytes': capture_bytes, 'temporary_bytes': temporary_bytes,
            'host_staging_bytes': host_staging_bytes, 'host_checkpoint_bytes': host_checkpoint_bytes,
            'host_measurement_bytes': host_measurement_bytes,
            'device_measurement_bytes': device_measurement_bytes,
            'host_dataset_bytes': host_dataset_bytes,
            'minimum_device_bytes': minimum_device, 'minimum_host_bytes': minimum_host,
            'projected_device_bytes': device_projected,
            'projected_host_bytes': host_extra if str(device).startswith('cuda:') else device_projected,
            'batch_size': batch_size, 'minimum_batch_size': minimum_batch_size,
            'training': bool(training), 'optimizer': optimizer if training else None,
            'capture': bool(capture), 'workflow': workflow,
            'assumptions': ['float32 parameters/inputs, BatchNorm counters int64',
                'model constructed on CPU before device transfer',
                'worker minimum includes root inputs and persistent CPU feature matrices; caller-owned build_model inputs are excluded',
                'projected batch may exceed mandatory minimum batch when resume progress is not yet loaded',
                'activation/autograd/workspace and host snapshot projections are conservative estimates',
                'allocator fragmentation, kernel workspace and concurrent processes can change actual peak; allocation remains authoritative']}


def admit_memory(resolved, device, **options):
    """Query the selected runtime device, then reject only impossible minima."""
    from .devices import memory_snapshot
    device = str(device)
    try:
        snapshot = memory_snapshot(device)
        available, host_available = snapshot['available_bytes'], snapshot['host_available_bytes']
        if type(available) is not int or available < 0 or type(host_available) is not int or host_available < 0:
            raise ValueError('invalid free byte counters')
    except Exception as error:
        raise ValueError('Cannot query selected device memory for ' + device + ': ' + str(error)) from error
    estimate = estimate_memory(resolved, device=device, **options)
    if estimate['minimum_device_bytes'] > available:
        raise ValueError('Hardware memory minimum model/input storage ' + str(estimate['minimum_device_bytes']) +
                         ' bytes exceeds available ' + str(available) + ' bytes on ' + device)
    if estimate['minimum_host_bytes'] > host_available:
        raise ValueError('Hardware memory minimum CPU staging ' + str(estimate['minimum_host_bytes']) +
                         ' bytes exceeds available host ' + str(host_available) + ' bytes')
    device_headroom = max(64 * 1024 * 1024, available // 10)
    host_headroom = max(64 * 1024 * 1024, host_available // 10)
    warnings = []
    if estimate['projected_device_bytes'] + device_headroom > available:
        warnings.append('Conservative projected peak plus headroom exceeds available ' + device + ' memory; actual PyTorch allocation may fail')
    if device.startswith('cuda:') and estimate['projected_host_bytes'] + host_headroom > host_available:
        warnings.append('Conservative host staging/checkpoint/measurement projection plus headroom exceeds available CPU memory')
    return {'policy': 'hardware-v5', 'device': device, 'snapshot': snapshot, 'estimate': estimate,
            'headroom_bytes': {'device': device_headroom, 'host': host_headroom},
            'admission': 'minimum_fits_projection_warning' if warnings else 'minimum_and_projection_fit',
            'warnings': warnings, 'scope': 'platform estimate; standalone exported PyTorch follows its allocator'}
