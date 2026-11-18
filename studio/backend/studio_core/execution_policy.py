"""Root-only batch guards, shared by probes and classification entry points."""
from .schema import graph_resource_limits
from functools import wraps


def check_config_batch(graph, batch_size):
    limits = graph_resource_limits(graph)
    if limits.enforce_batch_bound and (type(batch_size) is not int or not 1 <= batch_size <= limits.batch_bound):
        raise ValueError("batch_size must be between 1 and resource_batch_bound (" + str(limits.batch_bound) + ")")
    return limits


def check_input_batches(inputs, limits):
    if limits.enforce_batch_bound:
        for name, tensor in inputs.items():
            if getattr(tensor, "ndim", 0) < 1 or not 1 <= tensor.shape[0] <= limits.batch_bound:
                raise ValueError("Input " + name + " batch must be between 1 and resource_batch_bound (" + str(limits.batch_bound) + ")")


def hardware_memory_errors(function):
    """Turn allocator failures into worker-readable diagnostics for v5 runs."""
    @wraps(function)
    def guarded(graph, *args, **kwargs):
        try:
            return function(graph, *args, **kwargs)
        except Exception as error:
            if isinstance(graph, dict) and graph.get('resource_profile') == 'hardware-v5':
                import torch
                if isinstance(error, (MemoryError, torch.OutOfMemoryError)):
                    raise ValueError('Hardware memory allocation failed during execution; available memory may have changed or the allocator/workspace exceeded the estimate: ' + str(error)) from error
            raise
    return guarded
