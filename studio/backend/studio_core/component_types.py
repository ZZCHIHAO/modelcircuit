"""JSON-only component bundle shared by factories and the component catalog."""
from dataclasses import dataclass


TENSOR_COMPONENT = {'schema_version': 2, 'capability_profile': 'tensor-v1',
                    'operator_set': 'tensor-ops-v1', 'resource_profile': 'hardware-v5'}


def tensor_requirements(batch_bound):
    if type(batch_bound) is not int or not 1 <= batch_bound <= 64:
        raise ValueError('Tensor component resource_batch_bound must be an integer in 1..64')
    return {**TENSOR_COMPONENT, 'resource_batch_bound': batch_bound}


def validate_tensor_requirements(requirements):
    if type(requirements) is not dict or type(requirements.get('schema_version')) is not int:
        raise ValueError('Invalid tensor component requirements')
    expected = tensor_requirements(requirements.get('resource_batch_bound'))
    if requirements != expected:
        raise ValueError('Tensor component requires schema 2, tensor-v1, tensor-ops-v1 and hardware-v5')
    return expected


@dataclass
class ComponentBundle:
    entry: dict
    definitions: list[dict]
    requirements: dict
