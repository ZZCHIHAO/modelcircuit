"""Build one real root call around a JSON component bundle for tests."""
from copy import deepcopy


def bundle_graph(bundle, input_shapes, policy):
    entry = deepcopy(bundle.entry)
    entry['id'] = 'entry'
    definition = next(item for item in bundle.definitions if item['id'] == entry['definition_id'])
    ports = definition['graph']['inputs']
    if set(input_shapes) != {port['id'] for port in ports}:
        raise ValueError('input_shapes must name every definition input exactly')
    return {
        'schema_version': 1, 'id': 'component_fixture', 'revision': 1, **deepcopy(policy),
        'inputs': [{**deepcopy(port), 'shape': deepcopy(input_shapes[port['id']])} for port in ports],
        'instances': [entry],
        'calls': [{'id': 'entry', 'instance_id': 'entry',
                   'inputs': {port['id']: 'input:' + port['id'] for port in ports}}],
        'outputs': [{'id': port['id'], 'source': 'call:entry:' + port['id']}
                    for port in definition['graph']['outputs']],
        'definitions': deepcopy(bundle.definitions),
    }


def bundle_model(bundle, input_shapes, policy):
    from studio_core.runtime import build_model
    return build_model(bundle_graph(bundle, input_shapes, policy))
