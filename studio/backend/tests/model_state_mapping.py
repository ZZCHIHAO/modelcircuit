"""Explicit semantic state mapping, never paired by traversal order."""
from collections import OrderedDict


def _child(module, *path):
    for name in path:
        module = module.components[name]
    return module


def canonical_state_paths(model, model_id):
    """Map torchvision semantic keys to complete graph/standalone state keys.

    Coverage checks reject omissions, duplicate target modules, and accidental
    parameter sharing. Both buffers and parameters participate in strict replay.
    """
    root = _child(model, 'entry')
    modules = {}
    variant = model_id.split('.')[-1].removesuffix('_classifier')
    if variant in ('resnet18', 'resnet34', 'resnet50'):
        counts = (2, 2, 2, 2) if variant == 'resnet18' else (3, 4, 6, 3)
        modules.update(conv1=_child(root, 'stem', 'conv'), bn1=_child(root, 'stem', 'bn'))
        for number in range(1, 5):
            for index in range(counts[number - 1]):
                block = _child(root, 'stage' + str(number), 'block' + str(index))
                prefix = 'layer' + str(number) + '.' + str(index) + '.'
                for name in (('conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3') if variant == 'resnet50' else ('conv1', 'bn1', 'conv2', 'bn2')):
                    modules[prefix + name] = _child(block, name)
                if index == 0 and (number > 1 or variant == 'resnet50'):
                    modules[prefix + 'downsample.0'] = _child(block, 'projection', 'conv')
                    modules[prefix + 'downsample.1'] = _child(block, 'projection', 'bn')
        if 'fc' in root.components:
            modules['fc'] = _child(root, 'fc')
    elif variant.startswith('vgg'):
        # The reference's numeric feature indices are pinned independently to
        # torchvision; these explicit counts do not read the generated graph.
        counts = {'vgg11': (1, 1, 2, 2, 2), 'vgg13': (2, 2, 2, 2, 2), 'vgg16': (2, 2, 3, 3, 3),
                  'vgg19': (2, 2, 4, 4, 4)}[variant]
        index = 0
        for stage, count in enumerate(counts, 1):
            for conv in range(count):
                modules['features.' + str(index)] = _child(root, 'stage' + str(stage), 'conv' + str(conv))
                index += 2
            index += 1
    elif variant.startswith('squeezenet'):
        features = _child(root, 'feature_layers')
        modules['features.0'] = _child(features, 'conv0')
        indices = (3, 4, 5, 7, 8, 9, 10, 12) if variant == 'squeezenet1_0' else (3, 4, 6, 7, 9, 10, 11, 12)
        for index in indices:
            for name in ('squeeze', 'expand1x1', 'expand3x3'):
                modules['features.' + str(index) + '.' + name] = _child(features, 'fire' + str(index), name)
        if 'classifier' in root.components:
            modules['classifier.1'] = _child(root, 'classifier', 'conv')
    else:
        raise AssertionError('Missing independent mapping for ' + model_id)
    assert len({id(module) for module in modules.values()}) == len(modules), 'Duplicate stateful target module'
    module_paths = {id(module): path for path, module in model.named_modules()}
    state = model.state_dict()
    result = {}
    for canonical, module in modules.items():
        for name in module.state_dict():
            result[canonical + '.' + name] = module_paths[id(module)] + '.' + name
    assert len(set(result.values())) == len(result), 'Duplicate state target'
    assert set(result.values()) == set(state), 'State coverage mismatch: ' + str(set(state) ^ set(result.values()))
    return result


def canonical_state_dict(model, model_id):
    state = model.state_dict()
    return OrderedDict((name, state[path]) for name, path in canonical_state_paths(model, model_id).items())


def copy_platform_state(reference, model, model_id):
    state = canonical_state_dict(model, model_id)
    assert set(state) == set(reference.state_dict()), 'Reference state key coverage mismatch'
    reference.load_state_dict(state, strict=True)


def canonical_parameters(model, model_id):
    parameters = dict(model.named_parameters())
    return {name: parameters[path] for name, path in canonical_state_paths(model, model_id).items() if path in parameters}
