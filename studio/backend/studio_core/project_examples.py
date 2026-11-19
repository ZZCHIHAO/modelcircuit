"""Small synthetic ImageFolder project with a shared layer and recorded splits."""
from pathlib import Path


def create_example_project(destination, *, runtime_source=None):
    from PIL import Image
    from .experiment_spec import default_classification_config
    from .research_project import export_project

    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise ValueError('Select a new example project directory')
    destination.mkdir(parents=True)
    images = destination / 'example-data'
    for split_index, split in enumerate(('train', 'val', 'test')):
        for label, name in enumerate(('dark', 'bright')):
            folder = images / split / name
            folder.mkdir(parents=True)
            for index in range(2):
                base = 15 + split_index * 7 + index * 4 + label * 160
                pixels = bytes(base + ((pixel * (index + 1)) % 23) for pixel in range(64))
                Image.frombytes('L', (8, 8), pixels).save(folder / f'{index + 1}.png')
    graph = example_graph()
    config = default_classification_config('image_folder')
    config.update(epochs=2, batch_size=2, optimizer='adamw', lr=.005)
    config['data'].update(path=str(images), split_mode='directories', channels=1, image_size=[8, 8])
    project = export_project(destination / 'project', graph, config, runtime_source=runtime_source)
    return project, images, None


def example_graph():
    """One weight instance used twice, so example reimport exercises sharing."""
    return {'id': 'portable_classification', 'schema_version': 1, 'revision': 1,
            'resource_profile': 'classification-v2',
            'inputs': [{'id': 'x', 'shape': [None, 1, 8, 8], 'dtype': 'float32', 'position': {'x': 0, 'y': 80}}],
            'definitions': [],
            'instances': [{'id': 'flat', 'kind': 'Flatten', 'params': {'start_dim': 1}},
                          {'id': 'shared', 'kind': 'Linear', 'params': {'in_features': 64, 'out_features': 8}},
                          {'id': 'sum', 'kind': 'Add', 'params': {}},
                          {'id': 'classes', 'kind': 'Linear', 'params': {'in_features': 8, 'out_features': 2}}],
            'calls': [{'id': 'flat', 'instance_id': 'flat', 'inputs': {'x': 'input:x'}, 'position': {'x': 220, 'y': 130}},
                      {'id': 'left', 'instance_id': 'shared', 'inputs': {'x': 'call:flat'}, 'position': {'x': 440, 'y': 60}},
                      {'id': 'right', 'instance_id': 'shared', 'inputs': {'x': 'call:flat'}, 'position': {'x': 440, 'y': 200}},
                      {'id': 'sum', 'instance_id': 'sum', 'inputs': {'a': 'call:left', 'b': 'call:right'}, 'position': {'x': 660, 'y': 130}},
                      {'id': 'classes', 'instance_id': 'classes', 'inputs': {'x': 'call:sum'}, 'position': {'x': 880, 'y': 130}}],
            'outputs': [{'id': 'logits', 'source': 'call:classes', 'position': {'x': 1100, 'y': 130}}]}
