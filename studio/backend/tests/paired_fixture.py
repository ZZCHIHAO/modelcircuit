"""Generated local paired CSV/PNG test fixtures; never scientific data."""
import csv
from pathlib import Path

FIELDS = ['sample_id', 'image_path', 'f', 'constant', 'label', 'patient_id', 'split']


def write_rows(spec, rows):
    with Path(spec['path']).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def paired_fixture(directory):
    from PIL import Image
    root = Path(directory)
    images = root / 'images'; images.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(24):
        identifier = 'sample_' + str(index).zfill(3)
        relative = 'images/' + identifier + '.png'
        Image.new('RGB', (10, 11), (20 + index * 5, index, (index % 2) * 100 + 10)).save(root / relative)
        rows.append({'sample_id': identifier, 'image_path': relative, 'f': index % 3, 'constant': 1,
                     'label': 'ab'[index % 2], 'patient_id': 'patient_' + str(index // 2),
                     'split': 'train' if index < 12 else 'val' if index < 18 else 'test'})
    spec = {'kind': 'paired_csv', 'path': str(root / 'paired.csv'), 'image_root': '', 'image_column': 'image_path', 'id_column': 'sample_id',
            'feature_columns': ['f', 'constant'], 'target_column': 'label', 'group_column': 'patient_id', 'split_column': 'split',
            'standardize': True, 'image_size': [8, 8], 'channels': 3, 'validation_fraction': 0.2, 'test_fraction': 0.2}
    write_rows(spec, rows)
    return spec, rows


def paired_graph(image_port='image', tabular_port='clinical'):
    stages = [('conv', 'Conv2d', {'in_channels': 3, 'out_channels': 3, 'kernel_size': 3, 'padding': 1}),
              ('image_relu', 'ReLU', {}), ('pool', 'AdaptiveAvgPool2d', {'output_size': [1, 1]}), ('flatten', 'Flatten', {}),
              ('clinical_dense', 'Linear', {'in_features': 2, 'out_features': 4}), ('clinical_relu', 'ReLU', {}), ('drop', 'Dropout', {'p': 0.25}),
              ('fusion', 'Concat', {'dim': 1}), ('head', 'Linear', {'in_features': 7, 'out_features': 2})]
    args = [{'x': 'input:' + image_port}, {'x': 'call:conv'}, {'x': 'call:image_relu'}, {'x': 'call:pool'},
            {'x': 'input:' + tabular_port}, {'x': 'call:clinical_dense'}, {'x': 'call:clinical_relu'},
            {'a': 'call:flatten', 'b': 'call:drop'}, {'x': 'call:fusion'}]
    return {'schema_version': 1, 'resource_profile': 'classification-v2', 'id': 'paired_classifier', 'revision': 1,
            'inputs': [{'id': image_port, 'shape': [None, 3, 8, 8], 'dtype': 'float32'}, {'id': tabular_port, 'shape': [None, 2], 'dtype': 'float32'}],
            'instances': [{'id': identifier, 'kind': primitive, 'params': params} for identifier, primitive, params in stages],
            'calls': [{'id': identifier, 'instance_id': identifier, 'inputs': arguments} for (identifier, _, _), arguments in zip(stages, args)],
            'outputs': [{'id': 'logits', 'source': 'call:head'}], 'definitions': []}
