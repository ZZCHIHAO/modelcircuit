"""The CPU execution path for a single fitted scikit-learn estimator."""
import csv
import io
import math
import platform
from pathlib import Path
import time

from .classification_data import inspect_dataset
from .classification_runtime import _atomic_write, _write_json
from .experiment_spec import model_digest, validate_supervised_config
from .operator_rules import normalize_operator
from .operators import CLASSICAL_KINDS, FOREST_KINDS
from .ports import parse_source

ESTIMATOR_TASKS = {
    'SVC': ('classification', 'classification-v2', ('csv', 'table_csv')),
    'SVR': ('regression', 'regression-v1', ('regression_csv', 'table_csv')),
    'RandomForestClassifier': ('classification', 'classification-v2', ('table_csv',)),
    'RandomForestRegressor': ('regression', 'regression-v1', ('table_csv',)),
}


def estimator_graph(manifest, estimator=None):
    """Create a canvas graph for a checked table or legacy numeric CSV."""
    kind = manifest['kind']
    task = manifest['task'] if kind == 'table_csv' else 'regression' if kind == 'regression_csv' else 'classification'
    estimator = estimator or ('RandomForestRegressor' if task == 'regression' else 'RandomForestClassifier')
    supported = ESTIMATOR_TASKS.get(estimator)
    if supported is None or supported[0] != task or kind not in supported[2]:
        raise ValueError('请选择与数据和任务匹配的估计器。')
    name = '随机森林' if estimator.startswith('RandomForest') else '支持向量机'
    return {'schema_version': 1, 'id': estimator.lower(), 'name': f'{estimator} · {name}', 'revision': 1,
            'inputs': [{'id': 'x', 'shape': list(manifest['input_shape']), 'dtype': 'float32', 'position': {'x': 20, 'y': 140}}],
            'instances': [{'id': 'estimator', 'kind': estimator, 'params': normalize_operator(estimator, {})}],
            'calls': [{'id': 'fit_model', 'instance_id': 'estimator', 'inputs': {'x': 'input:x'}, 'position': {'x': 260, 'y': 140}}],
            'outputs': [{'id': 'prediction', 'source': 'call:fit_model', 'position': {'x': 540, 'y': 140}}], 'definitions': []}


def contains_classical(graph):
    """Recognize classical estimators even in graphs not eligible for fit."""
    pending, visited = [graph], set()
    while pending:
        scope = pending.pop()
        if not isinstance(scope, dict) or id(scope) in visited:
            continue
        visited.add(id(scope))
        instances = scope.get('instances')
        if isinstance(instances, list):
            for item in instances:
                kind = item.get('kind') if isinstance(item, dict) else None
                if isinstance(kind, str) and kind in CLASSICAL_KINDS:
                    return True
        definitions = scope.get('definitions')
        if isinstance(definitions, list):
            pending.extend(item.get('graph') for item in definitions if isinstance(item, dict))
    return False


def classical_graph_contract(graph):
    """Identify an executable one-estimator graph without importing a runtime."""
    message = 'Classical estimators require one root input, one estimator call and one prediction output in a schema v1 graph'
    if not isinstance(graph, dict) or type(graph.get('schema_version')) is not int or graph['schema_version'] != 1:
        raise ValueError(message)
    collections = [graph.get(key) for key in ('inputs', 'instances', 'calls', 'outputs')]
    if any(not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict) for items in collections) or graph.get('definitions'):
        raise ValueError(message)
    input_port, instance, call, output = (items[0] for items in collections)
    kind = instance.get('kind')
    if not isinstance(kind, str) or kind not in ESTIMATOR_TASKS or call.get('instance_id') != instance.get('id'):
        raise ValueError(message)
    input_id, call_id, instance_id = input_port.get('id'), call.get('id'), instance.get('id')
    if not all(isinstance(value, str) and value for value in (input_id, call_id, instance_id)):
        raise ValueError(message)
    shape = input_port.get('shape')
    if (not isinstance(shape, list) or len(shape) != 2 or shape[0] is not None or type(shape[1]) is not int
            or shape[1] < 1 or input_port.get('dtype') != 'float32'):
        raise ValueError('Classical estimators require a float32 root input with shape [batch, feature_count]')
    if call.get('inputs') != {'x': 'input:' + input_id} or output.get('id') != 'prediction':
        raise ValueError(message)
    source_kind, source_id, source_port = parse_source(output.get('source'))
    if source_kind != 'call' or source_id != call_id or source_port not in (None, 'out'):
        raise ValueError(message)
    params = normalize_operator(kind, instance.get('params', {}))
    return {'kind': kind, 'input_id': input_id, 'input_shape': list(shape), 'call_id': call_id,
            'instance_id': instance_id, 'params': params, 'output_source': output['source']}


def classical_contract(graph, config, manifest=None):
    """Check the fit contract against experiment settings and optional inspected data."""
    contract = classical_graph_contract(graph)
    kind = contract['kind']
    task, profile, accepted_data = ESTIMATOR_TASKS[kind]
    data = config.get('data')
    data_kind = data.get('kind') if isinstance(data, dict) else None
    if (config.get('profile') != profile or data_kind not in accepted_data
            or (data_kind == 'table_csv' and data.get('task', task) != task)):
        raise ValueError(f'{kind} requires {" or ".join(accepted_data)} data and the {profile} experiment profile')
    if config.get('device', 'cpu') != 'cpu':
        raise ValueError(f'{kind} requires CPU execution')
    if config.get('resume_checkpoint') is not None:
        raise ValueError(f'{kind} uses fit; Torch checkpoint resume is unavailable')
    if 'attention_probe_requests' in config:
        raise ValueError(f'{kind} cannot collect attention probes')
    if config.get('input_port', 'x') != contract['input_id'] or config.get('input_ports') is not None:
        raise ValueError(f'{kind} requires a single input matching the experiment input port')
    if manifest is not None and (manifest['kind'] != data_kind or manifest['input_shape'] != contract['input_shape']):
        raise ValueError(f'{kind} requires input shape {manifest["input_shape"]} for this dataset')
    return contract


def _scores(classification, expected, predicted):
    count = len(expected)
    if classification:
        return {'count': count, 'accuracy': sum(int(a) == int(b) for a, b in zip(expected, predicted)) / count}
    errors = [float(p) - float(t) for t, p in zip(expected, predicted)]
    squared = math.fsum(error * error for error in errors)
    mean = math.fsum(expected) / count
    total = math.fsum((target - mean) ** 2 for target in expected)
    return {'count': count, 'mae': math.fsum(map(abs, errors)) / count, 'rmse': math.sqrt(squared / count),
            'r2': 1 - squared / total if count > 1 and total > 0 and any(t != expected[0] for t in expected) else None}


def _evaluate_csv(manifest, estimator, score, predict, check, emit):
    import numpy as np

    splits = {split: [] for split in ('train', 'val', 'test')}
    for record in manifest['records']:
        check()
        splits[record['split']].append(record)
    mean, std = (np.asarray(manifest['preprocessing'][key], dtype=np.float64) for key in ('mean', 'std'))

    def data(split):
        records = splits[split]
        features = np.asarray((np.asarray([record['features'] for record in records], dtype=np.float64) - mean) / std,
                              dtype=np.float32)
        return features, [record['label'] for record in records]

    emit('fit', 1)
    train_x, train_y = data('train')
    # libsvm fit is a single blocking call. The RunManager also enforces the
    # cancellation grace period and deadline outside the worker process.
    estimator.fit(train_x, train_y)
    check()
    metrics = {}
    for step, split in enumerate(splits, 2):
        emit(split, step)
        features, targets = (train_x, train_y) if split == 'train' else data(split)
        predictions = predict(estimator, features)
        metrics[split] = score(targets, predictions)
    return metrics, splits['test'], predictions


def run_classical(graph, config, artifact_directory, progress=None, cancelled=None):
    """Evaluate one classical estimator against a frozen holdout."""
    config = validate_supervised_config(config)
    contract = classical_contract(graph, config)
    kind = contract['kind']
    classification = ESTIMATOR_TASKS[kind][0] == 'classification'
    table = config['data']['kind'] == 'table_csv'
    total_steps = config['data']['folds'] + 4 if table else 5
    deadline = time.monotonic() + config['max_runtime_seconds']

    def check():
        if cancelled is not None and cancelled():
            raise InterruptedError(f'{kind} experiment cancelled')
        if time.monotonic() >= deadline:
            raise TimeoutError(f'{kind} experiment exceeded max_runtime_seconds')
        return False

    def emit(phase, step):
        if progress is not None:
            progress({'phase': phase, 'step': step, 'total': total_steps})
        check()

    def predict(model, features):
        predictions = model.predict(features).tolist()
        check()
        if not all(math.isfinite(value) for value in predictions):
            raise ValueError(f'{kind} emitted non-finite predictions')
        return predictions

    def score(targets, predictions):
        return _scores(classification, targets, predictions)

    check()
    emit('inspect', 0)
    manifest = inspect_dataset(config['data'], seed=config['seed'], cancelled=check)
    expected_digest = config['expected_dataset_digest']
    if expected_digest is not None and expected_digest != manifest['dataset_digest']:
        raise ValueError('Dataset digest changed since inspection')
    classical_contract(graph, config, manifest)
    try:
        import numpy as np
        import sklearn
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.svm import SVC, SVR
    except ImportError as exc:
        raise ValueError(f'The scikit-learn runtime is unavailable for {kind}. Install the desktop dependencies.') from exc
    constructor = {'SVC': SVC, 'SVR': SVR, 'RandomForestClassifier': RandomForestClassifier,
                   'RandomForestRegressor': RandomForestRegressor}[kind]
    params = dict(contract['params'])
    if kind in FOREST_KINDS:
        params['random_state'] = config['seed']
    estimator = constructor(**params)
    if table:
        from .table_cv import evaluate_table
        metrics, test_records, predictions, fold_report = evaluate_table(manifest, estimator, score, predict, check, emit)
    else:
        metrics, test_records, predictions = _evaluate_csv(manifest, estimator, score, predict, check, emit)

    rows = []
    labels = manifest.get('class_names', [])
    for record, prediction in zip(test_records, predictions):
        check()
        target = record['label']
        row = {'sample_id': record['id']}
        if table:
            row['group'] = record['group']
        if classification:
            row.update(label=int(target), label_name=labels[int(target)],
                       prediction=int(prediction), prediction_name=labels[int(prediction)])
        else:
            row.update(target=float(target), prediction=float(prediction), residual=float(prediction) - float(target))
        rows.append(row)

    directory = Path(artifact_directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {key: str(directory / filename) for key, filename in (
        ('manifest', 'manifest.json'), ('run_spec', 'run_spec.json'), ('metrics', 'metrics.json'), ('predictions', 'predictions.csv'))}
    dataset = {'dataset_digest': manifest['dataset_digest'], 'counts': manifest['counts'],
               **({'class_names': labels} if classification else {'target_column': manifest['target_column']})}
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'scikit_learn': sklearn.__version__, 'numpy': np.__version__}
    spec = {'version': 1, 'profile': config['profile'], 'execution': 'classical', 'algorithm': kind,
            'model_digest': model_digest(graph), 'graph': graph, 'config': config, 'effective_estimator_params': estimator.get_params(),
            'dataset_digest': manifest['dataset_digest'], 'device': 'cpu', 'environment': environment}
    _write_json(artifacts['manifest'], manifest)
    _write_json(artifacts['run_spec'], spec)
    _write_json(artifacts['metrics'], {'algorithm': kind, 'metrics': metrics})
    if table:
        artifacts['cv_folds'] = str(directory / 'cv_folds.json')
        _write_json(artifacts['cv_folds'], fold_report)
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(Path(artifacts['predictions']), lambda stream: stream.write(text.getvalue().encode('utf-8')))
    emit('complete', total_steps)
    return {'status': 'succeeded', 'execution': 'classical', 'algorithm': kind, 'device': 'cpu', 'profile': config['profile'],
            'history': [], 'metrics': metrics, 'dataset': dataset, 'artifacts': artifacts,
            **({'support_vectors': len(estimator.support_)} if hasattr(estimator, 'support_') else {})}
