"""Unlabeled CSV fitting with editable PCA and KMeans graph nodes."""
import csv
import hashlib
import io
import math
from pathlib import Path

from .classification_data import MAX_CSV_BYTES, MAX_FEATURES, MAX_FLOAT32, MAX_SAMPLES, _cancel, _json_digest, _read_source
from .operator_rules import normalize_operator
from .ports import parse_source
from .schema import prepare_graph

PROFILE = 'unsupervised-v1'
DATA_KIND = 'unsupervised_csv'
ALGORITHMS = {'PCA': ('PCA',), 'KMeans': ('KMeans',), 'PCA+KMeans': ('PCA', 'KMeans')}
STAGE_IDS = {'PCA': 'pca', 'KMeans': 'kmeans'}
OUTPUT_IDS = {'PCA': 'projection', 'KMeans': 'clusters'}


def validate_unlabeled_spec(data, *, editable=False):
    defaults = {'kind': DATA_KIND, 'path': '', 'feature_columns': [], 'standardize': True}
    if not isinstance(data, dict) or not data.keys() <= defaults.keys():
        raise ValueError('无标签 CSV 只需选择数值列和是否标准化')
    spec = {**defaults, **data}
    path, columns = spec['path'], spec['feature_columns']
    if (spec['kind'] != DATA_KIND or not isinstance(path, str) or len(path) > 4096
            or '\0' in path or not (editable or path.strip())):
        raise ValueError('请选择无标签 CSV 文件')
    if (not isinstance(columns, list) or len(columns) > MAX_FEATURES
            or not all(isinstance(name, str) and name for name in columns) or len(set(columns)) != len(columns)
            or type(spec['standardize']) is not bool):
        raise ValueError('请选择不同的数值列并设置是否标准化')
    return spec


def validate_unsupervised_config(config, *, editable=False):
    defaults = {'profile': PROFILE, 'data': {}, 'input_port': 'x', 'device': 'cpu', 'seed': 42,
                'max_runtime_seconds': 3600, 'expected_dataset_digest': None, 'resume_checkpoint': None}
    if not isinstance(config, dict) or not config.keys() <= defaults.keys():
        raise ValueError('无监督实验设置无效')
    result = {**defaults, **config}
    seed, seconds, digest = result['seed'], result['max_runtime_seconds'], result['expected_dataset_digest']
    if (result['profile'] != PROFILE or result['device'] != 'cpu' or result['input_port'] != 'x'
            or type(seed) is not int or not 0 <= seed < 2**31
            or type(seconds) is not int or not 10 <= seconds <= 604800
            or result['resume_checkpoint'] is not None
            or digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                      or any(c not in '0123456789abcdef' for c in digest))):
        raise ValueError('请选择 CPU、有效的随机种子和运行时间')
    result['data'] = validate_unlabeled_spec(result['data'], editable=editable)
    return result


def inspect_unlabeled_csv(data, seed=42, cancelled=None):
    spec = validate_unlabeled_spec(data)
    path = Path(spec['path']).expanduser().resolve()
    raw = _read_source(path, MAX_CSV_BYTES, cancelled)
    reader = csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True)
    records = []
    try:
        header = next(reader, [])
        if not header or not all(header) or len(set(header)) != len(header):
            raise ValueError('CSV 表头须为不同的非空列名')
        columns = spec['feature_columns'] or header
        if len(columns) > MAX_FEATURES or not set(columns) <= set(header):
            raise ValueError('请选择 CSV 中存在的数值列')
        indices = [header.index(name) for name in columns]
        for row in reader:
            _cancel(cancelled)
            if len(row) != len(header) or len(records) >= MAX_SAMPLES:
                raise ValueError(f'CSV 第 {reader.line_num} 行列数错误或样本数超出范围')
            try:
                features = [float(row[index]) for index in indices]
                if any(not math.isfinite(value) or abs(value) > MAX_FLOAT32 for value in features):
                    raise ValueError
            except ValueError:
                raise ValueError(f'CSV 第 {reader.line_num} 行存在空值或非有限的数值特征') from None
            records.append({'id': f'row:{reader.line_num}', 'features': features, 'content_digest': _json_digest(features)})
    except csv.Error as error:
        raise ValueError(f'CSV 第 {reader.line_num} 行格式错误') from error
    if len(records) < 2:
        raise ValueError('无监督实验至少需要两行数据')
    manifest = {'version': 1, 'kind': DATA_KIND, 'seed': seed, 'source_digest': hashlib.sha256(raw).hexdigest(),
                'data_spec': {**spec, 'path': str(path), 'feature_columns': list(columns)},
                'input_shape': [None, len(columns)], 'counts': {'fit': len(records)}, 'records': records,
                'preprocessing': {'kind': 'standardize' if spec['standardize'] else 'identity', 'fit_split': 'all'}}
    manifest['dataset_digest'] = _json_digest(manifest)
    return manifest


def unsupervised_graph(manifest, algorithm='PCA+KMeans'):
    if manifest.get('kind') != DATA_KIND or algorithm not in ALGORITHMS:
        raise ValueError('请选择已检查的无标签 CSV 和 PCA 或 K-means')
    instances, calls, outputs = [], [], []
    source = 'input:x'
    for index, kind in enumerate(ALGORITHMS[algorithm]):
        node = STAGE_IDS[kind]
        params = {'n_components': min(2, manifest['input_shape'][1], manifest['counts']['fit'])} if kind == 'PCA' else {}
        instances.append({'id': node, 'kind': kind, 'params': normalize_operator(kind, params)})
        calls.append({'id': node, 'instance_id': node, 'inputs': {'x': source},
                      'position': {'x': 210 + index * 210, 'y': 120}})
        source = f'call:{node}:out'
        outputs.append({'id': OUTPUT_IDS[kind], 'source': source, 'position': {'x': 660, 'y': 100 + index * 130}})
    return {'schema_version': 1, 'id': 'unsupervised_table', 'revision': 1, 'name': algorithm,
            'inputs': [{'id': 'x', 'dtype': 'float32', 'shape': manifest['input_shape'], 'position': {'x': 20, 'y': 120}}],
            'instances': instances, 'calls': calls, 'outputs': outputs, 'definitions': []}


def unsupervised_contract(graph, manifest=None):
    """Return the ordered estimator stages, verifying real canvas wires."""
    resolved, _, _ = prepare_graph(graph)
    inputs, instances, calls = (resolved[name] for name in ('inputs', 'instances', 'calls'))
    if (resolved['schema_version'] != 1 or len(inputs) != 1 or inputs[0]['id'] != 'x'
            or inputs[0]['dtype'] != 'float32' or len(inputs[0]['shape']) != 2
            or resolved.get('definitions') or not 1 <= len(instances) == len(calls) <= 2):
        raise ValueError('无监督画布需要一个 x 表格输入和 PCA、K-means 或 PCA → K-means')
    if manifest is not None and inputs[0]['shape'] != manifest['input_shape']:
        raise ValueError('画布输入特征数与所选数值列不一致')
    nodes = {instance['id']: instance for instance in instances}
    stages = {}
    for call in calls:
        instance = nodes[call['instance_id']]
        kind = instance['kind']
        if kind not in STAGE_IDS or kind in stages or instance.get('frozen'):
            raise ValueError('PCA 和 K-means 在无监督画布中各可拟合一次')
        stages[kind] = (instance, call)
    previous = ('input', 'x')
    expected = {}
    sequence = []
    for kind in STAGE_IDS:
        if kind not in stages:
            continue
        instance, call = stages[kind]
        if parse_source(call['inputs']['x'])[:2] != previous:
            raise ValueError('请将数据连到第一组件，再将 PCA 输出连到 K-means')
        previous = ('call', call['id'])
        expected[OUTPUT_IDS[kind]] = previous
        sequence.append((instance['id'], kind, normalize_operator(kind, instance['params'])))
    actual = {output['id']: parse_source(output['source'])[:2] for output in resolved['outputs']}
    if not actual or not actual.keys() <= expected.keys() or any(expected[key] != source for key, source in actual.items()):
        raise ValueError('请把 PCA 接到 projection 输出，或把 K-means 接到 clusters 输出')
    return sequence


def run_unsupervised(graph, config, artifact_directory, progress=None, cancelled=None):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    from .classification_runtime import _atomic_write, _write_json
    from .experiment_spec import model_digest

    config = validate_unsupervised_config(config)
    emit = progress or (lambda event: None)
    emit({'phase': 'inspect', 'step': 0, 'total': 2})
    manifest = inspect_unlabeled_csv(config['data'], config['seed'], cancelled)
    if config['expected_dataset_digest'] not in (None, manifest['dataset_digest']):
        raise ValueError('数据在核对后已改变，请重新检查')
    stages = unsupervised_contract(graph, manifest)
    features = np.asarray([row['features'] for row in manifest['records']], dtype=np.float64)
    total = len(features)
    analysis = {'cluster_counts': [], 'explained_variance_ratio': [], 'rows': total}
    scores = {'count': total}
    labels = None
    axes = manifest['data_spec']['feature_columns']
    emit({'phase': 'fit', 'step': 1, 'total': 2})
    with threadpool_limits(limits=1):
        if manifest['data_spec']['standardize']:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            analysis['standardization'] = {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}
        for _, kind, params in stages:
            _cancel(cancelled)
            if kind == 'PCA':
                if params['n_components'] > min(features.shape) or not np.any(features != features[0]):
                    raise ValueError('PCA 需要有变化的数据，主成分数不能超过样本数或特征数')
                pca = PCA(**params, svd_solver='full').fit(features)
                features = pca.transform(features)
                analysis['explained_variance_ratio'] = pca.explained_variance_ratio_.tolist()
                scores['explained_variance'] = float(pca.explained_variance_ratio_.sum())
                axes = [f'PC{i + 1}' for i in range(features.shape[1])]
            else:
                if params['n_clusters'] > len(np.unique(features, axis=0)):
                    raise ValueError('簇数不能超过不同样本的数量')
                fitted = KMeans(**params, random_state=config['seed']).fit(features)
                labels = fitted.labels_.tolist()
                scores['inertia'] = float(fitted.inertia_)
                analysis['cluster_counts'] = np.bincount(fitted.labels_, minlength=params['n_clusters']).tolist()
                analysis['clustering_space_digest'] = hashlib.sha256(features.astype('<f8', copy=False).tobytes()).hexdigest()
    _cancel(cancelled)
    preview_indices = np.linspace(0, total - 1, min(total, 2048), dtype=int).tolist()
    analysis.update(axes=[axes[0], axes[1] if len(axes) > 1 else '0'],
                    points=[[float(features[i, 0]), float(features[i, 1]) if len(axes) > 1 else 0.0,
                             labels[i] if labels is not None else None] for i in preview_indices])
    directory = Path(artifact_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / filename for name, filename in
             (('manifest', 'manifest.json'), ('run_spec', 'run_spec.json'), ('metrics', 'metrics.json'),
              ('projection', 'projection.csv'))}
    metrics = {'fit': scores}
    runtime = {'profile': PROFILE, 'dataset_digest': manifest['dataset_digest'], 'model_digest': model_digest(graph),
               'graph': graph, 'seed': config['seed'], 'settings': config}
    _write_json(paths['manifest'], {**manifest, 'records': [
        {key: row[key] for key in ('id', 'content_digest')} for row in manifest['records']]})
    _write_json(paths['run_spec'], runtime)
    _write_json(paths['metrics'], {'metrics': metrics, 'analysis': analysis})
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['sample_id', *axes, *(['cluster'] if labels is not None else [])])
    for i, row in enumerate(manifest['records']):
        _cancel(cancelled)
        writer.writerow([row['id'], *features[i].tolist(), *([labels[i]] if labels is not None else [])])
    _atomic_write(paths['projection'], lambda stream: stream.write(output.getvalue().encode('utf-8')))
    emit({'phase': 'complete', 'step': 2, 'total': 2})
    return {'status': 'succeeded', 'profile': PROFILE, 'device': 'cpu', 'execution': 'unsupervised',
            'algorithm': '+'.join(kind for _, kind, _ in stages), 'history': [], 'metrics': metrics,
            'analysis': analysis, 'artifacts': {name: str(path) for name, path in paths.items()}}
