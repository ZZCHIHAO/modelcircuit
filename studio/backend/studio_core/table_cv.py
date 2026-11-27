"""Fit and evaluate classical pipelines on the frozen table CSV fold plan."""
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _pipeline(spec, estimator):
    numeric_count = len(spec['numeric_columns'])
    columns = []
    if numeric_count:
        steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
        if spec['standardize']:
            steps.append(('scale', StandardScaler()))
        columns.append(('numeric', Pipeline(steps), list(range(numeric_count))))
    if spec['categorical_columns']:
        encoder = OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
        category_indices = list(range(numeric_count, numeric_count + len(spec['categorical_columns'])))
        columns.append(('categorical', encoder, category_indices))
    transformer = ColumnTransformer(columns, sparse_threshold=1.0)
    return Pipeline([('preprocess', transformer), ('estimator', estimator)])


def _fitted_preprocessing(pipeline, spec):
    transformers = pipeline.named_steps['preprocess'].named_transformers_
    summary = {'numeric_medians': {}, 'numeric_center': {}, 'numeric_scale': {}, 'categorical_vocabulary': {}}
    if spec['numeric_columns']:
        steps = transformers['numeric'].named_steps
        names = spec['numeric_columns']
        summary['numeric_medians'] = dict(zip(names, steps['impute'].statistics_.tolist()))
        if 'scale' in steps:
            summary['numeric_center'] = dict(zip(names, steps['scale'].mean_.tolist()))
            summary['numeric_scale'] = dict(zip(names, steps['scale'].scale_.tolist()))
    if spec['categorical_columns']:
        categories = (values.tolist() for values in transformers['categorical'].categories_)
        summary['categorical_vocabulary'] = dict(zip(spec['categorical_columns'], categories))
    return summary


def evaluate_table(manifest, estimator, score, predict, check, emit):
    """Fit fresh transformers per fold, then refit on all development cases."""
    spec = manifest['data_spec']
    records = manifest['records']
    indices = {record['id']: index for index, record in enumerate(records)}
    features = np.asarray([row['numeric'] + row['categorical'] for row in records], dtype=object)
    targets = np.asarray([row['label'] for row in records])
    development = [index for index, record in enumerate(records) if record['split'] != 'test']
    test = [index for index, record in enumerate(records) if record['split'] == 'test']
    fold_results = []
    oof_targets, oof_predictions = [], []
    for fold in manifest['fold_plan']:
        emit('cv_fold', fold['fold'])
        train_indices = [indices[case_id] for case_id in fold['train_ids']]
        val_indices = [indices[case_id] for case_id in fold['val_ids']]
        fold_pipeline = _pipeline(spec, clone(estimator))
        try:
            fold_pipeline.fit(features[train_indices], targets[train_indices])
        except ValueError as exc:
            raise ValueError(f"Table CV fold {fold['fold']} fit failed: {exc}") from exc
        check()
        actual = targets[val_indices].tolist()
        predicted = predict(fold_pipeline, features[val_indices])
        oof_targets.extend(actual)
        oof_predictions.extend(predicted)
        fold_results.append({**fold, 'metrics': score(actual, predicted),
                             'preprocessing': _fitted_preprocessing(fold_pipeline, spec)})
    metrics = {'cv': score(oof_targets, oof_predictions)}

    step = spec['folds'] + 1
    emit('fit', step)
    final_pipeline = _pipeline(spec, estimator).fit(features[development], targets[development])
    check()
    emit('dev', step + 1)
    dev_predictions = predict(final_pipeline, features[development])
    metrics['dev'] = score(targets[development].tolist(), dev_predictions)
    emit('test', step + 2)
    predictions = predict(final_pipeline, features[test])
    metrics['test'] = score(targets[test].tolist(), predictions)
    report = {'version': 1, 'dataset_digest': manifest['dataset_digest'], 'folds': fold_results,
              'final_preprocessing': _fitted_preprocessing(final_pipeline, spec)}
    if manifest['task'] == 'classification':
        report.update(split_policy=manifest['split_policy'], warnings=manifest['warnings'])
    return metrics, [records[index] for index in test], predictions, report
