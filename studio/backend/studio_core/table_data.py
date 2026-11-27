"""Inspect bounded table CSV data and freeze patient-isolated partitions."""
import csv
import hashlib
import io
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from .classification_data import (
    MAX_CLASSES, MAX_CSV_BYTES, MAX_FEATURES, MAX_FLOAT32, MAX_SAMPLES, SPLITS,
    _cancel, _json_digest, _read_source,
)


_STRATIFICATION_ATTEMPTS = 24


def table_defaults(task='classification'):
    if task not in ('classification', 'regression'):
        raise ValueError('table_csv task must be classification or regression')
    return {'kind': 'table_csv', 'task': task, 'path': '', 'id_column': 'sample_id',
            'group_column': 'patient_id', 'split_column': None, 'target_column': 'label',
            'numeric_columns': [], 'categorical_columns': [], 'standardize': True,
            'folds': 5, 'test_fraction': 0.2}


def validate_table_spec(data, *, editable=False):
    if not isinstance(data, dict) or data.get('kind') != 'table_csv':
        raise ValueError('Expected table_csv data')
    defaults = table_defaults(data.get('task', 'classification'))
    if not data.keys() <= defaults.keys():
        raise ValueError('Unsupported table_csv data option')
    spec = {**defaults, **data}
    if (not isinstance(spec['path'], str) or len(spec['path']) > 4096
            or (not editable and not spec['path'].strip())):
        raise ValueError('Select a table CSV path of up to 4096 characters')

    def column(value):
        return isinstance(value, str) and 0 < len(value) <= 256 and bool(value.strip())

    role_names = [spec[key] for key in ('id_column', 'group_column', 'target_column')]
    if spec['split_column'] is not None:
        role_names.append(spec['split_column'])
    if not all(map(column, role_names)):
        raise ValueError('Case ID, patient ID, target and optional split must be valid column names')
    for key in ('numeric_columns', 'categorical_columns'):
        names = spec[key]
        if not isinstance(names, list) or len(names) > MAX_FEATURES or not all(map(column, names)):
            raise ValueError(key + ' must be a list of valid column names')
        spec[key] = list(names)
    features = spec['numeric_columns'] + spec['categorical_columns']
    if len(features) > MAX_FEATURES or (not editable and not features):
        raise ValueError(f'Select 1..{MAX_FEATURES} explicit table features')
    if len(set(features + role_names)) != len(features) + len(role_names):
        raise ValueError('Case ID, patient ID, target, split and feature columns must be distinct')
    if type(spec['standardize']) is not bool:
        raise ValueError('standardize must be a boolean')
    if type(spec['folds']) is not int or not 2 <= spec['folds'] <= 20:
        raise ValueError('folds must be an integer between 2 and 20')
    fraction = spec['test_fraction']
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ValueError('test_fraction must be between zero and one')
    return spec


def _number(value, case, column):
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f'Expected a numeric value for {case} in column {column!r}') from exc
    if not math.isfinite(number) or abs(number) > MAX_FLOAT32:
        raise ValueError(f'Non-finite or out-of-range value for {case} in column {column!r}')
    return number


def _read_table(spec, cancelled):
    path = spec['path']
    raw = _read_source(path, MAX_CSV_BYTES, cancelled)
    source_digest = hashlib.sha256(raw).hexdigest()
    try:
        reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True)
    except UnicodeDecodeError as exc:
        raise ValueError('Table CSV must be UTF-8: ' + path) from exc
    id_column, group_column, target_column, split_column = (spec[key] for key in
        ('id_column', 'group_column', 'target_column', 'split_column'))
    requested = {id_column, group_column, target_column, *spec['numeric_columns'], *spec['categorical_columns']}
    if split_column is not None:
        requested.add(split_column)
    records, seen = [], set()
    try:
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError('Table CSV contains duplicate column names: ' + path)
        if not requested <= set(fields):
            raise ValueError('Table CSV is missing columns: ' + ', '.join(sorted(requested - set(fields))))
        for row in reader:
            _cancel(cancelled)
            line = reader.line_num
            if len(records) >= MAX_SAMPLES:
                raise ValueError(f'Table CSV sample limit exceeded ({MAX_SAMPLES}) at row {line}')
            case_id = (row[id_column] or '').strip()
            case = f'case {case_id!r} at CSV row {line}' if case_id else f'CSV row {line}'
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Malformed table fields for {case}')
            if not case_id or len(case_id) > 256 or case_id in seen:
                raise ValueError(f'Missing, oversized or duplicate case ID for {case}')
            seen.add(case_id)
            group, target = row[group_column].strip(), row[target_column].strip()
            if not group or len(group) > 256:
                raise ValueError(f'Missing or oversized patient ID for {case}')
            if not target or len(target) > 256:
                raise ValueError(f'Missing or oversized target for {case}')
            if spec['task'] == 'regression':
                target = _number(target, case, target_column)
            split = None
            if split_column is not None:
                split = row[split_column].strip().lower()
                if split not in SPLITS:
                    raise ValueError(f'Invalid explicit split {split!r} for {case}. Use train, val or test')
            numeric = []
            for name in spec['numeric_columns']:
                value = row[name].strip()
                numeric.append(_number(value, case, name) if value else None)
            categorical = [row[name].strip() or None for name in spec['categorical_columns']]
            records.append({'id': case_id, 'group': group, 'split': split, 'label': target,
                            'source_path': path, 'row_number': line, 'source_digest': source_digest,
                            'content_digest': _json_digest([numeric, categorical]),
                            'numeric': numeric, 'categorical': categorical})
    except csv.Error as exc:
        raise ValueError(f'Malformed table CSV near row {reader.line_num}: {exc}') from exc
    return records, source_digest


def _class_assignment(groups, fold_count, test_count, seed, cancelled):
    """Balance group labels while requiring every class in each training fold."""
    histograms = {group: Counter(row['label'] for row in rows) for group, rows in groups.items()}
    patient_totals = Counter(label for histogram in histograms.values() for label in histogram)
    if not 2 <= len(patient_totals) <= MAX_CLASSES:
        raise ValueError(f'Table classification requires 2..{MAX_CLASSES} classes in the development set')
    underrepresented = sorted(label for label, count in patient_totals.items() if count < 2)
    if underrepresented:
        raise ValueError('Every class needs at least two development patients for grouped cross-validation: '
                         + ', '.join(underrepresented))
    case_totals = Counter(row['label'] for rows in groups.values() for row in rows)
    target_test_fraction = test_count / len(groups)
    ratios = [(1 - target_test_fraction) / fold_count] * fold_count + [target_test_fraction]
    holdout = fold_count
    rng = random.Random(seed)
    best, best_score = None, None

    for attempt in range(_STRATIFICATION_ATTEMPTS):
        _cancel(cancelled)
        ordered = sorted(groups)
        rng.shuffle(ordered)
        if attempt % 2 == 0:
            ordered.sort(key=lambda group: (min(patient_totals[label] for label in histograms[group]),
                                            -len(histograms[group]), -len(groups[group])))
        patient_counts = [Counter() for _ in ratios]
        case_counts = [Counter() for _ in ratios]
        group_counts = [0] * len(ratios)
        case_loads = [0] * len(ratios)
        max_fold_counts = Counter()
        membership, test_groups = {}, set()
        empty_folds = fold_count

        for index, group in enumerate(ordered):
            _cancel(cancelled)
            histogram = histograms[group]
            test_slots = test_count - group_counts[holdout]
            must_fill_slot = len(ordered) - index == test_slots + empty_folds
            test_patients = patient_counts[holdout]
            # A class must have at least one patient outside both this fold
            # and the holdout, even when its rows span mixed-label patients.
            available = [fold for fold in range(fold_count)
                         if (not must_fill_slot or group_counts[fold] == 0)
                         and all(patient_counts[fold][label] + test_patients[label] + 1 < patient_totals[label]
                                 for label in histogram)]
            if test_slots and all(patient_totals[label] - test_patients[label] - 1 >= 2
                                  and max_fold_counts[label] + test_patients[label] + 1 < patient_totals[label]
                                  for label in histogram):
                available.append(holdout)
            if not available:
                break

            def cost(partition):
                existing = case_counts[partition]
                coverage_added = sum(existing[label] == 0 for label in histogram)
                # Change in squared error from the desired fraction of each class.
                imbalance = math.fsum((count / case_totals[label]) *
                                      (2 * (existing[label] / case_totals[label] - ratios[partition])
                                       + count / case_totals[label]) for label, count in histogram.items())
                return (partition != holdout or coverage_added == 0, -coverage_added,
                        imbalance, case_loads[partition])

            selected = min(available, key=cost)
            if selected == holdout:
                test_groups.add(group)
            else:
                membership[group] = selected
                empty_folds -= group_counts[selected] == 0
            group_counts[selected] += 1
            case_loads[selected] += len(groups[group])
            case_counts[selected].update(histogram)
            patient_counts[selected].update(histogram.keys())
            if selected != holdout:
                for label in histogram:
                    max_fold_counts[label] = max(max_fold_counts[label], patient_counts[selected][label])
        else:
            missing_test = len(patient_totals) - len(case_counts[holdout]) if test_count else 0
            missing_val = sum(len(patient_totals) - len(case_counts[fold]) for fold in range(fold_count))
            unavoidable_val = sum(max(0, fold_count - count + patient_counts[holdout][label])
                                  for label, count in patient_totals.items())
            score = (missing_test, missing_val - unavoidable_val, missing_val)
            if best_score is None or score < best_score:
                best, best_score = (test_groups, membership), score
            unavoidable_test = sum(count < 3 for count in patient_totals.values()) if test_count else 0
            if score[:2] == (unavoidable_test, 0):
                break

    if best is None:
        raise ValueError(f'Could not form {fold_count} patient-isolated folds with every class in training. '
                         'Review the number of folds and patient coverage for each class. '
                         'For an automatic holdout, reduce the test fraction or specify the split column.')
    return best


def _partition(records, spec, seed, cancelled):
    groups = defaultdict(list)
    for record in records:
        _cancel(cancelled)
        group = groups[record['group']]
        if group and record['split'] != group[0]['split']:
            raise ValueError(f"Patient {record['group']!r} crosses explicit splits at case {record['id']!r} "
                             f"(CSV row {record['row_number']}); first case {group[0]['id']!r}")
        group.append(record)
    automatic = spec['split_column'] is None
    classification = spec['task'] == 'classification'
    fold_count = spec['folds']
    membership = None
    if automatic and len(groups) >= 2:
        test_count = max(1, min(len(groups) - 1, round(len(groups) * spec['test_fraction'])))
        if len(groups) - test_count < fold_count:
            raise ValueError(f'{fold_count} folds require at least {fold_count} development patients. '
                             f'The selected test fraction leaves {len(groups) - test_count}')
        if classification:
            test_groups, membership = _class_assignment(groups, fold_count, test_count, seed, cancelled)
        else:
            ordered = sorted(groups)
            random.Random(seed).shuffle(ordered)
            test_groups = set(ordered[:test_count])
        for record in records:
            record['split'] = 'test' if record['group'] in test_groups else 'train'
    counts = Counter(record['split'] for record in records)
    if not counts['train'] or not counts['test']:
        raise ValueError('Table CSV requires nonempty train and test partitions with distinct patients')
    development_groups = {group: items for group, items in groups.items() if items[0]['split'] != 'test'}
    if len(development_groups) < fold_count:
        raise ValueError(f'{fold_count} folds require at least {fold_count} development patients. '
                         f'Found {len(development_groups)} in train and val')
    if classification and not automatic:
        _, membership = _class_assignment(development_groups, fold_count, 0, seed, cancelled)
    return {split: counts[split] for split in SPLITS}, development_groups, membership


def _fold_plan(records, groups, fold_count, seed, cancelled, membership):
    if membership is None:
        ordered = sorted(groups)
        random.Random(seed).shuffle(ordered)
        ordered.sort(key=lambda group: -len(groups[group]))
        loads = [0] * fold_count
        membership = {}
        for group in ordered:
            _cancel(cancelled)
            fold = loads.index(min(loads))
            membership[group] = fold
            loads[fold] += len(groups[group])
    development = [record for record in records if record['split'] != 'test']
    plan = []
    for fold in range(fold_count):
        _cancel(cancelled)
        train, validation = [], []
        for row in development:
            destination = validation if membership[row['group']] == fold else train
            destination.append(row['id'])
        plan.append({'fold': fold + 1, 'train_ids': train, 'val_ids': validation})
    return plan


def _coverage_warnings(records, fold_plan, class_names):
    labels = {record['id']: class_names[record['label']] for record in records}
    expected = set(class_names)
    test_labels = {labels[row['id']] for row in records if row['split'] == 'test'}
    warnings = []
    if expected - test_labels:
        warnings.append({'code': 'missing_split_classes', 'split': 'test', 'classes': sorted(expected - test_labels)})
    for fold in fold_plan:
        missing = expected - {labels[case_id] for case_id in fold['val_ids']}
        if missing:
            warnings.append({'code': 'missing_split_classes', 'split': 'val', 'fold': fold['fold'], 'classes': sorted(missing)})
    return warnings


def inspect_table_dataset(data, seed=42, cancelled=None):
    """Freeze local raw values and a path-independent partition identity."""
    spec = validate_table_spec(data)
    if type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ValueError('Table dataset seed is outside the supported integer range')
    _cancel(cancelled)
    try:
        spec['path'] = str(Path(spec['path']).expanduser().resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError('Table CSV source is unavailable: ' + spec['path']) from exc
    records, source_digest = _read_table(spec, cancelled)
    counts, development_groups, membership = _partition(records, spec, seed, cancelled)
    target = {'target_column': spec['target_column']}
    if spec['task'] == 'classification':
        names = sorted({record['label'] for record in records if record['split'] != 'test'})
        labels = {name: index for index, name in enumerate(names)}
        for record in records:
            _cancel(cancelled)
            name = record['label']
            if name not in labels:
                raise ValueError(f"Test case {record['id']!r} has class {name!r} missing from development data")
            record['label'] = labels[name]
        target['class_names'] = names
    fold_plan = _fold_plan(records, development_groups, spec['folds'], seed, cancelled, membership)
    coverage = {}
    if spec['task'] == 'classification':
        coverage = {'split_policy': {'holdout': 'stratified_patient_groups' if spec['split_column'] is None else 'explicit',
                                     'cross_validation': 'stratified_patient_groups',
                                     'training_class_coverage': 'required', 'missing_evaluation_classes': 'reported'},
                    'warnings': _coverage_warnings(records, fold_plan, target['class_names'])}
    numeric, categorical = spec['numeric_columns'], spec['categorical_columns']
    preprocessing = {'numeric_columns': numeric, 'categorical_columns': categorical,
                     'numeric_imputer': 'median' if numeric else None,
                     'categorical_encoder': 'one_hot_ignore_unknown' if categorical else None,
                     'categorical_missing': 'separate_category' if categorical else None,
                     'standardize': spec['standardize'], 'fit': 'fold_train_then_development'}
    digest = _json_digest({'version': 1, 'source_digest': source_digest, 'seed': seed,
                           'data_spec': {key: value for key, value in spec.items() if key != 'path'},
                           'fold_plan': fold_plan, 'preprocessing': preprocessing,
                           'test_ids': [record['id'] for record in records if record['split'] == 'test'], **coverage})
    return {'version': 1, 'kind': 'table_csv', 'task': spec['task'], 'seed': seed, 'data_spec': spec,
            'dataset_digest': digest, 'source_digest': source_digest, 'counts': counts, 'records': records,
            'input_shape': [None, len(numeric) + len(categorical)], 'preprocessing': preprocessing,
            'fold_plan': fold_plan, **target, **coverage}
