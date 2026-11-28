"""Round trip for the editable PCA and KMeans project pipeline."""
import ast
from copy import deepcopy

from .operator_rules import normalize_operator
from .unsupervised_task import STAGE_IDS, unsupervised_contract


def export_unsupervised(graph):
    stages = unsupervised_contract(graph)
    layout = deepcopy(graph)
    for instance in layout['instances']:
        instance.pop('params', None)
    steps = []
    for _, kind, params in stages:
        extra = "svd_solver='full'" if kind == 'PCA' else 'random_state=random_state'
        args = ', '.join([*(f'{key}={value!r}' for key, value in params.items()), extra])
        steps.append(f"        ({STAGE_IDS[kind]!r}, {kind}({args})),")
    body = '\n'.join(steps)
    return f'''"""Editable ModelCircuit Studio unsupervised pipeline."""
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

STUDIO_UNSUPERVISED_LAYOUT = {layout!r}


def build_estimator(standardize=False, random_state=42):
    steps = [
{body}
    ]
    if standardize:
        steps.insert(0, ('scale', StandardScaler()))
    return Pipeline(steps)


def fit_model(features, standardize=False, random_state=42):
    return build_estimator(standardize, random_state).fit(features)
'''


def import_unsupervised(tree):
    try:
        layout = ast.literal_eval(tree.body[5].value)
        steps = tree.body[6].body[0].value.elts
        by_kind = {instance['kind']: instance for instance in layout['instances']}
        for step in steps:
            call = step.elts[1]
            kind = call.func.id
            optional = 'svd_solver' if kind == 'PCA' else 'random_state'
            parameters = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
                          if keyword.arg != optional}
            by_kind[kind]['params'] = normalize_operator(kind, parameters)
        if ast.dump(ast.parse(export_unsupervised(layout))) != ast.dump(tree):
            raise ValueError('Only edits to the generated PCA and KMeans parameters are supported')
        return layout
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RecursionError) as error:
        raise ValueError('Unsupported ModelCircuit unsupervised Python: ' + str(error)) from None
