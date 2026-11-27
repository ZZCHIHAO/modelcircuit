"""Python interchange for a single classical fit graph."""
import ast
from copy import deepcopy

from .operator_rules import normalize_operator
from .operators import CLASSICAL_KINDS, FOREST_KINDS, operator_spec


def export_classical(graph):
    from .classical_task import classical_graph_contract
    classical_graph_contract(graph)
    layout = deepcopy(graph)
    instance = layout['instances'][0]
    kind = instance.pop('kind')
    params = normalize_operator(kind, instance.pop('params', {}))
    forest = kind in FOREST_KINDS
    family, module = ('random forest', 'ensemble') if forest else ('support vector', 'svm')
    constructor_args = ', '.join(f'{name}={params[name]!r}' for name in operator_spec(kind).defaults)
    if forest:
        constructor_args += ', random_state=42'
    return f'''"""ModelCircuit Studio {family} model. Pass preprocessed feature rows."""
from sklearn.{module} import {kind}

STUDIO_CLASSICAL_LAYOUT = {layout!r}


def build_estimator():
    return {kind}({constructor_args})


def fit_model(features, targets):
    return build_estimator().fit(features, targets)


def predict(model, features):
    return model.predict(features).reshape(-1, 1)
'''


def import_classical(tree):
    """Read estimator arguments from executable syntax and validate all statements."""
    try:
        body = tree.body
        layout = ast.literal_eval(body[2].value)
        call = body[3].body[0].value
        kind = call.func.id
        if kind not in CLASSICAL_KINDS:
            raise ValueError('Unsupported estimator constructor')
        accepted = set(operator_spec(kind).defaults)
        if kind in FOREST_KINDS:
            accepted.add('random_state')
        keywords = {argument.arg: ast.literal_eval(argument.value) for argument in call.keywords}
        if len(call.keywords) != len(accepted) or keywords.keys() != accepted:
            raise ValueError('Unsupported estimator constructor')
        keywords.pop('random_state', None)
        layout['instances'][0].update(kind=kind, params=normalize_operator(kind, keywords))
        canonical = ast.parse(export_classical(layout))
        if ast.dump(canonical) != ast.dump(tree):
            raise ValueError('Only edits to the generated estimator parameters are supported')
        return layout
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RecursionError) as exc:
        raise ValueError('Unsupported ModelCircuit classical Python: ' + str(exc)) from None
