"""Python interchange for a single SVC or SVR fit graph."""
import ast
from copy import deepcopy

from .operator_rules import normalize_operator


def export_classical(graph):
    from .classical_task import classical_graph_contract
    classical_graph_contract(graph)
    layout = deepcopy(graph)
    instance = layout['instances'][0]
    kind = instance.pop('kind')
    params = normalize_operator(kind, instance.pop('params', {}))
    return f'''"""ModelCircuit Studio support vector model. Pass preprocessed feature rows."""
from sklearn.svm import {kind}

STUDIO_CLASSICAL_LAYOUT = {layout!r}


def build_estimator():
    return {kind}(C={params['C']!r}, kernel={params['kernel']!r})


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
        keywords = {argument.arg: ast.literal_eval(argument.value) for argument in call.keywords}
        if kind not in ('SVC', 'SVR') or len(call.keywords) != 2 or keywords.keys() != {'C', 'kernel'}:
            raise ValueError('Unsupported estimator constructor')
        layout['instances'][0].update(kind=kind, params=normalize_operator(kind, keywords))
        canonical = ast.parse(export_classical(layout))
        if ast.dump(canonical) != ast.dump(tree):
            raise ValueError('Only edits to the generated C and kernel values are supported')
        return layout
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RecursionError) as exc:
        raise ValueError('Unsupported ModelCircuit SVM Python: ' + str(exc)) from None
