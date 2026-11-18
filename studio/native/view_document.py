"""Synchronous, nestable view cursors over the one shared Document."""
from contextlib import contextmanager
from copy import deepcopy

from .document import _find, _require
from .workbench_state import validate_view


def resolve_view(document, view):
    """Resolve stable IDs against the current graph, never cached graph objects."""
    validate_view(view)
    target = view.target.definition_id
    scope = document.graph if target is None else _find(document.graph.get('definitions', []), target, 'chip definition')['graph']
    trail = document._resolve_call_path(view.call_path)
    if view.call_path:
        _require(trail[-1]['id'] == target, 'Call path does not reach the target definition')
    return scope, trail


@contextmanager
def view_scope(document, view):
    """Temporarily bind an explicit view; restore a valid cursor even on failure.

    Commands still use the document's unified history. Do not retain this
    context across callbacks, asynchronous work, or worker submission.
    """
    _, trail = resolve_view(document, view)
    saved_override = document._scope_override
    saved_trail, saved_selected = deepcopy(document.trail), document.selected
    document._scope_override = view.target.definition_id
    document.trail = trail
    document.selected = None
    try:
        if view.selected_refs:
            document.select(view.selected_refs[0])
        yield document
    finally:
        document._restore_cursor(saved_override, saved_trail, saved_selected)
