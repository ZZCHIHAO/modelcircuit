"""Bounded, GUI-independent graph views. This state never owns a document."""
from dataclasses import dataclass
from collections import Counter
import json
import re

from studio_core.schema import MAX_CHIP_DEPTH

_ID = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_REF = re.compile(r'^(input|call|output):[A-Za-z_][A-Za-z0-9_]*$')
_PANES = ('primary', 'secondary')
_LAYOUTS = ('single', 'columns', 'rows')
_VIEW_FIELDS = {'view_id', 'target', 'call_path', 'selected_refs', 'probe_port', 'pane'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identifier(value):
    _require(type(value) is str and bool(_ID.fullmatch(value)), 'Expected a stable identifier')


def _path(value):
    _require(type(value) in (list, tuple) and len(value) <= MAX_CHIP_DEPTH, 'Invalid or excessive call path')
    for item in value:
        _identifier(item)
    return list(value)


@dataclass(frozen=True)
class ScopeTarget:
    definition_id: str | None


@dataclass
class ViewState:
    view_id: str
    target: ScopeTarget
    call_path: list[str]
    selected_refs: list[str]
    probe_port: str | None
    pane: str
    context_epoch: int = 0


def validate_view(view):
    """Validate a view's finite storage contract without resolving graph IDs."""
    _require(type(view) is ViewState and type(view.target) is ScopeTarget, 'Invalid view state')
    _identifier(view.view_id)
    if view.target.definition_id is not None:
        _identifier(view.target.definition_id)
    _path(view.call_path)
    _require(type(view.call_path) is list, 'Stored call path must be a list')
    _require(type(view.selected_refs) is list and len(view.selected_refs) <= 20_000, 'Invalid selection list')
    _require(all(type(ref) is str and bool(_REF.fullmatch(ref)) for ref in view.selected_refs), 'Invalid selected reference')
    _require(len(set(view.selected_refs)) == len(view.selected_refs), 'Duplicate selected reference')
    if view.probe_port is not None:
        _identifier(view.probe_port)
    _require(view.pane in _PANES, 'Invalid pane')
    _require(type(view.context_epoch) is int and view.context_epoch >= 0, 'Invalid context epoch')
    if view.target.definition_id is None:
        _require(not view.call_path and view.pane == 'primary', 'Root view must be unbound and in primary')


class WorkbenchState:
    def __init__(self):
        self.views = {'root': ViewState('root', ScopeTarget(None), [], [], None, 'primary')}
        self.layout = 'single'
        self.active_by_pane = {'primary': 'root', 'secondary': None}
        self.focused_pane = 'primary'
        self.warnings = []

    @property
    def focused_view_id(self):
        candidate = self.active_by_pane.get(self.focused_pane)
        return candidate if candidate in self.views else 'root'

    def _view(self, view_id):
        _require(type(view_id) is str and view_id in self.views, 'Unknown view: ' + str(view_id))
        return self.views[view_id]

    def _repair_focus(self, *, fallback=False):
        for pane in _PANES:
            current = self.active_by_pane[pane]
            if current not in self.views or self.views[current].pane != pane:
                self.active_by_pane[pane] = 'root' if pane == 'primary' else next(
                    (key for key, view in self.views.items() if view.pane == pane), None)
        if fallback or self.layout == 'single' or self.active_by_pane[self.focused_pane] is None:
            self.focused_pane = 'primary'

    def open_root(self):
        self.activate('root')
        return 'root'

    def open_definition(self, definition_id, call_path=(), pane='primary'):
        _identifier(definition_id)
        path = _path(call_path)
        _require(pane in _PANES, 'Invalid pane')
        for key, view in self.views.items():
            if view.target.definition_id == definition_id:
                if view.call_path != path:
                    candidate = self.to_payload()
                    next(item for item in candidate['views'] if item['view_id'] == key)['call_path'] = path
                    self._parse(candidate)
                    view.call_path = path
                    view.context_epoch += 1
                self.activate(key)
                return key
        _require(len(self.views) < 16, 'At most 16 graph tabs may be open')
        base = 'definition_' + definition_id
        key, suffix = base, 2
        while key in self.views:
            key, suffix = base + '_' + str(suffix), suffix + 1
        candidate = self.to_payload()
        candidate['views'].append({'view_id': key, 'target': {'definition_id': definition_id},
                                   'call_path': path, 'selected_refs': [], 'probe_port': None, 'pane': pane})
        self._parse(candidate)
        self.views[key] = ViewState(key, ScopeTarget(definition_id), path, [], None, pane)
        self.activate(key)
        return key

    def activate(self, view_id):
        view = self._view(view_id)
        if self.layout == 'single' and view.pane == 'secondary':
            view.pane = 'primary'
            self._repair_focus()
        self.active_by_pane[view.pane] = view_id
        self.focused_pane = view.pane

    def close(self, view_id):
        self._view(view_id)
        _require(view_id != 'root', 'The root tab cannot be closed')
        was_active = view_id in self.active_by_pane.values()
        del self.views[view_id]
        self._repair_focus(fallback=was_active)

    def set_layout(self, mode):
        _require(mode in _LAYOUTS, 'Invalid layout')
        self.layout = mode
        self._repair_focus()

    def move_to_pane(self, view_id, pane):
        view = self._view(view_id)
        _require(pane in _PANES, 'Invalid pane')
        _require(view_id != 'root' or pane == 'primary', 'The root tab is fixed in primary')
        if view.pane == pane:
            return
        focused = self.focused_view_id == view_id
        view.pane = pane
        self._repair_focus()
        self.active_by_pane[pane] = view_id
        if focused:
            self.focused_pane = pane if self.layout != 'single' else 'primary'

    def reconcile(self, document, *, model_replaced=False):
        from .view_document import resolve_view
        from studio_core.ports import output_port_ids
        messages = []
        if model_replaced:
            previous_warnings = self.warnings
            self.__init__()
            self.warnings = previous_warnings
            messages.append('Model replaced; workbench reset to root')
        else:
            definitions = document.graph.get('definitions', [])
            definition_counts = Counter(item['id'] for item in definitions)
            lost_active = False
            for key, view in list(self.views.items()):
                if view.target.definition_id is not None and definition_counts[view.target.definition_id] != 1:
                    lost_active |= key in self.active_by_pane.values()
                    del self.views[key]
                    messages.append('Closed missing or ambiguous definition view: ' + key)
                    continue
                try:
                    scope, _ = resolve_view(document, view)
                except ValueError:
                    view.call_path = []
                    view.context_epoch += 1
                    scope, _ = resolve_view(document, view)
                    messages.append('Cleared invalid call context: ' + key)
                reference_counts = Counter(kind + ':' + item['id'] for kind, collection in (
                    ('input', 'inputs'), ('call', 'calls'), ('output', 'outputs')) for item in scope[collection])
                probe_ref = view.selected_refs[0] if view.selected_refs else None
                selected = [ref for ref in view.selected_refs if reference_counts[ref] == 1]
                if selected != view.selected_refs:
                    view.selected_refs = selected
                    messages.append('Cleared invalid selection: ' + key)
                if view.probe_port is not None:
                    calls = [call for call in scope['calls'] if selected and selected[0] == probe_ref == 'call:' + call['id']]
                    instances = [item for item in scope['instances'] if len(calls) == 1 and item['id'] == calls[0]['instance_id']]
                    instance = instances[0] if len(instances) == 1 else None
                    try:
                        valid_port = (instance is not None
                                      and (instance['kind'] != 'chip' or definition_counts[instance.get('definition_id')] == 1)
                                      and view.probe_port in output_port_ids(instance, definitions))
                    except ValueError:
                        valid_port = False
                    if not valid_port:
                        view.probe_port = None
                        messages.append('Cleared invalid probe port: ' + key)
            self._repair_focus(fallback=lost_active)
        self.warnings.extend(messages)
        return messages

    def to_payload(self):
        for key, view in self.views.items():
            validate_view(view)
            _require(key == view.view_id, 'View dictionary identity differs from view ID')
        payload = {'layout': self.layout, 'views': [
            {'view_id': view.view_id, 'target': {'definition_id': view.target.definition_id},
             'call_path': list(view.call_path), 'selected_refs': list(view.selected_refs),
             'probe_port': view.probe_port, 'pane': view.pane} for view in self.views.values()],
            'active_by_pane': dict(self.active_by_pane), 'focused_pane': self.focused_pane}
        self._parse(payload)
        return payload

    @classmethod
    def _parse(cls, payload):
        try:
            return cls._parse_fields(payload)
        except (TypeError, KeyError, RecursionError) as exc:
            raise ValueError('Workbench must contain valid finite view fields') from exc

    @classmethod
    def _parse_fields(cls, payload):
        _require(type(payload) is dict and set(payload) == {'layout', 'views', 'active_by_pane', 'focused_pane'}, 'Invalid workbench fields')
        _require(payload['layout'] in _LAYOUTS, 'Invalid layout')
        _require(type(payload['views']) is list and 1 <= len(payload['views']) <= 16, 'Invalid view count')
        result = cls()
        result.views = {}
        targets = set()
        for entry in payload['views']:
            _require(type(entry) is dict and set(entry) == _VIEW_FIELDS, 'Invalid view fields')
            target = entry['target']
            _require(type(target) is dict and set(target) == {'definition_id'}, 'Invalid scope target')
            view = ViewState(entry['view_id'], ScopeTarget(target['definition_id']), entry['call_path'], entry['selected_refs'], entry['probe_port'], entry['pane'])
            validate_view(view)
            _require(view.view_id not in result.views and view.target.definition_id not in targets, 'Duplicate view or definition')
            _require((view.view_id == 'root') == (view.target.definition_id is None), 'Invalid root identity')
            result.views[view.view_id] = ViewState(view.view_id, view.target, list(view.call_path), list(view.selected_refs), view.probe_port, view.pane)
            targets.add(view.target.definition_id)
        _require('root' in result.views, 'Root view is required')
        active = payload['active_by_pane']
        _require(type(active) is dict and set(active) == set(_PANES), 'Invalid active panes')
        for pane, key in active.items():
            _require(key is None or type(key) is str, 'Invalid active view ID')
        focus = payload['focused_pane']
        _require(focus in _PANES, 'Invalid focused pane')
        # Empty secondary and hidden secondary cannot retain focus after loading.
        result.layout, result.active_by_pane, result.focused_pane = payload['layout'], dict(active), focus
        result._repair_focus()
        if result.active_by_pane != active or result.focused_pane != focus:
            result.warnings.append('Repaired invalid active view or hidden/empty pane focus')
        try:
            size = len(json.dumps(payload, allow_nan=False).encode('utf-8'))
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError('Workbench must contain finite JSON') from exc
        _require(size <= 8 * 1024 * 1024, 'Workbench payload exceeds 8 MiB')
        return result

    @classmethod
    def from_payload(cls, payload, document):
        try:
            result = cls._parse(payload)
        except ValueError as exc:
            result = cls()
            result.warnings.append('Invalid workbench; restored root: ' + str(exc))
            return result
        result.reconcile(document)
        return result
