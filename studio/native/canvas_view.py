"""Session-only persistent editors and explicit, versioned event ownership."""
from dataclasses import dataclass, field
import json

import dearpygui.dearpygui as dpg

from .view_document import resolve_view


@dataclass(frozen=True)
class CanvasToken:
    view_id: str
    generation: int
    context_epoch: int
    document_epoch: int


@dataclass
class CanvasView:
    view_id: str
    editor_tag: str
    node_refs: dict = field(default_factory=dict)
    ref_nodes: dict = field(default_factory=dict)
    source_attributes: dict = field(default_factory=dict)
    target_attributes: dict = field(default_factory=dict)
    link_targets: dict = field(default_factory=dict)
    view_positions: dict = field(default_factory=dict)
    node_categories: dict = field(default_factory=dict)
    content_token: str | None = None
    generation: int = 0
    context_epoch: int = 0
    last_selection: tuple = ()
    visible: bool = False
    container_tag: str = ''
    edge_overlay: object = None
    selected_links: set = field(default_factory=set)


class CanvasHost:
    def __init__(self, document, workbench, *, render=None, dispatch=None, log=None, on_sync=None, themes=None):
        self.document, self.workbench = document, workbench
        self.views = {}
        self.document_epoch = 0
        self._serial = 0
        self._generation_clock = 0
        self._render, self._dispatch = render, dispatch
        self._log, self._on_sync = log, on_sync
        self._themes = themes if themes is not None else {}
        self._parents = {}

    def get(self, view_id):
        return self.views[view_id]

    def token(self, view_id):
        view = self.get(view_id)
        state = self.workbench.views[view_id]
        return CanvasToken(view_id, view.generation, state.context_epoch, self.document_epoch)

    def accept(self, token):
        valid = (isinstance(token, CanvasToken) and token.view_id in self.views
                 and token.view_id in self.workbench.views and token == self.token(token.view_id))
        if not valid and self._log:
            self._log('已丢弃过期画布回调；请在当前视图重新操作。', '提示')
        return valid

    def _callback(self, view_id, action):
        token = self.token(view_id)
        def callback(sender=None, app_data=None, user_data=None):
            if self.accept(token) and self._dispatch:
                return self._dispatch(token, action, sender, app_data)
        return callback

    def _content(self, state):
        scope, _ = resolve_view(self.document, state)
        definitions = {item['id']: item for item in self.document.graph.get('definitions', [])}
        dependencies, visited = [], set()
        def walk(graph):
            for instance in graph.get('instances', []):
                key = instance.get('definition_id') if instance.get('kind') == 'chip' else None
                if key in definitions and key not in visited:
                    visited.add(key)
                    definition = definitions[key]
                    dependencies.append(definition)
                    walk(definition['graph'])
        walk(scope)
        return json.dumps([scope, dependencies], sort_keys=True, ensure_ascii=False, allow_nan=False)

    def sync(self, parent_by_pane):
        self._parents = dict(parent_by_pane)
        self.workbench.reconcile(self.document)
        for key in list(self.views):
            if key not in self.workbench.views:
                self.dispose(key)
        width, height = 810, 405
        if dpg.does_item_exist('native_canvas_body'):
            config = dpg.get_item_configuration('native_canvas')
            body = dpg.get_item_configuration('native_canvas_body')
            width, height = max(200, config['width'] - 18), max(120, body['height'])
        split = self.workbench.layout != 'single'
        for index, pane in enumerate(('primary', 'secondary')):
            parent = parent_by_pane[pane]
            columns = self.workbench.layout == 'columns'
            rows = self.workbench.layout == 'rows'
            dpg.configure_item(parent, show=index == 0 or split,
                               pos=((width // 2 + 2) if index and columns else 0,
                                    (height // 2 + 2) if index and rows else 0),
                               width=(width // 2 - 2) if columns else width,
                               height=(height // 2 - 2) if rows else height)
        for key, state in self.workbench.views.items():
            if key not in self.views:
                self._serial += 1
                tag = f'native_canvas_{key}_d{self.document_epoch}_e{self._serial}'
                container = tag + '_container'
                self.views[key] = CanvasView(key, tag, container_tag=container)
                from .orthogonal_edges import OrthogonalEdgeOverlay
                self.views[key].edge_overlay = OrthogonalEdgeOverlay()
                dpg.add_child_window(tag=container, parent=parent_by_pane[state.pane], width=-1, height=-1,
                                     border=False, no_scrollbar=True, pos=(0, 56))
                dpg.add_node_editor(tag=tag, parent=container, width=-1, height=-1, minimap=True)
            view = self.views[key]
            parent = parent_by_pane[state.pane]
            current_parent = dpg.get_item_info(view.container_tag)['parent']
            current_parent = dpg.get_alias_id(current_parent) if isinstance(current_parent, str) else current_parent
            parent_id = dpg.get_alias_id(parent) if isinstance(parent, str) else parent
            if current_parent != parent_id:
                dpg.move_item(view.container_tag, parent=parent)
            content = self._content(state)
            changed = content != view.content_token
            if changed:
                view.edge_overlay.clear()
                self._generation_clock += 1
                view.generation = self._generation_clock
                dpg.delete_item(view.editor_tag, children_only=True)
                for mapping in (view.node_refs, view.ref_nodes, view.source_attributes, view.target_attributes,
                                view.link_targets, view.view_positions, view.node_categories):
                    mapping.clear()
                if self._render:
                    self._render(view)
                view.content_token = content
                view.last_selection = ()
                view.selected_links.clear()
            if changed or view.context_epoch != state.context_epoch:
                view.context_epoch = state.context_epoch
                dpg.configure_item(view.editor_tag, callback=self._callback(key, 'connect'),
                                   delink_callback=self._callback(key, 'disconnect'), user_data=self.token(key))
                if hasattr(self, 'catalog_panel'):
                    from .catalog_panel import PAYLOAD_TYPE
                    dpg.configure_item(view.container_tag, payload_type=PAYLOAD_TYPE,
                                       drop_callback=self.catalog_panel.drop_callback(self.token(key)))
            view.visible = self.workbench.active_by_pane[state.pane] == key and (state.pane == 'primary' or split)
            if not view.visible:
                view.edge_overlay.clear()
            parent_config = dpg.get_item_configuration(parent)
            dpg.configure_item(view.container_tag, show=view.visible,
                               width=parent_config['width'], height=max(50, parent_config['height'] - 56))
            dpg.configure_item(view.editor_tag, show=view.visible,
                               width=max(50, parent_config['width'] - 8), height=max(50, parent_config['height'] - 64))
            self._selection_theme(key)
        if self._on_sync:
            self._on_sync()

    def refresh_scope(self, definition_id):
        affected = {definition_id}
        definitions = self.document.graph.get('definitions', [])
        pending = [definition_id]
        while pending:
            target = pending.pop()
            for definition in definitions:
                key = definition['id']
                if key not in affected and any(item.get('kind') == 'chip' and item.get('definition_id') == target
                                               for item in definition['graph'].get('instances', [])):
                    affected.add(key); pending.append(key)
        affected.add(None)
        for key, view in self.views.items():
            if self.workbench.views[key].target.definition_id in affected:
                view.content_token = None

    def apply_selection(self, view_id, refs, *, intentional=False):
        view, state = self.get(view_id), self.workbench.views[view_id]
        refs = list(dict.fromkeys(refs))
        if any(ref not in view.ref_nodes for ref in refs):
            raise ValueError('Selection does not belong to this canvas')
        if refs or intentional:
            if refs:
                view.selected_links.clear()
            if refs != state.selected_refs:
                state.probe_port = None
            state.selected_refs = refs
            view.last_selection = tuple(refs)
            if intentional and not refs:
                dpg.clear_selected_nodes(view.editor_tag)
                dpg.clear_selected_links(view.editor_tag)
        self._selection_theme(view_id)
        if self._on_sync:
            self._on_sync()

    def _selection_theme(self, view_id):
        view, state = self.get(view_id), self.workbench.views[view_id]
        for ref, item in view.ref_nodes.items():
            category = 'selected' if ref in state.selected_refs else view.node_categories.get(ref, 'tensor')
            if category in self._themes:
                dpg.bind_item_theme(item, self._themes[category])

    def dispose(self, view_id):
        view = self.views.pop(view_id, None)
        if view and view.edge_overlay:
            view.edge_overlay.dispose()
        if view and dpg.does_item_exist(view.container_tag):
            dpg.delete_item(view.container_tag)

    def dispose_all(self):
        for key in list(self.views):
            self.dispose(key)

    def reset_document(self):
        self.document_epoch += 1
        self.dispose_all()
        self.workbench.reconcile(self.document, model_replaced=True)
