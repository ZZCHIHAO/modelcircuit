"""Small compatibility bridge from legacy UI commands to explicit view scopes."""
from contextlib import contextmanager
from functools import wraps

import dearpygui.dearpygui as dpg

from .canvas_view import CanvasToken
from .view_document import resolve_view, view_scope


MAPS = ('node_refs', 'ref_nodes', 'source_attributes', 'target_attributes', 'link_targets', 'view_positions')
MODALS = ('native_chip_tools', 'native_chip_dialog', 'native_chip_import', 'native_python',
          'native_python_open', 'native_python_save', 'native_python_replace', 'native_json_open',
          'native_json_save', 'native_data', 'native_data_open', 'native_starter_replace',
          'native_catalog_config', 'native_catalog_policy')


class CanvasCommands:
    def _canvas_aliases(self, key):
        view, state = self.canvas_host.get(key), self.workbench.views[key]
        for name in MAPS:
            setattr(self, name, getattr(view, name))
        self.selected_refs = state.selected_refs
        self.generation = view.generation
        self.last_selection = view.last_selection

    def _canvas_synced(self):
        key = self.workbench.focused_view_id
        view = self.canvas_host.get(key)
        current = dpg.get_alias_id('native_editor') if dpg.does_alias_exist('native_editor') else 0
        if current != dpg.get_alias_id(view.editor_tag):
            if dpg.does_alias_exist('native_editor'):
                dpg.remove_alias('native_editor')
            dpg.add_alias('native_editor', view.editor_tag)
        if self._bound_canvas_id is None:
            self._focus_document(key)
            self._canvas_aliases(key)
            focus = key, self.workbench.views[key].context_epoch
            if focus != self._presented_canvas_focus:
                self._presented_canvas_focus = focus
                if self._built:
                    # Shared palette/inspector controls must acquire the new
                    # owner's token even when no graph content is redrawn.
                    self._render_tree()
                    self._render_inspector()
                    self._render_probe()
        self._stamp_callbacks('native_main', self.canvas_host.token(key))
        for view_key, current_view in self.canvas_host.views.items():
            self._stamp_callbacks(current_view.editor_tag, self.canvas_host.token(view_key), include_bound=True)
        if dpg.does_item_exist('native_delete_handler'):
            dpg.set_item_user_data('native_delete_handler', self.canvas_host.token(key))
        if hasattr(self, 'workbench_panel'):
            self.workbench_panel.sync()
        if hasattr(self, 'catalog_panel'):
            self.catalog_panel.sync_quick_owners()

    def _focus_document(self, key):
        state = self.workbench.views[key]
        _, trail = resolve_view(self.document, state)
        self.document._scope_override = state.target.definition_id
        self.document.trail = trail
        self.document.selected = state.selected_refs[-1] if state.selected_refs else None

    def _stamp_callbacks(self, parent, token, *, include_bound=False):
        if not dpg.does_item_exist(parent):
            return
        callback = dpg.get_item_callback(parent)
        if callback is not None and (include_bound or getattr(callback, '_canvas_dynamic', False)):
            dpg.set_item_user_data(parent, token)
        for children in dpg.get_item_children(parent).values():
            for child in children:
                if not include_bound and any(dpg.get_alias_id(view.editor_tag) == child for view in self.canvas_host.views.values()):
                    continue
                self._stamp_callbacks(child, token, include_bound=include_bound)

    def _bind_modal(self, tag):
        key = self._bound_canvas_id or self.workbench.focused_view_id
        token = self.canvas_host.token(key)
        self._modal_tokens[tag] = token
        self._stamp_callbacks(tag, token, include_bound=True)

    @contextmanager
    def _canvas_context(self, key, *, adopt=False, restore_shared=True):
        previous = self._bound_canvas_id
        shared_before = self._shared_render_serial
        state = self.workbench.views[key]
        if adopt and previous is None and self.document.scope_id == state.target.definition_id:
            if self.selected_refs != state.selected_refs:
                state.selected_refs = list(self.selected_refs)
            elif self.document.selected and self.document.selected not in state.selected_refs:
                state.selected_refs = [self.document.selected]
        self._bound_canvas_id = key
        self._canvas_aliases(key)
        try:
            with view_scope(self.document, state):
                yield
                # Legacy direct callers may assign selected_refs; model commands
                # may instead select their newly-created node.
                if self._bound_canvas_id == key and key in self.workbench.views:
                    if self.selected_refs != state.selected_refs:
                        state.selected_refs = list(self.selected_refs)
                    elif self.document.selected and self.document.selected not in state.selected_refs:
                        state.selected_refs = [self.document.selected]
        finally:
            restored = previous if previous in self.workbench.views else None
            self._bound_canvas_id = restored
            focused = restored or self.workbench.focused_view_id
            if focused in self.canvas_host.views:
                self._canvas_aliases(focused)
                if restored is None:
                    self._focus_document(focused)
                    if (restore_shared and key != focused and self._built
                            and self._shared_render_serial != shared_before):
                        # A valid delayed owner command may edit another pane.
                        # Restore focus only if that command rendered shared
                        # controls. Pure scope reads/position polls preserve
                        # existing input widgets and unsubmitted text.
                        self._presented_canvas_focus = None
                        self._canvas_synced()

    def _draw_canvas(self, view):
        # CanvasHost calls on_sync after all editors are ready. Rendering an
        # individual nonfocused view must not rebuild shared controls midway.
        with self._canvas_context(view.view_id, restore_shared=False):
            self._render_canvas_contents(view)

    def _dispatch_canvas(self, token, action, sender, app_data):
        if not self.canvas_host.accept(token):
            return
        with self._canvas_context(token.view_id):
            try:
                return getattr(self, '_' + action)(sender=sender, app_data=app_data, user_data=token)
            except Exception as error:
                self.log(str(error), '错误')

    def _render_graph(self):
        self.canvas_host.sync(self.canvas_parent_by_pane)

    def _activate_document_cursor(self):
        if self.document.scope_id is None:
            key = 'root'
        else:
            key = self.workbench.open_definition(self.document.scope_id,
                                                [entry['call_id'] for entry in self.document.trail])
        self.workbench.activate(key)
        # Navigation establishes the new view's selection, never another view's.
        self.workbench.views[key].selected_refs = [self.document.selected] if self.document.selected else []
        self.canvas_host.sync(self.canvas_parent_by_pane)
        self._bound_canvas_id = key if self._bound_canvas_id is not None else None

    def _select_root(self, ref, **_):
        self.workbench.activate('root')
        self.canvas_host.sync(self.canvas_parent_by_pane)
        self.canvas_host.apply_selection('root', [ref], intentional=True)
        if self._bound_canvas_id is not None:
            self._bound_canvas_id = 'root'
            self._focus_document('root')
        self._refresh(graph=False)

    def _enter_root(self, call_id, **_):
        self.document.root(); self.document.enter(call_id)
        self._activate_document_cursor(); self._refresh(graph=False)

    def _enter(self, call_id, **_):
        self.document.enter(call_id)
        self._activate_document_cursor(); self._refresh(graph=False)

    def _back(self, **_):
        self.document.back()
        self._activate_document_cursor(); self._refresh(graph=False)

    def _root(self, **_):
        self.workbench.activate('root')
        self.canvas_host.sync(self.canvas_parent_by_pane)
        self._bound_canvas_id = 'root' if self._bound_canvas_id is not None else None
        self._focus_document('root')
        self._refresh(graph=False)

    def _crumb(self, index, **_):
        while len(self.document.trail) > index + 1:
            self.document.back()
        self._activate_document_cursor(); self._refresh(graph=False)

    def _reset_canvas_document(self):
        self.canvas_host.reset_document()
        self.document.attention_probe_requests = []
        self._bound_canvas_id = None
        self._modal_tokens.clear()
        self.probe_selection.clear()
        self.pending_starter = None
        if hasattr(self, 'catalog_panel'):
            self.catalog_panel.invalidate_requests()
        for tag in MODALS:
            if dpg.does_item_exist(tag):
                dpg.hide_item(tag)
        self.canvas_host.sync(self.canvas_parent_by_pane)

    def _commit_canvas_positions(self, key):
        view = self.canvas_host.get(key)
        with self._canvas_context(key):
            for identifier, ref in list(view.node_refs.items()):
                position = tuple(dpg.get_item_pos(identifier))
                old = view.view_positions.get(ref)
                if old and any(abs(position[index] - old[index]) > 1 for index in (0, 1)):
                    x, y = self._document_position(position)
                    self.document.move(ref, x, y)
                    view.view_positions[ref] = position
            view.content_token = self.canvas_host._content(self.workbench.views[key])

    def _clear_canvas_selection(self, **_):
        key = self._bound_canvas_id or self.workbench.focused_view_id
        self.canvas_host.get(key).selected_links.clear()
        self.canvas_host.apply_selection(key, [], intentional=True)
        self.document.select(None)
        self.selected_refs = []
        self._refresh(graph=False)

    @staticmethod
    def _canvas_input_active():
        # DPG 2.3.1 has no public is_any_item_active(). Public item states
        # include active only for widgets which support that state.
        return any(dpg.get_item_state(item).get('active', False) for item in dpg.get_all_items())


def bind_legacy_commands(cls):
    """Direct old Python methods bind focus; registered callbacks bind tokens."""
    names = ('_add', '_share', '_set_param', '_freeze', '_set_input', '_set_output',
             '_connect', '_disconnect', '_delete_selected', '_render_inspector', '_render_probe',
             '_probe_source', '_open_encapsulate', '_open_template', '_save_chip', '_prepare_chip_dialog',
             '_open_chip_tools', '_render_chip_tools', '_render_interface_rows', '_chip_action',
             '_interface_add_input', '_interface_add_output', '_interface_rename', '_interface_shape',
             '_interface_remove', '_public_apply', '_select_probe_output', '_socket_probe',
             '_library_import_path', '_library_save')
    for name in names:
        original = getattr(cls, name)
        @wraps(original)
        def scoped(self, *args, __original=original, **kwargs):
            if not self.canvas_host.views:
                return __original(self, *args, **kwargs)
            key = self._bound_canvas_id or self.workbench.focused_view_id
            with self._canvas_context(key, adopt=True):
                return __original(self, *args, **kwargs)
        setattr(cls, name, scoped)
    return cls
