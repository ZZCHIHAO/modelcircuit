"""Native tab strips and navigation over the shared, persistent canvas host."""
from copy import deepcopy
import dearpygui.dearpygui as dpg

from .view_document import resolve_view
from .workbench_state import ScopeTarget, ViewState

BLUE = (100, 200, 255)
MUTED = (145, 154, 164)


class WorkbenchPanel:
    def __init__(self, ui):
        self.ui = ui
        self._signature = None

    def _callback(self, function, *args):
        def callback(sender=None, app_data=None, user_data=None):
            try:
                return function(*args, app_data=app_data)
            except (ValueError, KeyError) as error:
                self.ui.log(str(error), '提示')
        return callback

    def build(self):
        for pane in ('primary', 'secondary'):
            with dpg.group(tag='native_tabs_' + pane, parent=self.ui.canvas_parent_by_pane[pane], pos=(2, 0)):
                pass

    def title(self, view_id):
        view = self.ui.workbench.views[view_id]
        if view.target.definition_id is None:
            return '主模型'
        definition = self.ui._definition(view.target.definition_id)
        return (definition or {}).get('name', view.target.definition_id)

    def sync(self):
        state = self.ui.workbench
        signature = (state.layout, state.focused_view_id, tuple(state.active_by_pane.items()),
                     tuple((key, view.pane, tuple(view.call_path), self.title(key), self.ui.canvas_host.token(key))
                           for key, view in state.views.items() if key in self.ui.canvas_host.views))
        if signature == self._signature:
            return
        self._signature = signature
        # A view moving secondary -> primary may retain the same tab ID.
        # Remove both old strips before creating either destination strip.
        for pane in ('primary', 'secondary'):
            parent = 'native_tabs_' + pane
            if dpg.does_item_exist(parent):
                dpg.delete_item(parent, children_only=True)
        for pane in ('primary', 'secondary'):
            parent = 'native_tabs_' + pane
            if not dpg.does_item_exist(parent):
                continue
            keys = [key for key, view in state.views.items() if view.pane == pane]
            active = state.active_by_pane[pane]
            if not keys:
                with dpg.group(tag='native_empty_' + pane, parent=parent):
                    dpg.add_text('此栏未打开图', color=MUTED)
                    dpg.add_button(label='在此栏打开芯片', tag='native_empty_open_' + pane,
                                   callback=self._callback(self._open_empty, pane))
                continue
            tabs = 'native_tabbar_' + pane
            with dpg.tab_bar(tag=tabs, parent=parent, tab_list_popup_button=True,
                             no_scrolling_buttons=False, draw_selected_overline=True,
                             callback=self._callback(self._tab_selected)):
                for key in keys:
                    title = self.title(key)
                    with dpg.tab(tag='native_tab_' + key, label=title[:14], order_mode=dpg.mvTabOrder_Fixed):
                        pass
                    with dpg.tooltip('native_tab_' + key):
                        dpg.add_text(title + (' · 正在编辑' if key == state.focused_view_id else ''))
            if active:
                dpg.set_value(tabs, dpg.get_alias_id('native_tab_' + active))
            labels = [self.title(key) + ' [' + key + ']' for key in keys]
            with dpg.group(horizontal=True, parent=parent):
                dpg.add_combo(items=labels, default_value=labels[keys.index(active)] if active in keys else labels[0],
                              width=110, tag='native_view_picker_' + pane,
                              callback=self._callback(self._pick, tuple(zip(labels, keys))))
                dpg.add_button(label='移栏', tag='native_view_move_' + pane, enabled=active != 'root',
                               callback=self._view_action(self._move_view, active, pane))
                dpg.add_button(label='关闭', tag='native_view_close_' + pane, enabled=active != 'root',
                               callback=self._view_action(self.close_view, active, pane))
                dpg.add_text('编辑' if pane == state.focused_pane else '对照', color=BLUE if pane == state.focused_pane else MUTED)

    def _pick(self, pairs, app_data=None):
        key = dict(pairs).get(app_data)
        if key:
            self.activate_view(key)

    def _tab_selected(self, app_data=None):
        for key in self.ui.workbench.views:
            if dpg.does_item_exist('native_tab_' + key) and dpg.get_alias_id('native_tab_' + key) == app_data:
                return self.activate_view(key)

    def _open_empty(self, pane, **_):
        candidates = [item['id'] for item in self.ui.document.graph.get('definitions', [])]
        if not candidates:
            self.ui._open_template()
            return
        key = self.open_definition(candidates[0])
        self.ui.workbench.move_to_pane(key, pane)
        self.activate_view(key)

    def _view_action(self, function, view_id, pane):
        token = self.ui.canvas_host.token(view_id)
        def callback(sender=None, app_data=None, user_data=None):
            if (not self.ui.canvas_host.accept(token)
                    or self.ui.workbench.active_by_pane[pane] != view_id
                    or self.ui.workbench.views[view_id].pane != pane):
                return False
            return function(view_id)
        return callback

    def _move_view(self, key):
        state = self.ui.workbench
        pane = state.views[key].pane
        if key and key != 'root':
            if state.layout == 'single':
                state.set_layout('columns')
            state.move_to_pane(key, 'secondary' if pane == 'primary' else 'primary')
            self.activate_view(key)

    def activate_view(self, view_id, **_):
        self.ui.workbench.activate(view_id)
        # A navigation callback may originate inside a temporarily bound canvas.
        if self.ui._bound_canvas_id is not None:
            self.ui._bound_canvas_id = view_id
        self.ui.canvas_host.sync(self.ui.canvas_parent_by_pane)
        self.ui._focus_document(view_id)
        self.ui._canvas_aliases(view_id)
        self.ui._canvas_synced()
        self.sync()
        return view_id

    def open_call(self, call_path, pane='primary'):
        trail = self.ui.document._resolve_call_path(call_path)
        if not trail:
            return self.activate_view('root')
        key = self.ui.workbench.open_definition(trail[-1]['id'], call_path, pane=pane)
        return self.activate_view(key)

    def open_definition(self, definition_id):
        if self.ui._definition(definition_id) is None:
            raise ValueError('目标已变化')
        key = self.ui.workbench.open_definition(definition_id)
        return self.activate_view(key)

    def set_layout(self, mode, **_):
        self.ui.workbench.set_layout(mode)
        self.ui.canvas_host.sync(self.ui.canvas_parent_by_pane)
        self.sync()

    def close_view(self, view_id):
        if view_id == 'root':
            self.ui.log('主模型页签始终保留。', '提示')
            return
        self.ui.workbench.close(view_id)
        self.ui.canvas_host.sync(self.ui.canvas_parent_by_pane)
        self.sync()

    def focus_location(self, location):
        try:
            if type(location) is not dict or set(location) != {'definition_id', 'call_path', 'node_ref', 'field'}:
                raise ValueError('目标已变化')
            target, path, ref, field = (location[name] for name in ('definition_id', 'call_path', 'node_ref', 'field'))
            view = ViewState('location', ScopeTarget(target), path, [], None, 'primary')
            scope, _ = resolve_view(self.ui.document, view)
            if ref is not None:
                if type(ref) is not str or ':' not in ref:
                    raise ValueError('目标已变化')
                kind, identifier = ref.split(':', 1)
                collection = {'call': 'calls', 'input': 'inputs', 'output': 'outputs'}.get(kind)
                matches = [item for item in scope.get(collection, []) if item['id'] == identifier] if collection else []
                if len(matches) != 1:
                    raise ValueError('目标已变化')
                item = matches[0]
                fields = set(item)
                if kind == 'call':
                    instance = next(i for i in scope['instances'] if i['id'] == item['instance_id'])
                    fields.update(instance.get('params', {}))
                    definition = self.ui._definition(instance.get('definition_id'))
                    fields.update(p['id'] for p in (definition or {}).get('public_parameters', []))
                if field is not None and (type(field) is not str or field not in fields):
                    raise ValueError('目标已变化')
            elif field is not None:
                raise ValueError('目标已变化')
            key = self.open_call(path) if path else self.open_definition(target) if target else self.activate_view('root')
            if ref:
                self.ui.canvas_host.apply_selection(key, [ref], intentional=True)
                self.ui._focus_document(key)
                self.ui._render_inspector()
                self.ui._render_probe()
                if field in self.ui.param_widgets:
                    dpg.focus_item(self.ui.param_widgets[field])
            return True
        except (ValueError, KeyError, StopIteration, TypeError):
            self.ui.log('目标已变化；请重新选择有效图位置。', '提示')
            return False

    def location_callback(self, location, *, document_epoch=None):
        epoch = self.ui.canvas_host.document_epoch if document_epoch is None else document_epoch
        saved = deepcopy(location)
        def callback(sender=None, app_data=None, user_data=None):
            if epoch != self.ui.canvas_host.document_epoch:
                self.ui.log('目标已变化；原工程位置已失效。', '提示')
                return False
            return self.focus_location(saved)
        return callback
