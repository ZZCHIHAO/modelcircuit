"""Searchable native component selector, drag payloads and atomic placement."""
from copy import deepcopy
from dataclasses import dataclass
import json
import math

import dearpygui.dearpygui as dpg

from studio_core.component_catalog import build_component, list_components
from studio_core.component_types import ComponentBundle
from .chip_library import _closure, project_requirements, requires_adoption, resource_policy
from .local_components import LocalComponentIndex

BLUE = (100, 200, 255)
MUTED = (145, 154, 164)
RED = (240, 143, 134)
PAYLOAD_TYPE = 'MLFORGE_COMPONENT_V1'
LIBRARIES = {
    '基础组件库': ('layers', 'activation', 'normalization', 'math_fusion', 'shape_pool'),
    '深度学习模型库': ('backbone', 'transformer', 'model'),
    '传统机器学习模型库': ('classical',),
    '网络结构与融合库': ('functional',),
}
CATEGORIES = {'全部库': None, **{name: groups for name, groups in LIBRARIES.items()},
              '全部': None, '基础层': 'layers', '激活': 'activation', '归一化': 'normalization',
              '数学与融合': 'math_fusion', '形状与池化': 'shape_pool', '功能芯片': 'functional',
              '视觉主干': 'backbone', 'Transformer': 'transformer', '完整模型': 'model',
              '工程芯片': 'project', '本地芯片': 'local'}


@dataclass(frozen=True)
class ConfigRequest:
    serial: int
    identifier: str
    token: object
    widgets: tuple


@dataclass(frozen=True)
class PolicyRequest:
    serial: int
    bundle: object
    token: object
    position: object


@dataclass(frozen=True)
class ExecutionPolicyRequest:
    serial: int
    token: object
    policy: object


def screen_rect(item):
    """DPG child windows expose parent-relative pos; widgets expose rect_min."""
    state = dpg.get_item_state(item)
    if 'rect_min' in state:
        origin = state['rect_min']
    else:
        origin = list(dpg.get_item_pos(item))
        parent = dpg.get_item_info(item)['parent']
        if parent:
            base = screen_rect(parent)[0]
            origin = [origin[0] + base[0], origin[1] + base[1]]
    return tuple(origin), tuple(state.get('rect_size', (0, 0)))


def button_caption(label, width=108):
    """Keep names readable in the narrow library; full titles stay in tooltips."""
    size = dpg.get_text_size(label)
    # The initial palette is built before DPG setup; metrics become available
    # on the first rendered frame. Search/category callbacks use real metrics.
    if size is None or size[0] <= width - 12:
        return label
    text = label
    while text and dpg.get_text_size(text + '…')[0] > width - 12:
        text = text[:-1]
    return text + '…'


class CatalogPanel:
    def __init__(self, ui, *, model_registry=None):
        self.ui = ui
        self.model_registry = model_registry
        self.visible_ids = []
        self.config_widgets = {}
        self.configs = {}
        self.pending = None
        self.config_target = None
        self.selected_id = None
        self.category = None
        self._request_serial = 0
        self._config_request = None
        self._policy_request = None
        self._quick_owner = None
        self.local_index = LocalComponentIndex(ui.data_directory / 'chips')

    def callback(self, function, *args):
        def run(sender=None, app_data=None, user_data=None):
            try:
                return function(*args, app_data=app_data)
            except Exception as error:
                self.ui.log(str(error), '错误')
                if dpg.does_item_exist('native_catalog_error'):
                    dpg.set_value('native_catalog_error', str(error))
        return run

    def owned_callback(self, function, identifier):
        key = self.ui.workbench.focused_view_id
        token = self.ui.canvas_host.token(key) if key in self.ui.canvas_host.views else None
        def run(sender=None, app_data=None, user_data=None):
            if token is None or not self.ui.canvas_host.accept(token) or token.view_id != self.ui.workbench.focused_view_id:
                return False
            return self.callback(function, identifier, token)(sender, app_data, user_data)
        return run

    def build_filters(self):
        dpg.add_combo(items=[name for name in CATEGORIES if name != '全部'], default_value='全部库', width=-1,
                      tag='native_catalog_category', callback=self.callback(self._category))
        with dpg.group(horizontal=True):
            for identifier, label in (('Linear', 'Linear'), ('ReLU', 'ReLU'), ('Add', 'Add')):
                button = dpg.add_button(label=label, tag='native_catalog_quick_' + identifier,
                                        callback=lambda *args: False)
                self._drag(button, self.payload(identifier))

    def sync_quick_owners(self):
        key = self.ui.workbench.focused_view_id
        if key not in self.ui.canvas_host.views:
            return
        token = self.ui.canvas_host.token(key)
        if self._quick_owner == token:
            return
        self._quick_owner = token
        for identifier in ('Linear', 'ReLU', 'Add'):
            tag = 'native_catalog_quick_' + identifier
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, callback=self.owned_callback(self._quick, identifier), user_data=token)

    def build_dialogs(self):
        with dpg.window(tag='native_catalog_config', label='组件配置与端口', show=False, modal=True,
                        width=510, height=530, no_collapse=True):
            with dpg.child_window(tag='native_catalog_details', width=-1, height=445):
                pass
            dpg.add_text('', tag='native_catalog_error', color=RED, wrap=480)
            with dpg.group(horizontal=True):
                dpg.add_button(label='按此配置添加', tag='native_catalog_apply', callback=lambda *args: False)
                dpg.add_button(label='取消', tag='native_catalog_config_cancel', callback=lambda *args: False)
        with dpg.window(tag='native_catalog_policy', label='应用组件所需的执行策略', show=False, modal=True,
                        width=530, height=255, no_collapse=True):
            dpg.add_text('', tag='native_catalog_policy_text', wrap=490, color=BLUE)
            dpg.add_text('本次操作将策略和组件作为一步撤销。实验批大小保持当前配置；运行前请检查。',
                         tag='native_catalog_policy_footer', wrap=490, color=MUTED)
            with dpg.group(horizontal=True):
                dpg.add_button(label='应用策略并添加', tag='native_catalog_policy_apply', callback=lambda *args: False)
                dpg.add_button(label='取消', tag='native_catalog_policy_cancel', callback=lambda *args: False)

    def payload(self, identifier):
        key = ('project_definition_id' if identifier.startswith('project:') else
               'local_component_id' if identifier.startswith('local:') else 'component_id')
        value = identifier.split(':', 1)[1] if key == 'project_definition_id' else identifier
        return {'version': 1, key: value, 'config': deepcopy(self.configs.get(identifier, {}))}

    def _drag(self, parent, payload):
        with dpg.drag_payload(parent=parent, payload_type=PAYLOAD_TYPE, drag_data=payload):
            dpg.add_text('拖到目标画布放置')

    def _category(self, app_data=None):
        if app_data not in CATEGORIES:
            raise ValueError('未知组件分类')
        self.category = CATEGORIES.get(app_data)
        dpg.set_value('native_catalog_category', app_data)
        self.render(dpg.get_value('native_search'))

    def render(self, query=''):
        parent = 'native_palette'
        if not dpg.does_item_exist(parent):
            return
        dpg.delete_item(parent, children_only=True)
        self.visible_ids = []
        dpg.add_button(label='新建芯片模板', width=-1, parent=parent, callback=self.ui.guarded(self.ui._open_template))
        policy_button = dpg.add_button(label='硬件容量策略…', width=-1, parent=parent,
                                       tag='native_catalog_hardware_policy',
                                       callback=self.owned_callback(self.open_hardware_policy, 'hardware-v5'))
        with dpg.tooltip(policy_button):
            dpg.add_text('将当前工程切换到按硬件容量运行，解除历史固定参数量阈值。\n修改可撤销，并会改变 checkpoint 兼容身份；实验设置保持当前值。', wrap=350)
        if not self.local_index.initialized:
            self.local_index.refresh()
        if self.category in (None, 'local'):
            button = dpg.add_button(label='刷新本地芯片', width=-1, parent=parent,
                                    tag='native_catalog_refresh_local', callback=self.callback(self.refresh_local))
            with dpg.tooltip(button):
                dpg.add_text('从此目录读取有限 JSON 架构包，不加载 Python 或权重。\n' + str(self.local_index.directory), wrap=350)
        static = [] if self.category in ('project', 'local') else list_components(
            str(query or ''), registry=self.model_registry, include_tensor=True)
        if isinstance(self.category, tuple):
            static = [spec for spec in static if spec.category in self.category]
        elif self.category is not None:
            static = [spec for spec in static if spec.category == self.category]
        for title, categories in LIBRARIES.items():
            specs = [spec for spec in static if spec.category in categories]
            if not specs:
                continue
            section = dpg.add_collapsing_header(label=title, parent=parent, default_open=True)
            for spec in specs:
                self._render_builtin(spec, section)
        for spec in static:
            if spec.id not in self.visible_ids:
                self._render_builtin(spec, parent)
        self._render_user_components(parent, query)

    def _render_builtin(self, spec, parent):
        self.visible_ids.append(spec.id)
        with dpg.group(horizontal=True, parent=parent):
            if spec.category == 'classical':
                label = spec.label + ' (' + spec.aliases[0] + ')'
            else:
                label = spec.label
            button = dpg.add_button(label=button_caption(label), width=108,
                                    tag='native_catalog_add_' + spec.id, callback=self.owned_callback(self._quick, spec.id))
            self._drag(button, self.payload(spec.id))
            dpg.add_button(label='配置', width=42, tag='native_catalog_config_' + spec.id,
                           callback=self.owned_callback(self.open_config, spec.id))
        with dpg.tooltip(button):
            dpg.add_text(spec.id + ' · ' + spec.description, wrap=350)
            dpg.add_text('输入 ' + ', '.join(spec.input_ports) + ' → 输出 ' + ', '.join(spec.output_ports))
            dpg.add_text('默认 ' + json.dumps(spec.config_defaults, ensure_ascii=False), wrap=350)

    def _render_user_components(self, parent, query):
        if self.category in (None, 'local'):
            for spec in self.local_index.list_components(str(query or '')):
                self.visible_ids.append(spec.id)
                with dpg.group(horizontal=True, parent=parent):
                    button = dpg.add_button(label=button_caption(spec.label), width=108, tag='native_catalog_add_' + spec.id,
                                            callback=self.owned_callback(self._quick, spec.id))
                    self._drag(button, self.payload(spec.id))
                    dpg.add_button(label='配置', width=42, tag='native_catalog_config_' + spec.id,
                                   callback=self.owned_callback(self.open_config, spec.id))
                with dpg.tooltip(button):
                    dpg.add_text(spec.label + '\n' + spec.description, wrap=350)
                    dpg.add_text('输入 ' + ', '.join(spec.input_ports) + ' → 输出 ' + ', '.join(spec.output_ports))
                    dpg.add_text('文件 ' + ', '.join(spec.source_names), wrap=350)
            for message in self.local_index.diagnostics[:5]:
                dpg.add_text(message, color=RED, wrap=150, parent=parent)
            if len(self.local_index.diagnostics) > 5:
                dpg.add_text('还有' + str(len(self.local_index.diagnostics) - 5) + '项库提示，请查看控制台。', color=MUTED, wrap=150, parent=parent)
            if self.category == 'local' and not self.visible_ids:
                dpg.add_text('可在“接口与芯片”中保存版本，或复制 JSON 架构包到本地库后刷新。', wrap=150, color=MUTED, parent=parent)
        if self.category in (None, 'project'):
            for definition in self.ui.document.graph.get('definitions', []):
                if str(query or '').casefold() not in (definition['id'] + ' ' + definition.get('name', '')).casefold():
                    continue
                identifier = 'project:' + definition['id']
                self.visible_ids.append(identifier)
                button = dpg.add_button(label='芯片 · ' + definition.get('name', definition['id']), width=-1, parent=parent,
                                        callback=self.owned_callback(self.open_config, identifier))
                self._drag(button, self.payload(identifier))
                with dpg.tooltip(button):
                    dpg.add_text('当前工程定义；新建独立实例。内部编辑影响全部定义引用。', wrap=350)
        if not self.visible_ids:
            dpg.add_text('没有匹配的可用组件', color=MUTED, parent=parent)

    def refresh_local(self, **_):
        # Explicit refresh ends configuration transactions, so a queued Apply
        # cannot place the previous snapshot after the user has refreshed it.
        self.invalidate_requests()
        for tag in ('native_catalog_config', 'native_catalog_policy'):
            if dpg.does_item_exist(tag):
                dpg.hide_item(tag)
        self.local_index.refresh()
        current = {spec.id for spec in self.local_index.list_components()}
        self.configs = {key: value for key, value in self.configs.items() if not key.startswith('local:') or key in current}
        self.render(dpg.get_value('native_search'))
        for message in self.local_index.diagnostics:
            self.ui.log(message, '提示')
        self.ui.log('本地芯片库已刷新：' + str(len(current)) + '个架构。', '完成')

    def _quick(self, identifier, token, **_):
        return self.place(self.payload(identifier), token)

    def open_hardware_policy(self, identifier, token, **_):
        if not self.ui.canvas_host.accept(token):
            return False
        self.cancel_config()
        graph = self.ui.document.graph
        current = {'resource_profile': graph.get('resource_profile')}
        if current['resource_profile'] in ('vision-v4', 'hardware-v5'):
            current['resource_batch_bound'] = graph['resource_batch_bound']
        policy = {'resource_profile': 'hardware-v5', 'resource_batch_bound': current.get('resource_batch_bound', 64)}
        self._request_serial += 1
        request = ExecutionPolicyRequest(self._request_serial, token, policy)
        self._policy_request = request
        self.pending = None
        dpg.configure_item('native_catalog_policy_apply', label='应用硬件容量策略',
                           callback=self.callback(self.confirm_policy, request), user_data=request)
        cancel = self.callback(self.cancel_policy, request)
        dpg.configure_item('native_catalog_policy_cancel', callback=cancel, user_data=request)
        dpg.configure_item('native_catalog_policy', label='当前工程执行策略', on_close=cancel)
        dpg.set_value('native_catalog_policy_text', '当前：' + self.policy_description(current)
                      + '\n变更后：' + self.policy_description(policy)
                      + '\n此次只修改工程执行策略，可撤销。原有 checkpoint 身份将改变。')
        dpg.set_value('native_catalog_policy_footer', '本次只修改执行策略，可单独撤销。实验批大小保持当前配置；运行前请检查。')
        dpg.show_item('native_catalog_policy')
        return True

    def open_config(self, identifier, token=None, **_):
        self.selected_id = identifier
        self.config_target = token or self.ui.canvas_host.token(self.ui.workbench.focused_view_id)
        self.config_widgets = {}
        dpg.delete_item('native_catalog_details', children_only=True)
        parent = 'native_catalog_details'
        if identifier.startswith('project:'):
            definition = self.ui._definition(identifier.split(':', 1)[1])
            if definition is None:
                raise ValueError('目标已变化')
            dpg.add_text(definition.get('name', definition['id']), color=BLUE, parent=parent)
            dpg.add_text('当前工程芯片 · 创建独立实例；内部定义保持共享。', wrap=460, parent=parent)
            graph = definition['graph']
            dpg.add_text('输入 ' + ', '.join(p['id'] for p in graph['inputs']) + ' → 输出 ' + ', '.join(p['id'] for p in graph['outputs']), wrap=460, parent=parent)
            for parameter in definition.get('public_parameters', []):
                self._config_field(parent, parameter['id'], parameter.get('default'))
        else:
            local = identifier.startswith('local:')
            matches = self.local_index.list_components() if local else list_components(registry=self.model_registry, include_tensor=True)
            spec = next((item for item in matches if item.id == identifier), None)
            if spec is None:
                raise ValueError('组件已变化，请刷新并重新选择。')
            dpg.add_text(spec.label + ' · ' + spec.id, color=BLUE, parent=parent)
            dpg.add_text(spec.description, wrap=460, parent=parent)
            dpg.add_text('输入 ' + ', '.join(spec.input_ports) + ' → 输出 ' + ', '.join(spec.output_ports), wrap=460, parent=parent)
            if not local and spec.category == 'classical':
                dpg.add_text('CPU 拟合与预测 · 适用于单估计器表格实验', wrap=460, color=MUTED, parent=parent)
            else:
                dpg.add_text('可展开：' + ('是' if local or spec.capabilities['expandable'] else '否') + ' · 支持前向 / 训练 / Python 往返 · 无预训练权重', wrap=460, color=MUTED, parent=parent)
            if local:
                dpg.add_text('版本快照来自 ' + ', '.join(spec.source_names) + '；文件变更后请刷新库。', wrap=460, color=MUTED, parent=parent)
            defaults = {**spec.config_defaults, **self.configs.get(identifier, {})}
            for name, value in defaults.items():
                self._config_field(parent, name, value)
            if 'resource_batch_bound' in defaults:
                dpg.add_text('批次上限是本次结构契约；所需执行策略不匹配时将先请求确认。', color=BLUE, wrap=460, parent=parent)
        dpg.set_value('native_catalog_error', '')
        self._request_serial += 1
        request = ConfigRequest(self._request_serial, identifier, self.config_target, tuple(self.config_widgets.items()))
        self._config_request = request
        dpg.configure_item('native_catalog_apply', callback=self.callback(self._apply_config, request), user_data=request)
        cancel = self.callback(self.cancel_config, request)
        dpg.configure_item('native_catalog_config_cancel', callback=cancel, user_data=request)
        dpg.configure_item('native_catalog_config', on_close=cancel)
        dpg.show_item('native_catalog_config')

    def _config_field(self, parent, name, value):
        if type(value) is bool:
            widget = dpg.add_checkbox(label=name, default_value=value, parent=parent)
        else:
            widget = dpg.add_input_text(label=name, default_value=json.dumps(value, ensure_ascii=False), width=205, parent=parent)
        self.config_widgets[name] = widget

    def _apply_config(self, request, **_):
        if self._config_request is not request or not self.ui.canvas_host.accept(request.token):
            return False
        config = {}
        for name, widget in request.widgets:
            value = dpg.get_value(widget)
            config[name] = value if type(value) is bool else json.loads(value, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('参数必须为有限 JSON')))
        payload = self.payload(request.identifier)
        payload['config'] = config
        # Validate first; preserve the editable dialog on invalid configuration.
        self._bundle(payload)
        self.configs[request.identifier] = config
        result = self.place(payload, request.token)
        if result or self.pending:
            self.cancel_config(request)
            self.render(dpg.get_value('native_search'))

    def cancel_config(self, request=None, **_):
        if request is not None and self._config_request is not request:
            return False
        self._config_request = None
        self.config_target = None
        dpg.hide_item('native_catalog_config')

    def invalidate_requests(self):
        self._config_request = self._policy_request = None
        self.config_target = self.pending = None

    def _bundle(self, payload):
        if type(payload) is not dict or payload.get('version') != 1 or type(payload.get('version')) is not int:
            raise ValueError('无效组件载荷')
        keys = set(payload)
        if keys not in ({'version', 'component_id', 'config'}, {'version', 'project_definition_id', 'config'},
                        {'version', 'local_component_id', 'config'}):
            raise ValueError('无效组件载荷字段')
        encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False)
        if len(encoded) > 65536 or type(payload['config']) is not dict:
            raise ValueError('组件配置超过限制或不是对象')
        if 'component_id' in payload:
            return build_component(payload['component_id'], payload['config'], registry=self.model_registry)
        if 'local_component_id' in payload:
            return self.local_index.build(payload['local_component_id'], payload['config'])
        identifier = payload['project_definition_id']
        definitions = {item['id']: item for item in self.ui.document.graph.get('definitions', [])}
        if type(identifier) is not str or identifier not in definitions:
            raise ValueError('目标已变化')
        closure = _closure(identifier, definitions)
        return ComponentBundle({'kind': 'chip', 'definition_id': identifier, 'params': deepcopy(payload['config'])},
                               [deepcopy(definitions[key]) for key in closure], project_requirements(self.ui.document.graph))

    def drop_callback(self, token):
        def dropped(sender=None, app_data=None, user_data=None):
            return self.place(app_data, token, screen_point=dpg.get_mouse_pos(local=False))
        return dropped

    def drop_position(self, token, point):
        if not self.ui.canvas_host.accept(token):
            raise ValueError('目标已变化')
        if type(point) not in (tuple, list) or len(point) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
            raise ValueError('无效放置位置')
        view = self.ui.canvas_host.get(token.view_id)
        # DPG 2.3 reports node_editor rect_min/rect_max as [0, 0]. Its
        # enclosing registered drop child has a measurable parent-relative
        # rectangle and is the actual native drop target.
        origin, size = screen_rect(view.container_tag)
        if not view.visible or not all(origin[i] <= point[i] <= origin[i] + size[i] for i in (0, 1)):
            raise ValueError('请放到可见目标画布内')
        # Native node rect and node-editor coordinates share the current pan.
        # Their measured difference is the actual transform, never a guessed
        # fixed toolbar offset. No public DPG zoom API is assumed.
        for node in view.ref_nodes.values():
            state = dpg.get_item_state(node)
            if 'rect_min' in state and state.get('rect_size', [0, 0])[0] > 0:
                rect = state['rect_min']
                local = dpg.get_item_pos(node)
                x, y = self.ui._document_position((point[0] - rect[0] + local[0], point[1] - rect[1] + local[1]))
                return {'x': x, 'y': y}
        raise ValueError('画布几何尚未就绪，请点击添加或等待下一帧')

    def place(self, payload, token, screen_point=None):
        try:
            if not self.ui.canvas_host.accept(token):
                return False
            bundle = self._bundle(payload)
            position = self.drop_position(token, screen_point) if screen_point is not None else None
            graph = self.ui.document.graph
            if requires_adoption(graph, bundle.requirements):
                requirement = resource_policy(bundle.requirements)
                current = resource_policy(graph)
                upgrade = bundle.requirements.get('schema_version') == 2 and graph['schema_version'] == 1
                self.pending = (deepcopy(bundle), token, position)
                self._request_serial += 1
                request = PolicyRequest(self._request_serial, deepcopy(bundle), token, deepcopy(position))
                self._policy_request = request
                dpg.configure_item('native_catalog_policy_apply', label='升级并添加' if upgrade else '应用策略并添加', callback=self.callback(self.confirm_policy, request), user_data=request)
                cancel = self.callback(self.cancel_policy, request)
                dpg.configure_item('native_catalog_policy_cancel', callback=cancel, user_data=request)
                dpg.configure_item('native_catalog_policy', label='升级为张量工作区' if upgrade else '应用组件所需的执行策略', on_close=cancel)
                dpg.set_value('native_catalog_policy_text', '目标：' + self.ui.workbench_panel.title(token.view_id)
                              + '\n当前：' + self.policy_description(current)
                              + '\n组件要求：' + self.policy_description(requirement)
                              + ('\n升级至 schema 2 / tensor-v1；为现有浮点接口补全轴名。' if upgrade else '\n应用后，本工程将采用上述执行策略。'))
                dpg.set_value('native_catalog_policy_footer', '本次操作将策略和组件作为一步撤销。实验批大小保持当前配置；运行前请检查。')
                dpg.show_item('native_catalog_policy')
                return False
            return self._insert(bundle, token, position)
        except (ValueError, TypeError, KeyError, RecursionError) as error:
            self.ui.log(str(error), '提示')
            return False

    @staticmethod
    def policy_description(policy):
        profile = policy.get('resource_profile')
        if profile == 'hardware-v5':
            return '按所选硬件容量评估，无固定参数量上限（批次契约 ' + str(policy['resource_batch_bound']) + '）'
        if profile == 'vision-v4':
            return '视觉实验策略（批次上限 ' + str(policy['resource_batch_bound']) + '）'
        if profile == 'classification-v2':
            return '分类实验策略（沿用原有资源限制）'
        return '基础实验策略（沿用原有资源限制）'

    def _insert(self, bundle, token, position, apply_policy=False):
        if not self.ui.canvas_host.accept(token):
            return False
        self.ui.workbench_panel.activate_view(token.view_id)
        with self.ui._canvas_context(token.view_id):
            ref = self.ui.document.add_component(bundle, position, apply_policy=apply_policy)
            self.ui.selected_refs = [ref]
            self.ui.workbench.views[token.view_id].selected_refs = [ref]
            self.ui._refresh()
        return True

    def confirm_policy(self, request, **_):
        if self._policy_request is not request:
            return False
        self._policy_request = None
        self.pending = None
        dpg.hide_item('native_catalog_policy')
        if isinstance(request, ExecutionPolicyRequest):
            if not self.ui.canvas_host.accept(request.token):
                return False
            self.ui.workbench_panel.activate_view(request.token.view_id)
            changed = self.ui.document.set_execution_policy(request.policy)
            self.ui._refresh()
            if changed:
                self.ui.log('当前工程已采用硬件容量策略；运行前将检查所选设备可用内存。', '完成')
            return changed
        return self._insert(request.bundle, request.token, request.position, apply_policy=True)

    def cancel_policy(self, request=None, **_):
        if request is not None and self._policy_request is not request:
            return False
        self._policy_request = None
        self.pending = None
        dpg.hide_item('native_catalog_policy')
