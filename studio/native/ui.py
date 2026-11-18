"""Native workbench. All DearPyGui operations stay on the GUI thread.

The document owns model semantics; the RunManager owns subprocesses. This module
only adapts those two contracts to the original ML Forge desktop layout.
"""
from __future__ import annotations

import json
import io
import math
import time
from array import array
from copy import deepcopy
from pathlib import Path

import dearpygui.dearpygui as dpg
from PIL import Image

from .data_panel import ExperimentControls
from .chip_panel import ChipControls
from .experiment_panel import progress_points, result_view
from .workspace_state import read_workspace_full, save_workspace, workspace_payload
from .workbench_state import WorkbenchState
from .canvas_view import CanvasHost, CanvasToken
from .canvas_commands import CanvasCommands, bind_legacy_commands
from .workbench_panel import WorkbenchPanel
from .catalog_panel import CatalogPanel, screen_rect
from studio_core.operators import operator_spec


BLUE = (100, 200, 255)
MINT = (93, 216, 192)
MUTED = (145, 154, 164)
AMBER = (235, 190, 110)
RED = (240, 143, 134)
TERMINAL = {"succeeded", "failed", "cancelled"}
KINDS = ("Linear", "ReLU", "GELU", "Add", "Concat", "LayerNorm", "Dropout", "Conv2d", "MaxPool2d", "AdaptiveAvgPool2d", "Flatten")
ARGUMENTS = {"Add": ("a", "b"), "Concat": ("a", "b")}
PARAMETERS = {"in_features": "输入维度", "hidden_features": "特征维度", "out_features": "输出维度", "normalized_shape": "归一化形状", "dim": "拼接维度", "p": "失活概率", "in_channels": "输入通道", "out_channels": "输出通道", "kernel_size": "卷积/池化核", "stride": "步长", "padding": "填充", "output_size": "输出空间尺寸", "start_dim": "展平起始维", "end_dim": "展平结束维"}
RUN_LABELS = {"queued": "等待启动", "preparing": "准备实验", "running": "正在训练", "cancel_requested": "正在停止", "succeeded": "实验完成", "failed": "实验失败", "cancelled": "已停止"}
VIEW_X_SCALE = 0.60
VIEW_Y_SCALE = 0.82
VIEW_X_OFFSET = 4
VIEW_Y_OFFSET = 10


def _shape(value):
    return "[" + ", ".join("N" if item is None else str(item) for item in value) + "]" if isinstance(value, (list, tuple)) else "待校验"


def _number(value):
    return f"{value:.6g}" if isinstance(value, (float, int)) and math.isfinite(value) else "—"


@bind_legacy_commands
class NativeWorkbench(CanvasCommands, ExperimentControls, ChipControls):
    def __init__(self, document, manager, data_directory, resource_directory, experiment=None, *, workbench_payload=None, model_registry=None):
        self.document = document
        self.manager = manager
        self.data_directory = Path(data_directory)
        self.resource_directory = Path(resource_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.node_refs = {}
        self.ref_nodes = {}
        self.source_attributes = {}
        self.target_attributes = {}
        self.link_targets = {}
        self.view_positions = {}
        self.selected_refs = []
        self.last_selection = ()
        self.generation = 0
        self.report = {}
        self.messages = []
        self._message_locations = []
        self._protected_save_paths = {Path(path).resolve() for path in getattr(document, 'workspace_protected_paths', ())}
        self.run_record = None
        self.run_config = None
        self.run_key = None
        self.result = None
        self.result_key = None
        self.progress_history = {}
        self.last_event = 0
        self.last_poll = 0.0
        self.last_draft = 0.0
        self.draft_signature = ""
        self.last_scope = None
        self.last_size = None
        self.python_code = ""
        self.python_initialized = False
        self.python_dirty = False
        self.closed = False
        self.node_themes = {}
        self.param_widgets = {}
        self._context_created = False
        self._built = False
        self._bound_canvas_id = None
        self._presented_canvas_focus = None
        self._shared_render_serial = 0
        self._modal_tokens = {}
        self.workbench = (WorkbenchState() if workbench_payload is None
                          else WorkbenchState.from_payload(workbench_payload, document))
        for warning in [*getattr(document, 'workspace_load_warnings', ()), *self.workbench.warnings]:
            self.messages.append((time.strftime('%H:%M:%S'), '提示', warning))
            self._message_locations.append(None)
        self.canvas_parent_by_pane = {'primary': 'native_canvas_primary', 'secondary': 'native_canvas_secondary'}
        self.canvas_host = CanvasHost(document, self.workbench, render=self._draw_canvas,
                                      dispatch=self._dispatch_canvas, log=self.log,
                                      on_sync=self._canvas_synced, themes=self.node_themes)
        self.workbench_panel = WorkbenchPanel(self)
        self.catalog_panel = CatalogPanel(self, model_registry=model_registry)
        self.canvas_host.catalog_panel = self.catalog_panel
        self._init_experiment(experiment)
        self._init_chip_controls()
        self._restore_probe_choices()

    def _restore_probe_choices(self):
        for key, view in self.workbench.views.items():
            if view.probe_port and view.selected_refs and view.selected_refs[0].startswith('call:'):
                self.probe_selection[(key, tuple(view.call_path), view.selected_refs[0][5:])] = view.probe_port

    def _restore_workbench(self, payload):
        restored = WorkbenchState() if payload is None else WorkbenchState.from_payload(payload, self.document)
        # Keep CanvasHost's shared state object and its document epoch. Reset
        # has already invalidated every queued callback from the old project.
        self.workbench.__dict__.update(restored.__dict__)
        self.probe_selection.clear()
        self._restore_probe_choices()
        for warning in restored.warnings:
            self.log(warning, '提示')
        if self._context_created:
            self.canvas_host.sync(self.canvas_parent_by_pane)

    def guarded(self, function, *bound):
        token = self.canvas_host.token(self._bound_canvas_id) if self._bound_canvas_id in self.canvas_host.views else None
        def callback(sender=None, app_data=None, user_data=None):
            try:
                event_token = user_data if isinstance(user_data, CanvasToken) else token
                if event_token is not None:
                    if not self.canvas_host.accept(event_token):
                        return
                    with self._canvas_context(event_token.view_id):
                        return function(*bound, sender=sender, app_data=app_data, user_data=user_data)
                return function(*bound, sender=sender, app_data=app_data, user_data=user_data)
            except Exception as error:
                self.log(str(error), "错误")
                for modal, label in (("native_chip_tools", "native_chip_tool_error"), ("native_chip_dialog", "native_chip_error"), ("native_data", "native_data_error")):
                    if dpg.does_item_exist(modal) and dpg.is_item_shown(modal):
                        dpg.set_value(label, str(error))
                if dpg.does_item_exist("native_status"):
                    dpg.set_value("native_status", "操作失败：" + str(error)[:160])
        callback._canvas_dynamic = token is None
        return callback

    def log(self, message, level="信息", *, location=None):
        self.messages.append((time.strftime("%H:%M:%S"), level, str(message)))
        self._message_locations.append(None if location is None else (deepcopy(location), self.canvas_host.document_epoch))
        self.messages = self.messages[-300:]
        self._message_locations = self._message_locations[-300:]
        if dpg.does_item_exist("native_console_content"):
            dpg.delete_item("native_console_content", children_only=True)
            location_index = 0
            for (stamp, kind, text), target in zip(self.messages, self._message_locations):
                color = RED if kind == "错误" else AMBER if kind == "提示" else BLUE if kind == "完成" else MUTED
                dpg.add_text(f"[{stamp}] [{kind}] {text}", color=color, parent="native_console_content")
                if target is not None:
                    dpg.add_button(label='定位图位置', tag='native_error_location_' + str(location_index),
                                   small=True, parent='native_console_content',
                                   callback=self.workbench_panel.location_callback(target[0], document_epoch=target[1]))
                    location_index += 1
            dpg.set_y_scroll("native_console_scroll", -1.0)

    def _font(self):
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
        if font_path.exists():
            with dpg.font_registry():
                with dpg.font(str(font_path), 15) as font:
                    # 2.3 automatically loads Chinese glyphs from the font.
                    # Explicit ranges are only needed by earlier DPG releases.
                    version = tuple(int(part) for part in dpg.get_dearpygui_version().split(".")[:2])
                    if version < (2, 3):
                        dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
                        dpg.add_font_range(0x2000, 0x206F)
                        dpg.add_font_chars([0x2192])
                dpg.bind_font(font)
        else:
            self.log("Chinese font missing: C:/Windows/Fonts/msyh.ttc. System default font is active; Chinese text may be unavailable.", "提示")

    def _theme(self):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                for item, color in ((dpg.mvThemeCol_WindowBg, (31, 31, 33)), (dpg.mvThemeCol_ChildBg, (36, 36, 38)), (dpg.mvThemeCol_FrameBg, (49, 49, 53)), (dpg.mvThemeCol_Button, (55, 55, 60)), (dpg.mvThemeCol_ButtonHovered, (65, 87, 105)), (dpg.mvThemeCol_ButtonActive, (62, 105, 133)), (dpg.mvThemeCol_Header, (49, 53, 60)), (dpg.mvThemeCol_HeaderHovered, (57, 76, 91)), (dpg.mvThemeCol_Text, (218, 223, 228)), (dpg.mvThemeCol_Border, (72, 73, 78)), (dpg.mvThemeCol_CheckMark, BLUE), (dpg.mvThemeCol_TabActive, (45, 92, 126))):
                    dpg.add_theme_color(item, color)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 0)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 1)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 5, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 7, 5)
        dpg.bind_theme(theme)
        with dpg.theme() as self.canvas_container_theme:
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0)
        for name, color in (("chip", (55, 132, 183)), ("primitive", (73, 130, 181)), ("activation", (184, 128, 60)), ("tensor", (48, 160, 145)), ("selected", (55, 148, 192))):
            with dpg.theme() as node_theme:
                with dpg.theme_component(dpg.mvNode):
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBar, color, category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, tuple(min(255, c + 22) for c in color), category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, tuple(min(255, c + 35) for c in color), category=dpg.mvThemeCat_Nodes)
                    dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, (54, 55, 59), category=dpg.mvThemeCat_Nodes)
            self.node_themes[name] = node_theme

    def _build(self):
        with dpg.window(tag="native_main", no_title_bar=True, no_move=True, no_resize=True, no_scrollbar=True, no_collapse=True):
            with dpg.group(horizontal=True, tag="native_toolbar", pos=(9, 8)):
                image = self.resource_directory / "icon.png"
                if image.exists():
                    # DPG's stb filename loader returns None for some Windows
                    # Unicode paths. Decode the actual resource bytes instead.
                    with Image.open(io.BytesIO(image.read_bytes())) as decoded:
                        rgba = decoded.convert("RGBA")
                        width, height = rgba.size
                        data = array("f", (value / 255.0 for value in rgba.tobytes()))
                    with dpg.texture_registry():
                        texture = dpg.add_static_texture(width, height, data, tag="native_logo_texture")
                    dpg.add_image(texture, width=24, height=24, tag="native_logo")
                dpg.add_text("ModelCircuit Studio", color=BLUE)
                dpg.add_spacer(width=10)
                for label, function in (("打开 JSON", self._open_json), ("保存 JSON", self._save_json), ("Python 双向编辑", self._open_python), ("校验模型", self._validate)):
                    dpg.add_button(label=label, callback=self.guarded(function))
                dpg.add_button(label="撤销", tag="native_undo", callback=self.guarded(self._undo), enabled=False)
                dpg.add_button(label="重做", tag="native_redo", callback=self.guarded(self._redo), enabled=False)
                dpg.add_spacer(width=10)
                dpg.add_button(label="运行实验", tag="native_run", callback=self.guarded(self._start_run))
                dpg.add_button(label="停止", tag="native_cancel", enabled=False, callback=self.guarded(self._cancel_run))
                dpg.add_text("CPU · S1 合成演示", tag="native_mode_summary", color=MUTED)
            dpg.add_text("", tag="native_project", pos=(12, 45), color=MUTED)
            with dpg.child_window(tag="native_library", pos=(8, 77), width=205, height=490, no_scrollbar=True):
                dpg.add_text("项目层级", color=BLUE)
                with dpg.child_window(tag='native_tree_scroll', width=-1, height=145, horizontal_scrollbar=True):
                    with dpg.group(tag="native_tree"):
                        pass
                dpg.add_separator()
                dpg.add_text('组件目录', color=BLUE)
                dpg.add_input_text(hint="搜索组件", width=-1, tag="native_search", callback=self.guarded(self._search_library))
                self.catalog_panel.build_filters()
                with dpg.child_window(tag='native_catalog_scroll', width=-1, height=-1, horizontal_scrollbar=True):
                    with dpg.group(tag="native_palette"):
                        pass
            with dpg.child_window(tag="native_canvas", pos=(221, 77), width=830, height=490, no_scrollbar=True):
                with dpg.group(horizontal=True):
                    for mode, label in (('single', '单栏'), ('columns', '左右'), ('rows', '上下')):
                        dpg.add_button(label=label, tag='native_layout_' + mode,
                                       callback=self.workbench_panel._callback(self.workbench_panel.set_layout, mode))
                    dpg.add_button(label='主模型', callback=self.guarded(self._root))
                    dpg.add_button(label="返回上层", callback=self.guarded(self._back))
                with dpg.group(horizontal=True):
                    dpg.add_button(label="删除所选", callback=self.guarded(self._delete_selected))
                    dpg.add_button(label="清除选择", callback=self.guarded(self._clear_canvas_selection))
                    dpg.add_button(label="封装所选调用", callback=self.guarded(self._open_encapsulate))
                    dpg.add_button(label="接口与芯片", callback=self.guarded(self._open_chip_tools))
                with dpg.group(horizontal=True, tag="native_breadcrumb"):
                    pass
                with dpg.child_window(tag='native_canvas_body', width=-1, height=405, border=False, no_scrollbar=True):
                    dpg.add_child_window(tag='native_canvas_primary', width=-1, height=-1, border=False, no_scrollbar=True)
                    dpg.add_child_window(tag='native_canvas_secondary', width=-1, height=-1, border=False, no_scrollbar=True, show=False)
                self.workbench_panel.build()
                dpg.bind_item_theme('native_canvas_body', self.canvas_container_theme)
            with dpg.child_window(tag="native_right", pos=(1059, 77), width=305, height=490):
                dpg.add_text("属性与实验", color=BLUE)
                dpg.add_progress_bar(tag="native_progress", default_value=0.0, width=-1, overlay="尚未运行")
                with dpg.tab_bar(tag="native_right_tabs"):
                    with dpg.tab(label="属性", tag="native_inspector_tab"):
                        with dpg.group(tag="native_inspector"):
                            pass
                    with dpg.tab(label="实验", tag="native_experiment_tab"):
                        self._build_experiment_controls()
            with dpg.child_window(tag="native_bottom", pos=(8, 575), width=1355, height=210):
                dpg.add_text("尚未运行 · 设计快照与实验快照分开记录", tag="native_result_version", color=MUTED)
                with dpg.tab_bar(tag="native_bottom_tabs"):
                    with dpg.tab(label="张量探针", tag="native_probe_tab"):
                        with dpg.group(tag="native_probe"):
                            pass
                    with dpg.tab(label="训练损失", tag="native_loss_tab"):
                        dpg.add_text("尚未采集", tag="native_loss_summary", color=MUTED)
                        with dpg.plot(tag="native_loss_plot", height=120, width=-1, no_title=True):
                            dpg.add_plot_legend(tag="native_loss_legend", horizontal=True)
                            dpg.add_plot_axis(dpg.mvXAxis, label="step", tag="native_loss_x")
                            with dpg.plot_axis(dpg.mvYAxis, label="loss", tag="native_loss_y"):
                                dpg.add_line_series([], [], label="train loss", tag="native_loss_series")
                                dpg.add_line_series([], [], label="val loss", tag="native_val_loss_series")
                    with dpg.tab(label="最终评估与文件", tag="native_artifact_tab"):
                        with dpg.child_window(height=-1, width=-1, horizontal_scrollbar=True):
                            dpg.add_text("尚未采集", tag="native_final_metrics", color=MUTED, wrap=1250)
                            dpg.add_text("", tag="native_artifacts", color=MUTED, wrap=1250)
                    with dpg.tab(label="参数与梯度", tag="native_parameter_tab"):
                        dpg.add_text("梯度：末次更新前；更新范数：初始到最终", tag="native_gradient_context", color=MUTED, wrap=1250)
                        with dpg.child_window(height=-1, width=-1, horizontal_scrollbar=True):
                            with dpg.group(tag="native_parameters"):
                                pass
                    with dpg.tab(label="控制台", tag="native_console_tab"):
                        with dpg.child_window(tag="native_console_scroll", height=-1, width=-1, horizontal_scrollbar=True):
                            with dpg.group(tag="native_console_content"):
                                pass
            dpg.add_text("就绪", tag="native_status", pos=(12, 795), color=MUTED)
        self._dialogs()
        self._build_data_dialogs()
        self._build_chip_dialogs()
        self.catalog_panel.build_dialogs()
        with dpg.handler_registry():
            dpg.add_key_release_handler(key=dpg.mvKey_Delete, tag='native_delete_handler', callback=self.guarded(self._keyboard_delete))

    def _dialogs(self):
        for tag, extension, callback in (("native_json_open", ".json", self._json_opened), ("native_json_save", ".json", self._json_saved), ("native_python_open", ".py", self._python_opened), ("native_python_save", ".py", self._python_saved)):
            with dpg.file_dialog(tag=tag, show=False, width=730, height=460, modal=True, default_path=str(self.data_directory), callback=self.guarded(callback)):
                dpg.add_file_extension(extension, color=BLUE)
                dpg.add_file_extension(".*")
        with dpg.window(tag="native_python", label="Python 与画布 · 静态解析", show=False, modal=True, width=950, height=660, no_collapse=True, on_close=self.guarded(self._python_hidden)):
            with dpg.group(horizontal=True):
                dpg.add_button(label="从画布重新生成", callback=self.guarded(self._regenerate_python))
                dpg.add_button(label="打开 .py", callback=self.guarded(lambda **_: dpg.show_item("native_python_open")))
                dpg.add_button(label="导出 .py", callback=self.guarded(lambda **_: dpg.show_item("native_python_save")))
                dpg.add_text("关闭窗口保留草稿", color=MUTED)
            dpg.add_input_text(tag="native_python_code", multiline=True, tab_input=True, width=-1, height=475, callback=self.guarded(self._python_changed))
            dpg.add_text("", tag="native_python_error", color=RED, wrap=900)
            dpg.add_text("支持已实现的 PyTorch 算子与显式 forward；任意 Python / 动态控制流会被拒绝。导入不执行代码，仅替换模型。", color=MUTED, wrap=900)
            with dpg.group(horizontal=True):
                dpg.add_button(label="重新解析到画布", callback=self.guarded(self._import_python))
                dpg.add_button(label="关闭并保留草稿", callback=self.guarded(self._python_hidden))
        with dpg.window(tag="native_python_replace", label="替换代码草稿", show=False, modal=True, width=450, height=150, no_resize=True):
            dpg.add_text("重新生成将替换当前 Python 草稿。\n生成失败会保留现有内容。", color=AMBER)
            with dpg.group(horizontal=True):
                dpg.add_button(label="保留草稿", callback=self.guarded(lambda **_: dpg.hide_item("native_python_replace")))
                dpg.add_button(label="替换并重新生成", callback=self.guarded(self._generate_python))
        with dpg.window(tag="native_chip_dialog", label="保存可复用芯片", show=False, modal=True, width=630, height=430, no_resize=True):
            dpg.add_text("保留原有图，在当前作用域新增独立实例。\n每行声明一个命名输出，选择所选调用的具体端口。", color=MUTED, wrap=580)
            dpg.add_input_text(label="名称", tag="native_chip_name", default_value="自定义芯片", width=290)
            dpg.add_input_text(label="定义 ID", tag="native_chip_id", default_value="custom_chip", width=290)
            with dpg.group(tag="native_encapsulation_settings"):
                with dpg.child_window(height=160, width=-1):
                    with dpg.group(tag="native_chip_outputs"):
                        pass
                dpg.add_button(label="新增命名输出", callback=self.guarded(self._encapsulation_add))
            dpg.add_text("", tag="native_chip_error", color=RED, wrap=420)
            with dpg.group(horizontal=True):
                dpg.add_button(label="保存并新增实例", callback=self.guarded(self._save_chip))
                dpg.add_button(label="取消", callback=self.guarded(lambda **_: dpg.hide_item("native_chip_dialog")))

    def _resize(self):
        # Viewport minimums include the Windows frame; lay out against the
        # measured client area so the right pane remains fully visible.
        width = max(1000, dpg.get_viewport_client_width() or 1380)
        height = max(720, dpg.get_viewport_client_height() or 820)
        if (width, height) == self.last_size:
            return
        self.last_size = (width, height)
        left, right, bottom = 205, 305, max(160, min(180, int(height * 0.19)))
        middle = max(355, height - 77 - bottom - 35)
        canvas = width - left - right - 40
        for tag, pos, size in (("native_library", (8, 77), (left, middle)), ("native_canvas", (left + 16, 77), (canvas, middle)), ("native_right", (width - right - 8, 77), (right, middle)), ("native_bottom", (8, middle + 85), (width - 16, bottom))):
            dpg.configure_item(tag, pos=pos, width=size[0], height=size[1])
        dpg.configure_item('native_canvas_body', height=max(220, middle - 95))
        dpg.configure_item('native_tree_scroll', height=max(85, min(145, int(middle * .29))))
        self.canvas_host.sync(self.canvas_parent_by_pane)
        dpg.configure_item("native_loss_plot", height=max(120, bottom - 100))
        dpg.set_item_pos("native_status", (12, height - 22))
        for dialog, dw, dh in (("native_python", 950, 660), ("native_chip_dialog", 630, 430), ("native_chip_tools", 940, 680), ("native_python_replace", 450, 150), ("native_data", 790, 690), ("native_starter_replace", 485, 190)):
            dpg.set_item_pos(dialog, (max(20, (width - dw) // 2), max(20, (height - dh) // 2)))
        for dialog, dw, dh in (('native_catalog_config', 510, 530), ('native_catalog_policy', 530, 255)):
            dpg.set_item_pos(dialog, (max(10, (width - dw) // 2), max(10, (height - dh) // 2)))

    def _scope(self):
        return self.document.scope

    def _probe_source(self):
        try:
            if self.document.scope_id is not None and not self.document.trail:
                return None
            call = self.document.selected_call
            if call and len(self._output_ports(call["id"])) > 1:
                port = self.probe_selection.get(self._probe_selection_key(call["id"]))
                if port not in self._output_ports(call["id"]):
                    return None
                prefix = "".join("call:" + entry["call_id"] + "/" for entry in self.document.trail)
                return prefix + "call:" + call["id"] + ":" + port
            return self.document.probe_source()
        except ValueError:
            # Disconnected named outputs are legitimate editable drafts.
            return None

    def _definition(self, identifier):
        return next((item for item in self.document.graph.get("definitions", []) if item["id"] == identifier), None)

    def _ports(self, instance):
        if instance["kind"] == "chip":
            definition = self._definition(instance.get("definition_id"))
            return [item["id"] for item in definition["graph"]["inputs"]] if definition else ["x"]
        return operator_spec(instance['kind']).input_ports

    def _refresh(self, graph=True):
        key = self._bound_canvas_id or self.workbench.focused_view_id
        state = self.workbench.views.get(key)
        if state is not None and self.document.scope_id == state.target.definition_id:
            if self.selected_refs != state.selected_refs:
                state.selected_refs = list(self.selected_refs)
            if self.document.selected and self.document.selected not in state.selected_refs:
                state.selected_refs = [self.document.selected]
        self.workbench.reconcile(self.document)
        if self._bound_canvas_id is not None and self._bound_canvas_id not in self.workbench.views:
            self._bound_canvas_id = self.workbench.focused_view_id
            self._focus_document(self._bound_canvas_id)
            self._canvas_aliases(self._bound_canvas_id)
        if self._bound_canvas_id in self.workbench.views:
            state = self.workbench.views[self._bound_canvas_id]
            if self.document.selected and self.document.selected not in state.selected_refs:
                state.selected_refs = [self.document.selected]
        self.report = self.document.validation()
        if graph:
            self._render_graph()
        self._render_tree()
        self._render_inspector()
        self._render_probe()
        self._render_loss()
        self._render_experiment_summary()
        self._refresh_chip_controls()
        valid = self.report.get("valid", False)
        count = self.report.get("parameter_count")
        graph_doc = self.document.graph
        dpg.set_value("native_project", f"{graph_doc.get('name', '模型')}   |   设计 r{graph_doc.get('revision', 1)}   |   {'结构有效' if valid else str(len(self.report.get('errors', []))) + ' 个校验问题'}   |   参数 {count if count is not None else '待校验'}")
        dpg.set_value("native_status", f"本地项目  |  {len(self._scope()['calls'])} 个调用  |  拖动端口连接；选中芯片双击进入  |  草稿自动保存到 {self.data_directory}")
        self._canvas_synced()
        self.workbench_panel.sync()

    def _render_tree(self):
        self._shared_render_serial += 1
        dpg.delete_item("native_tree", children_only=True)
        dpg.add_button(label="根模型 / " + self.document.graph.get("name", "模型"), width=-1, parent="native_tree", callback=self.guarded(self._root))
        for instance in self.document.graph["instances"]:
            calls = [call for call in self.document.graph["calls"] if call["instance_id"] == instance["id"]]
            label = instance.get("name", instance["kind"]) + (f" ×{len(calls)}" if len(calls) > 1 else "")
            with dpg.tree_node(label=label, parent="native_tree", default_open=instance["kind"] == "chip"):
                if calls:
                    dpg.add_button(label="检查实例参数", width=-1, callback=self.guarded(self._select_root, "call:" + calls[0]["id"]))
                if instance["kind"] == "chip":
                    for call in calls:
                        dpg.add_button(label="进入 " + call["id"], width=-1, callback=self.guarded(self._enter_root, call["id"]))
        dpg.delete_item("native_breadcrumb", children_only=True)
        # Root and chip breadcrumbs contain different widget types. Reserve
        # one fixed row so focusing a drop target cannot shift its canvas.
        dpg.add_spacer(width=0, height=24, parent='native_breadcrumb')
        dpg.add_button(label="模型", small=True, parent="native_breadcrumb", callback=self.guarded(self._root))
        for index, entry in enumerate(self.document.trail):
            definition = self._definition(entry["id"])
            dpg.add_text("/", parent="native_breadcrumb", color=MUTED)
            label = (definition or {}).get("name", entry["id"]) + " · " + str(entry.get("call_id", "定义"))
            dpg.add_button(label=label, small=True, parent="native_breadcrumb", callback=self.guarded(self._crumb, index))
        self._search_library(app_data=dpg.get_value("native_search"))

    def _search_library(self, sender=None, app_data=None, user_data=None):
        self.catalog_panel.render(str(app_data or ''))

    @staticmethod
    def _view_position(position):
        return (int((position.get("x", 0) + 40) * VIEW_X_SCALE + VIEW_X_OFFSET), int(position.get("y", 0) * VIEW_Y_SCALE + VIEW_Y_OFFSET))

    @staticmethod
    def _document_position(position):
        return ((position[0] - VIEW_X_OFFSET) / VIEW_X_SCALE - 40, (position[1] - VIEW_Y_OFFSET) / VIEW_Y_SCALE)

    def _render_canvas_contents(self, view):
        editor = view.editor_tag
        scope = self._scope()
        instance_map = {item["id"]: item for item in scope["instances"]}
        shape_map = self.report.get("shapes", {}) if not self.document.scope_id else {}
        desired = []
        for index, item in enumerate(scope["inputs"]):
            ref = "input:" + item["id"]
            desired.append((ref, item.get("name", item["id"]), item.get("position", {"x": -40, "y": 105 + 215 * index}), "tensor", [], ref, [_shape(shape_map.get(ref, item.get("shape"))) + " · " + item.get("dtype", "float32")]))
        for call in scope["calls"]:
            instance = instance_map.get(call["instance_id"])
            if not instance:
                continue
            ref = "call:" + call["id"]
            shared = [item for item in scope["calls"] if item["instance_id"] == instance["id"]]
            kind = instance["kind"]
            tensor_shape = _shape(shape_map[ref]) if ref in shape_map else ""
            details = [call["id"] + (" " + tensor_shape if kind == "chip" and tensor_shape else "")]
            if kind == "chip":
                definition = self._definition(instance.get("definition_id"))
                metadata = "v" + (definition or {}).get("version", "")
            else:
                metadata = kind + (" " + tensor_shape if tensor_shape else "")
            if len(shared) > 1:
                metadata += f" · 共享×{len(shared)}"
            if instance.get("frozen"):
                metadata += " · 冻结"
            details.append(metadata)
            category = "chip" if kind == "chip" else "activation" if kind in ("ReLU", "GELU", "Dropout") else "primitive"
            desired.append((ref, instance.get("name", kind), call.get("position", {"x": 250, "y": 100}), category, self._ports(instance), ref, details))
        right = max([800] + [call.get("position", {}).get("x", 0) + 240 for call in scope["calls"]])
        for index, output in enumerate(scope["outputs"]):
            ref = "output:" + output["id"]
            desired.append((ref, "输出 · " + output["id"], output.get("position", {"x": right, "y": 140 + 140 * index}), "tensor", ["value"], None, [_shape(shape_map.get(output.get("source")))]))
        sources = {}
        targets = {}
        for ref, label, position, category, ports, source, details in desired:
            tag = f"{view.editor_tag}_g{view.generation}_node_{len(self.node_refs)}"
            pos = self._view_position(position)
            with dpg.node(label=label, tag=tag, parent=editor, pos=pos) as node:
                for port in ports:
                    with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as attribute:
                        dpg.add_text(port, color=(195, 206, 214))
                    self.target_attributes[attribute] = ("output" if ref.startswith("output:") else "call", ref.split(":", 1)[1], port)
                    targets[(ref, port)] = attribute
                with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                    for detail in details:
                        dpg.add_text(str(detail), color=MUTED)
                    dpg.add_spacer(width=132 if category == "chip" else 105)
                if source:
                    output_ports = self._output_ports(ref.split(":", 1)[1]) if ref.startswith("call:") else [None]
                    for output_port in output_ports:
                        qualified = source + ":" + output_port if output_port is not None else source
                        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as attribute:
                            if len(output_ports) > 1:
                                dpg.add_button(label=output_port, small=True, callback=self.guarded(self._socket_probe, ref.split(":", 1)[1], output_port))
                            else:
                                dpg.add_text(output_port or source.split(':', 1)[1], color=MINT)
                        self.source_attributes[attribute] = qualified
                        sources[qualified] = attribute
                        if len(output_ports) == 1:
                            sources[source] = attribute
            identifier = dpg.get_alias_id(tag)
            self.node_refs[identifier] = ref; self.ref_nodes[ref] = identifier
            self.view_positions[ref] = pos
            view.node_categories[ref] = category
            dpg.bind_item_theme(node, self.node_themes["selected" if ref == self.document.selected else category])
        for call in scope["calls"]:
            for port, source in call.get("inputs", {}).items():
                if source in sources and ("call:" + call["id"], port) in targets:
                    link = dpg.add_node_link(sources[source], targets[("call:" + call["id"], port)], parent=editor, show=False)
                    self.link_targets[link] = ("call", call["id"], port)
        for output in scope["outputs"]:
            source = output.get("source")
            if source in sources:
                link = dpg.add_node_link(sources[source], targets[("output:" + output["id"], "value")], parent=editor, show=False)
                self.link_targets[link] = ("output", output["id"], "value")
        self.last_selection = ()
        self.selected_refs = self.workbench.views[view.view_id].selected_refs

    def _render_inspector(self):
        self._shared_render_serial += 1
        parent = "native_inspector"
        dpg.delete_item(parent, children_only=True)
        self.param_widgets.clear()
        focused = self.workbench.views[self._bound_canvas_id or self.workbench.focused_view_id]
        if focused.target.definition_id:
            dpg.add_text('定义：' + focused.target.definition_id, color=BLUE, wrap=260, parent=parent)
            dpg.add_text('调用路径：' + (' / '.join(focused.call_path) if focused.call_path else '无调用上下文（定义配置）'), color=MUTED, wrap=260, parent=parent)
            dpg.add_text('内部修改影响该定义的全部引用。', color=AMBER, wrap=260, parent=parent)
        canvas = self.canvas_host.get(focused.view_id)
        wire = next((link for link in canvas.selected_links if link in canvas.link_targets), None)
        if wire is not None:
            configuration = dpg.get_item_configuration(wire)
            source = canvas.source_attributes[configuration['attr_1']]
            kind, identifier, port = canvas.link_targets[wire]
            dpg.add_text('连接', color=BLUE, parent=parent)
            dpg.add_text(source + '\n→ ' + kind + ':' + identifier + ':' + port, color=MUTED, wrap=260, parent=parent)
            dpg.add_button(label='断开所选连接', width=-1, parent=parent, callback=self.guarded(self._delete_selected))
            return
        instance = self.document.selected_instance
        call = self.document.selected_call
        selected = self.document.selected
        if instance and call:
            dpg.add_text(instance.get("name", instance["kind"]), color=BLUE, parent=parent)
            dpg.add_text(f"实例 {instance['id']}\n调用 {call['id']}", color=MUTED, parent=parent)
            definition = self._definition(instance.get("definition_id")) if instance["kind"] == "chip" else None
            parameters = {item["id"]: item.get("default") for item in (definition or {}).get("public_parameters", [])}
            parameters.update(instance.get("params", {}))
            if parameters:
                dpg.add_text("公开参数" if definition else "算子参数", parent=parent)
                for name, value in parameters.items():
                    label = PARAMETERS.get(name, name)
                    widget = dpg.add_input_text(label=label, default_value=json.dumps(value, ensure_ascii=False), width=145, on_enter=True, parent=parent, callback=self.guarded(self._set_param, instance["id"], name))
                    self.param_widgets[name] = widget
                dpg.add_text("回车应用参数；不自动改写下游维度。", color=MUTED, wrap=260, parent=parent)
            for name, paths in (definition or {}).get("bindings", {}).items():
                dpg.add_text(name + " → " + ", ".join(paths), color=(122, 177, 204), wrap=260, parent=parent)
            dpg.add_checkbox(label="冻结此实例参数", default_value=bool(instance.get("frozen")), parent=parent, callback=self.guarded(self._freeze, instance["id"]))
            shared = [item["id"] for item in self._scope()["calls"] if item["instance_id"] == instance["id"]]
            dpg.add_text("同一实例调用：" + ", ".join(shared), color=MUTED, wrap=260, parent=parent)
            dpg.add_button(label="添加共享调用", width=-1, parent=parent, callback=self.guarded(self._share, call["id"]))
            if definition:
                dpg.add_button(label="进入此调用内部", width=-1, parent=parent, callback=self.guarded(self._enter, call["id"]))
                dpg.add_button(label="编辑芯片接口与公开参数", width=-1, parent=parent, callback=self.guarded(self._open_chip_tools, definition["id"]))
            output_ports = self._output_ports(call["id"])
            if len(output_ports) > 1:
                selected_port = self.probe_selection.get(self._probe_selection_key(call["id"]), "请选择输出端口")
                if selected_port not in output_ports:
                    selected_port = "请选择输出端口"
                dpg.add_combo(label="探针端口", tag="native_probe_output", items=["请选择输出端口"] + output_ports, default_value=selected_port, width=185, parent=parent, callback=self.guarded(self._select_probe_output, call["id"]))
            sources = ["未连接"] + [source for source in self._source_options() if source != "call:" + call["id"] and not source.startswith("call:" + call["id"] + ":")]
            for port in self._ports(instance):
                dpg.add_combo(label="输入 " + port, items=sources, default_value=call.get("inputs", {}).get(port, "未连接"), width=190, parent=parent, callback=self.guarded(self._set_input, call["id"], port))
        elif selected and selected.startswith("output:"):
            output = next(item for item in self._scope()["outputs"] if "output:" + item["id"] == selected)
            dpg.add_text("命名输出：" + output["id"], color=MINT, parent=parent)
            sources = ["未连接"] + self._source_options()
            dpg.add_combo(label="来源", items=sources, default_value=output.get("source") or "未连接", width=200, parent=parent, callback=self.guarded(self._set_output, output["id"]))
        elif selected and selected.startswith("input:"):
            item = next(item for item in self._scope()["inputs"] if "input:" + item["id"] == selected)
            dpg.add_text("模型输入：" + item["id"], color=MINT, parent=parent)
            dpg.add_text(_shape(item.get("shape")) + " · " + item.get("dtype", "float32"), parent=parent)
        else:
            dpg.add_text("选择节点以检查接口、共享关系和参数。", color=MUTED, wrap=260, parent=parent)
            dpg.add_text(f"输入 {len(self._scope()['inputs'])} / 调用 {len(self._scope()['calls'])} / 输出 {len(self._scope()['outputs'])}", color=MUTED, parent=parent)
            if self.document.scope_id:
                definition = self._definition(self.document.scope_id)
                for parameter in (definition or {}).get("public_parameters", []):
                    dpg.add_text(f"{parameter['id']} = {parameter.get('default')}", color=BLUE, parent=parent)
        dpg.add_spacer(height=5, parent=parent)

    def _render_probe(self):
        self._shared_render_serial += 1
        parent = "native_probe"
        dpg.delete_item(parent, children_only=True)
        sources = [source for source in (self.result or {}).get('probes', {}) if self._probe_location(source) is not None]
        if sources:
            # Only actual collected keys appear in this navigation picker.
            dpg.add_combo(label='已采集位置', tag='native_probe_sources', items=sources[:512], width=400,
                          parent=parent, callback=self.guarded(self._probe_source_selected))
        source = self._probe_source()
        stale = bool(self.result and self.result_key != self._experiment_identity())
        revision = self.run_record.get("revision") if self.run_record else None
        dpg.add_text((source or "请选择一个调用或张量接口") + (f"   结果 r{revision} · 当前 r{self.document.graph.get('revision')} · 已过期" if stale else f"   结果 r{revision}" if self.result else ""), color=AMBER if stale else BLUE, parent=parent)
        probe = (self.result or {}).get("probes", {}).get(source)
        context = result_view(self.result)["probe"]
        if context:
            dpg.add_text(context, color=MUTED, wrap=1250, parent=parent)
        if not probe:
            dpg.add_text("尚未采集", color=MUTED, parent=parent)
            call = self.document.selected_call
            text = ("此芯片有多个输出；请在属性页选择探针端口，或点击节点上的命名端口。" if call and len(self._output_ports(call["id"])) > 1 and source is None
                    else "运行实验后显示此具体调用的实测张量；芯片内部探针沿调用路径区分。")
            dpg.add_text(text, color=MUTED, parent=parent)
            return
        dpg.add_text(f"{_shape(probe.get('shape'))} · {probe.get('dtype')}   全量统计：{probe.get('count')} 个值   数值采样：{probe.get('sample_size')} 个值   NaN {probe.get('nan_count')} / Inf {probe.get('inf_count')}", color=MUTED, parent=parent)
        with dpg.group(horizontal=True, parent=parent):
            for name, key in (("最小值", "min"), ("最大值", "max"), ("均值", "mean"), ("标准差", "std")):
                dpg.add_text(name + "  " + _number(probe.get(key)) + "      ", color=MINT)
        preview = json.dumps(probe.get("values"), ensure_ascii=False)
        dpg.add_text("数值预览（截取显示）：" + preview[:1200], color=MUTED, wrap=1250, parent=parent)

    def _render_parameters(self):
        parent = "native_parameters"
        dpg.delete_item(parent, children_only=True)
        context = result_view(self.result)["parameters"]
        dpg.set_value("native_gradient_context", context or "梯度：末次更新前；更新范数：初始到最终")
        parameters = (self.result or {}).get("parameters", [])
        if not parameters:
            dpg.add_text("尚未采集", color=MUTED, parent=parent)
            return
        with dpg.table(parent=parent, header_row=True, borders_innerH=True, borders_outerH=True, borders_innerV=True, resizable=True):
            for title in ("真实参数名称", "形状", "可训练", "梯度范数", "更新范数"):
                dpg.add_table_column(label=title)
            for item in parameters:
                with dpg.table_row():
                    for value in (item["name"], _shape(item["shape"]), "是" if item["trainable"] else "已冻结", _number(item.get("grad_norm")), _number(item.get("update_norm"))):
                        dpg.add_text(str(value))

    def _render_loss(self):
        revision = self.document.graph.get("revision", 1)
        stale = bool(self.result and self.result_key != self._experiment_identity())
        if self.run_record:
            state = "已过期的实测结果" if stale else "实测结果" if self.result else RUN_LABELS.get(self.run_record["status"], self.run_record["status"])
            message = f"{state}   |   实验快照 r{self.run_record['revision']}   /   当前设计 r{revision}"
        else:
            message = f"尚未运行   |   当前设计 r{revision}   |   {self.experiment_mode}"
        dpg.set_value("native_result_version", message)
        dpg.configure_item("native_result_version", color=AMBER if stale else MUTED)
        active_config = self.run_config or self.experiment
        display = self.result or {"profile": active_config.get("profile")} if self.progress_history else self.result
        view = result_view(display, self.progress_history)
        dpg.set_value("native_loss_series", view["train"])
        dpg.set_value("native_val_loss_series", view["val"])
        dpg.configure_item("native_loss_x", label="epoch" if view["real"] else "step")
        text = view["summary"] + (" · 旧版本结果" if stale else "")
        if not self.result and self.progress_history:
            text = f"已采集 {len(self.progress_history)} 个实测进度点；test 仅在最佳 checkpoint 最终评估。" if view["real"] else f"已实测采集 {len(self.progress_history)} 个进度点"
        dpg.set_value("native_loss_summary", text)
        dpg.set_value("native_final_metrics", view["summary"])
        dpg.set_value("native_artifacts", view["artifacts"])
        if view["train"][0] or view["val"][0]:
            dpg.fit_axis_data("native_loss_x"); dpg.fit_axis_data("native_loss_y")

    def _connect(self, sender=None, app_data=None, user_data=None):
        first, second = app_data
        source = self.source_attributes.get(first) or self.source_attributes.get(second)
        target = self.target_attributes.get(second) or self.target_attributes.get(first)
        if not source or not target:
            raise ValueError("连接端口已改变，请重新拖线。")
        kind, identifier, port = target
        if kind == "call":
            self.document.set_input(identifier, port, source)
        else:
            self.document.set_output(identifier, source)
        self._refresh()

    def _disconnect(self, sender=None, app_data=None, user_data=None):
        target = self.link_targets.get(app_data)
        if not target:
            return
        kind, identifier, port = target
        if kind == "call":
            self.document.set_input(identifier, port, None)
        else:
            self.document.set_output(identifier, None)
        self._refresh()

    def _add(self, kind, definition_id, **_):
        ref = self.document.add(kind, definition_id)
        if ref:
            self.document.select(ref)
        self._refresh()

    def _share(self, call_id, **_):
        ref = self.document.share(call_id)
        if ref:
            self.document.select(ref)
        self._refresh()

    def _set_param(self, instance_id, name, sender=None, app_data=None, **_):
        value = json.loads(str(app_data))
        self.document.set_param(instance_id, name, value); self._refresh()

    def _freeze(self, instance_id, app_data=None, **_):
        self.document.set_frozen(instance_id, bool(app_data)); self._refresh()

    def _set_input(self, call_id, port, app_data=None, **_):
        self.document.set_input(call_id, port, None if app_data == "未连接" else app_data); self._refresh()

    def _set_output(self, identifier, app_data=None, **_):
        self.document.set_output(identifier, None if app_data == "未连接" else app_data); self._refresh()

    def _delete_selected(self, **_):
        key = self._bound_canvas_id or self.workbench.focused_view_id
        view = self.canvas_host.get(key)
        wire = next((link for link in view.selected_links if link in view.link_targets), None)
        if wire is not None:
            self._disconnect(app_data=wire)
            return
        refs = self.selected_refs or ([self.document.selected] if self.document.selected else [])
        self.document.remove_many(list(dict.fromkeys(refs)))
        self._refresh()

    def _keyboard_delete(self, **_):
        if self._canvas_modal_open():
            return
        key = self.workbench.focused_view_id
        if self._bound_canvas_id is not None and self._bound_canvas_id != key:
            self.log('已丢弃失去焦点的删除回调。', '提示')
            return
        view = self.canvas_host.get(key)
        if view.visible and dpg.is_item_hovered(view.editor_tag) and not self._canvas_input_active():
            self._delete_selected()

    def _validate(self, **_):
        self._refresh(graph=False)
        report = self.report
        if report.get("valid"):
            self.log(f"结构校验通过；参数总量 {report.get('parameter_count')}", "完成")
        else:
            for issue in report.get("errors", []):
                self.log(f"{issue.get('node_id', issue.get('field', '模型'))}: {issue['message']}", "错误",
                         location=self._issue_location(issue))
            dpg.set_value("native_bottom_tabs", dpg.get_alias_id("native_console_tab"))

    def _issue_location(self, issue):
        explicit = issue.get('location')
        if explicit is not None:
            return deepcopy(explicit)
        identifier = issue.get('node_id')
        if type(identifier) is not str:
            return None
        # Existing diagnostics do not always carry a scope. Offer a location
        # only if the node is unique across the complete editable document.
        scopes = [(None, self.document.graph)] + [(item['id'], item['graph'])
                 for item in self.document.graph.get('definitions', [])]
        candidates = {}
        for definition_id, scope in scopes:
            for kind, collection in (('input', 'inputs'), ('call', 'calls'), ('output', 'outputs')):
                for item in scope[collection]:
                    instances = [instance for instance in scope['instances']
                                 if kind == 'call' and instance['id'] == item['instance_id']]
                    if item['id'] != identifier and not (len(instances) == 1 and instances[0]['id'] == identifier):
                        continue
                    fields = set(item)
                    if len(instances) == 1:
                        fields.update(instances[0].get('params', {}))
                        definition = self._definition(instances[0].get('definition_id'))
                        fields.update(parameter['id'] for parameter in (definition or {}).get('public_parameters', []))
                    field = issue.get('field') if issue.get('field') in fields else None
                    ref = kind + ':' + item['id']
                    candidates[definition_id, ref] = {'definition_id': definition_id, 'call_path': [],
                                                       'node_ref': ref, 'field': field}
        return next(iter(candidates.values())) if len(candidates) == 1 else None

    def _probe_location(self, source):
        from studio_core.ports import output_port_ids
        from .view_document import resolve_view
        from .workbench_state import ScopeTarget, ViewState
        try:
            if type(source) is not str or len(source) > 4096:
                return None
            parts = source.split('/')
            path = []
            for part in parts[:-1]:
                kind, identifier = part.split(':')
                if kind != 'call':
                    return None
                path.append(identifier)
            trail = self.document._resolve_call_path(path)
            target = trail[-1]['id'] if trail else None
            scope, _ = resolve_view(self.document, ViewState('location', ScopeTarget(target), path, [], None, 'primary'))
            final = parts[-1].split(':')
            if len(final) not in (2, 3) or final[0] not in ('input', 'call'):
                return None
            kind, identifier = final[:2]
            matches = [item for item in scope['inputs' if kind == 'input' else 'calls'] if item['id'] == identifier]
            if len(matches) != 1 or (kind == 'input' and len(final) != 2):
                return None
            if kind == 'call':
                instances = [item for item in scope['instances'] if item['id'] == matches[0]['instance_id']]
                if len(instances) != 1:
                    return None
                ports = output_port_ids(instances[0], self.document.graph.get('definitions', []))
                if (len(final) == 3 and final[2] not in ports) or (len(final) == 2 and len(ports) != 1):
                    return None
            return {'definition_id': target, 'call_path': path, 'node_ref': kind + ':' + identifier, 'field': None}
        except (ValueError, KeyError, TypeError):
            return None

    def _probe_source_selected(self, app_data=None, **_):
        return self._focus_probe_source(app_data)

    def _focus_probe_source(self, source):
        location = self._probe_location(source)
        if location is None:
            self.log('目标已变化；探针位置不属于当前工程。', '提示')
            return False
        if not self.workbench_panel.focus_location(location):
            return False
        final = source.rsplit('/', 1)[-1].split(':')
        if len(final) == 3:
            self._select_probe_output(final[1], app_data=final[2])
        dpg.set_value('native_bottom_tabs', dpg.get_alias_id('native_probe_tab'))
        return True

    def _open_json(self, **_):
        self._bind_modal('native_json_open')
        dpg.show_item("native_json_open")

    def _save_json(self, **_):
        self._bind_modal('native_json_save')
        dpg.configure_item("native_json_save", default_filename=self.document.graph.get("id", "model") + ".json")
        dpg.show_item("native_json_save")

    @staticmethod
    def _file_path(app_data, extension=None):
        target = Path(app_data["file_path_name"])
        if extension and target.suffix.lower() != extension:
            target = target.with_suffix(extension)
        return target

    def _json_opened(self, app_data=None, **_):
        path = self._file_path(app_data)
        try:
            document, experiment, workbench_payload = read_workspace_full(path)
        except Exception as error:
            from .main import preserve_failed_workspace
            self._protected_save_paths.add(path.resolve())
            self.log(preserve_failed_workspace(path, error), '提示')
            raise
        self._protected_save_paths.update(getattr(document, 'workspace_protected_paths', ()))
        self.document.replace_graph(document.graph)
        self._reset_canvas_document()
        self._restore_workbench(workbench_payload)
        for warning in getattr(document, 'workspace_load_warnings', ()):
            self.log(warning, '提示')
        self.python_initialized = False; self.python_dirty = False
        self._install_experiment(experiment)
        self._sync_experiment_widgets()
        self._refresh(); self.log("已打开模型与实验设置", "完成")

    def _json_saved(self, app_data=None, **_):
        path = self._file_path(app_data, ".json")
        save_workspace(path, self.document, self._configuration(), workbench=self.workbench.to_payload())
        self._protected_save_paths.discard(path.resolve())
        self.log("已保存工作区：" + str(path), "完成")

    def _open_python(self, **_):
        for tag in ('native_python', 'native_python_open', 'native_python_save', 'native_python_replace'):
            self._bind_modal(tag)
        if not self.python_initialized:
            try:
                self.python_code = self.document.export_python(); self.python_initialized = True
            except Exception as error:
                dpg.set_value("native_python_error", str(error))
        dpg.set_value("native_python_code", self.python_code)
        dpg.show_item("native_python")

    def _python_changed(self, app_data=None, **_):
        self.python_code = str(app_data); self.python_initialized = True; self.python_dirty = True

    def _python_hidden(self, **_):
        self.python_code = dpg.get_value("native_python_code")
        dpg.hide_item("native_python")

    def _regenerate_python(self, **_):
        if self.python_dirty:
            dpg.show_item("native_python_replace")
        else:
            self._generate_python()

    def _generate_python(self, **_):
        dpg.hide_item("native_python_replace")
        try:
            code = self.document.export_python()
            self.python_code = code; self.python_initialized = True; self.python_dirty = False
            dpg.set_value("native_python_code", code); dpg.set_value("native_python_error", "")
        except Exception as error:
            dpg.set_value("native_python_error", str(error)); self.log(str(error), "错误")

    def _python_opened(self, app_data=None, **_):
        code = self._file_path(app_data).read_text(encoding="utf-8-sig")
        self.python_code = code; self.python_initialized = True; self.python_dirty = True
        dpg.set_value("native_python_code", code); dpg.set_value("native_python_error", "")

    def _python_saved(self, app_data=None, **_):
        self.python_code = dpg.get_value("native_python_code")
        path = self._file_path(app_data, ".py")
        path.write_text(self.python_code, encoding="utf-8")
        self.log("已导出 Python：" + str(path), "完成")

    def _import_python(self, **_):
        self.python_code = dpg.get_value("native_python_code")
        self.python_initialized = True
        try:
            self.document.import_python(self.python_code)
            self._reset_canvas_document()
            self.python_dirty = False
            dpg.set_value("native_python_error", ""); dpg.hide_item("native_python")
            self._refresh(); self.log("Python 已静态解析到原生画布", "完成")
        except Exception as error:
            self.python_dirty = True
            dpg.set_value("native_python_error", str(error)); self.log(str(error), "错误")

    def _open_template(self, **_):
        self.chip_mode = "template"; self.chip_calls = []
        self._prepare_chip_dialog([])

    def _open_encapsulate(self, **_):
        refs = self.selected_refs or ([self.document.selected] if self.document.selected else [])
        calls = [ref.split(":", 1)[1] for ref in refs if ref.startswith("call:")]
        if not calls:
            raise ValueError("请先在节点编辑器中选中至少一个调用。")
        self.chip_mode = "selection"; self.chip_calls = calls
        self._prepare_chip_dialog(calls)

    def _prepare_chip_dialog(self, calls):
        self._bind_modal('native_chip_dialog')
        existing = {item["id"] for item in self.document.graph["definitions"]}
        index, identifier = 1, "custom_chip"
        while identifier in existing:
            index += 1; identifier = "custom_chip_" + str(index)
        dpg.set_value("native_chip_name", "自定义芯片")
        dpg.set_value("native_chip_id", identifier)
        dpg.configure_item("native_encapsulation_settings", show=bool(calls))
        sources = [source for source in self._source_options() if source.startswith("call:") and source.split(":")[1] in calls]
        self.encapsulation_rows = [{"id": "out", "source": sources[-1]}] if sources else []
        self._render_encapsulation_rows()
        dpg.set_value("native_chip_error", "")
        dpg.show_item("native_chip_dialog")

    def _save_chip(self, **_):
        try:
            name, identifier = dpg.get_value("native_chip_name"), dpg.get_value("native_chip_id")
            if self.chip_mode == "template":
                ref = self.document.add_template(name, identifier)
            else:
                rows = self._encapsulation_draft()
                if not rows or len({row["id"] for row in rows}) != len(rows):
                    raise ValueError("请声明至少一个输出，且每行输出 ID 不重复。")
                ref = self.document.encapsulate(self.chip_calls, name, identifier, outputs={row["id"]: row["source"] for row in rows})
            if ref:
                self.document.select(ref)
            dpg.hide_item("native_chip_dialog"); self._refresh()
            self.log("芯片定义已保存，并在当前作用域新增独立实例", "完成")
        except Exception as error:
            dpg.set_value("native_chip_error", str(error)); self.log(str(error), "错误")

    def _start_run(self, **_):
        if self.run_record and self.run_record["status"] not in TERMINAL:
            raise ValueError("已有实验正在运行，请先停止或等待完成。")
        self.report = self.document.validation()
        if not self.report.get("valid"):
            self._validate(); return
        config = self._run_configuration()
        self.run_record = self.manager.submit(self.document.graph, config)
        self.run_config = dict(config)
        self.run_key = self._experiment_identity(); self.result = None; self.result_key = None
        self.progress_history = {}; self.last_event = 0
        dpg.configure_item("native_run", enabled=False); dpg.configure_item("native_cancel", enabled=True)
        dpg.set_value("native_bottom_tabs", dpg.get_alias_id("native_loss_tab"))
        self._render_loss(); self._render_probe(); self._render_parameters()
        self.log(f"实验已提交；快照 r{self.run_record['revision']}，{config.get('device', 'cpu')} / {config.get('profile', 'synthetic_s1')}")

    def _cancel_run(self, **_):
        if self.run_record and self.run_record["status"] not in TERMINAL:
            self.run_record = self.manager.cancel(self.run_record["run_id"])
            self.log("已请求停止；等待子进程退出确认", "提示")

    def _poll_run(self):
        if not self.run_record or self.run_record["status"] in TERMINAL:
            return
        record = self.manager.get(self.run_record["run_id"])
        self.run_record = record
        progress = record.get("progress") or {}
        step, total, loss = progress.get("step", 0), progress.get("total", 1), progress.get("loss")
        real = (self.run_config or {}).get("profile") == "classification-v2"
        for point in progress_points(record, self.last_event, real):
            self.progress_history[point["epoch"] if real else point["step"]] = dict(point)
        dpg.set_value("native_progress", step / max(1, total))
        phase = progress.get("phase")
        phase_text = (str(phase) + " · epoch " + str(progress.get("epoch", 0)) + "/" + str(progress.get("epochs", "—"))) if phase else f"{step}/{total}"
        dpg.configure_item("native_progress", overlay=f"{RUN_LABELS.get(record['status'], record['status'])} · {phase_text} · r{record['revision']}")
        for event in record.get("events", []):
            if event.get("seq", 0) > self.last_event and event.get("type") != "progress":
                self.log(event.get("message") or event.get("status") or event.get("type", "实验事件"), "错误" if event.get("type") == "error" else "信息")
            self.last_event = max(self.last_event, event.get("seq", 0))
        if record.get("result"):
            self.result = record["result"]; self.result_key = self.run_key
            self._render_probe(); self._render_parameters()
        self._render_loss()
        if record["status"] in TERMINAL:
            dpg.configure_item("native_run", enabled=True); dpg.configure_item("native_cancel", enabled=False)
            if record.get("error"):
                self.log(record["error"], "错误")
            else:
                self.log(RUN_LABELS.get(record["status"], record["status"]), "完成")

    def _save_draft(self):
        self.workbench.reconcile(self.document)
        view_payload = self.workbench.to_payload()
        try:
            config = self._configuration()
            payload = workspace_payload(self.document, config, workbench=view_payload)
        except (ValueError, TypeError) as error:
            # Preserve the model and last applied settings while a form field
            # contains incomplete JSON or a temporarily invalid numeric value.
            config = self.experiment
            payload = workspace_payload(self.document, config, workbench=view_payload)
            if self._context_created and dpg.does_item_exist("native_data_error"):
                dpg.set_value("native_data_error", "当前表单尚未应用；自动保存保留上次有效设置。" + str(error))
        signature = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        if signature != self.draft_signature:
            draft, workspace = self.data_directory / 'draft.json', self.data_directory / 'workspace.json'
            if draft.resolve() not in self._protected_save_paths:
                self.document.save_json(draft)
            if workspace.resolve() not in self._protected_save_paths:
                save_workspace(workspace, self.document, config, workbench=view_payload)
            self.draft_signature = signature

    @staticmethod
    def _socket_position(attribute, *, output):
        """Measure an actual ImNodes side pin from public rendered geometry."""
        node = dpg.get_item_info(attribute)['parent']
        state = dpg.get_item_state(node)
        minimum, maximum = state.get('rect_min'), state.get('rect_max')
        if minimum is None or maximum is None:
            return None
        size = state['rect_size']
        if not all(type(value) in (int, float) and math.isfinite(value) for value in (*minimum, *maximum, *size)):
            return None
        # DPG reports content min/max but a padded outer rect_size. The
        # side-pin center is at the outer node edge, including that padding.
        padding = (size[0] - (maximum[0] - minimum[0])) / 2
        x = maximum[0] + padding if output else minimum[0] - padding
        children = [child for group in dpg.get_item_children(attribute).values() for child in group]
        for child in children:
            rendered = dpg.get_item_state(child)
            if 'rect_min' in rendered and rendered.get('rect_size', [0, 0])[1] > 0:
                y = rendered['rect_min'][1] + rendered['rect_size'][1] / 2
                if math.isfinite(y):
                    return (x, y)
        return None

    @staticmethod
    def _node_rectangle(node):
        state = dpg.get_item_state(node)
        if 'rect_min' not in state or 'rect_max' not in state:
            return None
        minimum, maximum, size = state['rect_min'], state['rect_max'], state['rect_size']
        padx, pady = ((size[i] - (maximum[i] - minimum[i])) / 2 for i in (0, 1))
        return (minimum[0] - padx, minimum[1] - pady, maximum[0] + padx, maximum[1] + pady)

    @staticmethod
    def _canvas_modal_open():
        return any(dpg.get_item_configuration(item).get('modal', False) and dpg.is_item_shown(item)
                   for item in dpg.get_all_items())

    def _update_orthogonal_edges(self):
        modal_open = self._canvas_modal_open()
        for key, view in self.canvas_host.views.items():
            if not view.edge_overlay:
                continue
            if not view.visible or modal_open:
                view.edge_overlay.clear()
                continue
            origin, size = screen_rect(view.container_tag)
            if size[0] <= 8 or size[1] <= 8:
                view.edge_overlay.clear()
                continue
            clip = (origin[0], origin[1], origin[0] + size[0] - 8, origin[1] + size[1] - 8)
            edges = {}
            for link in view.link_targets:
                config = dpg.get_item_configuration(link)
                source = self._socket_position(config['attr_1'], output=True)
                target = self._socket_position(config['attr_2'], output=False)
                if source is not None and target is not None:
                    edges[link] = (source, target)
            rectangles = [self._node_rectangle(node) for node in view.ref_nodes.values()]
            config = dpg.get_item_configuration(view.editor_tag)
            if config.get('minimap'):
                # DPG exposes the minimap corner but no measured minimap rect.
                # Reserve a conservative quarter of the editor plus margins;
                # the native .2 minimap fits here at our tested viewports.
                reserve_w, reserve_h = size[0] * .25 + 16, size[1] * .25 + 16
                location = config['minimap_location']
                right_corner = location in (dpg.mvNodeMiniMap_Location_TopRight, dpg.mvNodeMiniMap_Location_BottomRight)
                bottom_corner = location in (dpg.mvNodeMiniMap_Location_BottomLeft, dpg.mvNodeMiniMap_Location_BottomRight)
                left = clip[2] - reserve_w if right_corner else clip[0]
                top = clip[3] - reserve_h if bottom_corner else clip[1]
                rectangles.append((left, top, left + reserve_w, top + reserve_h))
            hovered = view.edge_overlay.hit_test(dpg.get_mouse_pos(local=False))
            view.edge_overlay.update(edges, clip, node_rects=[rect for rect in rectangles if rect],
                                     selected=view.selected_links, hovered=hovered)

    def _select_orthogonal_wire(self, token, point):
        if self._canvas_modal_open() or self._canvas_input_active() or not self.canvas_host.accept(token):
            return False
        view = self.canvas_host.get(token.view_id)
        if not view.visible:
            return False
        link = view.edge_overlay.hit_test(point)
        had_wire = bool(view.selected_links)
        view.selected_links.clear()
        dpg.clear_selected_links(view.editor_tag)
        if link is not None and link in view.link_targets:
            self.workbench.activate(token.view_id)
            view.selected_links.add(link)
            self.canvas_host.apply_selection(token.view_id, [], intentional=True)
            self._focus_document(token.view_id)
            self._canvas_synced()
            self._render_inspector(); self._render_probe()
            return link
        if had_wire:
            self._render_inspector(); self._render_probe()
        return False

    def tick(self):
        """One nonblocking GUI tick; safe to invoke from a native harness."""
        self._resize()
        self._poll_background()
        self._update_orthogonal_edges()
        # Hover establishes pane focus; only the visible focused editor consumes
        # selection/double-click/Delete. Empty native getters do not erase the
        # independently stored logical selection.
        for key, view in self.canvas_host.views.items():
            if view.visible and dpg.is_item_hovered(view.editor_tag) and dpg.is_mouse_button_clicked(dpg.mvMouseButton_Left):
                self.workbench.activate(key)
                self._canvas_synced()
                if not self._canvas_input_active():
                    self._select_orthogonal_wire(self.canvas_host.token(key), dpg.get_mouse_pos(local=False))
        key = self.workbench.focused_view_id
        view = self.canvas_host.get(key)
        selected = tuple(view.node_refs[item] for item in dpg.get_selected_nodes(view.editor_tag) if item in view.node_refs) if view.visible else ()
        if selected and selected != view.last_selection:
            self.canvas_host.apply_selection(key, selected, intentional=True)
            self._render_inspector(); self._render_probe()
        if view.visible and dpg.is_mouse_button_double_clicked(dpg.mvMouseButton_Left) and dpg.is_item_hovered(view.editor_tag):
            call, instance = self.document.selected_call, self.document.selected_instance
            if call and instance and instance.get("kind") == "chip":
                self._enter(call["id"])
        now = time.monotonic()
        if now - self.last_poll >= 0.2:
            self.last_poll = now
            if not dpg.is_mouse_button_down(dpg.mvMouseButton_Left):
                for view_key, current in self.canvas_host.views.items():
                    if current.visible:
                        self._commit_canvas_positions(view_key)
            self._poll_run()
        if now - self.last_draft >= 2:
            self.last_draft = now; self._save_draft()

    def snapshot(self):
        """Model/UI identifiers for diagnostics, without modifying the document."""
        return {"revision": self.document.graph.get("revision"), "scope_id": self.document.scope_id, "selected": self.document.selected, "selected_nodes": list(self.selected_refs), "node_count": len(self.node_refs), "link_count": len(self.link_targets), "node_items": dict(self.ref_nodes), "probe_source": self._probe_source(), "run_status": (self.run_record or {}).get("status"), "run_revision": (self.run_record or {}).get("revision"), "has_measured_result": bool(self.result), "python_dirty": self.python_dirty, "experiment_mode": self.experiment_mode, "dataset_digest": (self.manifest or {}).get("dataset_digest"), "inspecting": self.inspecting, "available_devices": list(self.available_devices), "result_stale": bool(self.result and self.result_key != self._experiment_identity())}

    def build(self):
        """Create/show the native viewport, without entering the main loop.

        Embedders can call process_callbacks(), tick(), then
        dpg.render_dearpygui_frame(). Always finish with shutdown().
        The supplied manager remains owned by the caller.
        """
        if self._built:
            return
        dpg.create_context()
        self._context_created = True
        dpg.configure_app(manual_callback_management=True)
        self._font(); self._theme()
        icon = self.resource_directory / "icon.ico"
        # DPG 2.3.1's Windows viewport title path access-violates on Unicode.
        # The branded OS title stays ASCII; the entire workbench is Chinese.
        dpg.create_viewport(title="ModelCircuit Studio", width=1380, height=860, min_width=1100, min_height=760, small_icon=str(icon) if icon.exists() else "", large_icon=str(icon) if icon.exists() else "")
        self._build(); self._sync_experiment_widgets(); self._refresh(); self._render_parameters(); self._render_loss()
        dpg.setup_dearpygui(); dpg.show_viewport(); dpg.set_primary_window("native_main", True)
        self._resize(); self.log("ModelCircuit Studio 已就绪。", "完成")
        self._refresh_devices()
        dpg.render_dearpygui_frame()
        self._built = True

    def process_callbacks(self):
        """Drain DPG's manual callback queue on the calling GUI thread."""
        for job in dpg.get_callback_queue() or []:
            try:
                dpg.run_callbacks([job])
            except Exception as error:
                self.log(str(error), "错误")

    def shutdown(self):
        """Save the editable draft and release DPG; never close the manager."""
        if not self._context_created:
            return
        try:
            self._save_draft()
        finally:
            try:
                self.canvas_host.dispose_all()
            finally:
                try:
                    self.queries.close()
                finally:
                    self.closed = True; self._context_created = False; self._built = False
                    dpg.destroy_context()

    def run(self):
        try:
            self.build()
            while dpg.is_dearpygui_running():
                self.process_callbacks()
                try:
                    self.tick()
                except Exception as error:
                    self.log(str(error), "错误")
                dpg.render_dearpygui_frame()
        finally:
            self.shutdown()
