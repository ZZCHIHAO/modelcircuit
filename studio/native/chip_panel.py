"""Small native editors for scope ports, public chip bindings and local bundles."""
import json

import dearpygui.dearpygui as dpg

BLUE = (100, 200, 255)
MUTED = (145, 154, 164)
RED = (240, 143, 134)
TYPES = ["auto", "int", "float", "bool", "shape"]


class ChipControls:
    def _init_chip_controls(self):
        self.probe_selection = {}
        self.public_rows = []
        self.public_row_widgets = []
        self.encapsulation_rows = []
        self.encapsulation_widgets = []
        self.edit_definition = None
        self.interface_widgets = {}

    def _output_ports(self, call_id):
        try:
            if hasattr(self.document, "output_ports"):
                return self.document.output_ports(call_id)
            call = next(item for item in self._scope()["calls"] if item["id"] == call_id)
            instance = next(item for item in self._scope()["instances"] if item["id"] == call["instance_id"])
            if instance["kind"] == "chip":
                definition = self._definition(instance.get("definition_id"))
                return [item["id"] for item in (definition or {}).get("graph", {}).get("outputs", [])]
            return ["out"]
        except (ValueError, StopIteration):
            return []

    def _source_options(self):
        if hasattr(self.document, "source_options"):
            try:
                return self.document.source_options()
            except ValueError:
                # An incomplete editable definition must not prevent repairing
                # other interfaces. Keep sources with known declared ports.
                pass
        return (["input:" + item["id"] for item in self._scope()["inputs"]] +
                ["call:" + call["id"] + ":" + port for call in self._scope()["calls"] for port in self._output_ports(call["id"])])

    def _probe_selection_key(self, call_id):
        key = self._bound_canvas_id or self.workbench.focused_view_id
        return key, tuple(item["call_id"] for item in self.document.trail), call_id

    def _select_probe_output(self, call_id, app_data=None, **_):
        if app_data not in self._output_ports(call_id):
            self.probe_selection.pop(self._probe_selection_key(call_id), None)
        else:
            self.probe_selection[self._probe_selection_key(call_id)] = app_data
        key = self._bound_canvas_id or self.workbench.focused_view_id
        self.workbench.views[key].probe_port = app_data if app_data in self._output_ports(call_id) else None
        self.document.select("call:" + call_id)
        self.selected_refs = ["call:" + call_id]
        self._render_inspector(); self._render_probe()

    def _socket_probe(self, call_id, port, **_):
        self._select_probe_output(call_id, app_data=port)
        dpg.set_value("native_bottom_tabs", dpg.get_alias_id("native_probe_tab"))

    def _build_chip_dialogs(self):
        with dpg.window(tag="native_chip_tools", label="接口、公开参数与本地芯片库", show=False, modal=True, width=940, height=680, no_collapse=True):
            dpg.add_text("", tag="native_chip_scope", color=BLUE)
            dpg.add_combo(label="芯片定义", tag="native_chip_definition", items=[], width=360, callback=self.guarded(self._choose_definition))
            with dpg.tab_bar(tag="native_chip_tool_tabs"):
                with dpg.tab(label="当前作用域接口", tag="native_scope_ports"):
                    dpg.add_text("先断开被使用的接口，再删除；每个作用域至少保留一个输入和输出。", color=MUTED, wrap=890)
                    with dpg.child_window(height=390, width=-1, horizontal_scrollbar=True):
                        with dpg.group(tag="native_interface_rows"):
                            pass
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag="native_new_input_id", default_value="new_input", width=140, hint="输入 ID")
                        dpg.add_input_text(tag="native_new_input_shape", default_value="[null, 4]", width=210, hint="形状 JSON")
                        dpg.add_button(label="新增输入", callback=self.guarded(self._interface_add_input))
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag="native_new_output_id", default_value="new_output", width=140, hint="输出 ID")
                        dpg.add_combo(tag="native_new_output_source", items=["未连接"], default_value="未连接", width=300)
                        dpg.add_button(label="新增输出", callback=self.guarded(self._interface_add_output))
                with dpg.tab(label="公开参数与绑定", tag="native_public_parameters"):
                    dpg.add_text("默认值填写 JSON；绑定填写局部 instance.parameter 路径列表。应用时整体校验。", color=MUTED, wrap=890)
                    with dpg.child_window(height=360, width=-1, horizontal_scrollbar=True):
                        with dpg.group(tag="native_public_rows"):
                            pass
                    dpg.add_text("", tag="native_binding_paths", color=MUTED, wrap=880)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="新增参数行", callback=self.guarded(self._public_add_row))
                        dpg.add_button(label="应用全部参数与绑定", tag="native_public_apply", callback=self.guarded(self._public_apply))
                with dpg.tab(label="版本与本地芯片库", tag="native_chip_library"):
                    dpg.add_text("只保存模型定义和依赖，不包含训练权重。同 ID/版本文件不可覆盖为其他内容。", color=MUTED, wrap=890)
                    dpg.add_input_text(label="版本", tag="native_chip_version", default_value="1.0.0", width=190)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="保存选中芯片版本", tag="native_chip_version_save", callback=self.guarded(self._library_save))
                        dpg.add_button(label="从 JSON 导入芯片…", callback=self.guarded(lambda **_: dpg.show_item("native_chip_import")))
                    dpg.add_text("", tag="native_chip_saved_path", color=MUTED, wrap=880)
                    with dpg.child_window(height=330, width=-1):
                        with dpg.group(tag="native_local_chip_files"):
                            pass
            dpg.add_text("", tag="native_chip_tool_error", color=RED, wrap=890)
            dpg.add_button(label="关闭", callback=self.guarded(lambda **_: dpg.hide_item("native_chip_tools")))
        self.chip_directory = self.data_directory / "chips"
        self.chip_directory.mkdir(parents=True, exist_ok=True)
        with dpg.file_dialog(tag="native_chip_import", show=False, modal=True, width=760, height=470, default_path=str(self.chip_directory), callback=self.guarded(self._library_import)):
            dpg.add_file_extension(".json", color=BLUE)
            dpg.add_file_extension(".*")

    def _open_chip_tools(self, definition_id=None, **_):
        self._bind_modal('native_chip_tools')
        self._bind_modal('native_chip_import')
        self.edit_definition = definition_id or self.document.scope_id
        if self.edit_definition is None:
            instance = self.document.selected_instance
            if instance and instance.get("kind") == "chip":
                self.edit_definition = instance.get("definition_id")
        if self.edit_definition is None and self.document.graph.get("definitions"):
            self.edit_definition = self.document.graph["definitions"][0]["id"]
        self._render_chip_tools()
        dpg.show_item("native_chip_tools")

    def _render_chip_tools(self):
        definitions = [item["id"] for item in self.document.graph.get("definitions", [])]
        if self.edit_definition not in definitions:
            self.edit_definition = definitions[0] if definitions else None
        dpg.set_value("native_chip_scope", "当前作用域：" + (self.document.scope_id or "根模型"))
        dpg.configure_item("native_chip_definition", items=definitions)
        dpg.set_value("native_chip_definition", self.edit_definition or "")
        dpg.set_value("native_chip_tool_error", "")
        self._render_interface_rows(); self._load_public_rows(); self._render_local_library()

    def _refresh_chip_controls(self):
        dpg.configure_item("native_undo", enabled=bool(getattr(self.document, "can_undo", False)))
        dpg.configure_item("native_redo", enabled=bool(getattr(self.document, "can_redo", False)))
        if dpg.does_item_exist("native_chip_tools") and dpg.is_item_shown("native_chip_tools"):
            token = self._modal_tokens.get('native_chip_tools')
            key = self._bound_canvas_id or self.workbench.focused_view_id
            current = self.canvas_host.token(key)
            if token and (token.view_id, token.context_epoch, token.document_epoch) == (current.view_id, current.context_epoch, current.document_epoch):
                self._bind_modal('native_chip_tools')
                self._render_chip_tools()
            else:
                dpg.hide_item('native_chip_tools')

    def _undo(self, **_):
        if self.document.undo():
            self.probe_selection.clear(); self._refresh()

    def _redo(self, **_):
        if self.document.redo():
            self.probe_selection.clear(); self._refresh()

    def _render_interface_rows(self):
        parent = "native_interface_rows"
        dpg.delete_item(parent, children_only=True)
        self.interface_widgets = {}
        sources = ["未连接"] + self._source_options()
        for item in self._scope()["inputs"]:
            identifier = item["id"]
            with dpg.group(horizontal=True, parent=parent):
                dpg.add_text("输入", color=BLUE)
                name = dpg.add_input_text(default_value=identifier, width=140)
                if self.document.graph['schema_version']==2:
                    fields=self.tensor_panel.fields(item)
                else:
                    fields={'shape': dpg.add_input_text(default_value=json.dumps(item.get('shape'), ensure_ascii=False), width=210)}
                self.interface_widgets[("input", identifier)] = {"id": name, **fields}
                dpg.add_button(label="改 ID", callback=self.guarded(self._interface_rename, "input", identifier))
                dpg.add_button(label="应用契约" if self.document.graph['schema_version']==2 else "改形状",
                               callback=self.guarded(self._interface_shape, identifier))
                dpg.add_button(label="删除", callback=self.guarded(self._interface_remove, "input", identifier))
        dpg.add_separator(parent=parent)
        for item in self._scope()["outputs"]:
            identifier = item["id"]
            with dpg.group(horizontal=True, parent=parent):
                dpg.add_text("输出", color=BLUE)
                name = dpg.add_input_text(default_value=identifier, width=140)
                source = dpg.add_combo(items=sources, default_value=item.get("source") or "未连接", width=280, callback=self.guarded(self._set_output, identifier))
                self.interface_widgets[("output", identifier)] = {"id": name, "source": source}
                dpg.add_button(label="改 ID", callback=self.guarded(self._interface_rename, "output", identifier))
                dpg.add_button(label="删除", callback=self.guarded(self._interface_remove, "output", identifier))
        dpg.configure_item("native_new_output_source", items=sources)

    def _chip_action(self, action):
        try:
            value = action()
            self._refresh()
            return value
        except Exception as error:
            dpg.set_value("native_chip_tool_error", str(error))
            self.log(str(error), "错误")
            raise

    def _interface_add_input(self, **_):
        shape = json.loads(dpg.get_value("native_new_input_shape"))
        return self._chip_action(lambda: self.document.add_input(dpg.get_value("native_new_input_id"), shape))

    def _interface_add_output(self, **_):
        source = dpg.get_value("native_new_output_source")
        return self._chip_action(lambda: self.document.add_output(dpg.get_value("native_new_output_id"), None if source == "未连接" else source))

    def _interface_rename(self, kind, identifier, **_):
        value = dpg.get_value(self.interface_widgets[(kind, identifier)]["id"])
        method = self.document.rename_input if kind == "input" else self.document.rename_output
        return self._chip_action(lambda: method(identifier, value))

    def _interface_shape(self, identifier, **_):
        if self.document.graph['schema_version']==2:
            fields={key:value for key,value in self.interface_widgets[('input',identifier)].items() if key!='id'}
            return self._chip_action(lambda: self.document.set_tensor_input(identifier,self.tensor_panel.read(fields)))
        value = json.loads(dpg.get_value(self.interface_widgets[("input", identifier)]["shape"]))
        return self._chip_action(lambda: self.document.set_input_shape(identifier, value))

    def _interface_remove(self, kind, identifier, **_):
        method = self.document.remove_input if kind == "input" else self.document.remove_output
        return self._chip_action(lambda: method(identifier))

    def _choose_definition(self, app_data=None, **_):
        self.edit_definition = app_data or None
        self._load_public_rows()

    def _load_public_rows(self):
        definition = self._definition(self.edit_definition) or {}
        self.public_rows = [{"id": item["id"], "name": item.get("name", ""), "type": item.get("type", "auto"), "default": json.dumps(item.get("default"), ensure_ascii=False), "bindings": json.dumps(definition.get("bindings", {}).get(item["id"], []), ensure_ascii=False)} for item in definition.get("public_parameters", [])]
        dpg.set_value("native_chip_version", definition.get("version", "1.0.0"))
        paths = [item["id"] + "." + name for item in definition.get("graph", {}).get("instances", []) for name in item.get("params", {})]
        dpg.set_value("native_binding_paths", "可绑定路径：" + ", ".join(paths))
        dpg.configure_item("native_public_apply", enabled=bool(definition))
        self._render_public_rows()

    def _public_draft(self):
        return [{key: dpg.get_value(widget) for key, widget in row.items()} for row in self.public_row_widgets]

    def _render_public_rows(self):
        dpg.delete_item("native_public_rows", children_only=True)
        self.public_row_widgets = []
        with dpg.table(parent="native_public_rows", header_row=True, resizable=True, borders_innerH=True, borders_outerH=True):
            for label in ("ID", "名称", "类型", "默认值 JSON", "绑定路径列表 JSON", ""):
                dpg.add_table_column(label=label)
            for index, row in enumerate(self.public_rows):
                with dpg.table_row():
                    widgets = {}
                    for key in ("id", "name", "type", "default", "bindings"):
                        widgets[key] = (dpg.add_combo(items=TYPES, default_value=row[key], width=80) if key == "type" else dpg.add_input_text(default_value=row[key], width=-1))
                    self.public_row_widgets.append(widgets)
                    dpg.add_button(label="删", callback=self.guarded(self._public_remove_row, index))

    def _public_add_row(self, **_):
        self.public_rows = self._public_draft()
        self.public_rows.append({"id": "parameter_" + str(len(self.public_rows) + 1), "name": "", "type": "auto", "default": "8", "bindings": "[]"})
        self._render_public_rows()

    def _public_remove_row(self, index, **_):
        self.public_rows = self._public_draft()
        self.public_rows.pop(index); self._render_public_rows()

    def _public_apply(self, **_):
        parameters, bindings = [], {}
        for row in self._public_draft():
            declaration = {"id": row["id"], "type": row["type"], "default": json.loads(row["default"])}
            if row["name"]:
                declaration["name"] = row["name"]
            parameters.append(declaration)
            paths = json.loads(row["bindings"])
            if paths != []:
                bindings[row["id"]] = paths
        return self._chip_action(lambda: self.document.set_public_parameters(self.edit_definition, parameters, bindings))

    def _render_local_library(self):
        dpg.delete_item("native_local_chip_files", children_only=True)
        files = sorted(self.chip_directory.glob("*.json"))[:200]
        for path in files:
            with dpg.group(horizontal=True, parent="native_local_chip_files"):
                dpg.add_button(label="导入", callback=self.guarded(self._library_import_path, path))
                dpg.add_text(path.name, color=MUTED)
        if not files:
            dpg.add_text("本地芯片库为空。", color=MUTED, parent="native_local_chip_files")

    def _library_save(self, **_):
        if not self.edit_definition:
            raise ValueError("请选择要保存的芯片定义。")
        path = self.document.save_chip(self.edit_definition, dpg.get_value("native_chip_version"), self.chip_directory)
        self._refresh()
        dpg.set_value("native_chip_saved_path", str(path))
        self._render_local_library()
        self.catalog_panel.refresh_local()
        self.log("已保存芯片版本：" + str(path), "完成")
        return path

    def _library_import(self, app_data=None, **_):
        return self._library_import_path(self._file_path(app_data))

    def _library_import_path(self, path, **_):
        identifier = self.document.import_chip(path)
        self.edit_definition = identifier
        self._refresh()
        self._render_chip_tools()
        self.log("已导入芯片定义：" + identifier, "完成")
        return identifier

    def _encapsulation_draft(self):
        return [{key: dpg.get_value(widget) for key, widget in row.items()} for row in self.encapsulation_widgets]

    def _render_encapsulation_rows(self):
        dpg.delete_item("native_chip_outputs", children_only=True)
        self.encapsulation_widgets = []
        sources = [source for source in self._source_options() if source.startswith("call:") and source.split(":")[1] in self.chip_calls]
        for index, row in enumerate(self.encapsulation_rows):
            with dpg.group(horizontal=True, parent="native_chip_outputs"):
                identifier = dpg.add_input_text(default_value=row["id"], width=120)
                source = dpg.add_combo(items=sources, default_value=row["source"], width=280)
                self.encapsulation_widgets.append({"id": identifier, "source": source})
                dpg.add_button(label="删除", callback=self.guarded(self._encapsulation_remove, index))

    def _encapsulation_add(self, **_):
        self.encapsulation_rows = self._encapsulation_draft()
        sources = [source for source in self._source_options() if source.startswith("call:") and source.split(":")[1] in self.chip_calls]
        self.encapsulation_rows.append({"id": "out_" + str(len(self.encapsulation_rows) + 1), "source": sources[-1] if sources else ""})
        self._render_encapsulation_rows()

    def _encapsulation_remove(self, index, **_):
        self.encapsulation_rows = self._encapsulation_draft()
        self.encapsulation_rows.pop(index); self._render_encapsulation_rows()
