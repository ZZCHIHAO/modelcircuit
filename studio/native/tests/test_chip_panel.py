"""S3 native controls use the app's own callbacks, never desktop input."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import dearpygui.dearpygui as dpg

from native.document import Document
from native.paths import resource_root
from native.ui import NativeWorkbench


def two_output_graph():
    graph = Document().graph
    definition = graph["definitions"][0]
    definition["output_mode"] = "named"
    definition["graph"]["outputs"].append({"id": "encoded", "source": "call:project"})
    graph["calls"][2]["inputs"] = {"a": "call:encode_a:out", "b": "call:encode_b:encoded"}
    return graph


class ChipPanelTests(unittest.TestCase):
    def test_native_layout_exposes_chip_interface_history_and_paired_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            ui = NativeWorkbench(Document(), None, temporary, resource_root())
            try:
                ui.build()
                for tag in ("native_undo", "native_redo", "native_chip_tools", "native_scope_ports", "native_public_parameters", "native_chip_library", "native_paired_settings", "native_image_port", "native_tabular_port"):
                    self.assertTrue(dpg.does_item_exist(tag), tag)
            finally:
                ui.shutdown()

    def test_multi_output_sockets_and_probes_require_a_named_port(self):
        with tempfile.TemporaryDirectory() as temporary:
            ui = NativeWorkbench(Document(two_output_graph()), None, temporary, resource_root())
            try:
                ui.build()
                sources = set(ui.source_attributes.values())
                self.assertIn("call:encode_a:out", sources)
                self.assertIn("call:encode_a:encoded", sources)
                ui.document.select("call:encode_a")
                ui._render_inspector()
                self.assertIsNone(ui._probe_source())
                ui._select_probe_output("encode_a", app_data="encoded")
                self.assertEqual(ui._probe_source(), "call:encode_a:encoded")
                self.assertTrue(dpg.does_item_exist("native_probe_output"))
            finally:
                ui.shutdown()

    def test_interface_public_binding_library_and_history_callbacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            ui = NativeWorkbench(Document(), None, temporary, resource_root())
            try:
                ui.build(); ui._open_chip_tools()
                ui.selected_refs = ["input:a"]
                original = deepcopy(ui.document.graph)
                with self.assertRaises(ValueError):
                    ui._delete_selected()
                self.assertEqual(ui.document.graph, original)
                ui.selected_refs = []
                dpg.set_value("native_new_input_id", "extra")
                dpg.set_value("native_new_input_shape", "[null, 2]")
                ui._interface_add_input()
                dpg.set_value(ui.interface_widgets[("input", "extra")]["id"], "extra_renamed")
                ui._interface_rename("input", "extra")
                dpg.set_value(ui.interface_widgets[("input", "extra_renamed")]["shape"], "[null, 3]")
                ui._interface_shape("extra_renamed")
                ui._interface_remove("input", "extra_renamed")
                ui._undo(); self.assertIn("extra_renamed", [item["id"] for item in ui.document.scope["inputs"]])
                ui._redo(); self.assertNotIn("extra_renamed", [item["id"] for item in ui.document.scope["inputs"]])
                dpg.set_value("native_new_output_id", "extra_output")
                dpg.set_value("native_new_output_source", "call:fuse")
                ui._interface_add_output()
                dpg.set_value(ui.interface_widgets[("output", "extra_output")]["id"], "features_copy")
                ui._interface_rename("output", "extra_output")
                ui._interface_remove("output", "features_copy")
                ui._choose_definition(app_data="encoder_chip")
                dpg.set_value(ui.public_row_widgets[0]["type"], "int")
                ui._public_apply()
                self.assertEqual(ui.document.graph["definitions"][0]["public_parameters"][0]["type"], "int")
                before = deepcopy(ui.document.graph)
                dpg.set_value(ui.public_row_widgets[0]["bindings"], '["missing.parameter"]')
                with self.assertRaises(ValueError):
                    ui._public_apply()
                self.assertEqual(ui.document.graph, before)
                ui._load_public_rows()
                dpg.set_value("native_chip_tool_error", "")
                dpg.set_value("native_chip_version", "invalid-version")
                ui.guarded(ui._library_save)()
                self.assertTrue(dpg.get_value("native_chip_tool_error"))
                dpg.set_value("native_chip_version", "1.2.3")
                saved = ui._library_save()
                self.assertTrue(saved.is_file())
                self.assertIn("设计 r" + str(ui.document.graph["revision"]), dpg.get_value("native_project"))
                imported = ui._library_import_path(saved)
                self.assertIn(imported, [item["id"] for item in ui.document.graph["definitions"]])
            finally:
                ui.shutdown()

    def test_nested_encapsulation_exposes_two_selected_output_ports(self):
        with tempfile.TemporaryDirectory() as temporary:
            ui = NativeWorkbench(Document(), None, temporary, resource_root())
            try:
                ui.build(); ui._enter("encode_a")
                ui.selected_refs = ["call:project", "call:activate"]
                ui._open_encapsulate()
                dpg.set_value("native_chip_id", "nested_features")
                ui.encapsulation_rows = [{"id": "features", "source": "call:project"}, {"id": "activation", "source": "call:activate"}]
                ui._render_encapsulation_rows(); ui._save_chip()
                self.assertEqual(ui.document.scope_id, "encoder_chip")
                call = ui.document.selected_call
                self.assertEqual(ui.document.output_ports(call["id"]), ["features", "activation"])
                ui._select_probe_output(call["id"], app_data="features")
                self.assertEqual(ui._probe_source(), "call:encode_a/call:" + call["id"] + ":features")
            finally:
                ui.shutdown()

    def test_incomplete_chip_draft_remains_renderable_for_repair(self):
        graph = Document().graph
        graph["instances"][0]["definition_id"] = "missing_definition"
        document = Document(graph); document.select("call:encode_a")
        with tempfile.TemporaryDirectory() as temporary:
            ui = NativeWorkbench(document, None, temporary, resource_root())
            try:
                ui.build()
                self.assertFalse(ui.report["valid"])
                self.assertIn("input:a", ui._source_options())
                ui._open_chip_tools()
                self.assertTrue(dpg.is_item_shown("native_chip_tools"))
            finally:
                ui.shutdown()

    def test_unbound_public_declaration_survives_open_and_apply(self):
        graph = Document().graph
        graph["definitions"][0]["public_parameters"].append({"id": "unused", "default": 7})
        with tempfile.TemporaryDirectory() as temporary:
            ui = NativeWorkbench(Document(graph), None, temporary, resource_root())
            try:
                ui.build(); ui._open_chip_tools("encoder_chip")
                ui._public_apply()
                declaration = ui.document.graph["definitions"][0]
                self.assertNotIn("unused", declaration["bindings"])
                self.assertEqual(declaration["public_parameters"][-1]["default"], 7)
            finally:
                ui.shutdown()

    def test_multi_selection_delete_failure_preserves_model_and_canvas(self):
        with tempfile.TemporaryDirectory() as temporary:
            ui = NativeWorkbench(Document(), None, temporary, resource_root())
            try:
                ui.build(); ui.document.add_input("extra", [None, 2]); ui._refresh()
                ui.selected_refs = ["input:extra", "input:a"]
                before = deepcopy(ui.document.graph)
                with self.assertRaises(ValueError):
                    ui._delete_selected()
                self.assertEqual(ui.document.graph, before)
                self.assertIn("input:extra", ui.ref_nodes)
            finally:
                ui.shutdown()


if __name__ == "__main__":
    unittest.main()
