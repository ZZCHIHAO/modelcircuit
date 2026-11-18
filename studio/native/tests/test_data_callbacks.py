"""Application-owned DPG callback checks; no OS or browser input automation."""
import csv
from pathlib import Path
import tempfile
import threading
import time
import unittest

import dearpygui.dearpygui as dpg
from PIL import Image

from native.document import Document
from native.paths import resource_root
from native.ui import NativeWorkbench


class DataCallbackTests(unittest.TestCase):
    def test_csv_inspection_confirmation_and_workspace_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "features.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["f1", "f2", "label"])
                writer.writerows([[i + 0.1, i * 0.3 + 0.02, "a" if i % 2 else "b"] for i in range(40)])
            ui = NativeWorkbench(Document(), None, directory, resource_root())
            try:
                ui.build()
                dpg.set_value("native_bottom_tabs", dpg.get_alias_id("native_loss_tab"))
                for _ in range(3):
                    ui.tick(); dpg.render_dearpygui_frame()
                self.assertTrue(dpg.get_item_configuration("native_loss_legend")["horizontal"])
                self.assertGreaterEqual(dpg.get_item_rect_size("native_loss_plot")[1], 120)
                ui._mode_changed(app_data="CSV 数值分类")
                dpg.set_value("native_data_path", str(path))
                ui._read_columns()
                self._wait(ui, lambda: bool(ui.csv_header))
                self.assertEqual(ui._configuration()["data"]["feature_columns"], ["f1", "f2"])
                dpg.set_value("native_csv_features", '["f2"]')
                ui._experiment_changed()
                ui.csv_header = []
                ui._read_columns()
                self._wait(ui, lambda: bool(ui.csv_header))
                self.assertEqual(ui._configuration()["data"]["feature_columns"], ["f2"])
                dpg.set_value("native_lr", 0.000001)
                started = time.monotonic(); ui._inspect_data()
                self.assertLess(time.monotonic() - started, 0.1)
                self._wait(ui, lambda: ui.manifest is not None)
                self.assertEqual(sum(ui.manifest["counts"].values()), 40)
                before = ui.document.semantic_key()
                ui.document.graph["revision"] = 7
                ui._request_starter()
                self.assertEqual(ui.document.semantic_key(), before)
                self.assertTrue(dpg.is_item_shown("native_starter_replace"))
                ui._replace_starter()
                self.assertEqual(ui.document.graph["revision"], 8)
                self.assertEqual(len(ui.document.graph["inputs"]), 1)
                config = ui._run_configuration()
                self.assertEqual(config["expected_dataset_digest"], ui.manifest["dataset_digest"])
                code = ui.document.export_python()
                dpg.set_value("native_python_code", code)
                ui._import_python()
                self.assertEqual(ui._configuration(), config)
                saved = directory / "saved.json"
                ui._json_saved(app_data={"file_path_name": str(saved)})
                ui._mode_changed(app_data="S1 合成 CPU（演示）")
                ui._json_opened(app_data={"file_path_name": str(saved)})
                self.assertEqual(ui.experiment_mode, "csv")
                self.assertEqual(ui._configuration()["data"]["path"], str(path))
                self.assertIsNone(ui.manifest, "Reopening settings requires fresh content inspection")
                # Editing an on-enter field while an inspection is pending must
                # not leave both inspect buttons disabled after its stale reply.
                from native.workspace_state import data_identity
                gate = threading.Event()
                ui.inspecting = True
                ui.inspect_request_key = data_identity(ui._configuration())
                ui.queries.submit("inspect", lambda cancelled: (gate.wait(2), {"dataset_digest": "old"})[-1])
                dpg.set_value("native_data_path", str(directory / "changed.csv"))
                ui._save_draft()
                gate.set()
                self._wait(ui, lambda: not ui.inspecting)
                self.assertTrue(dpg.get_item_configuration("native_inspect")["enabled"])
                self.assertIsNone(ui.manifest)
                previous_features = dpg.get_value("native_csv_features")
                dpg.set_value("native_csv_features", "[")
                try:
                    ui._save_draft()
                finally:
                    dpg.set_value("native_csv_features", previous_features)
            finally:
                ui.shutdown()

    def test_image_inspection_and_cnn_starter_use_selected_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for class_index, name in enumerate(("a", "b")):
                folder = directory / "images" / name
                folder.mkdir(parents=True)
                for index in range(12):
                    Image.new("RGB", (18, 16), (index * 11, class_index * 100 + index, index * 3)).save(folder / f"{index}.png")
            ui = NativeWorkbench(Document(), None, directory, resource_root())
            try:
                ui.build()
                ui._mode_changed(app_data="ImageFolder 图片分类")
                dpg.set_value("native_data_path", str(directory / "images"))
                dpg.set_value("native_image_height", 16)
                dpg.set_value("native_image_width", 18)
                dpg.set_value("native_image_channels", "1")
                ui._inspect_data()
                self._wait(ui, lambda: ui.manifest is not None)
                self.assertEqual(ui.manifest["input_shape"], [None, 1, 16, 18])
                ui._request_starter(); ui._replace_starter()
                for _ in range(4):
                    ui.tick(); dpg.render_dearpygui_frame()
                bounds = [(ref, *dpg.get_item_pos(item), *dpg.get_item_rect_size(item)) for ref, item in ui.ref_nodes.items()]
                for index, first in enumerate(bounds):
                    for second in bounds[index + 1:]:
                        _, x, y, width, height = first
                        _, xx, yy, ww, hh = second
                        self.assertFalse(min(x + width, xx + ww) > max(x, xx) and min(y + height, yy + hh) > max(y, yy), (first, second))
                self.assertTrue(ui.document.validation()["valid"])
                self.assertEqual(ui.document.graph["inputs"][0]["shape"], [None, 1, 16, 18])
                self.assertEqual(ui._run_configuration()["data"]["channels"], 1)
            finally:
                ui.shutdown()

    @staticmethod
    def _wait(ui, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            ui.process_callbacks(); ui.tick(); dpg.render_dearpygui_frame()
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("Native query did not complete: " + str(ui.messages[-4:]))


if __name__ == "__main__":
    unittest.main()
