import csv
from pathlib import Path
import tempfile
import time
import unittest

import dearpygui.dearpygui as dpg
from PIL import Image

from native.document import Document
from native.paths import resource_root
from native.ui import NativeWorkbench


class PairedCallbackTests(unittest.TestCase):
    def test_paired_columns_inspection_starter_mapping_and_reopen(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            image_root = directory / "images"; image_root.mkdir()
            path = directory / "paired.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "patient_id", "image_path", "age", "score", "label", "split"])
                for index in range(40):
                    Image.new("RGB", (10, 8), (index * 5, index * 3, index + 1)).save(image_root / f"{index}.png")
                    writer.writerow([f"s{index}", f"p{index}", f"{index}.png", 30 + index % 3, index / 40, "a" if index % 2 else "b", "train" if index < 24 else "val" if index < 32 else "test"])
            ui = NativeWorkbench(Document(), None, directory, resource_root())
            try:
                ui.build(); ui._mode_changed(app_data="配对图片与表格分类")
                dpg.set_value("native_data_path", str(path))
                dpg.set_value("native_paired_image_root", str(image_root))
                dpg.set_value("native_image_height", 8); dpg.set_value("native_image_width", 10)
                dpg.set_value("native_image_channels", "1")
                ui._read_columns(); self._wait(ui, lambda: bool(ui.csv_header))
                dpg.set_value("native_csv_split", "split"); ui._csv_selection_changed()
                config = ui._configuration()
                self.assertEqual(config["data"]["feature_columns"], ["age", "score"])
                self.assertEqual(config["data"]["group_column"], "patient_id")
                self.assertEqual(config["input_ports"], {"image": "image", "tabular": "clinical"})
                ui._inspect_data(); self._wait(ui, lambda: ui.manifest is not None)
                self.assertEqual(ui.manifest["input_shapes"]["image"], [None, 1, 8, 10])
                before = ui.document.semantic_key(); ui._request_starter()
                self.assertEqual(ui.document.semantic_key(), before)
                ui._replace_starter()
                self.assertEqual([item["id"] for item in ui.document.graph["inputs"]], ["image", "clinical"])
                self.assertEqual(ui._run_configuration()["input_ports"], config["input_ports"])
                dpg.set_value("native_image_port", "wrong_image"); ui._experiment_changed()
                with self.assertRaises(ValueError):
                    ui._run_configuration()
                dpg.set_value("native_image_port", "image"); ui._experiment_changed()
                saved = directory / "saved.json"
                ui._json_saved(app_data={"file_path_name": str(saved)})
                ui._mode_changed(app_data="S1 合成 CPU（演示）")
                ui._json_opened(app_data={"file_path_name": str(saved)})
                self.assertEqual(ui._configuration()["input_ports"], config["input_ports"])
                self.assertEqual(ui._configuration()["data"]["image_root"], str(image_root))
                self.assertIsNone(ui.manifest)
            finally:
                ui.shutdown()

    @staticmethod
    def _wait(ui, condition):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            ui.process_callbacks(); ui.tick(); dpg.render_dearpygui_frame()
            if condition():
                return
        raise AssertionError(str(ui.messages[-4:]))


if __name__ == "__main__":
    unittest.main()
