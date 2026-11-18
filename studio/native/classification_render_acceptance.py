"""Real file-based native callback/framebuffer acceptance, without OS input."""
import csv
from pathlib import Path
import time

import dearpygui.dearpygui as dpg
from PIL import Image

from .document import Document
from .render_acceptance import _capture, _frames, _wait_ui
from .ui import NativeWorkbench
from .workspace_state import read_workspace


def _wait_until(ui, condition, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _frames(ui, 1)
        if condition():
            return
    raise TimeoutError("Native data/device query did not finish: " + str(ui.messages[-4:]))


def _fixtures(directory):
    """Small generated test fixtures, never presented as scientific results."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "features.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["f1", "f2", "label", "subject", "split"])
        for index in range(40):
            split = "train" if index < 24 else "val" if index < 32 else "test"
            writer.writerow([index / 40 + 0.001, (index % 2) * 0.8 + index / 200, "a" if index % 2 else "b", "sample_" + str(index), split])
    image_root = root / "images"
    for class_index, name in enumerate(("a", "b")):
        for index in range(12):
            folder = image_root / name
            folder.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (18, 16), (index * 17, class_index * 100 + index, 30 + index * 5)).save(folder / f"image-{index}.png")
    return csv_path, image_root


def _assert_result(ui, kind, device):
    result = ui.result
    assert result and result.get("profile") == "classification-v2", result
    assert result["data_source"] == kind
    assert result["device"] == device
    assert result["probe_context"]["split"] == "val"
    assert result["probe_context"]["mode"] == "eval"
    assert result["probe_context"]["sample_ids"]
    assert result["metrics"]["test"]["count"] > 0
    assert len(result["history"]) == 2
    assert all("train_loss" in item and "val_loss" in item for item in result["history"])
    assert len(dpg.get_value("native_loss_series")[0]) == 2
    assert len(dpg.get_value("native_val_loss_series")[0]) == 2
    assert dpg.get_item_configuration("native_loss_legend")["horizontal"]
    assert dpg.get_item_rect_size("native_loss_plot")[1] >= 120
    assert dpg.get_text_size("train loss     val loss")[0] + 80 < dpg.get_item_rect_size("native_loss_plot")[0]
    assert "final test" in dpg.get_value("native_final_metrics")
    assert not ui.snapshot()["result_stale"]
    for name, path in result["artifacts"].items():
        assert Path(path).is_file(), (name, path)
    ui.document.select("call:logits")
    ui._render_inspector(); ui._render_probe()
    source = ui.snapshot()["probe_source"]
    assert source == "call:logits"
    assert result["probes"][source]["source"] == "measured"
    return {"run_id": ui.run_record["run_id"], "device": result["device"], "dataset_digest": result["dataset"]["dataset_digest"],
            "counts": result["dataset"]["counts"], "metrics": result["metrics"], "best_epoch": result["best_epoch"],
            "artifacts": result["artifacts"], "probe_context": result["probe_context"],
            "train_validation_curves": True, "final_test_only_after_best_checkpoint": True}


def check_classification_render(manager, directory, resources, output_directory, device="cpu"):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path, image_root = _fixtures(output / "classification-fixtures")
    ui = NativeWorkbench(Document(), manager, directory, resources)
    runs, timings = {}, {}
    try:
        ui.build()
        dpg.set_value("native_right_tabs", dpg.get_alias_id("native_experiment_tab"))
        ui._mode_changed(app_data="CSV 数值分类")
        _wait_until(ui, lambda: ui.capabilities is not None)
        if device not in ui.available_devices:
            raise RuntimeError("Requested render-check device is unavailable: " + device + "; " + str(ui.capabilities.get("reason")))
        dpg.set_value("native_device", device)
        dpg.set_value("native_data_path", str(csv_path))
        ui._read_columns()
        _wait_until(ui, lambda: bool(ui.csv_header))
        dpg.set_value("native_csv_group", "subject")
        dpg.set_value("native_csv_split", "split")
        ui._csv_selection_changed()
        dpg.set_value("native_epochs", 2)
        dpg.set_value("native_max_runtime", 60)
        ui._experiment_changed()
        started = time.monotonic(); ui._inspect_data()
        timings["csv_inspect_dispatch_ms"] = round((time.monotonic() - started) * 1000, 3)
        _wait_until(ui, lambda: ui.manifest is not None)
        before = ui.document.semantic_key()
        ui._request_starter()
        assert ui.document.semantic_key() == before, "Preparing a starter must not replace the graph"
        assert dpg.is_item_shown("native_starter_replace")
        ui._replace_starter()
        assert len(ui.document.graph["inputs"]) == 1
        ui._start_run()
        run = _wait_ui(ui)
        if run["status"] != "succeeded":
            raise RuntimeError(str(run.get("error")))
        runs["csv"] = _assert_result(ui, "csv", device)
        # The model view can move freely; optimizer/device/data changes must
        # invalidate measurements, including when the graph revision is equal.
        ui.document.move("call:logits", 760, 160)
        assert not ui.snapshot()["result_stale"]
        dpg.set_value("native_optimizer", "sgd"); ui._experiment_changed()
        assert ui.snapshot()["result_stale"]
        dpg.set_value("native_optimizer", "adamw"); ui._experiment_changed()
        assert not ui.snapshot()["result_stale"]
        dpg.set_value("native_bottom_tabs", dpg.get_alias_id("native_loss_tab"))
        _capture(ui, output / "native-csv-classification.png")

        saved = output / "classification-workspace.json"
        ui._json_saved(app_data={"file_path_name": str(saved)})
        saved_document, saved_experiment = read_workspace(saved)
        assert saved_experiment == ui._configuration()
        assert saved_document.semantic_key() == ui.document.semantic_key()
        ui._mode_changed(app_data="S1 合成 CPU（演示）")
        ui._json_opened(app_data={"file_path_name": str(saved)})
        assert ui.experiment_mode == "csv"
        assert ui._configuration()["data"]["feature_columns"] == ["f1", "f2"]
        assert ui._configuration()["data"]["group_column"] == "subject"
        assert ui._configuration()["data"]["split_column"] == "split"
        assert ui.manifest is None, "Restored data must be inspected again"

        ui._mode_changed(app_data="ImageFolder 图片分类")
        dpg.set_value("native_device", device)
        dpg.set_value("native_data_path", str(image_root))
        dpg.set_value("native_image_height", 16)
        dpg.set_value("native_image_width", 18)
        dpg.set_value("native_image_channels", "1")
        dpg.set_value("native_epochs", 2)
        dpg.set_value("native_max_runtime", 60)
        ui._experiment_changed()
        started = time.monotonic(); ui._inspect_data()
        timings["image_inspect_dispatch_ms"] = round((time.monotonic() - started) * 1000, 3)
        _wait_until(ui, lambda: ui.manifest is not None)
        assert ui.manifest["input_shape"] == [None, 1, 16, 18]
        ui._request_starter(); ui._replace_starter()
        ui._start_run()
        run = _wait_ui(ui)
        if run["status"] != "succeeded":
            raise RuntimeError(str(run.get("error")))
        runs["image_folder"] = _assert_result(ui, "image_folder", device)
        _wait_until(ui, lambda: ui.capabilities is not None)
        if not ui.capabilities["cuda_available"]:
            assert ui.available_devices == ["cpu"]
            assert ui.capabilities.get("reason")
        dpg.set_value("native_bottom_tabs", dpg.get_alias_id("native_probe_tab"))
        _capture(ui, output / "native-image-classification.png")
        ui._save_draft()
        _, automatic_experiment = read_workspace(Path(directory) / "workspace.json")
        assert automatic_experiment["data"]["image_size"] == [16, 18]
        assert automatic_experiment["data"]["channels"] == 1
        return {"status": "passed", "kind": "native-classification-render-callbacks", "requested_device": device, "fixture_notice": "Generated CSV and PNG test fixtures; no scientific-performance claim",
                "runs": runs, "query_timings": timings, "workspace_experiment_restored": True,
                "starter_replacement_requires_confirmation": True, "settings_invalidate_measurements": True,
                "capabilities": ui.capabilities, "snapshot": ui.snapshot(),
                "screenshots": ["native-csv-classification.png", "native-image-classification.png"]}
    finally:
        ui.shutdown()
