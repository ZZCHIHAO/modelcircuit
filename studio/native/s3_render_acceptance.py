"""S3 paired/chip acceptance using only native callbacks and framebuffer APIs."""
import json
from pathlib import Path
import time

import dearpygui.dearpygui as dpg

from .document import Document
from .render_acceptance import _capture, _frames, _wait_ui
from .ui import NativeWorkbench
from .workspace_state import read_workspace


def _wait(ui, predicate):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        _frames(ui, 1)
        if predicate():
            return
    raise TimeoutError("S3 native query deadline exceeded: " + str(ui.messages[-5:]))


def check_s3_render(manager, directory, resources, output_directory, device="cpu"):
    from .s3_acceptance import create_paired_fixture
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    fixture = create_paired_fixture(output / "s3-fixtures")
    spec = fixture["data_spec"]
    ui = NativeWorkbench(Document(), manager, directory, resources)
    screenshots = ["native-s3-paired-starter.png", "native-s3-paired-data.png", "native-s3-chip-probe.png", "native-s3-chip-interfaces.png", "native-s3-chip-public.png"]
    try:
        ui.build(); ui._mode_changed(app_data="配对图片与表格分类")
        dpg.set_value("native_right_tabs", dpg.get_alias_id("native_experiment_tab"))
        _wait(ui, lambda: ui.capabilities is not None)
        if device not in ui.available_devices:
            raise RuntimeError("Requested S3 render device is unavailable: " + device + "; " + str(ui.capabilities.get("reason")))
        values = {"native_data_path": spec["path"], "native_paired_image_root": spec["image_root"],
                  "native_image_height": spec["image_size"][0], "native_image_width": spec["image_size"][1],
                  "native_image_channels": str(spec["channels"]), "native_epochs": 2, "native_batch": 8,
                  "native_max_runtime": 90, "native_device": device}
        for tag, value in values.items():
            dpg.set_value(tag, value)
        ui._read_columns(); _wait(ui, lambda: bool(ui.csv_header))
        for tag, key in (("native_csv_target", "target_column"), ("native_csv_group", "group_column"),
                         ("native_csv_split", "split_column"), ("native_paired_image_column", "image_column"), ("native_paired_id_column", "id_column")):
            dpg.set_value(tag, spec[key] or "无")
        ui._csv_selection_changed()
        dpg.set_value("native_csv_features", json.dumps(spec["feature_columns"]))
        ui._experiment_changed()
        started = time.monotonic(); ui._inspect_data()
        dispatch_ms = round((time.monotonic() - started) * 1000, 3)
        _wait(ui, lambda: ui.manifest is not None)
        assert set(ui.manifest["input_shapes"]) == {"image", "tabular"}
        before = ui.document.semantic_key(); ui._request_starter()
        assert ui.document.semantic_key() == before and dpg.is_item_shown("native_starter_replace")
        ui._replace_starter()
        assert [item["id"] for item in ui.document.graph["inputs"]] == ["image", "clinical"]
        _capture(ui, output / screenshots[0])
        ui._open_data_settings(); _capture(ui, output / screenshots[1]); dpg.hide_item("native_data")

        ui.selected_refs = ["call:encode_image", "call:encode_table", "call:fuse"]
        ui._open_encapsulate(); dpg.set_value("native_chip_id", "paired_fusion")
        port = ui.document.output_ports("encode_image")[0]
        ui.encapsulation_rows = [{"id": "fused", "source": "call:fuse"}, {"id": "image_embedding", "source": "call:encode_image:" + port}]
        ui._render_encapsulation_rows(); ui._save_chip()
        fusion_call = ui.document.selected_call["id"]
        assert ui.document.output_ports(fusion_call) == ["fused", "image_embedding"]
        ui._enter(fusion_call); ui.document.move("output:image_embedding", 800, 300); ui._root()
        ui._open_chip_tools("paired_fusion"); ui._public_add_row()
        row = ui.public_row_widgets[0]
        concat = next(item for item in ui._definition("paired_fusion")["graph"]["instances"] if item["kind"] == "Concat")
        for key, value in {"id": "concat_dim", "type": "int", "default": json.dumps(concat["params"]["dim"]), "bindings": json.dumps([concat["id"] + ".dim"])}.items():
            dpg.set_value(row[key], value)
        ui._public_apply()
        dpg.set_value("native_chip_version", "1.0.0"); chip_path = ui._library_save()
        before = ui.document.semantic_key()
        imported = ui._library_import_path(chip_path)
        assert imported == "paired_fusion" and ui.document.semantic_key() == before
        dpg.hide_item("native_chip_tools")
        ui._share(fusion_call)
        shared_call = ui.document.selected_call["id"]
        ui._set_input("classify", "x", app_data="call:" + fusion_call + ":fused")
        ui._set_output("features", app_data="call:" + shared_call + ":image_embedding")
        ui._start_run(); record = _wait_ui(ui)
        if record["status"] != "succeeded":
            raise RuntimeError(str(record.get("error")))
        result = ui.result
        assert result["data_source"] == "paired_csv" and result["device"] == device
        assert result["probe_context"]["device"] == device and result["probe_context"]["split"] == "val"
        assert result["probe_context"]["input_ports"] == {"image": "image", "tabular": "clinical"}
        assert len(result["history"]) == 2 and result["metrics"]["test"]["count"] > 0
        assert len(dpg.get_value("native_loss_series")[0]) == 2 and len(dpg.get_value("native_val_loss_series")[0]) == 2
        assert any(set(item["call_ids"]) == {fusion_call, shared_call} for item in result["shared_instances"])
        ui._select_probe_output(fusion_call, app_data="fused")
        qualified_probe = ui._probe_source()
        assert qualified_probe == "call:" + fusion_call + ":fused"
        assert result["probes"][qualified_probe]["source"] == "measured"
        for name in ("image", "clinical"):
            assert result["probes"]["input:" + name]["shape"][0] == len(result["probe_context"]["sample_ids"])
        assert all(Path(path).is_file() for path in result["artifacts"].values())

        config = ui._configuration()
        public_contract = {definition["id"]: {"parameters": [{key: parameter[key] for key in ("id", "type", "default") if key in parameter} for parameter in definition.get("public_parameters", [])], "bindings": definition.get("bindings", {})} for definition in ui.document.graph["definitions"]}
        code = ui.document.export_python()
        assert "ports-pytorch-v3" in code
        (output / "paired-model.py").write_text(code, encoding="utf-8")
        dpg.set_value("native_python_code", code); ui._import_python()
        assert not dpg.get_value("native_python_error"), dpg.get_value("native_python_error")
        assert ui._configuration() == config
        imported_contract = {definition["id"]: {"parameters": [{key: parameter[key] for key in ("id", "type", "default") if key in parameter} for parameter in definition.get("public_parameters", [])], "bindings": definition.get("bindings", {})} for definition in ui.document.graph["definitions"]}
        assert imported_contract == public_contract, "Python roundtrip changed declared public parameter types/defaults"
        assert not ui.snapshot()["result_stale"], "Unedited Python roundtrip changed the measured experiment identity"
        assert ui.document.output_ports(fusion_call) == ["fused", "image_embedding"]
        ui._enter(fusion_call)
        ui.document.select("output:fused"); ui._refresh()
        assert result["probes"][ui._probe_source()]["source"] == "measured"
        dpg.set_value("native_bottom_tabs", dpg.get_alias_id("native_probe_tab"))
        _frames(ui, 4)
        editor_width, editor_height = dpg.get_item_rect_size("native_editor")
        for identifier in ("fused", "image_embedding"):
            item = ui.ref_nodes["output:" + identifier]
            x, y = dpg.get_item_pos(item); width, height = dpg.get_item_rect_size(item)
            assert x >= 0 and y >= 0 and x + width <= editor_width and y + height <= editor_height, "Named output is clipped in the native canvas"
        _capture(ui, output / screenshots[2])
        ui._open_chip_tools("paired_fusion")
        dpg.set_value("native_chip_tool_tabs", dpg.get_alias_id("native_scope_ports")); _capture(ui, output / screenshots[3])
        dpg.set_value("native_chip_tool_tabs", dpg.get_alias_id("native_public_parameters")); _capture(ui, output / screenshots[4])
        dpg.hide_item("native_chip_tools"); ui._root()
        saved = output / "paired-workspace.json"
        ui._json_saved(app_data={"file_path_name": str(saved)})
        restored_document, restored_experiment = read_workspace(saved)
        assert restored_experiment == config
        assert restored_document.output_ports(fusion_call) == ["fused", "image_embedding"]
        ui._mode_changed(app_data="S1 合成 CPU（演示）"); ui._json_opened(app_data={"file_path_name": str(saved)})
        assert ui._configuration() == config and ui.manifest is None
        return {"status": "passed", "kind": "s3-native-render-callbacks", "fixture_notice": "Generated paired CSV/PNG fixture; no scientific-performance claim",
                "requested_device": device, "device": result["device"], "run_id": record["run_id"], "metrics": result["metrics"],
                "probe_context": result["probe_context"], "qualified_probe": qualified_probe, "shared_instances": result["shared_instances"],
                "dataset": result["dataset"], "artifacts": result["artifacts"], "chip_path": str(chip_path), "chip_library_roundtrip": True,
                "workspace_roundtrip": True, "python_roundtrip": True, "python_profile": "ports-pytorch-v3", "inspect_dispatch_ms": dispatch_ms,
                "python_roundtrip_keeps_measurements_current": True, "requires_reinspection_after_reopen": ui.manifest is None,
                "capabilities": ui.capabilities, "screenshots": screenshots, "snapshot": ui.snapshot()}
    finally:
        ui.shutdown()
