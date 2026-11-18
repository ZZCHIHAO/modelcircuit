"""Portable native workspace storage and data-aware experiment identities.

This module is deliberately independent of DearPyGui and torch. Legacy graph
JSON remains readable; explicit and automatic saves use the same envelope.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from studio_core.experiment_spec import SUPERVISED_PROFILES

from .document import Document
from .persistence import replace_with_retry


def synthetic_config():
    return {"steps": 20, "lr": 0.05, "seed": 42, "batch_size": 16}


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _editable_experiment(experiment):
    """Check widget storage types without requiring a selected runnable dataset."""
    def require(condition, message):
        if not condition:
            raise ValueError("Workspace experiment: " + message)
    require(type(experiment) is dict, "must be an object")
    _encoded(experiment)
    real = isinstance(experiment.get("profile"), str) and experiment["profile"] in SUPERVISED_PROFILES
    if "profile" in experiment:
        require(real, "unsupported profile")
    integers = {"epochs", "batch_size", "seed", "max_runtime_seconds"} if real else {"steps", "batch_size", "seed"}
    allowed = integers | {"lr"}
    if real:
        allowed |= {"profile", "data", "optimizer", "device", "input_port", "input_ports", "resume_checkpoint", "expected_dataset_digest"}
    require(experiment.keys() <= allowed, "unsupported setting")
    for key in integers & experiment.keys():
        require(type(experiment[key]) is int, key + " must be an integer")
        bounds = {"epochs": (1, 1000), "steps": (1, 200), "batch_size": (1 if real else 2, 64), "seed": (0, 2**31 - 1), "max_runtime_seconds": (10, 604800)}
        low, high = bounds[key]
        require(low <= experiment[key] <= high, key + " is outside the supported range")
    if "lr" in experiment:
        require(type(experiment["lr"]) in (int, float), "lr must be numeric")
        require((0.000001 if real else 0.001) <= experiment["lr"] <= 1, "lr is outside the supported range")
    if not real:
        return
    for key in ("optimizer", "device", "input_port"):
        if key in experiment:
            require(type(experiment[key]) is str, key + " must be a string")
    if "optimizer" in experiment:
        require(experiment["optimizer"] in ("sgd", "adamw"), "unsupported optimizer")
    if "device" in experiment:
        require(re.fullmatch(r"cpu|cuda:(0|[1-9][0-9]*)", experiment["device"]) is not None, "unsupported device")
    for key in ("resume_checkpoint", "expected_dataset_digest"):
        if key in experiment:
            require(experiment[key] is None or type(experiment[key]) is str, key + " must be a string or null")
    data = experiment.get("data")
    require(type(data) is dict, "data must be an object")
    kind = data.get("kind", "csv")
    allowed_kinds = ("regression_csv",) if experiment["profile"] == "regression-v1" else ("csv", "image_folder", "paired_csv")
    require(kind in allowed_kinds, "unsupported data kind")
    mapping=experiment.get('input_ports')
    if kind=='paired_csv':
        if mapping is not None:
            require(type(mapping) is dict and set(mapping)=={'image','tabular'} and
                    all(type(value) is str and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',value) for value in mapping.values()) and
                    len(set(mapping.values()))==2,'input_ports must name two distinct graph inputs')
    else:require(mapping is None,'input_ports is only supported for paired_csv')
    common = {"kind", "path", "validation_fraction", "test_fraction"}
    fields = {"target_column", "feature_columns", "group_column", "split_column", "standardize"} if kind in ("csv", "paired_csv", "regression_csv") else {"split_mode", "image_size", "channels"}
    if kind=='paired_csv':fields|={'image_root','image_column','id_column','image_size','channels'}
    require(data.keys() <= common | fields, "unsupported data setting")
    for key in ("path", "target_column", "image_root", "image_column", "id_column"):
        if key in data:
            require(type(data[key]) is str, key + " must be a string")
    for key in ("validation_fraction", "test_fraction"):
        if key in data:
            require(type(data[key]) in (int, float), key + " must be numeric")
            require(0 < data[key] < 1, key + " must be between 0 and 1")
    require(data.get("validation_fraction", 0.2) + data.get("test_fraction", 0.2) < 1, "train fraction must be positive")
    if kind in ("csv", "paired_csv", "regression_csv"):
        if "feature_columns" in data:
            require(type(data["feature_columns"]) is list and len(data["feature_columns"]) <= 512 and all(type(name) is str for name in data["feature_columns"]), "feature_columns must be a list of up to 512 strings")
        for key in ("group_column", "split_column"):
            if key in data:
                require(data[key] is None or type(data[key]) is str, key + " must be a string or null")
        if "standardize" in data:
            require(type(data["standardize"]) is bool, "standardize must be boolean")
    if kind in ('image_folder','paired_csv'):
        if "image_size" in data:
            require(type(data["image_size"]) is list and len(data["image_size"]) == 2 and all(type(value) is int and 8 <= value <= 512 for value in data["image_size"]), "image_size must contain height and width integers between 8 and 512")
        if "channels" in data:
            require(type(data["channels"]) is int and data["channels"] in (1, 3), "channels must equal 1 or 3")
        if "split_mode" in data:
            require(data["split_mode"] in ("stratified", "directories"), "unsupported split_mode")


def workspace_payload(document, experiment, workbench=None):
    # Validate editable storage without requiring a complete runnable graph.
    model = Document(document.graph).graph
    _editable_experiment(experiment)
    payload = {"workspace_version": 2, "model": model, "experiment": deepcopy(experiment)}
    if workbench is not None:
        from .workbench_state import WorkbenchState
        # Parsing checks the closed, finite view schema without serializing
        # runtime handles, warning messages or document state.
        WorkbenchState._parse(workbench)
        payload.update(workspace_version=3, workbench=deepcopy(workbench))
    if model['schema_version'] == 2 and document.attention_probe_requests:
        from .attention_probe_panel import normalize_probe_requests
        payload.update(workspace_version=3,
                       attention_probe_requests=normalize_probe_requests(document.attention_probe_requests))
        if workbench is None:
            from .workbench_state import WorkbenchState
            payload['workbench'] = WorkbenchState().to_payload()
    if len(_encoded(payload).encode('utf-8')) > 8 * 1024 * 1024:
        raise ValueError('Workspace JSON exceeds 8 MiB limit')
    return payload


def read_workspace_full(path):
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Workspace JSON exceeds 8 MiB limit")
    def invalid(value):
        raise ValueError("Workspace JSON numbers must be finite: " + value)
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            invalid(value)
        return parsed
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=invalid, parse_float=finite_float)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Invalid workspace JSON: " + str(error)) from error
    if type(payload) is not dict:
        raise ValueError("Workspace JSON must be an object")
    if "workspace_version" not in payload:
        return Document(payload), synthetic_config(), None
    version = payload["workspace_version"]
    if type(version) is not int or version not in (2, 3):
        raise ValueError("Workspace workspace_version must equal 2 or 3")
    fields = {"workspace_version", "model", "experiment"} | ({"workbench", "attention_probe_requests"} if version == 3 else set())
    required = {"workspace_version", "model", "experiment"}
    if not required <= payload.keys() or payload.keys() - fields:
        raise ValueError("Workspace requires exactly the versioned model, experiment and view fields")
    document = Document(payload["model"])
    workspace_payload(document, payload["experiment"])
    if 'attention_probe_requests' in payload:
        from .attention_probe_panel import normalize_probe_requests
        if document.graph['schema_version'] != 2:
            raise ValueError('Attention probe requests require schema 2')
        document.attention_probe_requests = normalize_probe_requests(payload['attention_probe_requests'])
    if version == 3:
        from .workbench_state import WorkbenchState
        restored = WorkbenchState.from_payload(payload.get('workbench'), document)
        if restored.warnings:
            document.workspace_protected_paths = {path.resolve()}
            document.workspace_load_warnings = [
                '工作区视图已恢复；自动保存与退出保存将保留原文件。手动保存可写入恢复后的视图：' + str(path)]
            if payload.get('workbench') is None:
                document.workspace_load_warnings.extend(restored.warnings)
    # The UI must see the original view payload to explain repairs. A damaged
    # view never discards a structurally valid model or editable experiment.
    # json.loads owns this value exclusively. Returning it directly lets the
    # bounded view parser reject deep corruption before any recursive copy.
    return document, deepcopy(payload["experiment"]), payload.get("workbench")


def read_workspace(path):
    return read_workspace_full(path)[:2]


def save_workspace(path, document, experiment, workbench=None):
    payload = workspace_payload(document, experiment, workbench=workbench)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if len(text.encode('utf-8')) > 8 * 1024 * 1024:
        raise ValueError('Workspace JSON exceeds 8 MiB limit')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        replace_with_retry(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def data_identity(config):
    """Only settings that change the inspected samples or their split."""
    return _encoded({"data": config.get("data"), "seed": config.get("seed", 42)})


def composite_key(document, experiment, manifest=None):
    value = {"model": document.semantic_key(), "experiment": experiment,
             "dataset_digest": (manifest or {}).get("dataset_digest")}
    return hashlib.sha256(_encoded(value).encode("utf-8")).hexdigest()


def compatibility_issue(document, manifest, input_port="x", input_ports=None):
    if not manifest:
        return "请先检查数据。"
    inputs = document.graph.get("inputs", [])
    from studio_core.classical_task import classical_contract, contains_classical
    if contains_classical(document.graph):
        profile = "regression-v1" if manifest.get("kind") == "regression_csv" else "classification-v2"
        try:
            classical_contract(document.graph, {"profile": profile, "data": manifest["data_spec"],
                                                 "input_port": input_port, "device": "cpu"}, manifest)
        except ValueError as exc:
            return str(exc)
        return ""
    if manifest.get('kind')=='paired_csv':
        mapping=input_ports or {'image':'image','tabular':'clinical'}
        if type(mapping) is not dict or set(mapping)!={'image','tabular'} or len(set(mapping.values()))!=2:
            return '影像/表格需要两个不同的具名输入映射。'
        if {item['id'] for item in inputs}!=set(mapping.values()):return '模型输入接口与两路数据映射不一致。'
        by_id={item['id']:item for item in inputs}
        if any(by_id[mapping[modality]].get('shape')!=shape for modality,shape in manifest.get('input_shapes',{}).items()):
            return '两路模型输入形状须分别匹配影像与表格，并使用动态批次 N。'
    elif len(inputs) != 1:
        return "此任务需要恰好一个模型输入；可从已检查数据创建起始模型。"
    elif inputs[0].get("id") != input_port:
        return "实验输入接口与模型输入不一致：" + str(input_port)
    elif inputs[0].get("shape") != manifest.get("input_shape"):
        return "模型输入形状须与数据一致，并使用动态批次 N。"
    report = document.validation()
    if not report.get("valid"):
        return "模型结构尚未通过校验。"
    if manifest.get("kind") == "regression_csv":
        from studio_core.schema import prepare_graph
        from studio_core.regression_task import RegressionTask
        try:
            resolved, shapes, _ = prepare_graph(document.graph)
            RegressionTask({"input_port": input_port}).validate_contract(resolved, shapes, manifest)
        except ValueError as exc:
            return "回归需要 float32 的具名 prediction 输出 [N, 1]。" + str(exc)
        return ""
    logits = next((item for item in document.graph.get("outputs", []) if item.get("id") == "logits"), None)
    expected = [None, len(manifest.get("class_names", []))]
    if not logits or report.get("shapes", {}).get(logits.get("source")) != expected:
        return "需要命名 logits 输出 [N, " + str(expected[1]) + "]，对应数据类别数。"
    return ""


def starter_graph(manifest, estimator=None):
    """Return an explicit supervised starter graph; never mutate the current graph."""
    if estimator is not None:
        kind = manifest.get('kind')
        if (kind, estimator) not in (('csv', 'SVC'), ('regression_csv', 'SVR')):
            raise ValueError('SVC 需要 CSV 数值分类，SVR 需要 CSV 数值回归。')
        return {'schema_version': 1, 'id': estimator.lower(), 'name': f'{estimator} · 支持向量机', 'revision': 1,
                'inputs': [{'id': 'x', 'shape': manifest['input_shape'], 'dtype': 'float32', 'position': {'x': 20, 'y': 140}}],
                'instances': [{'id': 'estimator', 'kind': estimator, 'params': {'C': 1.0, 'kernel': 'rbf'}}],
                'calls': [{'id': 'fit_model', 'instance_id': 'estimator', 'inputs': {'x': 'input:x'}, 'position': {'x': 260, 'y': 140}}],
                'outputs': [{'id': 'prediction', 'source': 'call:fit_model', 'position': {'x': 540, 'y': 140}}], 'definitions': []}
    if manifest.get('kind')=='paired_csv':
        from .starter_models import paired_starter_graph
        return paired_starter_graph(manifest)
    shape = manifest.get("input_shape")
    classes = manifest.get("class_names", [])
    kind = manifest.get("kind")
    regression = kind == "regression_csv"
    if not isinstance(shape, list) or (not regression and len(classes) < 2):
        raise ValueError("Inspect a supervised dataset before creating a starter model")
    output_id = "prediction" if regression else "logits"
    if regression and len(shape) == 2 and type(shape[1]) is int and shape[1] > 0:
        stages = [("prediction", "Linear", {"in_features": shape[1], "out_features": 1})]
        name = "CSV 线性回归"
    elif kind == "csv" and len(shape) == 2 and type(shape[1]) is int and shape[1] > 0:
        stages = [("dense", "Linear", {"in_features": shape[1], "out_features": 16}),
                  ("activation", "ReLU", {}),
                  ("logits", "Linear", {"in_features": 16, "out_features": len(classes)})]
        name = "CSV MLP 分类器"
    elif kind == "image_folder" and len(shape) == 4 and all(type(x) is int and x > 0 for x in shape[1:]):
        stages = [("conv", "Conv2d", {"in_channels": shape[1], "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1}),
                  ("activation", "ReLU", {}),
                  ("pool", "AdaptiveAvgPool2d", {"output_size": [1, 1]}),
                  ("flatten", "Flatten", {"start_dim": 1, "end_dim": -1}),
                  ("logits", "Linear", {"in_features": 8, "out_features": len(classes)})]
        name = "ImageFolder 小型 CNN"
    else:
        raise ValueError("Unsupported inspected dataset shape for a starter model")
    graph = {"schema_version": 1, "id": "data_regressor" if regression else "data_classifier", "name": name, "revision": 1,
             "resource_profile": "classification-v2",
             "inputs": [{"id": "x", "name": "数据输入", "shape": list(shape), "dtype": "float32", "position": {"x": -40 if kind == "image_folder" else 10, "y": 40 if kind == "image_folder" else 160}}],
             "instances": [], "calls": [], "outputs": [{"id": output_id, "source": "call:" + output_id}], "definitions": []}
    if kind == "image_folder":
        graph["outputs"][0]["position"] = {"x": 860, "y": 240}
    source = "input:x"
    for index, (identifier, primitive, params) in enumerate(stages):
        graph["instances"].append({"id": identifier, "kind": primitive, "name": primitive, "params": params})
        position = {"x": 260 + 300 * (index % 3), "y": 40 + 200 * (index // 3)} if kind == "image_folder" else {"x": 240 + 260 * index, "y": 160}
        graph["calls"].append({"id": identifier, "instance_id": identifier, "inputs": {"x": source}, "position": position})
        source = "call:" + identifier
    return graph
