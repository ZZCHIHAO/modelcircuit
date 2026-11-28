"""Background data/device queries and truthful experiment presentation.

No GUI calls occur here. The main GUI thread consumes completed query messages
and is solely responsible for applying them to widgets.
"""
import csv
import math
from pathlib import Path
import queue
import threading


class BackgroundQueries:
    def __init__(self):
        self._queue = queue.Queue()
        self._generations = {}
        self._cancel = {}
        self._closed = False

    def submit(self, name, function):
        if self._closed:
            raise ValueError("Background query controller is closed")
        self.invalidate(name)
        generation = self._generations[name]
        cancelled = threading.Event()
        self._cancel[name] = cancelled
        def run():
            try:
                value = function(cancelled.is_set)
                message = {"name": name, "generation": generation, "value": value, "error": None}
            except Exception as error:
                message = {"name": name, "generation": generation, "value": None, "error": str(error)}
            self._queue.put(message)
        threading.Thread(target=run, name="mlforge-" + name, daemon=True).start()
        return generation

    def invalidate(self, name):
        event = self._cancel.pop(name, None)
        if event:
            event.set()
        self._generations[name] = self._generations.get(name, 0) + 1

    def poll(self):
        messages = []
        while True:
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                break
            if not self._closed and message["generation"] == self._generations.get(message["name"]):
                self._cancel.pop(message["name"], None)
                messages.append(message)
        return messages

    def close(self):
        self._closed = True
        for event in self._cancel.values():
            event.set()
        self._cancel.clear()


def csv_columns(path):
    """Read just the bounded header; full numeric validation is inspection's job."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        line = stream.readline(65537)
    if not line or len(line) > 65536:
        raise ValueError("CSV header is empty or exceeds 64 KiB")
    columns = next(csv.reader([line]))
    if len(columns) > 517 or any(not column for column in columns) or len(set(columns)) != len(columns):
        raise ValueError("CSV requires distinct nonempty column names (up to 517 columns, including paired metadata)")
    return columns


def _number(value):
    return f"{value:.6g}" if isinstance(value, (float, int)) and math.isfinite(value) else "—"


def manifest_summary(manifest):
    if not manifest:
        return "尚未检查数据；运行前需检查。"
    counts = manifest.get("counts", {})
    def shape_text(shape):
        return "×".join("N" if item is None else str(item) for item in shape)
    shape = ("\n".join(name + " " + shape_text(value) for name, value in manifest["input_shapes"].items())
             if manifest.get("input_shapes") else shape_text(manifest.get("input_shape", [])))
    if manifest.get('kind') == 'unsupervised_csv':
        return f"无标签 CSV · 输入 {shape}\n{counts['fit']} 行参与拟合"
    classes = manifest.get("class_names", [])
    regression = manifest.get('kind') == 'regression_csv' or manifest.get('task') == 'regression'
    dataset_label = "CSV 数值回归" if regression else manifest.get('kind')
    target = ("目标 " + manifest['target_column'] if regression else
              f"{len(classes)} 类：" + ", ".join(str(value) for value in classes[:12]) + ("…" if len(classes) > 12 else ""))
    if manifest.get('kind') == 'segmentation_csv':
        dataset_label = '三维分割' if len(manifest['input_shape']) == 5 else '二维分割'
        target = '掩码值 ' + ', '.join(map(str, manifest['mask_values'][:12]))
        fmt = manifest.get('data_spec', {}).get('format')
        if fmt in ('dicom_ct', 'mri_nifti'):
            dataset_label = {'dicom_ct': 'CT 补丁分割', 'mri_nifti': 'MRI 补丁分割'}[fmt]
            patients = {split: set() for split in ('train', 'val', 'test')}
            for row in manifest['records']:
                patients[row['split']].add(row['group'])
            target += '\n患者数 ' + ' / '.join(str(len(groups)) for groups in patients.values())
    if manifest.get('kind') == 'table_csv':
        dataset_label = '表格回归' if regression else '表格分类'
        target += f" · {len(manifest['fold_plan'])} 折"
    elif manifest.get('kind') == 'multimodal_csv':
        dataset_label = '病例多模态分类'
    text = (f"{dataset_label} · 输入 {shape}\n"
            f"train {counts.get('train', 0)} / val {counts.get('val', 0)} / test {counts.get('test', 0)}\n"
            f"{target}\n摘要 " + str(manifest.get("dataset_digest", ""))[:16])
    ignored = manifest.get("ignored_files", [])
    if ignored:
        text += f"\n已忽略 {len(ignored)} 个系统/同步附属文件"
    elif manifest.get("warnings"):
        text += f"\n数据提示 {len(manifest['warnings'])} 项（见数据设置）"
    return text


def device_choices(capabilities):
    """Accept the public capabilities object, offering only executable devices."""
    choices = ["cpu"]
    for device in capabilities.get("devices", []):
        if isinstance(device, dict):
            identifier = device.get("id", device.get("device"))
            if isinstance(identifier, str) and identifier.startswith("cuda:") and device.get("available", True):
                choices.append(identifier)
    if capabilities.get("cuda_available") and len(choices) == 1:
        for index in range(capabilities.get("cuda_device_count", 0)):
            choices.append("cuda:" + str(index))
    return list(dict.fromkeys(choices))


def device_summary(capabilities):
    reason = capabilities.get("cuda_reason") or capabilities.get("reason") or capabilities.get("unavailable_reason")
    runtime = capabilities.get("torch_version", capabilities.get("runtime", ""))
    names = []
    for item in capabilities.get("devices", []):
        if isinstance(item, dict) and str(item.get("id", item.get("device", ""))).startswith("cuda:"):
            names.append(str(item.get("name", item.get("id", item.get("device")))))
    hardware = capabilities.get("hardware_gpus", capabilities.get("nvidia_hardware", capabilities.get("hardware", [])))
    if isinstance(hardware, list):
        for item in hardware:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
    text = "单设备执行 · " + str(runtime)
    if names:
        text += "\n" + " / ".join(dict.fromkeys(names))
    if reason:
        text += "\nCUDA：" + str(reason)
    elif not capabilities.get("cuda_available"):
        text += "\nCUDA 当前不可用；请查看运行时能力。"
    return text


def progress_points(record, after_seq, real):
    """S2 curves represent completed epochs, never train/validation batches."""
    def finite(value):
        return isinstance(value, (int, float)) and math.isfinite(value)
    if real:
        return [event for event in record.get("events", [])
                if event.get("seq", 0) > after_seq and event.get("type") == "progress"
                and event.get("phase") == "epoch" and type(event.get("epoch")) is int
                and finite(event.get("train_loss"))
                and (finite(event.get("val_loss")) or event.get("stage") == "hint")]
    progress = record.get("progress") or {}
    return [progress] if finite(progress.get("loss")) else []


def result_view(result, progress_history=None):
    result = result or {}
    regression = result.get("profile") == "regression-v1"
    segmentation = result.get('profile') == 'segmentation-v1'
    real = regression or segmentation or result.get("profile") in ("classification-v2", "fitnets-v1")
    history = result.get("history") or list((progress_history or {}).values())
    if history and not isinstance(history[0], dict):
        history = [{"step": step, "loss": value} for step, value in sorted((progress_history or {}).items())]
    def series(key):
        rows = [item for item in history if isinstance(item.get(key), (int, float)) and math.isfinite(item[key])]
        return [[item.get("epoch", item.get("step", 0)) if real else item.get("step", 0) for item in rows], [item[key] for item in rows]]
    current_history = history
    if result.get("profile") == "fitnets-v1":
        history = [row for row in current_history if row.get("stage") != "hint"]
    train, val = series("train_loss" if real else "loss"), series("val_loss") if real else [[], []]
    if result.get('execution') == 'unsupervised':
        fit = result['metrics']['fit']
        summary = f"{result['algorithm']} · CPU · {fit['count']} 个样本"
        if 'explained_variance' in fit:
            summary += f"\n累计解释方差 {fit['explained_variance']:.1%}"
        if 'inertia' in fit:
            summary += f"\n簇内平方和 {_number(fit['inertia'])}"
    elif result.get('execution') == 'classical':
        test = result.get('metrics', {}).get('test', {})
        summary = f"{result['algorithm']} · CPU · 已完成拟合"
        if regression:
            r2 = _number(test.get('r2')) if test.get('r2') is not None else '未定义'
            summary += f"\n测试集 MAE {_number(test.get('mae'))} · RMSE {_number(test.get('rmse'))}\nR² {r2} · n={test.get('count', '—')}"
        else:
            summary += f"\n测试集准确率 {_number(test.get('accuracy'))} · n={test.get('count', '—')}"
        cv = result.get('metrics', {}).get('cv')
        if cv and regression:
            summary += f"\n分组交叉验证 MAE {_number(cv.get('mae'))} · RMSE {_number(cv.get('rmse'))} · n={cv.get('count', '—')}"
        elif cv:
            summary += f"\n分组交叉验证准确率 {_number(cv.get('accuracy'))} · n={cv.get('count', '—')}"
    elif real:
        test = result.get("metrics", {}).get("test")
        summary = f"{result.get('device', '—')} · 最佳 epoch {result.get('best_epoch', '—')}"
        if result.get("profile") == "fitnets-v1":
            hint = [row for row in current_history if row.get("stage") == "hint"]
            if hint:
                summary = "FitNets · 提示训练 " + str(len(hint)) + " 轮 · 最终提示损失 " + _number(hint[-1].get("train_loss")) + "\n" + summary
        if segmentation and test:
            summary += (f"\n测试集前景 Dice {_number(test.get('foreground_dice'))} · IoU {_number(test.get('foreground_iou'))}"
                        f" · n={test.get('count', '—')}")
            if test.get('space') == 'original':
                summary += (f" · HD95 {_number(test.get('foreground_hd95_mm'))} mm\n"
                            "重叠和距离在原始空间计算。整卷损失在模型网格计算。")
            else:
                summary += '\n在模型网格上计算。原始网格掩码见「分割预测」。'
        elif regression and test:
            r2 = _number(test["r2"]) if test.get("r2") is not None else "未定义"
            summary += (f"\n测试集 MAE {_number(test.get('mae'))} · RMSE {_number(test.get('rmse'))}"
                        f"\nR² {r2} · n={test.get('count', '—')}")
        else:
            summary += (f" · final test loss {_number(test.get('loss'))} / accuracy {_number(test.get('accuracy'))} / n={test.get('count', '—')}" if test else " · held-out test 尚未评估")
    elif result:
        summary = f"初始损失 {_number(result.get('loss_before'))}   最终损失 {_number(result.get('loss_after'))}   同批次准确率 {_number(result.get('accuracy'))}   {result.get('device', 'cpu')} / synthetic_s1"
    else:
        summary = "尚未采集 · 不显示示例曲线"
    context = result.get("probe_context", {})
    probe = (f"实测 {context.get('split', '—')} · epoch {context.get('epoch', '—')} · {context.get('mode', '—')} · {context.get('device', result.get('device', '—'))} · {len(context.get('sample_ids', []))} 个样本"
             + " · 梯度来源：" + str(context.get("gradient_source", "—"))) if context else ""
    artifacts = "\n".join(str(name) + "：" + str(path) for name, path in result.get("artifacts", {}).items())
    parameter_context = (probe + " · 更新范数来源：" + str(context.get("parameter_update_source", "—"))) if context else ""
    return {"real": real, "train": train, "val": val, "summary": summary, "probe": probe, "parameters": parameter_context, "artifacts": artifacts}
