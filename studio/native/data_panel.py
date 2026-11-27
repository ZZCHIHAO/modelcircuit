"""Native experiment controls composed into the established workbench layout.

Queries return ordinary data to tick(); every DearPyGui call stays on the main
thread. Data fields live in a modal so the 305px inspector remains usable.
"""
from copy import deepcopy
import json

import dearpygui.dearpygui as dpg

from studio_core.experiment_spec import FITNETS_DEFAULTS, SUPERVISED_PROFILES
from studio_core.medical_volume_data import MEDICAL_FORMATS

from .cohort_panel import CohortPanel, KINDS as COHORT_KINDS
from .medical_panel import MedicalPatchPanel

from .experiment_panel import (BackgroundQueries, csv_columns, device_choices,
                               device_summary, manifest_summary)
from .workspace_state import (Document, compatibility_issue, composite_key,
                              data_identity, starter_graph, synthetic_config, workspace_payload)

MUTED = (145, 154, 164)
BLUE = (100, 200, 255)
AMBER = (235, 190, 110)
RED = (240, 143, 134)
MODES = {"S1 合成 CPU（演示）": "synthetic_s1", "CSV 数值分类": "csv", "CSV 数值回归": "regression_csv", "ImageFolder 图片分类": "image_folder", "配对图片与表格分类": "paired_csv", "科研影像 / 视频分类": "research_csv", "2D / 3D 医学影像分割": "segmentation_csv",
         "表格机器学习（分组交叉验证）": "table_csv", "病例多模态分类": "multimodal_csv"}
SEGMENTATION_FORMATS = {"二维图片 + PNG/TIFF 掩码": "image", "三维 NIfTI（固定网格）": "nifti",
                        "DICOM CT + NIfTI 掩码": "dicom_ct", "MRI NIfTI + NIfTI 掩码": "mri_nifti"}
_SEQUENCE_NONE = '（不使用）'
_SEQUENCE_TAGS = tuple(f'native_medical_sequence_{channel}' for channel in (2, 3, 4))
SEGMENTATION_COLUMNS = (("id_column", "样本 ID 列", "sample_id"), ("group_column", "患者 ID 列", "patient_id"),
                        ("image_column", "影像路径列", "image_path"), ("mask_column", "掩码路径列", "mask_path"),
                        ("split_column", "划分列（可选）", "自动划分"))
IMAGE_MODES = ('image_folder', 'paired_csv', 'research_csv', 'segmentation_csv')


class ExperimentControls:
    def _init_experiment(self, experiment=None):
        self.queries = BackgroundQueries()
        self.medical_patch_panel = MedicalPatchPanel(self)
        self.cohort_panel = CohortPanel(self)
        self.manifest = None
        self.inspection_key = None
        self.inspect_request_key = None
        self.inspecting = False
        self.capabilities = None
        self.available_devices = ["cpu"]
        self.pending_starter = None
        self.pending_teacher = False
        self.csv_header = []
        self._s1_experiment = synthetic_config()
        self._real_experiments = {}
        self._install_experiment(experiment or synthetic_config())

    @staticmethod
    def _supervised_defaults(kind):
        from studio_core.experiment_spec import default_classification_config, default_regression_config, default_segmentation_config
        if kind == 'segmentation_csv':
            return default_segmentation_config()
        return default_regression_config() if kind == "regression_csv" else default_classification_config(data_kind=kind)

    def _install_experiment(self, experiment):
        experiment = deepcopy(experiment)
        if experiment.get("profile") in SUPERVISED_PROFILES:
            kind = experiment.get("data", {}).get("kind", "csv")
            defaults = self._supervised_defaults(kind)
            defaults.update({key: value for key, value in experiment.items() if key != "data"})
            defaults["data"].update(experiment.get("data", {}))
            self.experiment = defaults
            self.experiment_mode = kind
            self._real_experiments[kind] = deepcopy(defaults)
        else:
            self.experiment_mode = "synthetic_s1"
            self.experiment = {**synthetic_config(), **experiment}
            self._s1_experiment = deepcopy(self.experiment)
        self.manifest = None
        self.inspection_key = None
        self.inspecting = False
        self.queries.invalidate("inspect")
        self.queries.invalidate("columns")
        self.medical_patch_panel.invalidate()

    def _widgets_available(self):
        return self._context_created and dpg.does_item_exist("native_task")

    def _build_experiment_controls(self):
        dpg.add_combo(label="任务", tag="native_task", items=list(MODES), default_value=list(MODES)[0], width=205, callback=self.guarded(self._mode_changed))
        with dpg.group(tag="native_s1_settings"):
            dpg.add_input_int(label="训练步数", tag="native_steps", default_value=20, min_value=1, max_value=200, min_clamped=True, max_clamped=True, width=120, callback=self.guarded(self._experiment_changed))
            dpg.add_text("S1 合成 CPU 双分类；用于验证模型操作。", color=MUTED, wrap=265)
        with dpg.group(tag="native_real_settings", show=False):
            with dpg.group(horizontal=True):
                dpg.add_button(label="数据设置", tag="native_data_settings", callback=self.guarded(self._open_data_settings))
                dpg.add_button(label="检查数据", tag="native_inspect", callback=self.guarded(self._inspect_data))
            dpg.add_text("尚未检查数据；运行前需检查。", tag="native_dataset_summary", color=MUTED, wrap=265)
            dpg.add_button(label='预览训练补丁…', tag='native_medical_patch_open', width=-1, show=False,
                           callback=self.guarded(self.medical_patch_panel.open))
            dpg.add_button(label="从数据创建起始模型…", tag="native_starter", width=-1, callback=self.guarded(self._request_starter))
            dpg.add_button(label="创建支持向量机起始图…", tag="native_svm_starter", show=False, width=-1,
                           callback=self.guarded(self._request_svm_starter))
            with dpg.group(tag="native_fitnet_templates", show=False):
                dpg.add_button(label="创建教师起始图…", tag="native_teacher_starter", width=-1,
                               callback=self.guarded(lambda **_: self._request_starter(teacher=True)))
                dpg.add_button(label="创建 FitNets 双分支图…", tag="native_fitnets_starter", width=-1,
                               callback=self.guarded(lambda **_: self._request_starter(fitnets=True)))
            dpg.add_text("", tag="native_data_compatibility", color=AMBER, wrap=265)
            with dpg.group(horizontal=True):
                dpg.add_combo(label="设备", tag="native_device", items=["cpu"], default_value="cpu", width=140, callback=self.guarded(self._experiment_changed))
                dpg.add_button(label="刷新", tag="native_device_refresh", callback=self.guarded(self._refresh_devices))
            dpg.add_text("设备能力尚未刷新。", tag="native_device_summary", color=MUTED, wrap=265)
            with dpg.group(horizontal=True, tag="native_epoch_controls"):
                dpg.add_input_int(label="Epochs", tag="native_epochs", default_value=10, min_value=1, max_value=1000, min_clamped=True, max_clamped=True, width=75, step=0, callback=self.guarded(self._experiment_changed))
                dpg.add_combo(label="优化器", tag="native_optimizer", items=["adamw", "sgd"], default_value="adamw", width=80, callback=self.guarded(self._experiment_changed))
        with dpg.group(horizontal=True, tag="native_gradient_controls"):
            dpg.add_input_float(label="LR", tag="native_lr", default_value=0.05, min_value=0.001, max_value=1, min_clamped=True, max_clamped=True, format="%.6f", width=95, step=0, callback=self.guarded(self._experiment_changed))
            dpg.add_input_int(label="批次", tag="native_batch", default_value=16, min_value=2, max_value=64, min_clamped=True, max_clamped=True, width=90, step=0, callback=self.guarded(self._experiment_changed))
        dpg.add_input_int(label="随机种子", tag="native_seed", default_value=42, min_value=0, max_value=2**31-1, min_clamped=True, max_clamped=True, width=120, callback=self.guarded(self._experiment_changed))

    def _build_data_dialogs(self):
        with dpg.window(tag="native_data", label="数据与实验设置", show=False, modal=True, width=790, height=690, no_collapse=True):
            dpg.add_text("先选择数据，再检查样本、划分和模型输入。", color=MUTED)
            with dpg.group(horizontal=True):
                dpg.add_input_text(label="数据路径", tag="native_data_path", width=580, on_enter=True, callback=self.guarded(self._experiment_changed))
                dpg.add_button(label="浏览…", callback=self.guarded(self._browse_data))
            with dpg.group(tag="native_csv_settings"):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="读取 CSV 列", tag="native_csv_columns", callback=self.guarded(self._read_columns))
                    dpg.add_text("先读取列，再选择标签、分组与 split。", color=MUTED)
                dpg.add_combo(label="标签列", tag="native_csv_target", items=[], width=310, callback=self.guarded(self._csv_selection_changed))
                dpg.add_combo(label="分组列（可选）", tag="native_csv_group", items=["无"], default_value="无", width=310, callback=self.guarded(self._csv_selection_changed))
                dpg.add_combo(label="split 列（可选）", tag="native_csv_split", items=["无"], default_value="无", width=310, callback=self.guarded(self._csv_selection_changed))
                dpg.add_input_text(label="数值特征列（JSON 列表）", tag="native_csv_features", default_value="[]", width=500, on_enter=True, callback=self.guarded(self._experiment_changed))
                dpg.add_checkbox(label="标准化（仅拟合 train）", tag="native_standardize", default_value=True, callback=self.guarded(self._experiment_changed))
                dpg.add_text('例如 ["f1", "f2"]。split 列取 train / val / test；分组不得跨 split。', color=MUTED, wrap=745)
            with dpg.group(tag="native_paired_settings", show=False):
                dpg.add_input_text(label="图片根目录（空则 CSV 所在目录）", tag="native_paired_image_root", width=460, on_enter=True, callback=self.guarded(self._experiment_changed))
                with dpg.group(horizontal=True):
                    dpg.add_combo(label="图片路径列", tag="native_paired_image_column", items=[], default_value="image_path", width=215, callback=self.guarded(self._csv_selection_changed))
                    dpg.add_combo(label="样本 ID 列", tag="native_paired_id_column", items=[], default_value="sample_id", width=215, callback=self.guarded(self._csv_selection_changed))
                dpg.add_text("配对数据必须选择分组列；同一行的图片与数值特征始终共同划分和加载。", color=MUTED, wrap=745)
            with dpg.group(tag="native_research_settings", show=False):
                dpg.add_input_text(label="样本根目录（默认 CSV 所在目录）", tag="native_research_root", width=440)
                with dpg.group(horizontal=True):
                    dpg.add_combo(items=["image", "gray16", "dicom", "nifti", "video"], label="格式", tag="native_research_format", width=135, callback=self.guarded(self._research_format_changed))
                    dpg.add_combo(items=["auto", "OCT", "CT", "MR"], label="模态", tag="native_research_modality", width=120)
                    dpg.add_input_int(label="深度 / 帧数", tag="native_research_depth", default_value=1, min_value=1, max_value=128, min_clamped=True, max_clamped=True, width=115)
                with dpg.group(horizontal=True):
                    dpg.add_input_text(label="强度下限", tag="native_research_min", width=95)
                    dpg.add_input_text(label="强度上限", tag="native_research_max", width=95)
                dpg.add_input_text(label="样本 ID / 分组 / 路径 / 标签 / 划分列", tag="native_research_columns", width=350)
                dpg.add_text("CSV 列名按顺序填写。路径相对样本根目录，分组必填。末尾划分列可省略。", color=MUTED)
            with dpg.group(tag="native_segmentation_settings", show=False):
                with dpg.group(horizontal=True):
                    dpg.add_combo(list(SEGMENTATION_FORMATS), label="影像与掩码格式", tag="native_segmentation_format",
                                  default_value=next(iter(SEGMENTATION_FORMATS)), width=245,
                                  callback=self.guarded(self._segmentation_format_changed))
                    dpg.add_input_int(label="深度 D", tag="native_segmentation_depth", default_value=16, width=115,
                                      min_value=2, max_value=128, min_clamped=True, max_clamped=True)
                dpg.add_input_text(label="路径根目录（默认 CSV 所在目录）", tag="native_segmentation_root", width=410)
                dpg.add_button(label="读取 CSV 列", tag="native_segmentation_read_columns", callback=self.guarded(self._read_columns))
                for start in range(0, len(SEGMENTATION_COLUMNS), 2):
                    with dpg.group(horizontal=True):
                        for key, label, default in SEGMENTATION_COLUMNS[start:start + 2]:
                            dpg.add_combo([default], label=label, tag='native_seg_' + key, width=175, default_value=default,
                                          callback=self.guarded(self._experiment_changed))
                dpg.add_input_text(label="掩码标签值（背景为 0）", tag="native_segmentation_values", width=280, default_value="0, 1")
                dpg.add_text("按从小到大填写整数标签。例 0, 255。划分列使用 train / val / test。", color=MUTED, wrap=730)
                with dpg.group(tag='native_medical_settings', show=False):
                    with dpg.group(tag='native_medical_ct', show=False, horizontal=True):
                        dpg.add_input_float(tag='native_ct_lower_hu', label='CT 下限 HU', default_value=-1000, width=120, step=0)
                        dpg.add_input_float(tag='native_ct_upper_hu', label='CT 上限 HU', default_value=400, width=120, step=0)
                    with dpg.group(tag='native_medical_mri', show=False):
                        dpg.add_text('影像路径列是参考序列。读取 CSV 列后，按模型通道顺序选择附加序列。', color=MUTED, wrap=730)
                        with dpg.group(horizontal=True):
                            for channel, tag in enumerate(_SEQUENCE_TAGS, 2):
                                dpg.add_combo([_SEQUENCE_NONE], tag=tag, label=f'通道 {channel}',
                                              default_value=_SEQUENCE_NONE, width=145)
                        dpg.add_checkbox(tag='native_medical_registered', label='附加序列已配准到参考影像')
                    with dpg.group(horizontal=True):
                        for axis in ('D', 'H', 'W'):
                            dpg.add_input_float(tag='native_medical_spacing_' + axis, label=f'{axis} 间距 mm',
                                                default_value=1, width=102, format='%.5g', step=0)
                    with dpg.group(horizontal=True):
                        dpg.add_input_int(tag='native_medical_patches', label='每病例补丁数', default_value=4, width=112,
                                          min_value=1, max_value=128, min_clamped=True, max_clamped=True)
                        dpg.add_input_float(tag='native_medical_foreground', label='前景采样比例', default_value=.5,
                                            width=95, step=0, format='%.3f')
                        dpg.add_input_float(tag='native_medical_overlap', label='滑窗重叠比例', default_value=.25,
                                            width=95, step=0, format='%.3f')
                    dpg.add_text('D × H × W 是补丁大小。分割评估会把整卷滑窗预测恢复到参考网格。', color=MUTED, wrap=730)
            self.cohort_panel.build()
            with dpg.group(tag="native_image_settings", show=False):
                with dpg.group(tag="native_image_directory_settings"):
                    dpg.add_combo(label="目录拆分", tag="native_image_split", items=["stratified", "directories"], default_value="stratified", width=230, callback=self.guarded(self._experiment_changed))
                    dpg.add_text("stratified：根目录/类别/图片\ndirectories：根目录/train、val、test/类别/图片", color=MUTED)
                with dpg.group(horizontal=True):
                    dpg.add_input_int(label="高 H", tag="native_image_height", default_value=32, width=145, min_value=8, max_value=512, min_clamped=True, max_clamped=True, callback=self.guarded(self._experiment_changed))
                    dpg.add_input_int(label="宽 W", tag="native_image_width", default_value=32, width=145, min_value=8, max_value=512, min_clamped=True, max_clamped=True, callback=self.guarded(self._experiment_changed))
                dpg.add_combo(label="通道", tag="native_image_channels", items=["1", "3"], default_value="3", width=145, callback=self.guarded(self._experiment_changed))
            with dpg.group(tag="native_split_fractions", horizontal=True):
                dpg.add_input_float(label="val 比例", tag="native_validation_fraction", default_value=0.2, format="%.3f", width=145, callback=self.guarded(self._experiment_changed))
                dpg.add_input_float(label="test 比例", tag="native_test_fraction", default_value=0.2, format="%.3f", width=145, callback=self.guarded(self._experiment_changed))
            dpg.add_text("比例用于自动划分。显式划分列或目录按原有划分检查。", tag="native_split_help", color=MUTED)
            dpg.add_input_text(label="模型输入接口", tag="native_input_port", default_value="x", width=180, on_enter=True, callback=self.guarded(self._experiment_changed))
            with dpg.group(tag="native_paired_ports", show=False, horizontal=True):
                dpg.add_input_text(label="图片输入端口", tag="native_image_port", default_value="image", width=170, on_enter=True, callback=self.guarded(self._experiment_changed))
                dpg.add_input_text(label="表格输入端口", tag="native_tabular_port", default_value="clinical", width=170, on_enter=True, callback=self.guarded(self._experiment_changed))
            dpg.add_input_int(label="最长运行秒数", tag="native_max_runtime", default_value=3600, min_value=10, max_value=604800, min_clamped=True, max_clamped=True, width=180, callback=self.guarded(self._experiment_changed))
            dpg.add_input_text(label="继续训练 checkpoint（可选 .pt）", tag="native_resume_checkpoint", width=470, on_enter=True, callback=self.guarded(self._experiment_changed))
            with dpg.group(tag="native_fitnet_options", show=False):
                dpg.add_checkbox(label="启用 FitNets 两阶段训练", tag="native_fitnets_enabled", callback=self.guarded(self._experiment_changed))
                dpg.add_input_text(label="教师 checkpoint", tag="native_teacher_checkpoint", width=460)
                with dpg.group(horizontal=True):
                    dpg.add_input_int(label="提示轮数", tag="native_hint_epochs", default_value=1, min_value=1, max_value=1000, min_clamped=True, max_clamped=True, width=80)
                    dpg.add_input_float(label="温度", tag="native_distill_temperature", default_value=4.0, width=115)
                    dpg.add_input_float(label="蒸馏权重", tag="native_distill_weight", default_value=.5, width=115)
            dpg.add_text("", tag="native_data_details", color=MUTED, wrap=745)
            dpg.add_text("", tag="native_data_error", color=RED, wrap=745)
            with dpg.group(horizontal=True):
                dpg.add_button(label="检查并冻结数据摘要", tag="native_data_inspect", callback=self.guarded(self._inspect_data))
                dpg.add_button(label="应用并关闭", tag="native_data_apply", callback=self.guarded(self._close_data_settings))
        self.medical_patch_panel.build()
        with dpg.file_dialog(tag="native_data_open", show=False, width=760, height=470, modal=True, callback=self.guarded(self._data_path_chosen)):
            dpg.add_file_extension(".csv", color=BLUE)
            dpg.add_file_extension(".*")
        with dpg.window(tag="native_starter_replace", label="替换为数据起始模型", show=False, modal=True, width=485, height=190, no_resize=True):
            dpg.add_text("将替换当前模型和芯片定义。\n实验数据与训练设置会保留。\n请先保存需要保留的当前模型。", color=AMBER, wrap=450)
            with dpg.group(horizontal=True):
                dpg.add_button(label="取消", callback=self.guarded(lambda **_: dpg.hide_item("native_starter_replace")))
                dpg.add_button(label="确认替换模型", tag="native_starter_confirm", callback=self.guarded(self._replace_starter))

    def _sync_experiment_widgets(self):
        if not self._widgets_available():
            return
        real = self.experiment_mode != "synthetic_s1"
        dpg.set_value("native_task", next(label for label, mode in MODES.items() if mode == self.experiment_mode))
        dpg.configure_item("native_s1_settings", show=not real)
        dpg.configure_item("native_real_settings", show=real)
        dpg.configure_item("native_lr", min_value=0.000001 if real else 0.001)
        dpg.configure_item("native_batch", min_value=1 if real else 2)
        for tag, key in (("native_steps", "steps"), ("native_lr", "lr"), ("native_seed", "seed"), ("native_batch", "batch_size")):
            if key in self.experiment:
                dpg.set_value(tag, self.experiment[key])
        if real:
            config, data = self.experiment, self.experiment["data"]
            values = {"native_epochs": config["epochs"], "native_optimizer": config["optimizer"], "native_device": config["device"], "native_data_path": data.get("path", ""),
                      "native_validation_fraction": data.get("validation_fraction", 0.2), "native_test_fraction": data.get("test_fraction", 0.2),
                      "native_input_port": config.get("input_port", "x"), "native_max_runtime": config.get("max_runtime_seconds", 3600), "native_resume_checkpoint": config.get("resume_checkpoint") or ""}
            if self.experiment_mode in ("csv", "paired_csv", "regression_csv"):
                values.update({"native_csv_target": data.get("target_column") or "", "native_csv_group": data.get("group_column") or "无", "native_csv_split": data.get("split_column") or "无", "native_csv_features": json.dumps(data.get("feature_columns") or [], ensure_ascii=False), "native_standardize": data.get("standardize", True)})
            if self.experiment_mode == 'research_csv':
                values.update(native_research_root=data['source_root'], native_research_format=data['format'],
                              native_research_modality=data['modality'], native_research_depth=data['depth'],
                              native_research_min='' if data['intensity_min'] is None else str(data['intensity_min']),
                              native_research_max='' if data['intensity_max'] is None else str(data['intensity_max']),
                              native_research_columns=','.join(data[k] or '' for k in ('id_column','group_column','path_column','target_column','split_column')))
            if self.experiment_mode == 'segmentation_csv':
                fmt_label = next(label for label, fmt in SEGMENTATION_FORMATS.items() if fmt == data['format'])
                values.update(native_segmentation_root=data['source_root'], native_segmentation_format=fmt_label,
                              native_segmentation_depth=data['depth'] if data['depth'] > 1 else 16,
                              native_segmentation_values=', '.join(map(str, data['mask_values'])))
                for key, _, default in SEGMENTATION_COLUMNS:
                    tag = 'native_seg_' + key
                    choice = data[key] or default
                    dpg.configure_item(tag, items=list(dict.fromkeys(dpg.get_item_configuration(tag)['items'] + [choice])))
                    values[tag] = choice
                self._show_segmentation_dimension(data['format'])
                window = data.get('ct_window') or [-1000., 400.]
                values.update(native_ct_lower_hu=window[0], native_ct_upper_hu=window[1],
                              native_medical_registered=data.get('registered', False),
                              native_medical_patches=data.get('patches_per_case', 4),
                              native_medical_foreground=data.get('foreground_fraction', .5),
                              native_medical_overlap=data.get('overlap', .25))
                for axis, spacing in zip('DHW', data.get('spacing_mm', [1., 1., 1.])):
                    values['native_medical_spacing_' + axis] = spacing
                for index, tag in enumerate(_SEQUENCE_TAGS):
                    choices = data.get('sequence_columns', [])
                    choice = choices[index] if index < len(choices) else _SEQUENCE_NONE
                    dpg.configure_item(tag, items=list(dict.fromkeys([_SEQUENCE_NONE, *self.csv_header, choice])))
                    values[tag] = choice
            else:
                dpg.configure_item('native_image_channels', enabled=True, show=True)
            fit = {**FITNETS_DEFAULTS, **config.get('fitnets', {})}
            values.update(native_fitnets_enabled=config['profile']=='fitnets-v1', native_teacher_checkpoint=fit['teacher_checkpoint'],
                          native_hint_epochs=fit['hint_epochs'], native_distill_temperature=fit['temperature'], native_distill_weight=fit['distill_weight'])
            if self.experiment_mode in IMAGE_MODES:
                values.update({"native_image_split": data.get("split_mode", "stratified"), "native_image_height": data.get("image_size", [32, 32])[0], "native_image_width": data.get("image_size", [32, 32])[1], "native_image_channels": str(data.get("channels", 3))})
            if self.experiment_mode == "paired_csv":
                ports = config.get("input_ports") or {"image": "image", "tabular": "clinical"}
                values.update({"native_paired_image_root": data.get("image_root", ""), "native_paired_image_column": data.get("image_column", "image_path"), "native_paired_id_column": data.get("id_column", "sample_id"), "native_image_port": ports["image"], "native_tabular_port": ports["tabular"]})
            if self.experiment_mode in COHORT_KINDS:
                self.cohort_panel.sync(config, self.csv_header)
            for tag, value in values.items():
                dpg.set_value(tag, value)
        dpg.configure_item("native_csv_settings", show=self.experiment_mode in ("csv", "paired_csv", "regression_csv"))
        dpg.configure_item("native_csv_target", label="连续目标列" if self.experiment_mode == "regression_csv" else "标签列")
        dpg.configure_item("native_csv_group", label="分组列（必选）" if self.experiment_mode == "paired_csv" else "分组列（可选）")
        dpg.configure_item("native_image_settings", show=self.experiment_mode in IMAGE_MODES)
        dpg.configure_item('native_segmentation_settings', show=self.experiment_mode == 'segmentation_csv')
        dpg.configure_item('native_research_settings', show=self.experiment_mode == 'research_csv')
        dpg.configure_item('native_fitnet_options', show=self.experiment_mode in ('csv','image_folder','research_csv'))
        dpg.configure_item("native_image_directory_settings", show=self.experiment_mode == "image_folder")
        dpg.configure_item("native_paired_settings", show=self.experiment_mode == "paired_csv")
        dpg.configure_item("native_paired_ports", show=self.experiment_mode == "paired_csv")
        dpg.configure_item("native_input_port", show=self.experiment_mode not in ('paired_csv', 'multimodal_csv'))
        dpg.configure_item("native_split_fractions", show=self.experiment_mode not in COHORT_KINDS)
        dpg.configure_item("native_split_help", show=self.experiment_mode not in COHORT_KINDS)
        self.cohort_panel.layout(self.experiment_mode)
        self._render_experiment_summary()

    def _show_segmentation_dimension(self, fmt):
        volume = fmt != 'image'
        dpg.configure_item('native_segmentation_depth', show=volume, label='补丁深度 D' if fmt in MEDICAL_FORMATS else '模型深度 D')
        dpg.configure_item('native_image_channels', show=not volume, enabled=not volume)
        dpg.configure_item('native_medical_settings', show=fmt in MEDICAL_FORMATS)
        dpg.configure_item('native_medical_ct', show=fmt == 'dicom_ct')
        dpg.configure_item('native_medical_mri', show=fmt == 'mri_nifti')

    def _segmentation_format_changed(self, app_data=None, **_):
        self._show_segmentation_dimension(SEGMENTATION_FORMATS[app_data])
        self._experiment_changed()

    def _research_format_changed(self, app_data=None, **_):
        if app_data in ('image','gray16'):
            dpg.set_value('native_research_depth',1)
        elif dpg.get_value('native_research_depth') == 1:
            dpg.set_value('native_research_depth',16)
        if app_data != 'video':
            dpg.set_value('native_image_channels','1')
        if app_data in ('image','video'):
            dpg.set_value('native_research_min','')
            dpg.set_value('native_research_max','')
        if app_data != 'dicom':
            dpg.set_value('native_research_modality','auto')
        self._experiment_changed()

    def _configuration(self):
        if not self._widgets_available():
            return deepcopy(self.experiment)
        # DPG stores input_float as float32; remove representation noise so
        # entering the valid lower bound 1e-6 does not become 9.99999997e-7.
        common = {"lr": float(format(dpg.get_value("native_lr"), ".7g")), "seed": dpg.get_value("native_seed"), "batch_size": dpg.get_value("native_batch")}
        if self.experiment_mode == "synthetic_s1":
            return {**common, "steps": dpg.get_value("native_steps")}
        config = deepcopy(self.experiment)
        config.update(common)
        use_fitnets = self.experiment_mode in ('csv','image_folder','research_csv') and dpg.get_value('native_fitnets_enabled')
        if use_fitnets:
            config.update(profile='fitnets-v1', fitnets={
                'teacher_checkpoint':dpg.get_value('native_teacher_checkpoint').strip(),
                'hint_epochs':dpg.get_value('native_hint_epochs'),
                'temperature':float(format(dpg.get_value('native_distill_temperature'),'.7g')),
                'distill_weight':float(format(dpg.get_value('native_distill_weight'),'.7g'))})
        elif config.get('profile') == 'fitnets-v1':
            config['profile'] = 'classification-v2'
            config.pop('fitnets',None)
        config.update({"epochs": dpg.get_value("native_epochs"), "optimizer": dpg.get_value("native_optimizer"), "device": dpg.get_value("native_device"), "input_port": "x" if self.experiment_mode == "paired_csv" else dpg.get_value("native_input_port"), "max_runtime_seconds": dpg.get_value("native_max_runtime"), "resume_checkpoint": dpg.get_value("native_resume_checkpoint").strip() or None})
        if self.experiment_mode == "paired_csv":
            config["input_ports"] = {"image": dpg.get_value("native_image_port"), "tabular": dpg.get_value("native_tabular_port")}
        data = {"kind": self.experiment_mode, "path": dpg.get_value("native_data_path").strip(), "validation_fraction": float(format(dpg.get_value("native_validation_fraction"), ".7g")), "test_fraction": float(format(dpg.get_value("native_test_fraction"), ".7g"))}
        if self.experiment_mode in COHORT_KINDS:
            data = self.cohort_panel.configuration(config, data)
        elif self.experiment_mode in ("csv", "paired_csv", "regression_csv"):
            try:
                features = json.loads(dpg.get_value("native_csv_features"))
            except json.JSONDecodeError as error:
                raise ValueError("数值特征列须为 JSON 字符串列表。") from error
            data.update({"target_column": dpg.get_value("native_csv_target"), "feature_columns": features,
                         "group_column": None if dpg.get_value("native_csv_group") == "无" else dpg.get_value("native_csv_group"),
                         "split_column": None if dpg.get_value("native_csv_split") == "无" else dpg.get_value("native_csv_split"), "standardize": dpg.get_value("native_standardize")})
        if self.experiment_mode == "image_folder":
            data.update({"split_mode": dpg.get_value("native_image_split"), "image_size": [dpg.get_value("native_image_height"), dpg.get_value("native_image_width")], "channels": int(dpg.get_value("native_image_channels"))})
        elif self.experiment_mode == 'segmentation_csv':
            fmt = SEGMENTATION_FORMATS[dpg.get_value('native_segmentation_format')]
            values = dpg.get_value('native_segmentation_values')
            try:
                values = [int(part.strip()) for part in values.split(',')]
            except ValueError:
                pass  # Preserve the draft value for the validation error.
            data.update(format=fmt, source_root=dpg.get_value('native_segmentation_root').strip(), mask_values=values,
                        depth=dpg.get_value('native_segmentation_depth') if fmt != 'image' else 1,
                        image_size=[dpg.get_value('native_image_height'), dpg.get_value('native_image_width')],
                        channels=int(dpg.get_value('native_image_channels')) if fmt == 'image' else 1)
            if fmt in MEDICAL_FORMATS:
                def number(tag):
                    return float(format(dpg.get_value(tag), '.7g'))

                sequences = [dpg.get_value(tag) for tag in _SEQUENCE_TAGS] if fmt == 'mri_nifti' else []
                sequences = [column for column in sequences if column != _SEQUENCE_NONE]
                data.update(sequence_columns=sequences, channels=1 + len(sequences),
                            registered=fmt == 'mri_nifti' and dpg.get_value('native_medical_registered'),
                            ct_window=[number('native_ct_lower_hu'), number('native_ct_upper_hu')] if fmt == 'dicom_ct' else None,
                            spacing_mm=[number('native_medical_spacing_' + axis) for axis in 'DHW'],
                            patches_per_case=dpg.get_value('native_medical_patches'),
                            foreground_fraction=number('native_medical_foreground'), overlap=number('native_medical_overlap'))
            for key, _, default in SEGMENTATION_COLUMNS:
                choice = dpg.get_value('native_seg_' + key)
                data[key] = None if key == 'split_column' and choice == default else choice
        elif self.experiment_mode == 'research_csv':
            data.update(format=dpg.get_value('native_research_format'), source_root=dpg.get_value('native_research_root').strip(),
                        modality=dpg.get_value('native_research_modality'), depth=dpg.get_value('native_research_depth'),
                        image_size=[dpg.get_value('native_image_height'),dpg.get_value('native_image_width')],
                        channels=int(dpg.get_value('native_image_channels')))
            names=[name.strip() for name in dpg.get_value('native_research_columns').split(',')]
            for index, key in enumerate(('id_column','group_column','path_column','target_column','split_column')):
                data[key] = (names[index] if index < len(names) else '') or (None if key=='split_column' else '')
            for tag,key in (('native_research_min','intensity_min'),('native_research_max','intensity_max')):
                raw=dpg.get_value(tag).strip()
                try:
                    data[key] = float(raw) if raw else None
                except ValueError:
                    data[key] = raw  # preserve an invalid draft for the validation error
        elif self.experiment_mode == "paired_csv":
            data.update({"image_root": dpg.get_value("native_paired_image_root").strip(), "image_column": dpg.get_value("native_paired_image_column"), "id_column": dpg.get_value("native_paired_id_column"), "image_size": [dpg.get_value("native_image_height"), dpg.get_value("native_image_width")], "channels": int(dpg.get_value("native_image_channels"))})
        config["data"] = data
        return config

    def _experiment_identity(self):
        config = self._configuration()
        manifest = self.manifest if self.inspection_key == data_identity(config) else None
        return composite_key(self.document, config, manifest)

    def _experiment_changed(self, **_):
        candidate = self._configuration()
        workspace_payload(self.document, candidate)
        if data_identity(candidate) != data_identity(self.experiment):
            self.manifest = None; self.inspection_key = None; self.inspecting = False
            candidate.pop("expected_dataset_digest", None)
            if candidate.get("profile") in SUPERVISED_PROFILES:
                candidate["expected_dataset_digest"] = None
            self.queries.invalidate("inspect")
            self.medical_patch_panel.invalidate()
        self.experiment = candidate
        self._render_experiment_summary()
        self._render_loss(); self._render_probe()

    def _mode_changed(self, app_data=None, **_):
        mode = MODES[app_data]
        current = self._configuration()
        if self.experiment_mode == "synthetic_s1":
            self._s1_experiment = current
        else:
            self._real_experiments[self.experiment_mode] = current
        next_config = self._s1_experiment if mode == "synthetic_s1" else self._real_experiments.get(mode) or self._supervised_defaults(mode)
        self._install_experiment(next_config)
        self._sync_experiment_widgets()
        self._render_loss(); self._render_probe()
        if mode != "synthetic_s1" and self.capabilities is None:
            self._refresh_devices()

    def _open_data_settings(self, **_):
        self._bind_modal('native_data')
        self._sync_experiment_widgets()
        dpg.show_item("native_data")

    def _close_data_settings(self, **_):
        self._experiment_changed()
        dpg.hide_item("native_data")

    def _browse_data(self, **_):
        self._bind_modal('native_data_open')
        dpg.configure_item("native_data_open", directory_selector=self.experiment_mode == "image_folder")
        dpg.show_item("native_data_open")

    def _data_path_chosen(self, app_data=None, **_):
        dpg.set_value("native_data_path", app_data["file_path_name"])
        self._experiment_changed()
        if self.experiment_mode in ("csv", "paired_csv", "regression_csv", "segmentation_csv", *COHORT_KINDS):
            self._read_columns()

    def _read_columns(self, **_):
        path = dpg.get_value("native_data_path").strip()
        self._columns_path = path
        dpg.set_value("native_data_error", "正在读取 CSV 列…")
        self.queries.submit("columns", lambda cancelled: csv_columns(path))

    def _csv_selection_changed(self, preserve=False, **_):
        excluded = {dpg.get_value("native_csv_target"), dpg.get_value("native_csv_group"), dpg.get_value("native_csv_split")}
        if self.experiment_mode == "paired_csv":
            excluded.update((dpg.get_value("native_paired_image_column"), dpg.get_value("native_paired_id_column")))
        if self.csv_header:
            features = []
            if preserve:
                try:
                    current = json.loads(dpg.get_value("native_csv_features"))
                    if isinstance(current, list):
                        features = [name for name in current if name in self.csv_header and name not in excluded]
                except json.JSONDecodeError:
                    pass
            if not features:
                features = [name for name in self.csv_header if name not in excluded]
            dpg.set_value("native_csv_features", json.dumps(features, ensure_ascii=False))
        self._experiment_changed()

    def _inspect_data(self, **_):
        if self.experiment_mode == "synthetic_s1":
            raise ValueError("请选择一个使用真实数据的实验任务。")
        from studio_core.experiment_spec import validate_supervised_config
        editable = self._configuration()
        inspect_config = editable
        if editable.get('profile') == 'fitnets-v1':
            inspect_config = {key:value for key,value in editable.items() if key != 'fitnets'}
            inspect_config['profile'] = 'classification-v2'
        config = validate_supervised_config(inspect_config)
        self.experiment = editable
        self.manifest = None; self.inspection_key = None; self.inspecting = True
        self.medical_patch_panel.invalidate()
        self.inspect_request_key = data_identity(config)
        data, seed = deepcopy(config["data"]), config["seed"]
        def inspect(cancelled):
            from studio_core.classification_data import inspect_dataset
            return inspect_dataset(data, seed=seed, cancelled=cancelled)
        self.queries.submit("inspect", inspect)
        self._render_experiment_summary()
        dpg.set_value("native_data_error", "正在后台检查数据；可继续编辑画布。")

    def _refresh_devices(self, **_):
        def query(cancelled):
            from studio_core.devices import device_capabilities
            return device_capabilities()
        self.queries.submit("devices", query)
        dpg.set_value("native_device_summary", "正在后台查询运行时与显卡…")

    def _poll_background(self):
        for message in self.queries.poll():
            name, error, value = message["name"], message["error"], message["value"]
            if name == 'medical-patch':
                self.medical_patch_panel.receive(value, error)
                continue
            if error:
                if name == "devices":
                    dpg.set_value("native_device_summary", "设备查询失败：" + error)
                else:
                    dpg.set_value("native_data_error", error)
                if name == "inspect":
                    self.inspecting = False
                    self._render_experiment_summary()
                self.log(error, "错误")
                continue
            if name == "columns":
                if dpg.get_value("native_data_path").strip() != self._columns_path:
                    continue
                self.csv_header = value
                if self.experiment_mode in COHORT_KINDS:
                    self.cohort_panel.columns(value)
                    dpg.set_value('native_data_error', f'已读取 {len(value)} 列。请核对病例、患者和目标列。')
                    continue
                if self.experiment_mode == 'segmentation_csv':
                    missing = []
                    for key, label, default in SEGMENTATION_COLUMNS:
                        tag = 'native_seg_' + key
                        items = [default, *value] if key == 'split_column' else value
                        dpg.configure_item(tag, items=items)
                        if dpg.get_value(tag) not in items:
                            dpg.set_value(tag, default if default in items else '')
                            if key != 'split_column' and default not in items:
                                missing.append(label)
                    for tag in _SEQUENCE_TAGS:
                        items = [_SEQUENCE_NONE, *value]
                        dpg.configure_item(tag, items=items)
                        if dpg.get_value(tag) not in items:
                            dpg.set_value(tag, _SEQUENCE_NONE)
                    message = ('请选择' + '、'.join(missing)) if missing else '请核对样本、患者、影像及掩码列。'
                    dpg.set_value('native_data_error', f'已读取 {len(value)} 列。' + message)
                    continue
                dpg.configure_item("native_csv_target", items=value)
                for tag in ("native_csv_group", "native_csv_split"):
                    dpg.configure_item(tag, items=["无"] + value)
                if self.experiment_mode == "paired_csv":
                    for tag, default in (("native_paired_image_column", "image_path"), ("native_paired_id_column", "sample_id")):
                        dpg.configure_item(tag, items=value)
                        if dpg.get_value(tag) not in value:
                            dpg.set_value(tag, default if default in value else "")
                    if dpg.get_value("native_csv_group") not in value:
                        dpg.set_value("native_csv_group", "patient_id" if "patient_id" in value else "无")
                if dpg.get_value("native_csv_target") not in value:
                    target = "" if self.experiment_mode == "regression_csv" else "label" if "label" in value else value[-1]
                    dpg.set_value("native_csv_target", target)
                self._csv_selection_changed(preserve=True)
                dpg.set_value("native_data_error", f"已读取 {len(value)} 列。请选择目标列并核对数值特征。")
            elif name == "inspect":
                if data_identity(self._configuration()) != self.inspect_request_key:
                    self.inspecting = False
                    self.manifest = None; self.inspection_key = None
                    self._render_experiment_summary()
                    dpg.set_value("native_data_error", "数据设置已改变，请重新检查。")
                    continue
                self.inspecting = False
                self.manifest, self.inspection_key = value, self.inspect_request_key
                self.experiment["expected_dataset_digest"] = value["dataset_digest"]
                self._render_experiment_summary()
                dpg.set_value("native_data_error", "检查完成。运行时会再次核对实际内容摘要。")
                self._render_loss(); self._render_probe()
                self.log("数据已检查：" + str(value.get("counts")), "完成")
            elif name == "devices":
                self.capabilities = value
                self.available_devices = device_choices(value)
                dpg.configure_item("native_device", items=self.available_devices)
                dpg.set_value("native_device_summary", device_summary(value))
                self._render_experiment_summary()

    def _render_experiment_summary(self):
        if not self._widgets_available():
            return
        config = self._configuration()
        real = self.experiment_mode != "synthetic_s1"
        manifest = self.manifest if self.inspection_key == data_identity(config) else None
        summary = "正在后台检查数据…" if self.inspecting else manifest_summary(manifest)
        dpg.set_value("native_dataset_summary", summary)
        dpg.configure_item("native_inspect", enabled=not self.inspecting)
        dpg.configure_item("native_data_inspect", enabled=not self.inspecting)
        starter_label = {'segmentation_csv': '创建分割起始图…', 'table_csv': '创建随机森林起始图…',
                         'multimodal_csv': '按所选端口创建分类图…'}.get(self.experiment_mode, '从数据创建起始模型…')
        dpg.configure_item("native_starter", enabled=bool(manifest), label=starter_label)
        fit_templates = self.experiment_mode in ('csv','image_folder','research_csv')
        dpg.configure_item('native_fitnet_templates', show=fit_templates)
        for tag in ('native_teacher_starter','native_fitnets_starter'):
            dpg.configure_item(tag, enabled=bool(manifest) and fit_templates)
        svm_available = self.experiment_mode in ('csv', 'regression_csv')
        svm = 'SVR' if self.experiment_mode == 'regression_csv' else 'SVC'
        dpg.configure_item('native_svm_starter', show=svm_available, enabled=bool(manifest), label=f'创建 {svm} 起始图…')
        classical = self._classical_graph()
        fit = classical or self.experiment_mode == 'table_csv'
        dpg.configure_item('native_epoch_controls', show=not fit)
        dpg.configure_item('native_gradient_controls', show=not fit)
        dpg.configure_item('native_resume_checkpoint', show=not fit)
        dpg.configure_item('native_device', enabled=self.experiment_mode != 'table_csv')
        if self.experiment_mode == 'table_csv':
            dpg.set_value('native_device', 'cpu')
        issue = compatibility_issue(self.document, manifest, config.get("input_port", "x"), input_ports=config.get("input_ports")) if real and manifest else ""
        dpg.set_value("native_data_compatibility", issue)
        if manifest:
            if manifest.get('kind') == 'table_csv':
                recipe = manifest['data_spec']
                details = (f"{recipe['folds']} 折分组验证 · {len(recipe['numeric_columns'])} 个数值列 · "
                           f"{len(recipe['categorical_columns'])} 个类别列\n最终在独立测试集评估。")
            elif manifest.get('kind') == 'multimodal_csv':
                details = '\n'.join(
                    f"{modality} → {config['input_ports'].get(modality, '本次不使用')} · {' × '.join(axes)}"
                    for modality, axes in manifest['input_axes'].items())
            elif manifest.get("kind") == "regression_csv":
                details = "连续目标：" + manifest["target_column"] + " · 特征预处理仅使用训练集拟合。"
            elif manifest.get('kind') == 'segmentation_csv':
                details = '各划分标签像素/体素数（标签顺序 ' + ', '.join(map(str, manifest['mask_values'])) + '）\n'
                details += '  '.join(split + ' ' + str(counts) for split, counts in manifest['class_pixels'].items())
            else:
                details = "类别分布：" + json.dumps(manifest.get("class_counts", {}), ensure_ascii=False)
            if manifest.get("warnings"):
                def warning_text(item):
                    if isinstance(item, dict) and item.get('code') == 'missing_split_classes':
                        stage = f"第 {item['fold']} 折验证集" if 'fold' in item else '测试集'
                        return stage + '没有类别 ' + '、'.join(map(str, item['classes']))
                    return str(item)
                details += "\n数据提示：" + "。".join(map(warning_text, manifest["warnings"]))
            if manifest.get("ignored_files"):
                details += "\n忽略文件（前 10 项）：" + json.dumps(manifest["ignored_files"][:10], ensure_ascii=False)
            dpg.set_value("native_data_details", details[:1800] + ("…（完整列表见 manifest 文件）" if len(details) > 1800 else ""))
        else:
            dpg.set_value("native_data_details", "")
        medical_ready = bool(manifest and manifest['data_spec'].get('format') in MEDICAL_FORMATS)
        dpg.configure_item('native_medical_patch_open', show=medical_ready)
        if dpg.does_item_exist("native_mode_summary"):
            if fit:
                mode_text = '传统机器学习 · CPU 拟合与预测'
            else:
                mode_text = (config.get('device', 'cpu') + ' · ' + self.experiment_mode) if real else 'CPU · S1 合成演示'
            dpg.set_value('native_mode_summary', mode_text)

    def _run_configuration(self):
        config = self._configuration()
        if self.experiment_mode == "synthetic_s1":
            if self._classical_graph():
                raise ValueError("传统机器学习估计器需要 CSV 分类或回归实验。")
            self.experiment = config
            return config
        from studio_core.experiment_spec import validate_supervised_config
        config = validate_supervised_config(config)
        if not self.manifest or self.inspection_key != data_identity(config):
            raise ValueError("运行前请检查当前数据与划分设置。")
        issue = compatibility_issue(self.document, self.manifest, config["input_port"], input_ports=config.get("input_ports"))
        if issue:
            raise ValueError(issue)
        if config["device"] not in self.available_devices:
            raise ValueError("当前设备不可用，请刷新设备能力并选择 CPU 或可用的单个 CUDA 设备。")
        if self._classical_graph():
            from studio_core.classical_task import classical_contract
            classical_contract(self.document.graph, config, self.manifest)
        config["expected_dataset_digest"] = self.manifest["dataset_digest"]
        self.experiment = config
        return config

    def _classical_graph(self):
        from studio_core.classical_task import contains_classical
        return contains_classical(self.document.graph)

    def _request_svm_starter(self, **_):
        estimator = 'SVR' if self.experiment_mode == 'regression_csv' else 'SVC'
        self._request_starter(estimator=estimator)

    def _request_starter(self, estimator=None, teacher=False, fitnets=False, **_):
        self._bind_modal('native_starter_replace')
        config = self._configuration()
        if not self.manifest or self.inspection_key != data_identity(config):
            raise ValueError("请先检查当前数据，再创建匹配的起始模型。")
        self.pending_teacher = teacher
        if teacher or fitnets:
            from .fitnets_templates import starter
            self.pending_starter = starter(self.manifest, teacher=teacher)
        else:
            self.pending_starter = starter_graph(self.manifest, estimator=estimator, input_ports=config.get('input_ports'))
        candidate = Document(self.pending_starter)
        report = candidate.validation()
        if not report.get("valid"):
            raise ValueError("起始模型无法通过校验：" + str(report.get("errors")))
        dpg.show_item("native_starter_replace")

    def _replace_starter(self, **_):
        if self.pending_starter is None:
            raise ValueError("请先从已检查数据准备起始模型。")
        fit = self.pending_starter.get('id') == 'fitnets_starter'
        dpg.set_value('native_fitnets_enabled', fit)
        if fit or self.pending_teacher or self.experiment_mode == 'segmentation_csv':
            dpg.set_value('native_resume_checkpoint','')
        self.document.replace_graph(self.pending_starter)
        self._reset_canvas_document()
        self.pending_starter = None
        self.python_initialized = False; self.python_dirty = False
        self.experiment["input_port"] = "x"
        dpg.set_value("native_input_port", "x")
        if self.experiment_mode == "paired_csv":
            self.experiment["input_ports"] = {"image": "image", "tabular": "clinical"}
            dpg.set_value("native_image_port", "image"); dpg.set_value("native_tabular_port", "clinical")
        if self._classical_graph():
            dpg.set_value("native_device", "cpu")
            dpg.set_value("native_resume_checkpoint", "")
        dpg.hide_item("native_starter_replace")
        self._refresh()
        self.log("已替换为与当前数据匹配的起始模型。", "完成")
