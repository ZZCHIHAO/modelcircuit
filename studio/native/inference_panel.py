"""Configure an independent inference job and browse one original-grid frame at a time."""
from array import array
import json
from pathlib import Path
import shutil

import dearpygui.dearpygui as dpg
import numpy as np

from studio_core.inference_spec import PROFILE
from studio_core.segmentation_predictions import _overlay


MODES = {'按模型自动': 'auto', '整图缩放': 'fixed', '原始网格滑窗': 'sliding'}
_TERMINAL = {'succeeded', 'failed', 'cancelled'}
_PHASES = {'load': '载入模型', 'decode': '读取影像', 'fixed': '整图推理', 'sliding': '滑窗推理', 'saved': '已保存'}
_MUTED = (145, 154, 164)
_MAX_REPORT_BYTES = 8 * 1024 ** 2


class InferencePanel:
    def __init__(self, app):
        self.app = app
        self.record = None
        self.path = None
        self.report = None
        self.case = 0
        self.slice = 0
        self._devices = ()
        self.medical_format = None
        self.sequence_columns = []

    def build_controls(self):
        dpg.add_text('训练运行记录', color=_MUTED)
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag='infer_run_spec', hint='run_spec.json', width=190, on_enter=True,
                               callback=self.app.guarded(self._sync_recipe))
            dpg.add_button(label='选择…', tag='infer_choose_run', callback=self.app.guarded(lambda **_: self._open('run')))
        dpg.add_button(label='使用本次训练的最佳权重', tag='infer_use_best', enabled=False, callback=self.app.guarded(self._use_best))
        dpg.add_text('检查点（默认同目录 best.pt）', color=_MUTED)
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag='infer_checkpoint', hint='可选 .pt', width=190)
            dpg.add_button(label='选择…', callback=self.app.guarded(lambda **_: self._open('checkpoint')))
        dpg.add_text('新影像或同格式文件夹', tag='infer_source_help', color=_MUTED, wrap=265)
        dpg.add_input_text(tag='infer_source', width=-1)
        with dpg.group(horizontal=True):
            dpg.add_button(label='选择影像…', tag='infer_choose_image', callback=self.app.guarded(lambda **_: self._open('image')))
            dpg.add_button(label='选择文件夹…', tag='infer_choose_folder', callback=self.app.guarded(lambda **_: self._open('folder')))
        with dpg.group(tag='infer_sequences', show=False):
            for channel in (2, 3, 4):
                with dpg.group(tag=f'infer_sequence_{channel}_row', show=False):
                    dpg.add_text('', tag=f'infer_sequence_{channel}_label', wrap=265, color=_MUTED)
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag=f'infer_sequence_{channel}', width=190)
                        dpg.add_button(label='选择…', callback=self.app.guarded(self._open, f'sequence_{channel}'))
        dpg.add_combo(list(MODES), tag='infer_mode', default_value=next(iter(MODES)), label='模式', width=157,
                      callback=self.app.guarded(self._mode_changed))
        dpg.add_slider_int(label='重叠 %', tag='infer_overlap', default_value=25, min_value=0, max_value=75, width=158, show=False)
        with dpg.group(horizontal=True):
            dpg.add_combo(['cpu'], default_value='cpu', tag='infer_device', label='设备', width=126)
            dpg.add_button(label='刷新', callback=self.app.guarded(self.app._refresh_devices))
        with dpg.group(horizontal=True):
            dpg.add_button(label='开始推理', tag='infer_start', callback=self.app.guarded(self.start))
            dpg.add_button(label='取消', tag='infer_cancel', enabled=False, callback=self.app.guarded(self.cancel))
        dpg.add_progress_bar(tag='infer_progress', width=-1, overlay='等待输入')
        dpg.add_text('', tag='infer_status', wrap=264, color=_MUTED)
        dpg.add_button(label='打开已保存的推理结果…', tag='infer_choose_report', callback=self.app.guarded(lambda **_: self._open('report')))

    def build_results(self):
        with dpg.tab(label='推理结果', tag='infer_results_tab', show=False):
            dpg.add_text('', tag='infer_summary', color=_MUTED)
            with dpg.group(horizontal=True):
                dpg.add_combo([''], tag='infer_case', label='影像', width=250, callback=self.app.guarded(self._choose_case))
                dpg.add_slider_int(label='原始切片', tag='infer_slice', min_value=1, max_value=1, width=180, clamped=True,
                                   callback=self.app.guarded(self._choose_slice))
                dpg.add_button(label='导出掩码…', tag='infer_export', callback=self.app.guarded(lambda **_: self._open('export')))
            dpg.add_text('左  原始影像     右  掩码叠加', color=_MUTED)
            dpg.add_group(tag='infer_image_host')

    def build_resources(self):
        dpg.add_texture_registry(tag='infer_textures', show=False)
        dialogs = [('run', '选择 run_spec.json', '.json'), ('checkpoint', '选择检查点', '.pt'), ('image', '选择影像', '.*'),
                   ('folder', '选择影像文件夹', None), ('report', '打开 inference.json', '.json'), ('export', '选择掩码保存目录', None)]
        dialogs.extend((f'sequence_{channel}', f'选择 MRI 通道 {channel}', '.*') for channel in (2, 3, 4))
        for name, label, extension in dialogs:
            with dpg.file_dialog(tag='infer_picker_' + name, label=label, show=False, modal=True, width=740, height=460,
                                 default_path=str(self.app.data_directory), directory_selector=extension is None,
                                 callback=self.app.guarded(lambda app_data=None, key=name, **_: self._chosen(key, app_data))):
                if extension is not None:
                    dpg.add_file_extension(extension)

    def _open(self, name, **_):
        tag = 'infer_picker_' + name
        self.app._bind_modal(tag)
        dpg.show_item(tag)

    def _chosen(self, key, payload):
        path = payload.get('file_path_name') or payload.get('current_path')
        if not path:
            raise ValueError('请选择路径。')
        if key == 'report':
            self.show_report(path)
        elif key == 'export':
            self._export(Path(path))
        else:
            field = {'run': 'run_spec', 'checkpoint': 'checkpoint'}.get(key, key if key.startswith('sequence_') else 'source')
            dpg.set_value('infer_' + field, path)
            if key == 'run':
                dpg.set_value('infer_checkpoint', '')
                self._sync_recipe(reset=True)
        dpg.hide_item('infer_picker_' + key)

    def _use_best(self, **_):
        result = self.app.result or {}
        if result.get('profile') != 'segmentation-v1':
            raise ValueError('请先完成一个分割训练。')
        files = result['artifacts']
        dpg.set_value('infer_run_spec', files['run_spec'])
        dpg.set_value('infer_checkpoint', files['best_checkpoint'])
        self._sync_recipe(reset=True)

    def _sync_recipe(self, reset=False, **_):
        from studio_core.medical_volume_data import MEDICAL_FORMATS
        from studio_core.segmentation_inference import MAX_SPEC_BYTES

        raw_path = dpg.get_value('infer_run_spec').strip()
        spec = {}
        if raw_path:
            try:
                with Path(raw_path).expanduser().open('rb') as stream:
                    raw = stream.read(MAX_SPEC_BYTES + 1)
                if len(raw) > MAX_SPEC_BYTES:
                    raise ValueError('训练记录过大。')
                spec = json.loads(raw)['config']['data']
                if not isinstance(spec, dict):
                    raise TypeError('Invalid data spec')
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                raise ValueError('请选择有效的分割训练记录 run_spec.json。') from exc
        fmt = spec.get('format')
        fmt = fmt if fmt in MEDICAL_FORMATS else None
        columns = spec.get('sequence_columns', []) if fmt == 'mri_nifti' else []
        if not isinstance(columns, list) or len(columns) > 3 or not all(isinstance(value, str) for value in columns):
            raise ValueError('训练记录的 MRI 序列设置无效。')
        if reset or fmt != self.medical_format or columns != self.sequence_columns:
            for channel in (2, 3, 4):
                dpg.set_value(f'infer_sequence_{channel}', '')
        self.medical_format, self.sequence_columns = fmt, columns
        dpg.configure_item('infer_sequences', show=bool(columns))
        for channel in (2, 3, 4):
            index = channel - 2
            dpg.configure_item(f'infer_sequence_{channel}_row', show=index < len(columns))
            if index < len(columns):
                dpg.set_value(f'infer_sequence_{channel}_label', f'通道 {channel} · {columns[index]}')
        dpg.set_value('infer_source_help', {'dicom_ct': 'CT 序列文件夹或多帧 DICOM 文件',
                                           'mri_nifti': 'MRI 参考 NIfTI（通道 1）'}.get(fmt, '新影像或同格式文件夹'))
        dpg.configure_item('infer_choose_folder', show=fmt != 'mri_nifti')
        dpg.configure_item('infer_mode', enabled=fmt is None)
        if fmt:
            dpg.set_value('infer_mode', next(iter(MODES)))
        self._mode_changed(dpg.get_value('infer_mode'))

    def _mode_changed(self, app_data=None, **_):
        dpg.configure_item('infer_overlap', show=MODES[app_data] == 'sliding')

    def start(self, **_):
        if self.record and self.record['status'] not in _TERMINAL:
            raise ValueError('已有推理任务正在运行。')
        device = dpg.get_value('infer_device')
        if device not in self.app.available_devices:
            raise ValueError('请选择可用的 CPU 或 CUDA 设备。')
        self._sync_recipe()
        sequences = [dpg.get_value(f'infer_sequence_{channel}').strip() for channel in range(2, 2 + len(self.sequence_columns))]
        if any(not value for value in sequences):
            raise ValueError('请按通道顺序选择全部 MRI 附加序列。')
        mode = MODES[dpg.get_value('infer_mode')]
        self.record = self.app.manager.submit_segmentation_inference({
            'run_spec': dpg.get_value('infer_run_spec').strip(), 'checkpoint': dpg.get_value('infer_checkpoint').strip() or None,
            'source': dpg.get_value('infer_source').strip(), 'mode': mode, 'sequence_paths': sequences,
            'overlap': dpg.get_value('infer_overlap') / 100 if mode == 'sliding' else None, 'device': device})
        dpg.configure_item('infer_start', enabled=False)
        dpg.configure_item('infer_cancel', enabled=True)
        dpg.set_value('infer_status', '')
        dpg.set_value('infer_progress', 0.0)
        dpg.configure_item('infer_progress', overlay='等待启动')

    def cancel(self, **_):
        if self.record and self.record['status'] not in _TERMINAL:
            self.app.manager.cancel(self.record['run_id'])

    def poll(self):
        devices = tuple(self.app.available_devices)
        if devices != self._devices:
            dpg.configure_item('infer_device', items=list(devices))
            self._devices = devices
        result = self.app.result or {}
        dpg.configure_item('infer_use_best', enabled=result.get('profile') == 'segmentation-v1')
        if not self.record or self.record['status'] in _TERMINAL:
            return
        self.record = self.app.manager.get(self.record['run_id'])
        status = self.record['status']
        event = self.record.get('progress', {})
        done, total = event.get('step', 0), event.get('total', 0)
        tiles, tile_total = event.get('batch', 0), event.get('batches', 0)
        phase = event.get('phase')
        fraction = tiles / tile_total if tile_total and phase == 'sliding' else 0.0
        dpg.set_value('infer_progress', min(1.0, (done + fraction) / max(1, total)))
        if status in _TERMINAL:
            dpg.configure_item('infer_start', enabled=True)
            dpg.configure_item('infer_cancel', enabled=False)
        if status == 'succeeded':
            result = self.record['result']
            dpg.configure_item('infer_progress', overlay='推理完成')
            dpg.set_value('infer_status', f"已保存 {result['cases']} 个预测掩码")
            try:
                self.show_report(result['artifacts']['inference'])
            except (OSError, ValueError, KeyError) as exc:
                dpg.set_value('infer_status', '预测已保存。无法打开预览，' + str(exc))
        elif status == 'failed':
            dpg.configure_item('infer_progress', overlay='推理失败')
            dpg.set_value('infer_status', str(self.record.get('error') or '推理失败'))
            self.app.log(str(self.record.get('error') or '推理失败'), '错误')
        elif status in ('cancelled', 'cancel_requested'):
            dpg.configure_item('infer_progress', overlay='已取消' if status == 'cancelled' else '正在取消')
        elif phase:
            detail = f' · 窗口 {tiles}/{tile_total}' if phase == 'sliding' and tile_total else ''
            dpg.configure_item('infer_progress', overlay=_PHASES.get(phase, phase) + (f' {done}/{total}' if total else ''))
            dpg.set_value('infer_status', detail.lstrip(' ·'))

    def show_report(self, path):
        path = Path(path).resolve()
        with path.open('rb') as stream:
            raw = stream.read(_MAX_REPORT_BYTES + 1)
        if len(raw) > _MAX_REPORT_BYTES:
            raise ValueError('推理报告超过预览大小范围。')
        report = json.loads(raw)
        if not isinstance(report, dict) or report.get('version') != 1 or report.get('profile') != PROFILE:
            raise ValueError('请选择分割推理生成的 inference.json。')
        cases = report.get('cases')
        if not isinstance(cases, list) or not 1 <= len(cases) <= 2048 or any(
                not isinstance(case, dict) or not isinstance(case.get('source_name'), str) for case in cases):
            raise ValueError('推理报告没有有效的影像清单。')
        names = [f"{index}. {case['source_name']}" for index, case in enumerate(cases, 1)]
        image, depth = self._image(path, cases[0], 0)
        self.path, self.report, self.case, self.slice = path, report, 0, 0
        dpg.configure_item('infer_case', items=names)
        dpg.set_value('infer_case', names[0])
        dpg.show_item('infer_results_tab')
        self._render(image, depth)
        dpg.set_value('native_bottom_tabs', dpg.get_alias_id('infer_results_tab'))

    @staticmethod
    def _artifact(report, case, field, subfolder):
        root = report.parent / subfolder
        path = (report.parent / case[field]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('推理文件缺失或不在对应的产物目录。')
        return path

    def _image(self, report, case, index):
        shape = case.get('view_shape')
        if (not isinstance(shape, list) or len(shape) != 4 or any(type(axis) is not int or axis < 1 for axis in shape)
                or max(shape[1:3]) > 224 or shape[3] != 4 or not 0 <= index < shape[0]):
            raise ValueError('切片索引或预览尺寸无效。')
        path = self._artifact(report, case, 'view_file', 'views')
        frames = np.load(path, allow_pickle=False, mmap_mode='r')
        if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8 or list(frames.shape) != shape:
            if hasattr(frames, 'close'):
                frames.close()
            raise ValueError('切片预览与报告的尺寸或格式不一致。')
        frame = frames[index].copy()
        del frames
        rgb = frame[..., :3]
        labels = frame[..., 3].astype(np.int16)
        width = rgb.shape[1]
        composite = np.full((rgb.shape[0], width * 2 + 6, 4), 255, dtype=np.uint8)
        composite[:, :width, :3] = rgb
        composite[:, width:width + 6, :3] = (25, 32, 44)
        composite[:, width + 6:, :3] = _overlay(rgb, labels)
        return composite, shape[0]

    def _render(self, pixels, depth):
        height, width = pixels.shape[:2]
        for tag in ('infer_image', 'infer_texture'):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        texture = array('f', (value / 255 for value in pixels.tobytes()))
        dpg.add_static_texture(width, height, texture, parent='infer_textures', tag='infer_texture')
        scale = min(680 / width, 131 / height)
        dpg.add_image('infer_texture', parent='infer_image_host', tag='infer_image',
                      width=max(1, int(width * scale)), height=max(1, int(height * scale)))
        dpg.configure_item('infer_slice', max_value=depth, show=depth > 1)
        dpg.set_value('infer_slice', self.slice + 1)
        case = self.report['cases'][self.case]
        mode = '滑窗' if self.report.get('mode') == 'sliding' else '整图'
        shape = '×'.join(map(str, case['source_shape']))
        thumbnail = ' · 缩略显示' if case['view_shape'][1:3] != case['source_shape'][-2:] else ''
        dpg.set_value('infer_summary', f"{self.report['device']} · {mode} · {shape} · {case['tile_count']} 个窗口" +
                      (f' · 原始第 {self.slice + 1}/{depth} 层' if depth > 1 else '') + thumbnail)

    def _choose_case(self, app_data=None, **_):
        index = int(app_data.split('. ', 1)[0]) - 1
        image, depth = self._image(self.path, self.report['cases'][index], 0)
        self.case, self.slice = index, 0
        self._render(image, depth)

    def _choose_slice(self, app_data=None, **_):
        index = app_data - 1
        image, depth = self._image(self.path, self.report['cases'][self.case], index)
        self.slice = index
        self._render(image, depth)

    def _export(self, destination):
        source = self._artifact(self.path, self.report['cases'][self.case], 'mask_file', 'masks')
        target = destination / ('seg-' + self.path.parent.name + '-' + source.name)
        with source.open('rb') as stream, target.open('xb') as output:
            try:
                shutil.copyfileobj(stream, output)
            except OSError:
                output.close()
                target.unlink(missing_ok=True)
                raise
        self.app.log('原始网格掩码已导出到 ' + str(target), '完成')
