"""Portable project controls. All DearPyGui calls stay on the main thread."""
from copy import deepcopy
from datetime import datetime
from itertools import islice
from math import isfinite
import os
from pathlib import Path
import stat

import dearpygui.dearpygui as dpg

from .experiment_panel import BackgroundQueries
from .workspace_state import data_identity


_PREFIX = 'native_project_'
_MUTED = (145, 154, 164)
_TERMINAL = {'succeeded', 'failed', 'cancelled'}
_RUN_LABELS = {'queued': '等待启动', 'preparing': '准备运行', 'running': '正在训练',
               'cancel_requested': '正在停止', 'succeeded': '运行完成', 'failed': '运行失败', 'cancelled': '已停止'}
_RUN_PHASES = {'bind': '核对数据', 'train': '训练', 'val': '验证', 'test': '最终评估'}
_KINDS = {'image_folder': 'ImageFolder 图片分类'}
_SEGMENTATION_LABELS = {'image': '二维 PNG/TIFF 分割', 'nifti': '三维 NIfTI 分割'}
_REPORT_NAME = 'project-run.json'
_HISTORY_LIMIT = 40
_METRIC_LIMIT = 80


def _tag(name):
    return _PREFIX + name


def _number(value, *, signed=False):
    try:
        if type(value) in (int, float) and isfinite(value):
            return format(value, '+.6g' if signed else '.6g')
    except OverflowError:
        pass
    return '—'


def _metric_values(mapping, prefix=''):
    for key, value in mapping.items():
        name = prefix + key
        if isinstance(value, dict):
            yield from _metric_values(value, name + '.')
        elif type(value) in (int, float):
            yield name, value


def _recent_reports(directory, cancelled):
    dates = []
    with os.scandir(directory) as entries:
        for index, entry in enumerate(entries):
            if index >= 8192 or cancelled():
                break
            if not entry.is_dir(follow_symlinks=False):
                continue
            report = Path(entry.path) / _REPORT_NAME
            try:
                info = report.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                dates.append((info.st_mtime_ns, report))
    dates.sort(reverse=True)
    return [path.resolve() for _, path in dates[:_HISTORY_LIMIT]]


def _read_report(path, cancelled):
    from studio_core.project_runner import read_project_run

    report = read_project_run(path, verify=True, cancelled=cancelled)
    metrics = report.get('metrics', {})
    history = report.get('history', [])
    if (not isinstance(metrics, dict) or not isinstance(history, list) or len(history) > 1000
            or any(not isinstance(epoch, dict) for epoch in history)):
        raise ValueError('运行记录中的指标或训练周期格式无效。')
    for key in ('model_digest', 'dataset_digest', 'device'):
        if report.get(key) is not None and not isinstance(report[key], str):
            raise ValueError('运行记录中的 ' + key + ' 格式无效。')
    return report


def _training_artifacts(report):
    if report.get('operation') == 'train' and report.get('status') == 'succeeded':
        return {**(report.get('artifacts') or {}), 'finished_at': report.get('finished_at')}
    return {}


def _ordered_run_specs(candidates):
    """Order finished jobs by report time or their final metrics write."""
    completed = []
    for files in candidates:
        if not files.get('run_spec') or not files.get('best_checkpoint'):
            continue
        try:
            spec = Path(files['run_spec']).resolve()
            checkpoint = Path(files['best_checkpoint']).resolve()
            if not spec.is_file() or not checkpoint.is_file() or spec.with_name('best.pt') != checkpoint:
                continue
            metrics = Path(files['metrics']) if files.get('metrics') else checkpoint
            stamp = metrics.stat().st_mtime_ns if metrics.is_file() else checkpoint.stat().st_mtime_ns
        except OSError:
            continue
        finished = stamp
        when = files.get('finished_at')
        if isinstance(when, str):
            try:
                recorded = datetime.fromisoformat(when)
                if recorded.utcoffset() is not None:
                    finished = int(recorded.timestamp() * 10 ** 9)
            except (OSError, ValueError, OverflowError):
                pass
        completed.append((finished, stamp, spec))
    return tuple(dict.fromkeys(spec for _, _, spec in sorted(completed, reverse=True)))


def _export_checkpoint(candidates, paths, directory, cancelled, *, inspected_paths=()):
    from studio_core.project_runner import read_project_run

    candidates = list(candidates)
    if directory is not None and Path(directory).is_dir():
        paths = [*paths, *_recent_reports(directory, cancelled)]
    for path in dict.fromkeys(paths):
        if path in inspected_paths:
            continue
        if cancelled():
            raise InterruptedError('工程导出已取消。')
        try:
            candidates.append(_training_artifacts(read_project_run(path)))
        except (OSError, ValueError):
            continue  # A historical report can be moved or deleted independently.
    checkpoints = _ordered_run_specs(candidates)
    if not checkpoints:
        raise ValueError('请先完成训练，或取消附带权重。')
    return checkpoints


class ProjectPanel:
    def __init__(self, app):
        self.app = app
        self.queries = BackgroundQueries()
        self.project = None
        self.binding = None
        self.record = None
        self._operation = None
        self._devices = ()
        self._recent = []
        self._opened = []
        self._choices = {'': None}
        self._reports = {'left': None, 'right': None}
        self._report_errors = {}

    def build_controls(self):
        dpg.add_text('导出当前模型与实验设置', color=_MUTED)
        dpg.add_input_text(tag=_tag('export_path'), hint='新工程目录', width=-1)
        dpg.add_checkbox(label='附带最近匹配训练的最佳权重', tag=_tag('include_weights'))
        with dpg.group(horizontal=True):
            self._button('选择上级目录…', 'choose_export', self._open, 'export')
            self._button('导出工程', 'export', self.export)
        self._button('创建示例工程…', 'create_example', self._open, 'example')
        dpg.add_separator()
        dpg.add_text('打开工程并重新绑定数据', color=_MUTED)
        dpg.add_input_text(tag=_tag('path'), hint='project.json', width=-1,
                           callback=self.app.guarded(self._binding_changed))
        with dpg.group(horizontal=True):
            self._button('选择工程…', 'choose_project', self._open, 'project')
            self._button('打开工程', 'open', self.open_project)
        dpg.add_text('', tag=_tag('summary'), wrap=252, color=_MUTED)
        dpg.add_input_text(tag=_tag('data'), hint='目标数据路径', width=-1,
                           callback=self.app.guarded(self._binding_changed))
        self._button('选择数据…', 'choose_data', self._choose_data, enabled=False)
        with dpg.group(tag=_tag('source_root_group'), show=False):
            self._button('使用工程清单，选择影像目录…', 'choose_pair_directory', self._open, 'pair_directory')
            dpg.add_input_text(tag=_tag('source_root'), hint='CSV 影像与掩码根目录（可选）', width=-1,
                               callback=self.app.guarded(self._binding_changed))
            self._button('选择影像根目录…', 'choose_source_root', self._open, 'source_root')
        self._button('核对数据并恢复画布', 'bind', self.bind, enabled=False)
        with dpg.group(horizontal=True):
            dpg.add_combo(['cpu'], default_value='cpu', tag=_tag('device'), label='设备', width=128)
            self._button('刷新', 'refresh_devices', self.app._refresh_devices)
        with dpg.group(horizontal=True):
            self._button('运行保存的工程', 'run', self.start, enabled=False)
            self._button('停止', 'cancel', self.cancel, enabled=False)
        dpg.add_progress_bar(tag=_tag('progress'), width=-1, overlay='尚未运行')
        dpg.add_text('', tag=_tag('run_status'), wrap=252, color=_MUTED)
        dpg.add_text('', tag=_tag('status'), wrap=252, color=_MUTED)
        self._button('查看工程运行记录', 'show_history', self.show_history)

    def build_results(self):
        with dpg.tab(label='工程运行记录', tag=_tag('results_tab')):
            with dpg.group(horizontal=True):
                self._button('刷新本地记录', 'refresh_history', self.refresh_history)
                for side, label in (('left', 'A'), ('right', 'B')):
                    dpg.add_combo([''], default_value='', width=158, label=label, tag=_tag('history_' + side),
                                  callback=self.app.guarded(self._history_selected, side))
                    self._button('打开记录 ' + label + '…', 'choose_' + side, self._open, side)
            dpg.add_text('请选择一份运行记录。', tag=_tag('compare_status'), wrap=930, color=_MUTED)
            with dpg.child_window(width=-1, height=-1, border=False, horizontal_scrollbar=True):
                dpg.add_group(tag=_tag('results'))

    def build_resources(self):
        dialogs = (
            ('project', '选择 project.json', '.json'),
            ('export', '选择新工程的上级目录', None),
            ('example', '选择示例工程的上级目录', None),
            ('data_directory', '选择 ImageFolder 数据目录', None),
            ('data_csv', '选择影像与掩码清单 CSV', '.csv'),
            ('pair_directory', '选择工程清单对应的影像与掩码目录', None),
            ('source_root', '选择影像与掩码根目录', None),
            ('left', '选择 project-run.json 作为记录 A', '.json'),
            ('right', '选择 project-run.json 作为记录 B', '.json'),
        )
        for key, label, extension in dialogs:
            with dpg.file_dialog(tag=_tag('picker_' + key), label=label, show=False, modal=True, width=740,
                                 height=460, default_path=str(self.app.data_directory),
                                 directory_selector=extension is None, callback=self.app.guarded(self._chosen, key)):
                if extension is not None:
                    dpg.add_file_extension(extension)

    def _button(self, label, name, action, *args, enabled=True):
        dpg.add_button(label=label, tag=_tag(name), callback=self.app.guarded(action, *args), enabled=enabled)

    def _open(self, name, **_):
        tag = _tag('picker_' + name)
        self.app._bind_modal(tag)
        dpg.show_item(tag)

    def _choose_data(self, **_):
        if not self._project_selected():
            return
        self._open('data_directory' if self._dataset_kind() == 'image_folder' else 'data_csv')

    def _chosen(self, key, app_data=None, **_):
        payload = app_data or {}
        path = payload.get('file_path_name') or payload.get('current_path')
        if not path:
            return
        dpg.hide_item(_tag('picker_' + key))
        if key == 'example':
            suffix = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.create_example(Path(path) / ('ModelCircuitExample-' + suffix))
        elif key == 'export':
            dpg.set_value(_tag('export_path'), str(Path(path) / 'ModelCircuitProject'))
        elif key in self._reports:
            self.open_run(path, side=key)
        else:
            field = {'project': 'path', 'data_directory': 'data', 'data_csv': 'data',
                     'pair_directory': 'data', 'source_root': 'source_root'}[key]
            dpg.set_value(_tag(field), path)
            if key == 'pair_directory':
                dpg.set_value(_tag('source_root'), '')
            self._binding_changed()

    @staticmethod
    def _value(name):
        return dpg.get_value(_tag(name)).strip()

    def _error(self, message, *, field='status'):
        dpg.set_value(_tag(field), str(message))
        self.app.log(str(message), '错误')

    def _submit(self, operation, status, task):
        if self._operation:
            return
        self.queries.submit(operation, task)
        self._operation = operation
        dpg.set_value(_tag('status'), status)
        self._sync_actions()

    def export(self, **_):
        from studio_core.research_project import export_project

        if self._operation:
            return
        destination = self._value('export_path')
        if not destination:
            return self._error('请填写导出目录。')
        try:
            config = self.app._configuration()
            graph = deepcopy(self.app.document.graph)
            workbench = self.app.workbench.to_payload()
            manifest = self.app.manifest if self.app.inspection_key == data_identity(config) else None
            config['expected_dataset_digest'] = manifest['dataset_digest'] if manifest else None
            with_weights = dpg.get_value(_tag('include_weights'))
            if with_weights:
                result = self.app.result or {}
                reports = [report for report in self._reports.values() if report]
                candidates = [dict(result.get('artifacts') or {}), *map(_training_artifacts, reports)]
                known = {report['path'] for report in reports}
                opened = [path for path in self._opened if path not in known]
                run_directory = getattr(self.app.manager, 'directory', None)
            runtime_source = self.app.project_runtime_source()
        except (OSError, ValueError) as error:
            return self._error(error)

        def save(cancelled):
            checkpoint = None
            if with_weights:
                checkpoint = _export_checkpoint(candidates, opened, run_directory, cancelled, inspected_paths=known)
            return export_project(destination, graph, config, workbench=workbench, checkpoint=checkpoint,
                                  runtime_source=runtime_source)

        self._submit('export', '正在检查数据并导出工程…', save)

    def open_project(self, **_):
        from studio_core.research_project import read_project

        if self._operation:
            return
        path = self._value('path')
        if not path:
            return self._error('请选择 project.json。')
        self._clear_project()
        self._submit('open', '正在读取工程与模型…', lambda _: (path, read_project(path)))

    def create_example(self, destination):
        from studio_core.project_examples import create_example_project
        from studio_core.research_project import read_project

        if self._operation:
            return
        runtime_source = self.app.project_runtime_source()

        def create(_):
            path, data, root = create_example_project(destination, runtime_source=runtime_source)
            return read_project(path), data, root

        self._submit('example', '正在创建示例工程与数据…', create)

    def _dataset_kind(self):
        return self.project['config']['data']['kind'] if self.project else None

    def _project_selected(self):
        return self.project is not None and self._value('path') == str(self.project['path'])

    def _binding_fields(self):
        root = self._value('source_root') if self._dataset_kind() == 'segmentation_csv' else ''
        return self._value('data'), root or None

    def _binding_changed(self, **_):
        self.binding = None
        if self._project_selected():
            dpg.set_value(_tag('status'), '请核对数据路径并重新绑定。')
        else:
            self._clear_project()
            dpg.set_value(_tag('status'), '请先打开工程。')
        self._sync_actions()

    def bind(self, **_):
        from studio_core.research_project import bind_project

        if self._operation:
            return
        if not self._project_selected():
            return self._error('请先打开工程。')
        project = self.project
        data, root = self._binding_fields()
        if not data:
            return self._error('请选择目标数据。')
        self.binding = None

        def inspect(cancelled):
            config, manifest = bind_project(project, data, source_root=root, cancelled=cancelled)
            return (data, root), config, manifest

        self._submit('bind', '正在核对数据内容与原有划分…', inspect)

    def _clear_project(self):
        self.project = self.binding = None
        dpg.set_value(_tag('summary'), '')
        dpg.hide_item(_tag('source_root_group'))

    def _set_project(self, project, data='', root=None):
        self.project = project
        self.binding = None
        kind = self._dataset_kind()
        dpg.set_value(_tag('path'), str(project['path']))
        dpg.set_value(_tag('data'), str(data))
        dpg.set_value(_tag('source_root'), str(root) if root else '')
        dpg.configure_item(_tag('source_root_group'), show=kind == 'segmentation_csv')
        has_pairs = kind == 'segmentation_csv' and project.get('paired_csv') is not None
        dpg.configure_item(_tag('choose_pair_directory'), show=has_pairs)
        dpg.configure_item(_tag('choose_data'), label='选择 CSV…' if kind == 'segmentation_csv' else '选择数据…')
        if kind == 'image_folder':
            data_hint = 'ImageFolder 数据目录'
        else:
            data_hint = '配对数据 CSV 或影像目录' if has_pairs else '配对数据 CSV'
        dpg.configure_item(_tag('data'), hint=data_hint)
        label = _KINDS.get(kind, kind)
        if kind == 'segmentation_csv':
            label = _SEGMENTATION_LABELS.get(project['config']['data'].get('format'), '配对影像分割')
        dpg.set_value(_tag('summary'), f"{label} · 工程 {project['project_id'][:12]}")
        dpg.set_value(_tag('status'), project.get('graph_error') or '已读取工程，请绑定本机数据以恢复画布。')

    def _finish_operation(self, name, value):
        if name == 'export':
            dpg.set_value(_tag('path'), str(value))
            self._clear_project()
            dpg.set_value(_tag('status'), f'已导出 {value}。可打开工程并重新绑定数据。')
            self.app.log(f'已导出科研工程 {value}', '完成')
        elif name == 'open':
            requested, project = value
            if self._value('path') == requested:
                self._set_project(project)
        elif name == 'example':
            self._set_project(*value)
            self.bind()
        elif name == 'bind':
            fields, config, manifest = value
            if self._project_selected() and fields == self._binding_fields():
                self.app._install_portable_project(self.project, config, manifest)
                self.binding = fields
                counts = manifest['counts']
                dpg.set_value(_tag('status'), f"数据已核对，画布已恢复。训练 {counts['train']} / 验证 {counts['val']} / 测试 {counts['test']}")
                self.app.log(f"科研工程已绑定 {self.project['path']}", '完成')

    def _can_run(self):
        return (not self._operation and not self._running() and self._project_selected()
                and self.binding is not None and self.binding == self._binding_fields())

    def _running(self):
        return self.record is not None and self.record['status'] not in _TERMINAL

    def _sync_actions(self):
        busy = self._operation is not None
        selected = self._project_selected()
        for name in ('export', 'open', 'create_example', 'choose_export', 'choose_project', 'path', 'export_path',
                     'include_weights', 'data', 'source_root'):
            dpg.configure_item(_tag(name), enabled=not busy)
        for name in ('choose_data', 'choose_pair_directory', 'choose_source_root', 'bind'):
            dpg.configure_item(_tag(name), enabled=not busy and selected)
        dpg.configure_item(_tag('run'), enabled=self._can_run())
        dpg.configure_item(_tag('cancel'), enabled=self._running() and self.record['status'] != 'cancel_requested')

    def start(self, **_):
        if not self._can_run():
            return self._error('请先绑定当前工程的数据，或等待已有运行结束。', field='run_status')
        device = self._value('device')
        if device not in self.app.available_devices:
            return self._error('请选择可用的 CPU 或 CUDA 设备。', field='run_status')
        data, root = self.binding
        config = {'project_path': str(self.project['path']), 'data_path': data, 'device': device,
                  'expected_model_digest': self.project['model_digest'],
                  'max_runtime_seconds': self.project['config']['max_runtime_seconds']}
        if root:
            config['source_root'] = root
        try:
            self.record = self.app.manager.submit_project(config)
        except (OSError, ValueError) as error:
            return self._error(error, field='run_status')
        dpg.set_value(_tag('progress'), 0.0)
        dpg.configure_item(_tag('progress'), overlay='等待启动')
        dpg.set_value(_tag('run_status'), '')
        self._sync_actions()

    def cancel(self, **_):
        if self._running():
            # Let poll observe the terminal state even if the worker just exited.
            try:
                self.app.manager.cancel(self.record['run_id'])
            except (KeyError, OSError, ValueError) as error:
                self._error(error, field='run_status')

    def _poll_run(self):
        if not self._running():
            return
        try:
            record = self.app.manager.get(self.record['run_id'])
        except KeyError:
            record = None
        if record is None:
            self.record = None
            dpg.configure_item(_tag('progress'), overlay='运行记录已移出队列')
            self.refresh_history()
            return
        self.record = record
        status = record['status']
        progress = record.get('progress') or {}
        step, total = progress.get('step', 0), progress.get('total', 0)
        label = _RUN_LABELS.get(status, status)
        if status == 'running':
            label = _RUN_PHASES.get(progress.get('phase'), label)
            if progress.get('epochs'):
                label += f" · 周期 {progress.get('epoch', 0)}/{progress['epochs']}"
        dpg.configure_item(_tag('progress'), overlay=label)
        dpg.set_value(_tag('progress'), 1.0 if status == 'succeeded' else min(1.0, step / max(1, total)))
        if status == 'succeeded':
            path = ((record.get('result') or {}).get('artifacts') or {}).get('project_run')
            if path:
                self.open_run(path)
                self.show_history()
                dpg.set_value(_tag('run_status'), str(path))
            else:
                self._error('运行已完成，但未返回工程运行记录。', field='run_status')
        elif status == 'failed':
            self._error(record.get('error') or '工程运行失败。', field='run_status')

    def show_history(self, **_):
        if dpg.does_item_exist('native_bottom_tabs'):
            dpg.set_value('native_bottom_tabs', dpg.get_alias_id(_tag('results_tab')))
        self.refresh_history()

    def refresh_history(self, **_):
        directory = self.app.manager.directory
        self.queries.submit('history', lambda cancelled: _recent_reports(directory, cancelled))

    def open_run(self, path, *, side='left'):
        self._reports[side] = None
        self._report_errors.pop(side, None)
        self._render_reports()
        dpg.set_value(_tag('compare_status'), '正在读取运行记录…')
        self.queries.submit(side, lambda cancelled: _read_report(path, cancelled))

    def _history_selected(self, side, app_data=None, **_):
        path = self._choices.get(app_data)
        if path:
            self.open_run(path, side=side)
        else:
            self.queries.invalidate(side)
            self._reports[side] = None
            self._report_errors.pop(side, None)
            self._render_reports()

    def _sync_history(self):
        current = [report['path'] for report in self._reports.values() if report]
        paths = dict.fromkeys([*self._recent, *self._opened, *current])
        self._choices = {'': None, **{f'{index}. {path.parent.name}': path for index, path in enumerate(paths, 1)}}
        labels = {path: label for label, path in self._choices.items()}
        for side, report in self._reports.items():
            tag = _tag('history_' + side)
            dpg.configure_item(tag, items=list(self._choices))
            dpg.set_value(tag, labels.get(report['path'], '') if report else '')

    @staticmethod
    def _table(parent, columns, rows):
        with dpg.table(parent=parent, header_row=True, borders_innerH=True, clipper=True,
                       policy=dpg.mvTable_SizingStretchProp):
            for column in columns:
                dpg.add_table_column(label=column)
            for row in rows:
                with dpg.table_row():
                    for value in row:
                        dpg.add_text(value)

    @staticmethod
    def _comparison_loss(parent, left, right):
        curves = []
        for side, label, report in (('left', 'A', left), ('right', 'B', right)):
            for split, title in (('train', '训练'), ('val', '验证')):
                points = []
                for point in report.get('history', []):
                    epoch, loss = point.get('epoch'), point.get(split + '_loss')
                    if type(epoch) is int and 1 <= epoch <= 1000 and _number(loss) != '—':
                        points.append((epoch, loss))
                if points:
                    curves.append((side + '_' + split, f'{label} · {title}', sorted(points)))
        if not curves:
            return
        with dpg.collapsing_header(label='A / B 损失曲线', tag=_tag('loss_comparison'), parent=parent):
            with dpg.plot(width=-1, height=192, no_title=True):
                dpg.add_plot_legend(horizontal=True)
                x_axis = dpg.add_plot_axis(dpg.mvXAxis, label='训练周期')
                with dpg.plot_axis(dpg.mvYAxis, label='损失') as y_axis:
                    for key, label, points in curves:
                        epochs, losses = zip(*points)
                        add_series = dpg.add_line_series if len(points) > 1 else dpg.add_scatter_series
                        add_series(epochs, losses, label=label, tag=_tag('loss_' + key))
                dpg.fit_axis_data(x_axis)
                dpg.fit_axis_data(y_axis)

    def _open_directory(self, directory, **_):
        if not directory.is_dir():
            raise ValueError('运行记录所在的目录已移动或删除。')
        if not hasattr(os, 'startfile'):
            raise ValueError('当前系统无法通过此按钮打开目录。')
        os.startfile(str(directory.resolve()), 'explore')

    def _show_inference(self, path, **_):
        self.app.inference_panel.show_report(path)

    def _show_predictions(self, path, **_):
        if path.stat().st_size > 8 * 1024 ** 2:
            raise ValueError('预测清单超过预览大小范围。')
        result = {'profile': 'segmentation-v1', 'artifacts': {'predictions': path}}
        self.app.segmentation_panel.sync(result, True, True)
        dpg.set_value('native_bottom_tabs', dpg.get_alias_id('native_segmentation_tab'))

    def _report_actions(self, parent, side, report):
        artifacts = report.get('artifacts') or {}
        with dpg.group(horizontal=True, parent=parent):
            self._button('打开所在目录', 'open_directory_' + side, self._open_directory, report['path'].parent)
            if report.get('missing_outputs') or report.get('changed_outputs'):
                return
            if report['operation'] == 'predict' and artifacts.get('inference'):
                self._button('在推理页查看', 'view_output_' + side, self._show_inference, artifacts['inference'])
            if report['operation'] == 'train' and report.get('profile') == 'segmentation-v1' and artifacts.get('predictions'):
                self._button('查看测试集分割预测', 'view_output_' + side, self._show_predictions, artifacts['predictions'])

    def _render_reports(self):
        parent = _tag('results')
        dpg.delete_item(parent, children_only=True)
        left, right = self._reports.values()
        message = ('请选择一份运行记录。' if not (left or right) else '可打开另一份记录以对照评估结果。')
        if left and right:
            from studio_core.project_runner import compare_project_runs

            try:
                comparison = compare_project_runs(left, right)
            except (TypeError, ValueError, OverflowError):
                comparison = {'comparable': False, 'reason': '测试指标格式无效。'}
            if comparison['comparable']:
                message = '评估协议与数据一致。差值按 B − A 计算。'
                self._table(parent, ('指标', '记录 A', '记录 B', '差值'), [
                    (key, _number(values['left']), _number(values['right']), _number(values['delta'], signed=True))
                    for key, values in islice(comparison['metrics'].items(), _METRIC_LIMIT)])
                self._comparison_loss(parent, left, right)
            else:
                message = '不可直接对照。' + str(comparison['reason'] or '')
        dpg.set_value(_tag('compare_status'), '\n'.join(self._report_errors.values()) or message)
        for side, label, report in (('left', 'A', left), ('right', 'B', right)):
            if report is None:
                continue
            history = report.get('history') or []
            operation = {'train': '训练', 'predict': '推理'}.get(report['operation'], report['operation'])
            status = _RUN_LABELS.get(report.get('status'), '')
            dpg.add_text(f"记录 {label} · {operation} · {status} · {report.get('device') or '—'}", parent=parent)
            model_id = (report.get('model_digest') or '—')[:12]
            dataset_id = (report.get('dataset_digest') or '—')[:12]
            dpg.add_text(f"模型 {model_id} · 数据 {dataset_id} · 随机种子 {_number(report.get('seed'))} · {len(history)} 个周期",
                         color=_MUTED, parent=parent)
            dpg.add_text(str(report['path']), color=_MUTED, parent=parent)
            checks = (('missing_artifacts', '缺少产物'), ('changed_artifacts', '产物已变更'),
                      ('missing_outputs', '缺少掩码或预览'), ('changed_outputs', '掩码或预览已变更'))
            issues = [f'{label} {len(report[key])} 项' for key, label in checks if report.get(key)]
            if issues:
                dpg.add_text(' · '.join(issues), tag=_tag('integrity_' + side), color=_MUTED, parent=parent)
            self._report_actions(parent, side, report)
            if report.get('error'):
                dpg.add_text(str(report['error']), parent=parent)
            rows = [(name, _number(value))
                    for name, value in islice(_metric_values(report.get('metrics') or {}), _METRIC_LIMIT)]
            if rows:
                self._table(parent, ('指标', '结果'), rows)
            if history:
                with dpg.collapsing_header(label=f'记录 {label} 的训练周期', parent=parent) as header:
                    self._table(header, ('周期', '训练损失', '验证损失'), [
                        (_number(point.get('epoch')), _number(point.get('train_loss')), _number(point.get('val_loss')))
                        for point in history])
            if report.get('artifacts'):
                with dpg.collapsing_header(label=f'记录 {label} 的产物文件', parent=parent):
                    for key, path in report['artifacts'].items():
                        dpg.add_text(f'{key}  {path}', color=_MUTED)

    def poll(self):
        devices = tuple(self.app.available_devices)
        if devices != self._devices:
            dpg.configure_item(_tag('device'), items=list(devices))
            if self._value('device') not in devices:
                dpg.set_value(_tag('device'), 'cpu')
            self._devices = devices
        for message in self.queries.poll():
            name, value, error = message['name'], message['value'], message['error']
            if name == self._operation:
                self._operation = None
                if error is None:
                    try:
                        self._finish_operation(name, value)
                    except (OSError, ValueError) as exc:
                        self._error(exc)
                else:
                    self._error(error)
            elif name == 'history':
                if error is None:
                    self._recent = value
                    self._sync_history()
                else:
                    self._error('无法读取本地运行记录。' + error, field='compare_status')
            elif name in self._reports:
                self._reports[name] = value
                if error is None:
                    path = value['path']
                    if path in self._opened:
                        self._opened.remove(path)
                    self._opened = [path, *self._opened[:_HISTORY_LIMIT - 1]]
                else:
                    self._report_errors[name] = '无法打开运行记录。' + error
                    self.app.log(error, '错误')
                self._sync_history()
                self._render_reports()
        self._poll_run()
        self._sync_actions()

    def close(self):
        self.queries.close()
