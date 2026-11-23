"""Native S12 controls. SSH and file validation run in background queries."""
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic
from uuid import uuid4

import dearpygui.dearpygui as dpg

from studio_core.remote_client import RemoteClient, RemoteProfile, batch_plan, check_job_id
from .experiment_panel import BackgroundQueries
from .persistence import replace_with_retry

PREFIX = 'native_remote_'
BLUE = (100, 200, 255)
MUTED = (145, 154, 164)
AMBER = (235, 190, 110)
TERMINAL = {'succeeded', 'failed', 'cancelled'}
LABELS = {'queued': '排队', 'preparing': '正在核对数据', 'running': '正在训练',
          'cancel_requested': '正在停止', 'cancelling': '正在停止', 'succeeded': '训练完成',
          'failed': '训练失败', 'cancelled': '已停止'}


def tag(name):
    return PREFIX + name


class RemotePanel:
    def __init__(self, app):
        self.app = app
        self.queries = BackgroundQueries()
        self.client = None
        self.inventory = None
        self.operation = None
        self.record = None
        self.restore_settings = False
        self.job_id = ''
        self.next_poll = 0.0
        self.settings_file = app.data_directory / 'remote-execution.json'

    def _button(self, label, name, action, **options):
        return dpg.add_button(label=label, tag=tag(name), callback=self.app.guarded(action), **options)

    def show(self, **_):
        dpg.set_value('native_right_tabs', dpg.get_alias_id(tag('tab')))

    def build_controls(self):
        with dpg.collapsing_header(label='服务器连接', tag=tag('connection_settings'), default_open=True):
            for name, label, hint in [('host', 'SSH 别名', 'research-server'),
                                      ('python', 'Python 路径', '/path/to/venv/bin/python'),
                                      ('workspace', '工作目录', '/path/to/workspace/jobs-home')]:
                dpg.add_input_text(label=label, tag=tag(name), hint=hint, width=150,
                                   callback=self.app.guarded(self._settings_edited))
            self._button('连接并读取显卡', 'connect', self.connect, width=-1)
        dpg.add_text('', tag=tag('connection'), color=BLUE, wrap=248)
        self._button('查看进度与日志', 'show_result', self.show_log, width=-1)
        dpg.add_separator()
        dpg.add_text('选择显卡 · Linux CUDA DDP', color=BLUE)
        dpg.add_group(tag=tag('gpus'))
        with dpg.group(horizontal=True):
            dpg.add_input_int(label='每卡批次', tag=tag('batch'), default_value=2, width=52, step=0,
                              min_value=1, max_value=64, min_clamped=True, max_clamped=True,
                              callback=self.app.guarded(self.refresh_batch))
            dpg.add_input_int(label='累积次数', tag=tag('accumulation'), default_value=1, width=52, step=0,
                              min_value=1, max_value=32, min_clamped=True, max_clamped=True,
                              callback=self.app.guarded(self.refresh_batch))
        dpg.add_text('请选择至少两张显卡', tag=tag('batch_plan'), color=MUTED, wrap=245)
        dpg.add_input_text(tag=tag('project'), hint='本地科研工程 project.json', width=-1)
        with dpg.group(horizontal=True):
            self._button('选择工程…', 'choose_project', self.choose_project)
            self._button('用已打开工程', 'use_project', self.use_project)
        dpg.add_input_text(tag=tag('data'), hint='服务器已有的 ImageFolder 目录', width=-1)
        with dpg.group(horizontal=True):
            dpg.add_input_int(label='目标轮数', tag=tag('epochs'), default_value=10, width=67, step=0,
                              min_value=1, max_value=1000, min_clamped=True, max_clamped=True)
            self._button('启动多卡训练', 'submit', self.start, enabled=False)
        dpg.add_separator()
        dpg.add_input_text(tag=tag('job'), label='任务 ID', hint='提交后自动填入', width=155)
        self._button('按任务 ID 重新连接', 'status', self.attach, width=-1, enabled=False)
        self._load()

    def build_results(self):
        with dpg.tab(label='远程作业', tag=tag('results_tab')):
            with dpg.group(horizontal=True):
                self._button('停止', 'cancel', self.cancel, enabled=False)
                self._button('从检查点续训', 'resume', self.resume, enabled=False)
                self._button('下载并打开结果', 'fetch', self.fetch, enabled=False)
                dpg.add_text('等待连接服务器', tag=tag('summary'), color=MUTED)
            dpg.add_progress_bar(tag=tag('progress'), width=-1, overlay='等待提交')
            with dpg.child_window(width=-1, height=-1, border=False, horizontal_scrollbar=True):
                dpg.add_text('', tag=tag('log'), color=MUTED)

    def build_resources(self):
        with dpg.file_dialog(tag=tag('picker'), label='选择科研工程 project.json', width=730, height=465,
                             show=False, modal=True, default_path=str(self.app.data_directory),
                             callback=self.app.guarded(self._picked)):
            dpg.add_file_extension('.json')

    def choose_project(self, **_):
        self.app._bind_modal(tag('picker'))
        dpg.show_item(tag('picker'))

    def _picked(self, app_data=None, **_):
        value = (app_data or {}).get('file_path_name')
        if value:
            dpg.set_value(tag('project'), value)
        dpg.hide_item(tag('picker'))

    def use_project(self, **_):
        project = self.app.project_panel.project
        if not project:
            raise ValueError('请在工程页先打开一份科研工程，或点击「选择工程…」')
        dpg.set_value(tag('project'), str(project['path']))

    def _value(self, name):
        return dpg.get_value(tag(name)).strip()

    def _load(self):
        try:
            if self.settings_file.stat().st_size > 64 * 1024:
                return
            state = json.loads(self.settings_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return
        if not isinstance(state, dict):
            return
        for field in ('host', 'python', 'workspace', 'project', 'data', 'job'):
            if isinstance(state.get(field), str):
                dpg.set_value(tag(field), state[field])
        for field, maximum in [('batch', 64), ('accumulation', 32), ('epochs', 1000)]:
            if type(state.get(field)) is int and 1 <= state[field] <= maximum:
                dpg.set_value(tag(field), state[field])

    def _save(self):
        if self.client is None:
            return
        state = asdict(self.client.profile)
        state.update({name: self._value(name) for name in ('project', 'data')})
        state.update({name: dpg.get_value(tag(name)) for name in ('batch', 'accumulation', 'epochs')})
        state['job'] = self.job_id
        temporary = self.settings_file.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        replace_with_retry(temporary, self.settings_file)

    def _runtime_source(self):
        frozen = self.app.project_runtime_source()
        if frozen is not None:
            return frozen
        from studio_core import remote_client
        return Path(remote_client.__file__).parent

    def _submit(self, name, task):
        if self.operation is not None:
            raise ValueError('正在执行上一项服务器操作')
        self.queries.submit(name, task)
        self.operation = name
        self._refresh_buttons()

    def connect(self, **_):
        profile = RemoteProfile(self._value('host'), self._value('python'), self._value('workspace'))
        client = RemoteClient(profile, self._runtime_source())
        self._submit('connect', lambda cancelled: (client, client.connect(cancelled=cancelled)))
        dpg.set_value(tag('connection'), '正在连接并核对工作进程…')

    def _selected_gpus(self):
        if self.inventory is None:
            return []
        return [gpu['uuid'] for gpu in self.inventory['gpus'] if dpg.get_value(tag('gpu_' + gpu['uuid']))]

    def _draw_gpus(self):
        dpg.delete_item(tag('gpus'), children_only=True)
        for gpu in self.inventory['gpus']:
            description = f"{gpu['index']} · {gpu['name']}"
            with dpg.group(parent=tag('gpus')):
                dpg.add_checkbox(label=description, tag=tag('gpu_' + gpu['uuid']),
                                 callback=self.app.guarded(self.refresh_batch))
                dpg.add_text(f"显存已用 {gpu['memory_used_mib']} / {gpu['memory_total_mib']} MiB · GPU {gpu['utilization']}%",
                             color=MUTED, wrap=245)
        self.refresh_batch()

    def refresh_batch(self, **_):
        gpus = self._selected_gpus()
        try:
            plan = batch_plan(gpus, dpg.get_value(tag('batch')), dpg.get_value(tag('accumulation')))
            summary = f"{len(gpus)} 张卡 · 每次读取 {plan['per_step']} 样本 · 每次更新 {plan['effective']} 样本"
        except ValueError:
            summary = '请选择至少两张不同显卡'
        dpg.set_value(tag('batch_plan'), summary)
        self._refresh_buttons()

    def _refresh_buttons(self):
        if not dpg.does_item_exist(tag('connect')):
            return
        ready = self._matches_connection()
        idle = self.operation is None
        can_train = ready and idle and self.inventory['ddp'] and len(self._selected_gpus()) >= 2
        status = self.record.get('status') if self.record else None
        active = status is not None and status not in TERMINAL
        for name, enabled in [('connect', idle), ('status', ready and idle), ('submit', can_train and not active),
                              ('cancel', ready and idle and active and status not in {'cancel_requested', 'cancelling'}),
                              ('resume', can_train and self.record is not None and status in TERMINAL and self.record.get('resume_available')),
                              ('fetch', ready and idle and status == 'succeeded')]:
            dpg.configure_item(tag(name), enabled=bool(enabled))

    def _matches_connection(self):
        if self.client is None:
            return False
        try:
            return RemoteProfile(self._value('host'), self._value('python'), self._value('workspace')) == self.client.profile
        except ValueError:
            return False

    def _settings_edited(self, **_):
        if self.client is not None and not self._matches_connection():
            dpg.set_value(tag('connection'), '连接参数已修改，请连接后操作')
        self._refresh_buttons()

    def _connect_required(self):
        if not self._matches_connection():
            raise ValueError('请先连接当前填写的服务器')
        return self.client

    def _start(self, resume):
        client = self._connect_required()
        if not self.inventory['ddp']:
            raise ValueError('服务器 Python 环境未提供 CUDA 与 NCCL DDP')
        if self.record and self.record['status'] not in TERMINAL:
            raise ValueError('当前任务正在运行')
        project, data, gpus = self._value('project'), self._value('data'), self._selected_gpus()
        batch, accumulation, epochs = [dpg.get_value(tag(key)) for key in ('batch', 'accumulation', 'epochs')]
        batch_plan(gpus, batch, accumulation)
        job = uuid4().hex
        if not project:
            raise ValueError('请选择本地科研工程')
        self.job_id, self.record = job, None
        self.restore_settings = False
        dpg.set_value(tag('job'), job)
        self._save()
        self._submit('submit', lambda cancelled: client.submit(
            project, data, gpus, batch, accumulation, epochs, resume_from=resume, request_id=job, cancelled=cancelled))
        dpg.set_value(tag('summary'), '正在提交任务 ' + job)
        self.show_log()

    def start(self, **_):
        self._start(None)

    def resume(self, **_):
        if not self.record or self.record['status'] not in TERMINAL or not self.record.get('resume_available'):
            raise ValueError('请选择已结束并具有检查点的任务')
        self._start(self.record['job_id'])

    def attach(self, **_):
        client = self._connect_required()
        job = check_job_id(self._value('job'))
        if job != self.job_id:
            self.record = None
        self.job_id = job
        self.restore_settings = True
        self._save()
        self._submit('status', lambda cancelled: client.status(job, cancelled=cancelled))
        self.show_log()

    def cancel(self, **_):
        client = self._connect_required()
        if not self.record or self.record['status'] in TERMINAL:
            raise ValueError('请选择正在运行的远程任务')
        job = check_job_id(self.record['job_id'])
        self._submit('cancel', lambda cancelled: client.cancel(job, cancelled=cancelled))
        dpg.set_value(tag('summary'), '已请求服务器停止任务 ' + job)

    def fetch(self, **_):
        client = self._connect_required()
        if not self.record or self.record['status'] != 'succeeded':
            raise ValueError('请先选择已完成的远程任务')
        job = check_job_id(self.record['job_id'])
        directory = self.app.manager.directory / ('remote-' + job)
        report = directory / 'project-run.json'
        if report.is_file():
            return self._open_report(report)
        self._submit('fetch', lambda cancelled: client.fetch(job, directory, cancelled=cancelled))
        dpg.set_value(tag('summary'), '正在下载并核验运行结果')

    def _open_report(self, path):
        self.app.project_panel.open_run(path)
        self.app.project_panel.show_history()

    def show_log(self, **_):
        dpg.set_value('native_bottom_tabs', dpg.get_alias_id(tag('results_tab')))

    def _present_record(self, record):
        if record.get('job_id') != self.job_id or not isinstance(record.get('progress'), dict):
            raise ValueError('服务器返回了无法匹配的任务状态')
        self.record = record
        if self.restore_settings:
            assigned = set(record.get('gpu_uuids', ()))
            for item in self.inventory['gpus']:
                dpg.set_value(tag('gpu_' + item['uuid']), item['uuid'] in assigned)
            for field, setting in [('batch', 'per_device_batch_size'), ('accumulation', 'accumulation_steps')]:
                if type(record.get(setting)) is int:
                    dpg.set_value(tag(field), record[setting])
            self.restore_settings = False
            self.refresh_batch()
        progress = record['progress']
        epoch, total = progress.get('epoch', 0), progress.get('epochs', 0)
        state = LABELS.get(record['status'], record['status'])
        summary = f"{state} · {self.job_id}"
        if record.get('resume_available'):
            summary += ' · 有可用检查点'
        if record.get('error'):
            summary += ' · ' + str(record['error'])[:500]
        dpg.set_value(tag('summary'), summary)
        dpg.set_value(tag('progress'), min(1, epoch / total) if total > 0 else 0)
        dpg.configure_item(tag('progress'), overlay=f"第 {epoch} / {total} 轮 · 更新 {progress.get('updates', 0)} 次" if total else state)
        lines = []
        for key, label in [('train_loss', '训练损失'), ('val_loss', '验证损失')]:
            if type(progress.get(key)) in (int, float):
                lines.append(f'{label} {progress[key]:.6g}')
        lines.append(str(record.get('log', ''))[-10000:])
        dpg.set_value(tag('log'), '\n'.join(lines).strip())

    def poll(self):
        for message in self.queries.poll():
            name, value, error = message['name'], message['value'], message['error']
            self.operation = None
            self.next_poll = monotonic() + 1.6
            if error:
                dpg.set_value(tag('connection' if name == 'connect' else 'summary'), error)
                self.app.log(error, '错误')
            elif name == 'connect':
                previous = self.client.profile if self.client is not None else None
                self.client, self.inventory = value
                desired = self._value('job') if previous in (None, self.client.profile) else ''
                self.record = None
                self.job_id = desired
                dpg.set_value(tag('job'), desired)
                capability = 'CUDA DDP 可用' if self.inventory['ddp'] else 'CUDA DDP 不可用'
                dpg.set_value(tag('connection'), f'{self.client.profile.host} · {capability}')
                self._draw_gpus()
                dpg.set_value(tag('connection_settings'), False)
                self._save()
                if desired:
                    try:
                        self.attach()
                    except ValueError as exc:
                        dpg.set_value(tag('summary'), str(exc))
            elif name == 'fetch':
                self._open_report(value)
            else:
                self._present_record(value)
            self._refresh_buttons()
        if (self._matches_connection() and self.record and self.record['status'] not in TERMINAL
                and not self.operation and monotonic() >= self.next_poll):
            job, client = self.record['job_id'], self.client
            self._submit('status', lambda cancelled: client.status(job, cancelled=cancelled))

    def close(self):
        self.queries.close()
