"""Native controls for graph diagnostics and bounded probe measurements."""
import json

import dearpygui.dearpygui as dpg

from studio_core.debug_spec import DATA_KINDS, MAX_PROBES, compare_debug_results


MUTED = (145, 154, 164)
ORANGE = (235, 190, 110)
MINT = (93, 216, 192)
TERMINAL = {'succeeded', 'failed', 'cancelled'}


def _tag(name):
    return 'native_debug_' + name


def _number(number):
    return '—' if number is None else f'{number:.6g}'


def _sample_line(sample):
    if sample is None:
        return '未采集'
    if not sample['sample_size']:
        return f"{sample['dtype']} {sample['shape']}  采样起点超出张量范围"
    start = sample['offset']
    end = start + sample['sample_size']
    suffix = f'（索引 {start}–{end - 1}）' if sample['truncated'] else ''
    preview = ', '.join(_number(value) for value in sample['values'][:12])
    return (f"{sample['dtype']} {sample['shape']}{suffix}  均值 {_number(sample['mean'])}  "
            f"最小 {_number(sample['min'])}  最大 {_number(sample['max'])}  样例 {preview}")


class DebugPanel:
    def __init__(self, app):
        self.app = app
        self.sources = []
        self.devices = ()
        self.record = None
        self.result = None
        self.epoch = None
        self.result_revision = None
        self.result_stale = False
        self.baseline = None

    def _button(self, label, name, action, **options):
        dpg.add_button(label=label, tag=_tag(name), callback=self.app.guarded(action), **options)

    def build_controls(self):
        dpg.add_combo(label='调试设备', tag=_tag('device'), items=['cpu'], default_value='cpu', width=128)
        dpg.add_combo(label='样本来源', tag=_tag('data'), items=['合成样本', '实验数据', '图像训练集'],
                      default_value='合成样本', width=141, callback=self.app.guarded(self._data_changed))
        dpg.add_input_int(label='样本数', tag=_tag('batch'), default_value=2, min_value=1,
                          max_value=64, min_clamped=True, max_clamped=True, width=94)
        with dpg.group(tag=_tag('data_options'), show=False):
            dpg.add_combo(['train', 'val', 'test'], default_value='train', label='数据划分', tag=_tag('split'),
                          width=124, callback=self.app.guarded(self._data_changed))
            with dpg.group(horizontal=True):
                dpg.add_input_int(tag=_tag('sample_index'), default_value=1, min_value=1,
                                  min_clamped=True, width=78, step=0)
                self._button('添加第 N 个样本', 'add_sample', self._add_sample)
            dpg.add_input_text(tag=_tag('sample_ids'), hint='样本 ID，逗号分隔', width=-1)
        with dpg.group(horizontal=True):
            self._button('检查结构', 'structure', self.app._validate)
            self._button('样例前向', 'forward', self.app_action('forward'))
            self._button('训练一步', 'step', self.app_action('step'))
        dpg.add_text('点击画布上的输出端口挂探针', color=MUTED, wrap=245)
        dpg.add_combo(tag=_tag('pins'), items=[], width=184)
        with dpg.group(horizontal=True):
            self._button('定位探针', 'locate', self.locate)
            self._button('移除探针', 'remove', self.remove)
        with dpg.collapsing_header(label='探针采样'):
            dpg.add_input_int(label='展平起点', tag=_tag('offset'), default_value=0, min_value=0,
                              min_clamped=True, max_value=2**31 - 1, max_clamped=True, width=112, step=0)
            dpg.add_text('按逻辑顺序取起点后的最多 128 个值。', color=MUTED, wrap=242)
        with dpg.collapsing_header(label='参数扫描'):
            self._button('从当前组件选择参数', 'select_sweep', self.select_sweep, width=-1)
            dpg.add_text('请选择主图上的组件', tag=_tag('sweep_instance'), wrap=242, color=MUTED)
            dpg.add_combo(label='参数', tag=_tag('parameter'), items=[], width=152)
            dpg.add_input_text(label='取值', tag=_tag('values'), hint='0.25, 0.5, 0.75', width=160)
            self._button('比较取值', 'sweep', self.app_action('sweep'))
        self._button('停止调试', 'cancel', self.cancel, enabled=False)
        dpg.add_text('', tag=_tag('status'), color=MUTED, wrap=245)

    def build_results(self):
        with dpg.tab(label='调试测量', tag=_tag('results_tab')):
            with dpg.group(horizontal=True):
                dpg.add_text('等待样例前向、训练一步或参数扫描', tag=_tag('summary'), color=MUTED)
                dpg.add_combo(tag=_tag('result_port'), items=[], width=232,
                              callback=self.app.guarded(self.render_result))
            with dpg.group(horizontal=True):
                self._button('设为对照 A', 'baseline', self._pin_baseline, enabled=False)
                self._button('清除对照', 'clear_baseline', self._clear_baseline, enabled=False)
                dpg.add_text('', tag=_tag('comparison'), color=MUTED, wrap=560)
            with dpg.child_window(width=-1, height=-1, border=False, horizontal_scrollbar=True):
                dpg.add_group(tag=_tag('result_content'))

    def app_action(self, mode):
        def run(**_):
            self.start(mode)
        return run

    def show(self):
        dpg.set_value('native_right_tabs', dpg.get_alias_id(_tag('controls_tab')))
        dpg.set_value('native_bottom_tabs', dpg.get_alias_id(_tag('results_tab')))

    def add_probe(self, ref, **_):
        self._sync_epoch()
        source = self.app.document.probe_source(ref)
        if source is None:
            raise ValueError('打开芯片的具体调用后才能挂内部探针')
        if source not in self.sources:
            if len(self.sources) == MAX_PROBES:
                raise ValueError(f'当前已有 {MAX_PROBES} 个探针，请先移除一个')
            self.sources.append(source)
        self._set_pins(source)
        dpg.set_value(_tag('status'), '已选探针 · ' + source)
        if self.result is not None and source in self.result['probes']:
            dpg.set_value(_tag('result_port'), source)
            self.render_result()
        self.show()

    def _set_pins(self, selected=''):
        dpg.configure_item(_tag('pins'), items=self.sources)
        dpg.set_value(_tag('pins'), selected or (self.sources[0] if self.sources else ''))

    def locate(self, **_):
        source = dpg.get_value(_tag('pins'))
        if source:
            if not self.app._focus_probe_source(source):
                raise ValueError('该端口已在设计中移除')
            self.show()

    def remove(self, **_):
        source = dpg.get_value(_tag('pins'))
        if source in self.sources:
            self.sources.remove(source)
            self._set_pins()

    def _data_changed(self, **_):
        dpg.configure_item(_tag('data_options'), show=dpg.get_value(_tag('data')) == '实验数据')
        dpg.set_value(_tag('sample_ids'), '')

    def _sample_ids(self):
        raw = dpg.get_value(_tag('sample_ids')).strip()
        if raw.startswith('['):
            try:
                return json.loads(raw)
            except ValueError as error:
                raise ValueError('样本 ID 的 JSON 列表格式有误') from error
        return [item.strip() for item in raw.split(',') if item.strip()]

    def _checked_manifest(self, config):
        from .workspace_state import data_identity

        if (self.app.experiment_mode not in DATA_KINDS or self.app.manifest is None
                or self.app.inspection_key != data_identity(config)):
            raise ValueError('请先在实验页检查数值 CSV、回归 CSV 或 ImageFolder 数据')
        return self.app.manifest

    def _add_sample(self, **_):
        manifest = self._checked_manifest(self.app._configuration())
        split = dpg.get_value(_tag('split'))
        rows = [item for item in manifest['records'] if item['split'] == split]
        index = dpg.get_value(_tag('sample_index')) - 1
        if not 0 <= index < len(rows):
            raise ValueError(f'{split} 共有 {len(rows)} 个样本')
        record = rows[index]
        selected = self._sample_ids()
        if record['id'] in selected:
            return
        if len(selected) >= 64:
            raise ValueError('一次最多选择 64 个样本')
        selected.append(record['id'])
        dpg.set_value(_tag('sample_ids'), json.dumps(selected, ensure_ascii=False))
        dpg.set_value(_tag('batch'), max(len(selected), dpg.get_value(_tag('batch'))))
        source = record.get('relative_path', f"CSV 第 {record.get('row_number', index + 2)} 行")
        dpg.set_value(_tag('status'), f'已添加 {split} · {source}')

    def _pin_baseline(self, **_):
        if self.result and self.result['mode'] in ('forward', 'step'):
            self.baseline = self.result
            self.render_result()

    def _clear_baseline(self, **_):
        self.baseline = None
        self.render_result()

    def select_sweep(self, **_):
        instance = self.app.document.selected_instance
        if instance is None or self.app.document.scope_id is not None:
            raise ValueError('请先在主图选中组件，芯片可扫描其公开参数')
        from studio_core.debug_spec import numeric_parameters
        parameters = list(numeric_parameters(self.app.document.graph, instance))
        if not parameters:
            raise ValueError('所选组件没有可扫描的数值参数')
        dpg.set_value(_tag('sweep_instance'), instance['id'])
        dpg.configure_item(_tag('parameter'), items=parameters)
        dpg.set_value(_tag('parameter'), parameters[0])

    def start(self, mode):
        if self.record is not None and self.record['status'] not in TERMINAL:
            raise ValueError('当前调试还在运行')
        self._sync_epoch()
        app = self.app
        current = app._configuration()
        config = {'mode': mode, 'device': dpg.get_value(_tag('device')),
                  'batch_size': dpg.get_value(_tag('batch')), 'seed': current['seed'],
                  'lr': current['lr'], 'optimizer': current.get('optimizer', 'sgd'), 'sources': list(self.sources),
                  'probe_offset': dpg.get_value(_tag('offset'))}
        data_choice = dpg.get_value(_tag('data'))
        if data_choice == '实验数据':
            manifest = self._checked_manifest(current)
            config.update(experiment={**current, 'expected_dataset_digest': manifest['dataset_digest']},
                          split=dpg.get_value(_tag('split')), sample_ids=self._sample_ids())
        elif data_choice == '图像训练集':
            if app.experiment_mode != 'image_folder':
                raise ValueError('请在实验页选择图像分类并配置 ImageFolder 训练集')
            config['experiment'] = current
        if mode == 'sweep':
            try:
                values = json.loads('[' + dpg.get_value(_tag('values')) + ']')
            except ValueError as error:
                raise ValueError('用英文逗号分隔扫描取值，例如 0.25, 0.5, 0.75') from error
            config['sweep'] = {'instance': dpg.get_value(_tag('sweep_instance')),
                               'parameter': dpg.get_value(_tag('parameter')), 'values': values}
        self.record = app.manager.submit_debug(app.document.graph, config)
        self.result = None
        dpg.set_value(_tag('status'), '正在执行 · ' + self.record['run_id'][:8])
        self._busy(True)
        self.render_result()
        self.show()

    def _busy(self, active):
        for name in ('forward', 'step', 'sweep'):
            dpg.configure_item(_tag(name), enabled=not active)
        dpg.configure_item(_tag('cancel'), enabled=active)

    def cancel(self, **_):
        if self.record and self.record['status'] not in TERMINAL:
            self.record = self.app.manager.cancel(self.record['run_id'])
            dpg.set_value(_tag('status'), '正在停止调试')

    def _sync_epoch(self):
        epoch = self.app.canvas_host.document_epoch
        if self.epoch is not None and self.epoch != epoch:
            self.sources.clear()
            self._set_pins()
            dpg.set_value(_tag('sweep_instance'), '请选择主图上的组件')
            dpg.configure_item(_tag('parameter'), items=[])
            if self.result is not None:
                self.result_stale = self.result['model_digest'] != self.app.document.semantic_key()
                self.render_result()
        self.epoch = epoch

    def poll(self):
        self._sync_epoch()
        devices = tuple(self.app.available_devices)
        if devices != self.devices:
            self.devices = devices
            dpg.configure_item(_tag('device'), items=list(devices))
            if dpg.get_value(_tag('device')) not in devices:
                dpg.set_value(_tag('device'), 'cpu')
        if self.record and self.record['status'] not in TERMINAL:
            self.record = self.app.manager.get(self.record['run_id'])
            if self.record['status'] in TERMINAL:
                self._busy(False)
                self.result = self.record['result']
                self.result_revision = self.app.document.graph['revision']
                self.result_stale = bool(self.result and self.result['model_digest'] != self.app.document.semantic_key())
                sources = list(self.result['probes']) if self.result else []
                dpg.configure_item(_tag('result_port'), items=sources)
                dpg.set_value(_tag('result_port'), sources[0] if sources else '')
                dpg.set_value(_tag('status'), self.record.get('error') or ('调试完成' if self.result else '调试已停止'))
                self.render_result()
        if self.result and self.result_revision != self.app.document.graph['revision']:
            self.result_revision = self.app.document.graph['revision']
            self.result_stale = self.result['model_digest'] != self.app.document.semantic_key()
            self.render_result()

    def render_result(self, **_):
        parent = _tag('result_content')
        dpg.delete_item(parent, children_only=True)
        dpg.configure_item(_tag('baseline'), enabled=bool(self.result and self.result['mode'] in ('forward', 'step')))
        dpg.configure_item(_tag('clear_baseline'), enabled=self.baseline is not None)
        dpg.set_value(_tag('comparison'), '已保留 A。调整模型后运行同一批输入。' if self.baseline else '')
        if self.result is None:
            dpg.set_value(_tag('summary'), '调试运行中' if self.record and self.record['status'] not in TERMINAL
                          else '等待样例前向、训练一步或参数扫描')
            if self.record and self.record.get('error'):
                dpg.add_text(self.record['error'], color=ORANGE, wrap=680, parent=parent)
            return
        result = self.result
        kind = {'image_folder': '图片', 'csv': '数值 CSV', 'regression_csv': '回归 CSV'}.get(
            result['batch_kind'], '合成样本')
        if result.get('split'):
            kind += ' · ' + result['split']
        dpg.set_value(_tag('summary'), f"{kind} ×{result['sample_count']}  {result['device']}  "
                      f"更新 {result['optimizer_steps']} 次  总耗时 {result['elapsed_ms']:.0f} ms")
        if self.result_stale:
            dpg.add_text('设计已修改 · 以下测量对应运行时的快照', color=ORANGE, parent=parent)
        if result['loss'] is not None:
            dpg.add_text('本批损失 ' + _number(result['loss']), color=MINT, parent=parent)
        if result.get('sample_ids'):
            with dpg.tree_node(label='本批样本 ID', parent=parent):
                dpg.add_text('\n'.join(result['sample_ids']), wrap=640)
        if self.baseline is not None and self.baseline is not result:
            comparison = compare_debug_results(self.baseline, result)
            if comparison['comparable']:
                summary = 'A 与 B 使用同一批输入'
                if comparison['loss_delta'] is not None:
                    summary += ' · B − A 损失 ' + _number(comparison['loss_delta'])
                dpg.set_value(_tag('comparison'), summary)
                with dpg.tree_node(label='对照 A 的输出采样', parent=parent, default_open=True):
                    for name, sample in self.baseline['outputs'].items():
                        dpg.add_text(name + ' · ' + _sample_line(sample))
                common_pin = dpg.get_value(_tag('result_port'))
                if common_pin in self.baseline['probes']:
                    measurement = self.baseline['probes'][common_pin]
                    dpg.add_text('A 前向 · ' + _sample_line(measurement['activation']), parent=parent)
                    if result['optimizer_steps']:
                        dpg.add_text('A 梯度 · ' + _sample_line(measurement['gradient']), parent=parent)
            else:
                dpg.set_value(_tag('comparison'), comparison['reason'])
        if result['sweep']:
            with dpg.table(parent=parent, header_row=True, borders_innerV=True, row_background=True):
                for label in ('参数取值', '输出端口', '均值', 'L2 范数', '分类损失', '采样范围'):
                    dpg.add_table_column(label=label)
                for row in result['sweep']:
                    for port, sample in row['outputs'].items():
                        with dpg.table_row():
                            for value in (_number(row['value']), port, _number(sample['mean']), _number(sample['l2']),
                                          _number(row['loss']), f"{sample['sample_size']}/{sample['count']}"):
                                dpg.add_text(value)
            return
        port = dpg.get_value(_tag('result_port'))
        probe = result['probes'].get(port)
        if probe:
            title = port + ('  ·  参数实例 ' + probe['instance'] if probe['instance'] else '')
            if probe['axes']:
                title += '  ·  轴 ' + ', '.join(probe['axes'])
            dpg.add_text(title, color=MINT, parent=parent)
            dpg.add_text('前向 · ' + _sample_line(probe['activation']), parent=parent)
            if result['optimizer_steps']:
                dpg.add_text('反向 · ' + _sample_line(probe['gradient']), parent=parent)
        else:
            for name, sample in result['outputs'].items():
                dpg.add_text(name + ' · ' + _sample_line(sample), parent=parent)
        if result['parameters']:
            with dpg.tree_node(label='参数更新明细', parent=parent, default_open=True):
                for row in result['parameters']:
                    before, after = row['before'], row['after']
                    label = row['parameter'] + (' · 冻结' if not row['trainable'] else '')
                    with dpg.tree_node(label=label + '  ·  ΔW ' + _number(row['update_norm']) +
                                       ('（采样）' if before['truncated'] else '')):
                        dpg.add_text('更新前 · ' + _sample_line(before))
                        dpg.add_text('梯度 · ' + _sample_line(row['gradient']))
                        dpg.add_text('更新后 · ' + _sample_line(after))
            if result.get('parameters_omitted'):
                dpg.add_text(f"已优先采样探针实例，其余 {result['parameters_omitted']} 个参数张量未采集", color=MUTED, parent=parent)
