"""Edit the CSV fields and the independent graph bindings for S14."""

import json

import dearpygui.dearpygui as dpg

from studio_core.experiment_spec import MULTIMODAL_PORTS
from studio_core.multimodal_data import MODALITIES, multimodal_defaults
from studio_core.table_data import table_defaults

KINDS = ('table_csv', 'multimodal_csv')
TABLE_TASKS = {'分类': 'classification', '回归': 'regression'}
MODALITY_NAMES = {'volume': '三维体数据', 'image': '二维影像', 'tabular': '临床表格'}
IMAGE_SOURCES = {'图片文件': 'raster', '体数据中央切片': 'center_slice', '体数据均值投影': 'mean_projection'}
VOLUME_FORMATS = {'DICOM CT': 'dicom_ct', 'MRI NIfTI': 'mri_nifti'}
_ROLES = {'id_column': '病例 ID 列', 'group_column': '患者 ID 列', 'target_column': '标签或目标列',
          'split_column': '划分列', 'volume_column': '三维影像路径列', 'image_column': '二维影像路径列'}
_AUTO = '（自动分组）'
_NONE = '（不使用）'
_MUTED = (145, 154, 164)


def _tag(field):
    return 'native_cohort_' + field


def _text_columns(raw):
    if raw.lstrip().startswith('['):
        value = json.loads(raw)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError('特征列须为列名或 JSON 字符串列表。')
        return value
    return [part.strip() for part in raw.replace('，', ',').split(',') if part.strip()]


def _label(options, value):
    return next(label for label, key in options.items() if key == value)


class CohortPanel:
    def __init__(self, app):
        self.app = app

    def _combo(self, field, items, label, *, callback=None):
        dpg.add_combo(items, tag=_tag(field), label=label, width=177,
                      default_value=items[0], callback=callback)

    def _feature_field(self, field, label):
        dpg.add_input_text(tag=_tag(field), label=label, width=430, hint='列名用逗号分开')

    def _ints(self, prefix, axes, values, maxima):
        with dpg.group(horizontal=True):
            for axis, value, maximum in zip(axes, values, maxima):
                dpg.add_input_int(tag=_tag(prefix + axis), label=axis, width=113, default_value=value,
                                  min_value=1, max_value=maximum, min_clamped=True, max_clamped=True, step=0)

    def build(self):
        refresh = self.app.guarded(lambda **_: self.layout(self.app.experiment_mode))
        with dpg.group(tag=_tag('settings'), show=False):
            dpg.add_button(label='读取 CSV 列', tag=_tag('read_columns'), callback=self.app.guarded(self.app._read_columns))
            with dpg.group(horizontal=True):
                self._combo('id_column', ['sample_id'], _ROLES['id_column'])
                self._combo('group_column', ['patient_id'], _ROLES['group_column'])
            with dpg.group(horizontal=True):
                self._combo('target_column', ['label'], _ROLES['target_column'])
                self._combo('split_column', [_AUTO], _ROLES['split_column'])
            dpg.add_text('病例 ID 匹配输入，患者 ID 决定划分。同一患者的多次检查使用同一个患者 ID。', color=_MUTED, wrap=730)
            with dpg.group(tag=_tag('table_settings')):
                with dpg.group(horizontal=True):
                    self._combo('task', list(TABLE_TASKS), '模型任务')
                    dpg.add_input_int(tag=_tag('folds'), label='交叉验证折数', default_value=5, width=84,
                                      min_value=2, max_value=20, min_clamped=True, max_clamped=True)
                    dpg.add_input_double(tag=_tag('table_test_fraction'), label='自动留出测试比例', default_value=.2,
                                        width=95, step=0, format='%.3f')
                self._feature_field('numeric_columns', '数值列')
                self._feature_field('categorical_columns', '类别列')
                dpg.add_text('空值由各折训练数据填补。测试集在最终评估时使用。', color=_MUTED, wrap=730)
            with dpg.group(tag=_tag('multimodal_settings')):
                with dpg.group(horizontal=True):
                    for modality in MODALITIES:
                        dpg.add_checkbox(tag=_tag('has_' + modality), label=MODALITY_NAMES[modality], callback=refresh)
                with dpg.group(horizontal=True):
                    for modality in MODALITIES:
                        dpg.add_input_text(tag=_tag('port_' + modality), label=MODALITY_NAMES[modality] + '端口',
                                           width=107, default_value=MULTIMODAL_PORTS[modality])
                dpg.add_text('勾选病例拥有的模态。将某个输入端口留空，可以在同一划分下训练其他分支。', color=_MUTED, wrap=730)
                dpg.add_input_text(tag=_tag('root'), label='影像根目录', width=430, hint='默认 CSV 所在目录')
                with dpg.group(horizontal=True):
                    for field, label in [('validation_fraction', '自动验证比例'), ('test_fraction', '自动测试比例')]:
                        dpg.add_input_double(tag=_tag(field), label=label, default_value=.2, width=105, step=0, format='%.3g')
                with dpg.group(tag=_tag('image_settings')):
                    with dpg.group(horizontal=True):
                        self._combo('image_source', list(IMAGE_SOURCES), '二维影像来源', callback=refresh)
                        self._combo('image_column', ['image_path'], _ROLES['image_column'])
                    self._ints('image_size_', 'HW', (32, 32), (512, 512))
                    self._combo('image_channels', ['1', '3'], '二维通道数')
                with dpg.group(tag=_tag('volume_settings')):
                    with dpg.group(horizontal=True):
                        self._combo('volume_format', list(VOLUME_FORMATS), '体数据格式', callback=refresh)
                        self._combo('volume_column', ['volume_path'], _ROLES['volume_column'])
                    self._ints('volume_size_', 'DHW', (16, 32, 32), (128, 512, 512))
                    with dpg.group(horizontal=True):
                        for axis in 'DHW':
                            dpg.add_input_double(tag=_tag('spacing_' + axis), label=axis + ' 间距 mm',
                                                width=108, default_value=1., step=0, format='%.5g')
                    with dpg.group(tag=_tag('ct'), horizontal=True):
                        dpg.add_input_double(tag=_tag('lower_hu'), label='CT 下限 HU', default_value=-1000, width=124, step=0)
                        dpg.add_input_double(tag=_tag('upper_hu'), label='CT 上限 HU', default_value=400, width=124, step=0)
                    with dpg.group(tag=_tag('mri')):
                        with dpg.group(horizontal=True):
                            for channel in (2, 3, 4):
                                self._combo('sequence_' + str(channel), [_NONE], f'MRI 通道 {channel}')
                        dpg.add_checkbox(tag=_tag('registered'), label='附加序列已配准到参考影像')
                self._feature_field('feature_columns', '临床数值列')
            dpg.add_checkbox(tag=_tag('standardize'), label='标准化数值特征', default_value=True)

    def layout(self, kind):
        table, multi = kind == 'table_csv', kind == 'multimodal_csv'
        dpg.configure_item(_tag('settings'), show=table or multi)
        if not multi:
            return
        has = lambda key: dpg.get_value(_tag('has_' + key))
        derived = dpg.get_value(_tag('image_source')) != next(iter(IMAGE_SOURCES))
        volume = has('volume') or (has('image') and derived)
        ct = dpg.get_value(_tag('volume_format')) == next(iter(VOLUME_FORMATS))
        for field, shown in {'image_settings': has('image'), 'image_column': not derived,
                             'volume_settings': volume, 'ct': ct, 'mri': not ct,
                             'feature_columns': has('tabular'), 'standardize': has('tabular')}.items():
            dpg.configure_item(_tag(field), show=shown)
        for modality in MODALITIES:
            dpg.configure_item(_tag('port_' + modality), show=has(modality))

    def columns(self, header):
        for role in (*_ROLES, 'sequence_2', 'sequence_3', 'sequence_4'):
            tag = _tag(role)
            sentinel = _AUTO if role == 'split_column' else _NONE if role.startswith('sequence_') else None
            value = dpg.get_value(tag)
            items = list(dict.fromkeys(([sentinel] if sentinel else []) + list(header) + ([value] if value else [])))
            dpg.configure_item(tag, items=items)

    def sync(self, config, header):
        data = config['data']
        kind = data['kind']
        table = kind == 'table_csv'
        dpg.configure_item(_tag('table_settings'), show=table)
        dpg.configure_item(_tag('multimodal_settings'), show=not table)
        dpg.configure_item(_tag('standardize'), show=True)
        values = {role: data[role] if data[role] is not None else (_AUTO if role == 'split_column' else '')
                  for role in _ROLES if role in data}
        values['standardize'] = data['standardize']
        if table:
            values.update(task=_label(TABLE_TASKS, data['task']), folds=data['folds'],
                          table_test_fraction=data['test_fraction'])
            feature_fields = ('numeric_columns', 'categorical_columns')
        else:
            values.update(root=data['source_root'], volume_format=_label(VOLUME_FORMATS, data['volume_format']),
                          image_source=_label(IMAGE_SOURCES, data['image_source']), image_channels=str(data['image_channels']),
                          registered=data['registered'], validation_fraction=data['validation_fraction'],
                          test_fraction=data['test_fraction'])
            mapping = config.get('input_ports') or {}
            for modality in MODALITIES:
                enabled = modality in data['modalities']
                values['has_' + modality] = enabled
                values['port_' + modality] = mapping.get(modality, '' if enabled else MULTIMODAL_PORTS[modality])
            for prefix, axes, entries in (('volume_size_', 'DHW', data['volume_shape']),
                                           ('spacing_', 'DHW', data['spacing_mm']),
                                           ('image_size_', 'HW', data['image_size'])):
                values.update({prefix + axis: value for axis, value in zip(axes, entries)})
            values['lower_hu'], values['upper_hu'] = data['ct_window'] or (-1000., 400.)
            for channel in (2, 3, 4):
                index = channel - 2
                values['sequence_' + str(channel)] = (data['sequence_columns'][index]
                                                       if index < len(data['sequence_columns']) else _NONE)
            feature_fields = ('feature_columns',)
        for field in feature_fields:
            columns = data[field]
            literal = any(name != name.strip() or name.startswith('[') or ',' in name or '，' in name for name in columns)
            values[field] = json.dumps(columns, ensure_ascii=False) if literal else ', '.join(columns)
        for field, value in values.items():
            dpg.set_value(_tag(field), value)
        self.columns(header)
        self.layout(kind)

    def configuration(self, config, common):
        def value(key):
            return dpg.get_value(_tag(key))

        kind = common['kind']
        split = value('split_column')
        roles = {role: value(role) for role in ('id_column', 'group_column', 'target_column')}
        roles.update(split_column=None if split == _AUTO else split, standardize=value('standardize'))
        if kind == 'table_csv':
            task = TABLE_TASKS[value('task')]
            data = table_defaults(task)
            data.update(roles, path=common['path'], folds=value('folds'), test_fraction=value('table_test_fraction'),
                        numeric_columns=_text_columns(value('numeric_columns')), categorical_columns=_text_columns(value('categorical_columns')))
            config.update(profile='regression-v1' if task == 'regression' else 'classification-v2',
                          device='cpu', resume_checkpoint=None, input_ports=None)
        else:
            data = {**multimodal_defaults(), **config.get('data', {})}
            modalities = [name for name in MODALITIES if value('has_' + name)]
            # Keep a restored project's modality ordering when only port mappings change.
            previous = data['modalities']
            modalities.sort(key=lambda name: previous.index(name) if name in previous else len(previous))
            data.update(common, **roles, modalities=modalities, source_root=value('root').strip(),
                        validation_fraction=value('validation_fraction'), test_fraction=value('test_fraction'))
            derived = False
            if 'image' in modalities:
                image_source = IMAGE_SOURCES[value('image_source')]
                derived = image_source != 'raster'
                data.update(image_source=image_source, image_channels=int(value('image_channels')),
                            image_size=[value('image_size_' + axis) for axis in 'HW'])
                if not derived:
                    data['image_column'] = value('image_column')
            if 'volume' in modalities or derived:
                fmt = VOLUME_FORMATS[value('volume_format')]
                mri = fmt == 'mri_nifti'
                sequences = [value('sequence_' + str(channel)) for channel in (2, 3, 4)] if mri else []
                data.update(volume_format=fmt, volume_column=value('volume_column'),
                            volume_shape=[value('volume_size_' + axis) for axis in 'DHW'],
                            spacing_mm=[value('spacing_' + axis) for axis in 'DHW'],
                            ct_window=None if mri else [value('lower_hu'), value('upper_hu')],
                            sequence_columns=[column for column in sequences if column != _NONE],
                            registered=mri and value('registered'))
            if 'tabular' in modalities:
                data['feature_columns'] = _text_columns(value('feature_columns'))
            config.update(input_port='x',
                          input_ports={name: value('port_' + name).strip() for name in modalities if value('port_' + name).strip()})
        return data
