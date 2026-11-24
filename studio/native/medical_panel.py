"""Bounded training patch preview. Volume decoding stays off the UI thread."""
from array import array

import dearpygui.dearpygui as dpg

from studio_core.medical_volume_data import MEDICAL_FORMATS, preview_medical_patch


class MedicalPatchPanel:
    def __init__(self, app):
        self.app = app
        self.metadata = None

    def build(self):
        dpg.add_texture_registry(tag='medical_patch_textures', show=False)
        with dpg.window(tag='medical_patch_dialog', label='训练补丁预览', show=False, modal=True,
                        width=650, height=470, no_collapse=True):
            with dpg.group(horizontal=True):
                for field, label in [('case', '病例'), ('epoch', '轮次'), ('index', '补丁')]:
                    dpg.add_input_int(tag='medical_patch_' + field, label=label, width=110,
                                      default_value=1, min_value=1, min_clamped=True, step=0)
            with dpg.group(horizontal=True):
                dpg.add_input_int(tag='medical_patch_channel', label='通道', width=110, default_value=1,
                                  min_value=1, min_clamped=True, step=0)
                dpg.add_input_int(tag='medical_patch_slice', label='补丁切片', width=110, default_value=1,
                                  min_value=1, min_clamped=True, step=0)
                dpg.add_button(label='查看', tag='medical_patch_refresh', callback=self.app.guarded(self.refresh))
            dpg.add_text('', tag='medical_patch_help', wrap=610)
            dpg.add_text('', tag='medical_patch_status', wrap=610)
            dpg.add_group(tag='medical_patch_image_host')
            dpg.add_button(label='关闭', callback=self.app.guarded(self.close))

    def _clear(self):
        self.metadata = None
        for tag in ('medical_patch_image', 'medical_patch_texture'):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)

    def invalidate(self):
        self.app.queries.invalidate('medical-patch')
        if self.app._context_created and dpg.does_item_exist('medical_patch_dialog'):
            self._clear()
            dpg.hide_item('medical_patch_dialog')

    def close(self, **_):
        self.invalidate()

    def _manifest(self):
        manifest = self.app.manifest
        if not manifest or manifest['data_spec'].get('format') not in MEDICAL_FORMATS:
            raise ValueError('请先检查 CT 或 MRI 分割数据。')
        return manifest

    def open(self, **_):
        manifest = self._manifest()
        channels, depth = manifest['input_shape'][1:3]
        names = [manifest['data_spec']['image_column'], *manifest['data_spec']['sequence_columns']]
        dpg.set_value('medical_patch_help', '通道顺序 ' + ' → '.join(names) + '。左侧为影像，右侧为标注叠加。')
        for field, maximum in [('case', manifest['counts']['train']), ('epoch', 1000),
                               ('index', manifest['data_spec']['patches_per_case']), ('channel', channels), ('slice', depth)]:
            dpg.configure_item('medical_patch_' + field, max_value=maximum, max_clamped=True)
            dpg.set_value('medical_patch_' + field, (depth + 1) // 2 if field == 'slice' else 1)
        self.app._bind_modal('medical_patch_dialog')
        dpg.show_item('medical_patch_dialog')
        self.refresh()

    def refresh(self, **_):
        manifest = self._manifest()
        case_number, epoch, patch, channel, slice_number = [dpg.get_value('medical_patch_' + field) for field in
                                                           ('case', 'epoch', 'index', 'channel', 'slice')]
        records = [row for row in manifest['records'] if row['split'] == 'train']
        if not 1 <= case_number <= len(records) or not 1 <= epoch <= 1000:
            raise ValueError('病例序号或训练轮次超出范围。')
        if not 1 <= channel <= manifest['input_shape'][1] or not 1 <= slice_number <= manifest['input_shape'][2]:
            raise ValueError('通道或切片超出补丁范围。')
        case_id = records[case_number - 1]['id']
        self._clear()
        dpg.set_value('medical_patch_status', '正在按训练配置采样…')

        def load(cancelled):
            import numpy as np
            from PIL import Image
            from studio_core.segmentation_predictions import _overlay

            image, labels, metadata = preview_medical_patch(manifest, case_id=case_id, epoch=epoch,
                                                            patch_index=patch - 1, cancelled=cancelled)
            values = (np.clip(image[channel - 1, slice_number - 1], 0, 1) * 255).astype(np.uint8)
            base = np.repeat(values[..., None], 3, axis=-1)
            left = Image.fromarray(base)
            right = Image.fromarray(_overlay(base, labels[slice_number - 1]))
            for panel in (left, right):
                panel.thumbnail((286, 225))
            width, height = left.size
            combined = Image.new('RGBA', (2 * width + 8, height), (24, 27, 31, 255))
            combined.paste(left, (0, 0))
            combined.paste(right, (width + 8, 0))
            return combined.size, combined.tobytes(), metadata

        self.app.queries.submit('medical-patch', load)

    def receive(self, value, error):
        if not dpg.is_item_shown('medical_patch_dialog'):
            return
        if error:
            dpg.set_value('medical_patch_status', error)
            return
        (width, height), rgba, self.metadata = value
        dpg.add_static_texture(width, height, array('f', (v / 255.0 for v in rgba)),
                               tag='medical_patch_texture', parent='medical_patch_textures')
        scale = min(570 / width, 224 / height)
        dpg.add_image('medical_patch_texture', tag='medical_patch_image', parent='medical_patch_image_host',
                      width=max(1, round(width * scale)), height=max(1, round(height * scale)))
        row = self.metadata

        def axes(values):
            return ' × '.join(f'{value:.5g}' for value in values)

        sampling = {'uniform': '均匀', 'foreground': '前景'}.get(row['sampling'], row['sampling'])
        dpg.set_value('medical_patch_status',
                      f"{row['case_id']} · 训练集 · {sampling}采样 · {'含前景' if row['contains_foreground'] else '无前景'}\n"
                      f"起点 D/H/W {axes(row['patch_start_dhw'])} · 末端填充 {axes(row['pad_after_dhw'])}\n"
                      f"参考间距 {axes(row['reference_spacing_mm'])} mm · 模型间距 {axes(row['model_spacing_mm'])} mm")
