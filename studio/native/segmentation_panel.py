"""Native preview of held-out segmentation predictions and mask export."""
from array import array
import csv
from pathlib import Path
import shutil

import dearpygui.dearpygui as dpg
from PIL import Image


class SegmentationPanel:
    def __init__(self, app):
        self.app = app
        self.index_path = None
        self.rows = []
        self.index = 0

    def build(self):
        with dpg.tab(label='分割预测', tag='native_segmentation_tab', show=False):
            dpg.add_text('训练完成后在此查看保留测试集的预测。', tag='native_segmentation_result')
            with dpg.group(tag='native_segmentation_browser', show=False):
                with dpg.group(horizontal=True):
                    dpg.add_button(label='上一样本', tag='native_segmentation_previous',
                                   callback=self.app.guarded(lambda **_: self.select(self.index - 1)))
                    dpg.add_input_int(label='样本序号', tag='native_segmentation_sample', width=110,
                                      min_value=1, min_clamped=True, max_clamped=True,
                                      callback=self.app.guarded(self._select_number))
                    dpg.add_button(label='下一样本', tag='native_segmentation_next',
                                   callback=self.app.guarded(lambda **_: self.select(self.index + 1)))
                    dpg.add_button(label='导出原始网格掩码…', tag='native_segmentation_export',
                                   callback=self.app.guarded(self._show_export))
                dpg.add_text('左  输入影像     中  标注叠加     右  预测叠加')
                dpg.add_group(tag='native_segmentation_image_host')

    def build_resources(self):
        dpg.add_texture_registry(tag='native_segmentation_textures', show=False)
        with dpg.file_dialog(tag='native_segmentation_export_dialog', directory_selector=True, show=False,
                             width=740, height=460, modal=True, default_path=str(self.app.data_directory),
                             callback=self.app.guarded(self._export)):
            pass

    def sync(self, result, active, stale):
        is_result = (result or {}).get('profile') == 'segmentation-v1'
        dpg.configure_item('native_segmentation_tab', show=active or is_result)
        path = (result or {}).get('artifacts', {}).get('predictions') if is_result else None
        if path == self.index_path:
            if self.rows:
                self._summary(stale)
            return
        self.index_path = path
        self.rows = []
        self.index = 0
        self._clear_image()
        dpg.hide_item('native_segmentation_browser')
        if not path:
            dpg.set_value('native_segmentation_result', '训练完成后在此查看保留测试集的预测。')
            return
        try:
            with Path(path).open(encoding='utf-8', newline='') as stream:
                self.rows = list(csv.DictReader(stream))
            if not self.rows:
                raise ValueError('预测清单为空。')
            self.select(0, stale=stale)
            dpg.show_item('native_segmentation_browser')
        except (OSError, ValueError, KeyError) as error:
            self.rows = []
            dpg.set_value('native_segmentation_result', '无法打开预测产物。' + str(error))

    def _clear_image(self):
        for tag in ('native_segmentation_image', 'native_segmentation_texture'):
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)

    def _artifact(self, field):
        directory = Path(self.index_path).resolve().parent / 'masks'
        path = (directory.parent / self.rows[self.index][field]).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError('预测文件缺失或不在本次实验目录。')
        return path

    def _summary(self, stale):
        row = self.rows[self.index]
        scope = f" · 切片 {row['slice_index']}/{row['slice_count']}（模型网格）" if row['slice_count'] != '1' else ''
        dpg.set_value('native_segmentation_result', f"样本 {row['sample_id']} · {self.index + 1}/{len(self.rows)}" + scope +
                      (' · 来自旧实验快照' if stale else ''))

    def _select_number(self, app_data=None, **_):
        self.select(app_data - 1)

    def select(self, index, *, stale=None):
        if not self.rows:
            return
        index = min(max(index, 0), len(self.rows) - 1)
        previous = self.index
        self.index = index
        try:
            with Image.open(self._artifact('preview_file')) as image:
                if image.width > 800 or image.height > 192:
                    raise ValueError('分割预览超过尺寸限制。')
                rgba = image.convert('RGBA')
                width, height = rgba.size
                pixels = array('f', (value / 255.0 for value in rgba.tobytes()))
        except Exception:
            self.index = previous
            raise
        self._clear_image()
        dpg.add_static_texture(width, height, pixels, tag='native_segmentation_texture', parent='native_segmentation_textures')
        scale = min(120 / height, 820 / width)
        dpg.add_image('native_segmentation_texture', tag='native_segmentation_image', parent='native_segmentation_image_host',
                      width=max(1, int(width * scale)), height=max(1, int(height * scale)))
        dpg.configure_item('native_segmentation_previous', enabled=index > 0)
        dpg.configure_item('native_segmentation_next', enabled=index + 1 < len(self.rows))
        dpg.configure_item('native_segmentation_sample', max_value=len(self.rows))
        dpg.set_value('native_segmentation_sample', index + 1)
        if stale is None:
            stale = self.app.result_key != self.app._experiment_identity()
        self._summary(stale)

    def _show_export(self, **_):
        self.app._bind_modal('native_segmentation_export_dialog')
        dpg.show_item('native_segmentation_export_dialog')

    def _export(self, app_data=None, **_):
        source = self._artifact('mask_file')
        directory = Path(app_data.get('file_path_name') or app_data['current_path'])
        target = directory / ('seg-' + Path(self.index_path).parent.name + '-' + source.name)
        with source.open('rb') as stream:
            with target.open('xb') as output:
                try:
                    shutil.copyfileobj(stream, output)
                except OSError:
                    output.close()
                    target.unlink(missing_ok=True)
                    raise
        dpg.hide_item('native_segmentation_export_dialog')
        self.app.log('原始网格掩码已导出到 ' + str(target), '完成')
