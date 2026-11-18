"""Persist requested attention slices separately from actual worker results."""
import json
import math

from studio_core.probe_capture import MAX_SIDE, normalize_probe_requests as normalize_capture_requests


_SELECTION = ('sample_index', 'head_index', 'query_range', 'key_range')
def normalize_probe_requests(requests):
    normalized = normalize_capture_requests(requests)
    for item in normalized:
        if item['source'].count('/') > 8:
            raise ValueError('Attention probe path exceeds the chip nesting limit')
        for field in _SELECTION[2:]:
            start, stop = item[field]
            if stop - start > MAX_SIDE:
                raise ValueError(f'{field} must span at most {MAX_SIDE} values')
    return normalized


def matching_capture(capture, request, captured_request=None):
    if not capture:
        return None
    if request is None:
        return capture
    if captured_request is not None:
        return capture if captured_request == request else None
    selection = capture.get('selection') or {}
    return capture if all(selection.get(key) == request[key] for key in _SELECTION) else None


class AttentionProbePanel:
    def __init__(self, ui):
        self.ui = ui

    def requests(self):
        return normalize_probe_requests(self.ui.document.attention_probe_requests)

    def run_requests(self):
        return [item for item in self.requests() if self.ui._probe_location(item['source']) is not None]

    def apply(self, source, sample, head, ranges, **_):
        import dearpygui.dearpygui as dpg

        request = {'source': source, 'sample_index': dpg.get_value(sample), 'head_index': dpg.get_value(head),
                   **{key: json.loads(dpg.get_value(widget)) for key, widget in ranges.items()},
                   'statistics': ['mean', 'min', 'max']}
        candidate = [item for item in self.requests() if item['source'] != source] + [request]
        self.ui.document.attention_probe_requests = normalize_probe_requests(candidate)
        self.ui._render_probe()

    def remove(self, source, **_):
        self.ui.document.attention_probe_requests = [item for item in self.requests() if item['source'] != source]
        self.ui._render_probe()

    def render(self, parent, source, capture, stale):
        import dearpygui.dearpygui as dpg

        requests = self.requests()
        missing = [item for item in requests if self.ui._probe_location(item['source']) is None]
        for item in missing:
            with dpg.group(horizontal=True, parent=parent):
                dpg.add_text(item['source'] + ' · 来源已移除 · 尚无本次结果')
                dpg.add_button(label='移除请求', callback=self.ui.guarded(self.remove, item['source']))
        if source is None:
            return
        request = next((item for item in requests if item['source'] == source), None)
        selection = request or (capture or {}).get('selection') or {}
        frozen = next((item for item in (self.ui.run_config or {}).get('attention_probe_requests', [])
                       if item['source'] == source), None)
        captured = matching_capture(capture, request, frozen) if not stale else None
        if captured:
            with dpg.group(horizontal=True, parent=parent):
                if not captured.get('omitted'):
                    self._heatmap(captured.get('values', []))
                with dpg.group():
                    actual = captured.get('selection') or {}
                    context = (self.ui.result or {}).get('probe_context', {})
                    sample_index = actual.get('sample_index')
                    samples = context.get('sample_ids', [])
                    sample_id = samples[sample_index] if type(sample_index) is int and 0 <= sample_index < len(samples) else '—'
                    dpg.add_text(f"{captured.get('shape')} · {captured.get('dtype')} · 样本 ID {sample_id}"
                                 + (' · 已截断' if captured.get('truncated') else ''))
                    if captured.get('omitted'):
                        dpg.add_text('未采集：' + str(captured.get('reason', '—')))
                    else:
                        dpg.add_text(f"头 {actual.get('head_index')} · 查询 {actual.get('query_range')} · 键 {actual.get('key_range')}")
        with dpg.group(horizontal=True, parent=parent):
            sample = dpg.add_input_int(tag='native_attention_sample', label='样本', width=70, default_value=selection.get('sample_index', 0),
                                       min_value=0, min_clamped=True)
            head = dpg.add_input_int(tag='native_attention_head', label='头', width=70, default_value=selection.get('head_index', 0),
                                     min_value=0, min_clamped=True)
            ranges = {key: dpg.add_input_text(tag='native_attention_' + key, label=label, width=74,
                                             default_value=json.dumps(selection.get(key, [0, 1])))
                      for key, label in (('query_range', '查询 [起,止)'), ('key_range', '键 [起,止)'))}
            dpg.add_button(tag='native_attention_apply', label='下次采集',
                           callback=self.ui.guarded(self.apply, source, sample, head, ranges))
            if request:
                dpg.add_button(label='移除请求', callback=self.ui.guarded(self.remove, source))
        if not captured:
            dpg.add_text('尚无本次结果 · 所选切片需要重新运行', parent=parent)

    @staticmethod
    def _heatmap(values):
        import dearpygui.dearpygui as dpg

        if not values or type(values[0]) is not list or not values[0]:
            return
        rows, columns = len(values), len(values[0])
        if rows > 64 or columns > 64 or any(type(row) is not list or len(row) != columns for row in values):
            return
        flat = [value if type(value) in (int, float) and math.isfinite(value) else 0.0
                for row in values for value in row]
        low, high = min(flat), max(flat)
        with dpg.plot(tag='native_attention_heatmap', width=210, height=90, no_menus=True, no_title=True):
            dpg.add_plot_axis(dpg.mvXAxis, no_tick_labels=True)
            axis = dpg.add_plot_axis(dpg.mvYAxis, no_tick_labels=True)
            dpg.add_heat_series(flat, rows, columns, parent=axis, format='', scale_min=low,
                                scale_max=high if high > low else low + 1)
