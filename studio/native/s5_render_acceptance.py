"""Run the S5 desktop workflow through application callbacks and framebuffer capture."""
import argparse
import json
from pathlib import Path
from time import monotonic

from .main import close_workers
from .paths import resource_root
from .s4_acceptance import empty_graph
from .s5_acceptance import fixture_config


def run(manager, directory, output, device):
    import dearpygui.dearpygui as dpg
    from .document import Document
    from .render_acceptance import _capture, _frames, _wait_ui
    from .ui import NativeWorkbench
    from .workspace_state import read_workspace_full

    graph = empty_graph()
    ui = NativeWorkbench(Document(graph), manager, directory, resource_root(), experiment=fixture_config(directory, device))
    callbacks = []

    def invoke(tag, data=None):
        callback = dpg.get_item_callback(tag)
        if not callable(callback):
            raise AssertionError(f'Missing registered callback: {tag}')
        callbacks.append(str(tag))
        return callback(tag, data, dpg.get_item_user_data(tag))

    def wait_for(predicate):
        deadline = monotonic() + 25
        while monotonic() < deadline:
            _frames(ui, 1)
            if predicate():
                return
        raise TimeoutError(f'UI query did not settle: {ui.messages[-4:]}')

    try:
        ui.build()
        _frames(ui, 6)
        invoke('native_search', 'vit')
        invoke('native_catalog_config_vit.small_classifier')
        _capture(ui, output / 'vit-config.png')
        invoke('native_catalog_apply')
        invoke('native_catalog_policy_apply')
        call = ui.document.selected_call['id']
        view = ui.canvas_host.get('root')
        for source, target in [('input:x', ('call', call, 'x')), (f'call:{call}:logits', ('output', 'logits', 'value'))]:
            left = next(key for key, value in view.source_attributes.items() if value == source)
            right = next(key for key, value in view.target_attributes.items() if value == target)
            invoke(view.editor_tag, [left, right])
            view = ui.canvas_host.get('root')
        report = ui.document.validation()
        if not report['valid'] or report['parameter_count'] != 74562:
            raise AssertionError(f'Unexpected ViT graph: {report}')
        source = f'call:{call}/call:encoder/call:block0/call:self_attention/call:attention:out'
        if not ui._focus_probe_source(source):
            raise AssertionError('Cannot navigate to actual attention call')
        _frames(ui, 4)
        dpg.set_value('native_attention_head', 1)
        dpg.set_value('native_attention_query_range', '[0,4]')
        dpg.set_value('native_attention_key_range', '[0,6]')
        invoke('native_attention_apply')
        requested = ui.attention_probe_panel.run_requests()
        if len(requested) != 1 or requested[0]['source'] not in (source, source.removesuffix(':out')):
            raise AssertionError(f'UI queued a different tensor: {requested}')
        source = requested[0]['source']
        invoke('native_layout_columns')
        invoke('native_view_move_primary')
        _capture(ui, output / 'attention-graph.png')
        invoke('native_inspect')
        wait_for(lambda: ui.manifest is not None and device in ui.available_devices)
        invoke('native_run')
        record = _wait_ui(ui)
        if record['status'] != 'succeeded':
            raise AssertionError(f'UI training failed: {record.get("error")}')
        observed = record['result']['probes'][source]
        if observed['omitted'] or observed['captured_bytes'] != 96:
            raise AssertionError('UI result did not contain the requested slice')
        ui._focus_probe_source(source)
        _capture(ui, output / 'attention-probe.png')
        if not any(dpg.get_item_type(item).endswith('mvHeatSeries') for item in dpg.get_all_items()):
            raise AssertionError('Measured attention heatmap was not rendered')
        dpg.set_viewport_width(1100)
        dpg.set_viewport_height(760)
        _capture(ui, output / 'attention-minimum.png')
        if not dpg.is_item_visible('native_attention_heatmap'):
            raise AssertionError('Attention heatmap is not visible at the minimum window size')
        workspace = output / 'vit-workspace.json'
        invoke('native_json_save', {'file_path_name': str(workspace)})
        document, _, views = read_workspace_full(workspace)
        if document.semantic_key() != ui.document.semantic_key() or views['layout'] != 'columns' or len(document.attention_probe_requests) != 1:
            raise AssertionError('Saved workspace lost graph, pane or request')
        invoke('native_json_open', {'file_path_name': str(workspace)})
        if ui.workbench.layout != 'columns' or ui.document.attention_probe_requests != document.attention_probe_requests:
            raise AssertionError('Reopened workspace lost pane or request')
        return {'device': device, 'callbacks': callbacks, 'frames': ['vit-config.png', 'attention-graph.png', 'attention-probe.png', 'attention-minimum.png'],
                'parameter_count': 74562, 'workspace_reopened': True, 'probe_shape': observed['shape'], 'probe_bytes': 96}
    finally:
        ui.shutdown()


def main():
    from server import RunManager
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.parent.resolve()
    output.mkdir(parents=True, exist_ok=True)
    directory = output / (args.output.stem + '-data')
    manager = RunManager(directory / 'runs')
    try:
        report = run(manager, directory, output, args.device)
    finally:
        close_workers(manager)
    report.update(status='passed', workers_stopped=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
