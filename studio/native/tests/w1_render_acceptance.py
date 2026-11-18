"""Application-owned save/open/restart and framebuffer proof; no OS input."""
from copy import deepcopy
import json
from pathlib import Path

import dearpygui.dearpygui as dpg

from native.catalog_panel import screen_rect
from native.document import Document
from native.main import load_saved_workspace_full
from native.paths import create_runtime_paths, resource_root
from native.render_acceptance import _capture, _frames
from native.ui import NativeWorkbench
from native.workspace_state import read_workspace_full


def run(output):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    paths = create_runtime_paths(output / 'session')
    report = {'status': 'running', 'input': 'application callbacks and framebuffers only',
              'callbacks': [], 'frames': [], 'saved': {}, 'exact_nonzero_pan_zoom_verified': False,
              'training_started': False}

    def invoke(tag, data=None):
        callback = dpg.get_item_callback(tag)
        assert callback, tag
        result = callback(tag, data, dpg.get_item_user_data(tag))
        report['callbacks'].append(tag)
        return result

    ui = NativeWorkbench(Document(), None, paths['data'], resource_root())
    ui.build()
    try:
        initial_model = deepcopy(ui.document.graph)
        initial_identity = ui.document.semantic_key()
        key = ui.workbench_panel.open_call(['encode_b'])
        ui.canvas_host.apply_selection(key, ['call:project'], intentional=True)
        ui._focus_document(key)
        ui._select_probe_output('project', app_data='out')
        for mode in ('single', 'columns', 'rows'):
            invoke('native_layout_' + mode)
            if mode != 'single' and ui.workbench.views[key].pane == 'primary':
                invoke('native_view_move_primary')
            ui.workbench_panel.activate_view(key)
            target = output / (mode + '.json')
            ui._save_json()
            invoke('native_json_save', {'file_path_name': str(target)})
            dpg.hide_item('native_json_save')
            payload = json.loads(target.read_text(encoding='utf-8'))
            assert payload['workspace_version'] == 3
            assert payload['model'] == initial_model
            assert ui.document.semantic_key() == initial_identity
            report['saved'][mode] = payload['workbench']
        old_token = ui.canvas_host.token(key)
        invoke('native_layout_single')
        ui._open_json()
        invoke('native_json_open', {'file_path_name': str(output / 'rows.json')})
        dpg.hide_item('native_json_open')
        assert ui.workbench.to_payload() == report['saved']['rows']
        assert not ui.canvas_host.accept(old_token)
        report['explicit_reopen_invalidated_old_token'] = True
        # The form is deliberately incomplete at both autosave and shutdown.
        applied = deepcopy(ui.experiment)
        ui.experiment_mode = 'csv'
        dpg.set_value('native_csv_features', '{incomplete')
        ui._save_draft()
        saved = json.loads(paths['workspace'].read_text(encoding='utf-8'))
        assert saved['workspace_version'] == 3 and saved['experiment'] == applied
        assert saved['workbench'] == report['saved']['rows']
    finally:
        ui.shutdown()
    saved = json.loads(paths['workspace'].read_text(encoding='utf-8'))
    assert saved['workspace_version'] == 3 and saved['experiment'] == applied
    document, experiment, payload = load_saved_workspace_full(paths)
    assert payload == report['saved']['rows']
    report['startup_full_restored'] = True
    report['invalid_form_auto_and_exit_saved_v3'] = True

    for mode in ('single', 'columns', 'rows'):
        document, experiment, payload = read_workspace_full(output / (mode + '.json'))
        ui = NativeWorkbench(document, None, output / ('restart-' + mode), resource_root(),
                             experiment=experiment, workbench_payload=payload)
        ui.build()
        try:
            assert ui.workbench.to_payload() == report['saved'][mode]
            assert ui.document.semantic_key() == initial_identity
            assert ui._probe_source() == 'call:encode_b/call:project'
            for width, height, size in ((1380, 860, 'default'), (1100, 760, 'minimum')):
                dpg.set_viewport_width(width); dpg.set_viewport_height(height)
                ui.last_size = None; _frames(ui, 5)
                tags = ['native_search', 'native_tree_scroll', 'native_catalog_scroll', 'native_canvas_body',
                        'native_canvas_primary', 'native_right', 'native_run', 'native_layout_rows',
                        'native_view_picker_' + ui.workbench.focused_pane]
                if mode != 'single':
                    tags.append('native_canvas_secondary')
                rectangles = {tag: {'origin': screen_rect(tag)[0], 'size': screen_rect(tag)[1]} for tag in tags}
                for tag, rectangle in rectangles.items():
                    x, y = rectangle['origin']; w, h = rectangle['size']
                    assert w > 10 and h > 10, (tag, rectangle)
                    assert x + w <= dpg.get_viewport_client_width() + 2, (tag, rectangle)
                    assert y + h <= dpg.get_viewport_client_height() + 2, (tag, rectangle)
                if mode != 'single':
                    (x, y), (w, h) = screen_rect('native_canvas_primary')
                    (sx, sy), (sw, sh) = screen_rect('native_canvas_secondary')
                    assert x + w <= sx or sx + sw <= x or y + h <= sy or sy + sh <= y
                name = f'{size}-{mode}.png'
                _capture(ui, output / name)
                selected = ui.workbench.views[ui.workbench.focused_view_id].selected_refs
                assert selected == ['call:project']
                report['frames'].append({'file': name, 'viewport': [dpg.get_viewport_client_width(), dpg.get_viewport_client_height()],
                                         'layout': mode, 'focused_view_id': ui.workbench.focused_view_id,
                                         'selected_refs': selected, 'rectangles': rectangles})
            if mode == 'rows':
                # These measured probe entries come from a real CPU eval
                # forward on generated inputs, with no training or worker.
                import torch
                from studio_core.runtime import build_model, _probe
                model = build_model(ui.document.graph, device='cpu')
                model.eval()
                with torch.no_grad():
                    model(a=torch.ones(2, 4), b=torch.zeros(2, 4))
                ui.result = {'probes': {source: _probe(value) for source, value in model.last_values.items()}}
                ui.result_key = ui._experiment_identity()
                ui._render_probe()
                source = 'call:encode_a/call:project'
                invoke('native_probe_sources', source)
                focused = ui.workbench.views[ui.workbench.focused_view_id]
                assert focused.call_path == ['encode_a'] and focused.selected_refs == ['call:project']
                assert ui._probe_source() == source
                assert ui.run_record is None
                report['actual_cpu_probe_navigation'] = {'source': source, 'shape': ui.result['probes'][source]['shape'],
                                                          'generated_inputs': [[2, 4], [2, 4]], 'mode': 'eval'}
                _capture(ui, output / 'minimum-measured-probe.png')
        finally:
            ui.shutdown()
    report['status'] = 'passed'
    (output / 'frame-report.json').write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    from PIL import Image, ImageOps, ImageDraw
    sheet = Image.new('RGB', (960, 3 * 320), '#171b21')
    for index, frame in enumerate(report['frames']):
        with Image.open(output / frame['file']) as image:
            thumbnail = ImageOps.contain(image.convert('RGB'), (480, 290))
        x, y = (index % 2) * 480, (index // 2) * 320
        sheet.paste(thumbnail, (x, y + 25))
        ImageDraw.Draw(sheet).text((x + 8, y + 6), frame['file'], fill='white')
    sheet.save(output / 'contact-sheet.png')
    print(json.dumps({'status': report['status'], 'frames': len(report['frames']), 'callbacks': len(report['callbacks'])}))


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
