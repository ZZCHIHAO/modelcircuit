"""Shipped S4 catalog/workbench diagnostic using native callbacks/framebuffers."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile

import dearpygui.dearpygui as dpg

from .catalog_panel import screen_rect
from .document import Document
from .render_acceptance import _capture, _frames
from .s4_acceptance import _cleanup_fixture, _require, _write_json, check_catalog, empty_graph
from .ui import NativeWorkbench


def check_s4_render(manager, directory, resources, output_directory, device='cpu'):
    """Exercise registered application callbacks; do not emulate physical input."""
    from .chip_library import save_bundle
    from studio_core.devices import resolve_device
    from studio_core.module_templates import make_module
    output, directory = Path(output_directory).resolve(), Path(directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    report_path = output / 's4-render-acceptance.json'
    report = {'status': 'running', 'kind': 'native-s4-render-callbacks', 'device': device,
              'frozen': bool(getattr(sys, 'frozen', False)), 'callbacks': [], 'frames': [], 'fixtures_cleaned': False,
              'notice': 'Application-owned callbacks and framebuffers only; physical gestures and exact pan/zoom are not verified',
              'gpu_execution': {'executed': False, 'reason': 'Native catalog/workbench diagnostic; no training is launched'},
              'report_path': str(report_path)}
    fixture, ui = None, None
    try:
        resolve_device(device)
        report['catalog'] = check_catalog()
        fixture = tempfile.TemporaryDirectory(prefix='s4-native-', dir=directory)
        fixture_root = Path(fixture.name)
        report['fixture_directory'] = str(fixture_root)
        bundle = make_module('fusion.weighted_sum', {'features': 4})
        chip = save_bundle({'definitions': bundle.definitions, 'resource_profile': 'vision-v4', 'resource_batch_bound': 2},
                           bundle.entry['definition_id'], '1.0.0', fixture_root / 'chips')
        (fixture_root / 'chips' / 'broken.json').write_text('{oops', encoding='utf-8')
        chip_evidence = output / 's4-local-chip.json'
        shutil.copy2(chip, chip_evidence)
        ui = NativeWorkbench(Document(empty_graph()), manager, fixture_root, resources)
        ui.build()

        def invoke(tag, data=None):
            callback = dpg.get_item_callback(tag)
            _require(callable(callback), 'native item has no registered callback: ' + str(tag))
            value = callback(tag, data, dpg.get_item_user_data(tag))
            report['callbacks'].append(str(tag))
            return value

        def capture(name):
            tags = ('native_search', 'native_catalog_category', 'native_canvas_body', 'native_layout_rows')
            # Resizing and closing a modal can invalidate rectangles for a few frames.
            for _ in range(30):
                _frames(ui, 1)
                rectangles = {tag: screen_rect(tag) for tag in tags}
                if all(min(size) > 0 for _, size in rectangles.values()):
                    break
            geometry = {}
            for tag, (origin, size) in rectangles.items():
                _require(min(size) > 0, f'{name}: native layout has empty geometry: {tag}')
                _require(origin[0] + size[0] <= dpg.get_viewport_client_width() + 2 and
                         origin[1] + size[1] <= dpg.get_viewport_client_height() + 2, 'native layout clipped: ' + tag)
                geometry[tag] = {'origin': origin, 'size': size}
            target = output / (name + '.png')
            _capture(ui, target)
            report['frames'].append({'file': str(target), 'client': [dpg.get_viewport_client_width(), dpg.get_viewport_client_height()],
                                     'category': ui.catalog_panel.category, 'rectangles': geometry})

        def link(source, target):
            view = ui.canvas_host.get('root')
            left = next(key for key, value in view.source_attributes.items() if value == source)
            right = next(key for key, value in view.target_attributes.items() if value == target)
            invoke(view.editor_tag, [left, right])

        invoke('native_catalog_category', '视觉主干')
        _require(len(ui.catalog_panel.visible_ids) == 9, 'native backbone category is incomplete')
        invoke('native_catalog_config_backbone.vgg19')
        capture('native-s4-vgg19-config')
        invoke('native_catalog_config_cancel')
        invoke('native_catalog_category', '完整模型')
        _require(len(ui.catalog_panel.visible_ids) == 5, 'native full-model category is incomplete')
        invoke('native_catalog_config_model.squeezenet1_1_classifier')
        dpg.set_value(ui.catalog_panel.config_widgets['input_shape'], '[null,3,32,32]')
        dpg.set_value(ui.catalog_panel.config_widgets['num_classes'], '2')
        invoke('native_catalog_apply')
        invoke('native_catalog_policy_apply')
        _require(ui.document.graph.get('resource_profile') == 'hardware-v5', 'catalog did not adopt hardware-v5')
        call = ui.document.selected_call['id']
        link('input:x', ('call', call, 'x'))
        link('call:' + call + ':logits', ('output', 'logits', 'value'))
        _require(ui.document.validation()['valid'], 'configured native model is not connected/valid')
        # Use the button registered by the production inspector to expand it.
        ui.document.select('call:' + call)
        ui._refresh()
        enter = [tag for tag in dpg.get_all_items() if dpg.get_item_label(tag) == '进入此调用内部']
        _require(len(enter) == 1, 'selected model has no unique expansion callback')
        invoke(enter[0])
        key = ui.workbench.focused_view_id
        _require(key != 'root' and len(ui.workbench.views) == 2, 'model expansion did not open a chip tab')
        invoke('native_layout_columns')
        invoke('native_view_move_primary')
        _require(ui.workbench.views[key].pane == 'secondary', 'expanded tab did not move to split pane')
        invoke('native_tabbar_primary', dpg.get_alias_id('native_tab_root'))
        saved_views, saved_model = deepcopy(ui.workbench.to_payload()), ui.document.semantic_key()
        workspace = output / 's4-workspace-v3.json'
        invoke('native_json_save', {'file_path_name': str(workspace)})
        dpg.hide_item('native_json_save')
        stored = json.loads(workspace.read_text(encoding='utf-8'))
        _require(stored['workspace_version'] == 3 and stored['model']['resource_profile'] == 'hardware-v5', 'workspace v3/hardware policy not saved')
        invoke('native_json_open', {'file_path_name': str(workspace)})
        dpg.hide_item('native_json_open')
        _require(ui.workbench.to_payload() == saved_views and ui.document.semantic_key() == saved_model,
                 'v3 workspace reopen changed model or split/tab state')
        report['workspace'] = {'workspace_version': 3, 'views_and_model_identity_equal': True, 'file': str(workspace),
                               'expanded_view': key, 'layout': ui.workbench.layout}
        capture('native-s4-model-split')
        dpg.set_viewport_width(1100)
        dpg.set_viewport_height(760)
        ui.last_size = None
        capture('native-s4-model-minimum')
        ui.document.replace_graph(empty_graph(image=False))
        ui._reset_canvas_document()
        ui._refresh()
        invoke('native_catalog_category', '本地芯片')
        _require(len(ui.catalog_panel.visible_ids) == 1, 'local JSON chip was not indexed')
        local_id = ui.catalog_panel.visible_ids[0]
        invoke('native_catalog_config_' + local_id)
        dpg.set_value(ui.catalog_panel.config_widgets['alpha'], '0.75')
        invoke('native_catalog_apply')
        invoke('native_catalog_policy_apply')
        call = ui.document.selected_call['id']
        link('input:a', ('call', call, 'a'))
        link('input:b', ('call', call, 'b'))
        link('call:' + call + ':features', ('output', 'features', 'value'))
        _require(ui.document.validation()['valid'], 'local chip callback did not produce a valid graph')
        _require(ui.document.selected_instance['params']['alpha'] == .75, 'local chip configuration was lost')
        _require(any('broken.json' in issue for issue in ui.catalog_panel.local_index.diagnostics), 'damaged local chip was silently ignored')
        capture('native-s4-local-chip')
        import torch
        from studio_core.python_io import import_python
        from studio_core.runtime import build_model
        prior_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            x, y = torch.arange(8, dtype=torch.float32).reshape(2, 4), torch.ones(2, 4)
            expected = x + .75 * (y - x)
            for graph in (ui.document.graph, import_python(ui.document.export_python())):
                actual = build_model(graph, capture=False)(a=x, b=y)['features']
                torch.testing.assert_close(actual, expected)
        finally:
            torch.set_num_threads(prior_threads)
        report['local_extension'] = {'valid': True, 'alpha': .75, 'damaged_file_diagnostic': True,
                                     'actual_output_and_python_reimport': True, 'chip_evidence': str(chip_evidence)}
        report['status'] = 'passed'
        return report
    except Exception as error:
        report['status'], report['error'] = 'failed', str(error)
        raise
    finally:
        try:
            if ui is not None:
                ui.shutdown()
        finally:
            try:
                if fixture is not None:
                    _cleanup_fixture(fixture)
                report['fixtures_cleaned'] = True
            except Exception as error:
                report['status'] = 'failed'
                report['cleanup_error'] = str(error)
                report.setdefault('error', 'Fixture cleanup failed: ' + str(error))
                raise
            finally:
                _write_json(report_path, report)
