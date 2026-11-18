"""Real native model-library frames and app callbacks; no OS input automation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import dearpygui.dearpygui as dpg

from native.catalog_panel import screen_rect
from native.chip_library import save_bundle
from native.document import Document
from native.paths import resource_root
from native.render_acceptance import _capture, _frames
from native.ui import NativeWorkbench
from studio_core.module_templates import make_module


def empty_graph(image=True):
    inputs = [{'id': 'x', 'shape': [None, 3, 32, 32], 'dtype': 'float32', 'position': {'x': 10, 'y': 80}}] if image else [
        {'id': name, 'shape': [None, 4], 'dtype': 'float32', 'position': {'x': 10, 'y': 80 + index * 180}}
        for index, name in enumerate(('a', 'b'))]
    return {'schema_version': 1, 'id': 'model_catalog_journey', 'revision': 1,
            'inputs': inputs, 'instances': [], 'calls': [], 'definitions': [],
            'outputs': [{'id': 'logits' if image else 'features', 'position': {'x': 880, 'y': 140}}]}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {'frames': [], 'callbacks': [], 'drops': [], 'notice': 'App-owned callbacks/framebuffers only. No physical gesture or exact pan/zoom claim.'}
    with tempfile.TemporaryDirectory(prefix='mlforge-model-catalog-') as temporary:
        directory = Path(temporary)
        bundle = make_module('fusion.weighted_sum', {'features': 4})
        save_bundle({'definitions': bundle.definitions, 'resource_profile': 'vision-v4', 'resource_batch_bound': 2},
                    bundle.entry['definition_id'], '1.0.0', directory / 'chips')
        (directory / 'chips' / 'broken.json').write_text('{oops', encoding='utf-8')
        ui = NativeWorkbench(Document(), None, directory, resource_root())
        ui.build()

        def invoke(tag, data=None):
            result = dpg.get_item_callback(tag)(tag, data, dpg.get_item_user_data(tag))
            report['callbacks'].append(tag)
            return result

        def capture(name):
            _frames(ui, 4)
            tags = ['native_search', 'native_catalog_category', 'native_canvas_body', 'native_layout_rows']
            geometry = {tag: {'origin': screen_rect(tag)[0], 'size': screen_rect(tag)[1]} for tag in tags}
            for value in geometry.values():
                assert all(axis > 0 for axis in value['size'])
                assert value['origin'][0] + value['size'][0] <= dpg.get_viewport_client_width() + 2
                assert value['origin'][1] + value['size'][1] <= dpg.get_viewport_client_height() + 2
            _capture(ui, output / (name + '.png'))
            report['frames'].append({'file': name + '.png', 'client': [dpg.get_viewport_client_width(), dpg.get_viewport_client_height()],
                                     'category': ui.catalog_panel.category, 'rectangles': geometry})

        def drop(identifier, offset=(260, 100)):
            panel = ui.catalog_panel
            view = ui.canvas_host.get('root')
            _frames(ui, 4)
            anchor = view.ref_nodes[next(ref for ref in view.ref_nodes if ref.startswith('input:'))]
            origin = dpg.get_item_rect_min(anchor)
            local = dpg.get_item_pos(anchor)
            point = [origin[0] + offset[0], origin[1] + offset[1]]
            expected = ui._document_position((local[0] + offset[0], local[1] + offset[1]))
            button = 'native_catalog_add_' + identifier
            payload_tag = next(child for group in dpg.get_item_children(button).values() for child in group
                               if dpg.get_item_info(child)['type'].endswith('mvDragPayload'))
            payload = deepcopy(dpg.get_item_configuration(payload_tag)['drag_data'])
            callback = dpg.get_item_configuration(view.container_tag)['drop_callback']
            with patch('native.catalog_panel.dpg.get_mouse_pos', return_value=point):
                callback(view.container_tag, payload, ui.canvas_host.token('root'))
            if panel.pending:
                invoke('native_catalog_policy_apply')
            position = ui.document.selected_call['position']
            assert all(abs(position[axis] - expected[index]) < 1e-6 for index, axis in enumerate(('x', 'y')))
            report['drops'].append({'id': identifier, 'point': point, 'document_position': position, 'maximum_coordinate_error': 0})
            return ui.document.selected_call['id']

        def link(source, target):
            view = ui.canvas_host.get('root')
            a = next(key for key, value in view.source_attributes.items() if value == source)
            b = next(key for key, value in view.target_attributes.items() if value == target)
            invoke(view.editor_tag, [a, b])

        try:
            capture('default-existing-root')
            original = ui.document.semantic_key()
            experiment = deepcopy(ui._configuration())
            invoke('native_catalog_hardware_policy')
            capture('default-hardware-policy-upgrade')
            invoke('native_catalog_policy_apply')
            assert ui.document.graph['resource_profile'] == 'hardware-v5'
            assert ui.document.semantic_key() != original
            assert ui._configuration() == experiment
            report['explicit_legacy_upgrade'] = {'profile': 'hardware-v5', 'batch_bound': 64,
                                                  'experiment_unchanged': True, 'undoable': True}
            ui._undo()
            assert ui.document.semantic_key() == original
            ui.document.replace_graph(empty_graph())
            ui._reset_canvas_document()
            ui._refresh()
            invoke('native_catalog_category', '视觉主干')
            assert len(ui.catalog_panel.visible_ids) == 9
            capture('default-backbone-library')
            dpg.set_y_scroll('native_catalog_scroll', 9999)
            capture('default-backbone-library-overflow')
            invoke('native_catalog_config_backbone.vgg19')
            capture('default-vgg19-configuration')
            invoke('native_catalog_config_cancel')
            invoke('native_catalog_category', '完整模型')
            identifier = 'model.squeezenet1_1_classifier'
            invoke('native_catalog_config_' + identifier)
            dpg.set_value(ui.catalog_panel.config_widgets['input_shape'], '[null,3,32,32]')
            dpg.set_value(ui.catalog_panel.config_widgets['num_classes'], '2')
            invoke('native_catalog_apply')
            invoke('native_catalog_policy_apply')
            ui._undo()
            ui.workbench_panel.activate_view('root')
            call = drop(identifier)
            link('input:x', ('call', call, 'x'))
            link('call:' + call + ':logits', ('output', 'logits', 'value'))
            assert ui.document.validation()['valid']
            key = ui.workbench_panel.open_call([call])
            ui.workbench_panel.set_layout('columns')
            ui.workbench.move_to_pane(key, 'secondary')
            ui._render_graph()
            ui.workbench_panel.activate_view('root')
            saved_views = deepcopy(ui.workbench.to_payload())
            saved_model = ui.document.semantic_key()
            target = directory / 'hardware-model-workspace.json'
            invoke('native_json_save', {'file_path_name': str(target)})
            dpg.hide_item('native_json_save')
            stored = json.loads(target.read_text(encoding='utf-8'))
            assert stored['workspace_version'] == 3
            assert stored['model']['resource_profile'] == 'hardware-v5'
            invoke('native_json_open', {'file_path_name': str(target)})
            dpg.hide_item('native_json_open')
            assert ui.workbench.to_payload() == saved_views
            assert ui.document.semantic_key() == saved_model
            report['hardware_workspace_reopen'] = {'workspace_version': 3, 'views_and_model_identity_equal': True}
            capture('default-squeezenet-model-and-chip')
            for width, height, prefix in ((1100, 760, 'minimum'), (1380, 860, 'default')):
                dpg.set_viewport_width(width)
                dpg.set_viewport_height(height)
                ui.last_size = None
                _frames(ui, 4)
                capture(prefix + '-squeezenet-split')
            ui.document.replace_graph(empty_graph(image=False))
            ui._reset_canvas_document()
            ui._refresh()
            invoke('native_catalog_category', '本地芯片')
            identifier = ui.catalog_panel.visible_ids[0]
            call = drop(identifier)
            link('input:a', ('call', call, 'a'))
            link('input:b', ('call', call, 'b'))
            link('call:' + call + ':features', ('output', 'features', 'value'))
            assert ui.document.validation()['valid']
            capture('default-local-extension-connected')
            report['local_extension'] = {'valid': True, 'inputs': ['a', 'b'], 'outputs': ['features'],
                                         'damaged_file_diagnostic': any('broken.json' in issue for issue in ui.catalog_panel.local_index.diagnostics)}
            from studio_core.runtime import build_model
            from studio_core.python_io import import_python
            import torch
            torch.set_num_threads(1)
            graph = deepcopy(ui.document.graph)
            source = ui.document.export_python()
            imported = import_python(source)
            x, y = torch.arange(8, dtype=torch.float32).reshape(2, 4), torch.ones(2, 4)
            expected = .5 * x + .5 * y
            actual = build_model(graph, capture=False)(a=x, b=y)
            torch.testing.assert_close(actual['features'], expected)
            torch.testing.assert_close(build_model(imported, capture=False)(a=x, b=y)['features'], expected)
            report['local_extension']['actual_output_and_python_reimport'] = True
            report['status'] = 'passed'
            (output / 'frame-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(json.dumps({'status': 'passed', 'frames': len(report['frames']), 'callbacks': len(report['callbacks']), 'drops': len(report['drops'])}))
        finally:
            ui.shutdown()


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
