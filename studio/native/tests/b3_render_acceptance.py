"""Application-owned B3 frames, rectangles and registered callback evidence.

Run with studio + backend on PYTHONPATH. No OS input, browser or training.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import dearpygui.dearpygui as dpg

from native.catalog_panel import screen_rect
from native.document import Document
from native.paths import resource_root
from native.render_acceptance import _capture, _frames
from native.ui import NativeWorkbench
from native.view_document import resolve_view


def run(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        ui = NativeWorkbench(Document(), None, directory, resource_root())
        ui.build()
        report = {'frames': [], 'callbacks': [], 'drops': [], 'input': 'application callbacks and framebuffers only',
                  'exact_nonzero_pan_zoom_verified': False}
        def invoke(tag, data=None):
            callback = dpg.get_item_callback(tag)
            assert callback
            result = callback(tag, data, dpg.get_item_user_data(tag))
            report['callbacks'].append(tag)
            return result
        def frame(name):
            _frames(ui, 4)
            tags = ['native_search', 'native_tree_scroll', 'native_catalog_scroll', 'native_canvas_body',
                    'native_layout_single', 'native_layout_columns', 'native_layout_rows', 'native_view_picker_primary']
            if dpg.does_item_exist('native_view_picker_secondary') and ui.workbench.layout != 'single':
                tags.append('native_view_picker_secondary')
            rects = {tag: {'origin': screen_rect(tag)[0], 'size': screen_rect(tag)[1]} for tag in tags}
            for tag in ['native_search', 'native_layout_rows', 'native_view_picker_primary']:
                rect = rects[tag]
                assert rect['origin'][0] + rect['size'][0] <= dpg.get_viewport_client_width() + 2
                assert rect['origin'][1] + rect['size'][1] <= dpg.get_viewport_client_height() + 2
            _capture(ui, output / (name + '.png'))
            report['frames'].append({'file': name + '.png', 'viewport': [dpg.get_viewport_client_width(), dpg.get_viewport_client_height()],
                                     'layout': ui.workbench.layout, 'focused_view_id': ui.workbench.focused_view_id, 'rectangles': rects})
        try:
            key = ui.workbench_panel.open_call(['encode_a'])
            ui.workbench_panel.activate_view('root')
            for width, height, prefix in ((1380, 860, 'default'), (1100, 760, 'minimum')):
                dpg.set_viewport_width(width); dpg.set_viewport_height(height)
                ui.last_size = None; _frames(ui, 4)
                for mode in ('single', 'columns', 'rows'):
                    invoke('native_layout_' + mode)
                    if mode != 'single':
                        ui.workbench.move_to_pane(key, 'secondary'); ui._render_graph()
                    ui.workbench_panel.activate_view('root')
                    frame(prefix + '-' + mode)
                    # Registered drop target uses a measured point in this
                    # exact editor at this viewport and layout.
                    view = ui.canvas_host.get(key if mode != 'single' else 'root')
                    origin, size = screen_rect(view.container_tag)
                    point = [origin[0] + min(60, size[0] / 2), origin[1] + min(45, size[1] / 2)]
                    anchor = next(iter(view.ref_nodes.values()))
                    anchor_rect = dpg.get_item_rect_min(anchor)
                    anchor_local = dpg.get_item_pos(anchor)
                    expected = ui._document_position((point[0] - anchor_rect[0] + anchor_local[0], point[1] - anchor_rect[1] + anchor_local[1]))
                    drop = dpg.get_item_configuration(view.container_tag)['drop_callback']
                    token = ui.canvas_host.token(view.view_id)
                    count = len(resolve_view(ui.document, ui.workbench.views[view.view_id])[0]['calls'])
                    with patch('native.catalog_panel.dpg.get_mouse_pos', return_value=point):
                        assert drop(view.container_tag, {'version': 1, 'component_id': 'Identity', 'config': {}}, token)
                    assert len(ui.document.scope['calls']) == count + 1
                    selected = ui.document.selected
                    actual = dict(ui.document.selected_call['position'])
                    assert abs(actual['x'] - expected[0]) < 2 and abs(actual['y'] - expected[1]) < 2
                    _frames(ui, 3)
                    placed_rect = dpg.get_item_rect_min(ui.canvas_host.get(view.view_id).ref_nodes[selected])
                    assert all(abs(placed_rect[i] - point[i]) <= 2 for i in (0, 1)), (prefix, mode, point, placed_rect, expected, actual)
                    report['drops'].append({'frame': prefix + '-' + mode, 'point': point, 'expected_position': expected,
                                            'actual_position': actual, 'rendered_node_origin': placed_rect, 'view_id': view.view_id})
                    report['callbacks'].append('native drop ' + prefix + '-' + mode)
            # Real add / parameter editing / connect callbacks followed by
            # back / close retain the same model and view ownership contracts.
            ui.workbench_panel.activate_view('root')
            invoke('native_catalog_add_primitive.Linear')
            ref = ui.document.selected
            invoke(ui.param_widgets['out_features'], '12')
            assert ui.document.selected_instance['params']['out_features'] == 12
            view = ui.canvas_host.get('root')
            source = next(item for item, value in view.source_attributes.items() if value == 'input:a')
            target = next(item for item, value in view.target_attributes.items() if value == ('call', ref[5:], 'x'))
            invoke(view.editor_tag, [source, target])
            assert ui.document.selected_call['inputs']['x'] == 'input:a'
            # Long tree scrolling cannot move the search/category controls.
            for _ in range(32):
                ui.document.add('Identity')
            ui._refresh(); ui.workbench_panel.set_layout('single')
            search_before = screen_rect('native_search')
            dpg.set_y_scroll('native_tree_scroll', 99999)
            _frames(ui, 3)
            assert screen_rect('native_search') == search_before
            frame('minimum-long-tree')
            # Sixteen long-title definitions stay available via native tab
            # overflow and a complete per-pane picker; last item is clickable.
            graph = deepcopy(ui.document.graph)
            template = graph['definitions'][0]
            for i in range(14):
                item = deepcopy(template); item['id'] = 'long_definition_' + str(i)
                item['name'] = '用于验证超长名称页签仍可达的功能芯片 ' + str(i)
                graph['definitions'].append(item)
            ui.document.replace_graph(graph); ui._reset_canvas_document()
            for definition in graph['definitions']:
                ui.workbench_panel.open_definition(definition['id'])
            items = dpg.get_item_configuration('native_view_picker_primary')['items']
            invoke('native_view_picker_primary', items[-1])
            assert ui.workbench.focused_view_id == 'definition_long_definition_13'
            frame('minimum-tab-overflow')
            # A valid deep call path is explicit in the inspector.
            graph = deepcopy(Document().graph)
            leaf = graph['definitions'][0]
            for identifier, child in (('middle_chip', 'encoder_chip'), ('outer_chip', 'middle_chip')):
                graph['definitions'].append({'id': identifier, 'version': '1.0.0', 'public_parameters': [], 'bindings': {},
                    'graph': {'inputs': [{'id': 'x', 'shape': [None, 4], 'dtype': 'float32'}],
                              'instances': [{'id': 'nested', 'kind': 'chip', 'definition_id': child, 'params': {}}],
                              'calls': [{'id': 'inside', 'instance_id': 'nested', 'inputs': {'x': 'input:x'}}],
                              'outputs': [{'id': 'out', 'source': 'call:inside'}]}})
            graph['instances'][0]['definition_id'] = 'outer_chip'; graph['instances'][0]['params'] = {}
            ui.document.replace_graph(graph); ui._reset_canvas_document()
            key = ui.workbench_panel.open_call(['encode_a', 'inside', 'inside'])
            ui.canvas_host.apply_selection(key, ['call:project'], intentional=True)
            ui._focus_document(key); ui._render_inspector()
            widget = ui.param_widgets['out_features']; dpg.set_value(widget, 'not yet submitted')
            for mode in ('columns', 'rows', 'single'):
                ui.workbench_panel.set_layout(mode); _frames(ui, 3)
                assert widget == ui.param_widgets['out_features']
                assert dpg.get_value(widget) == 'not yet submitted'
            frame('minimum-deep-chip')
            ui._back(); report['callbacks'].append('legacy back')
            assert key in ui.workbench.views
            ui.workbench_panel.close_view(key); report['callbacks'].append('close view')
            report['status'] = 'passed'
            (output / 'frame-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'status': report['status'], 'frames': len(report['frames']), 'callbacks': len(report['callbacks'])}))
        finally:
            ui.shutdown()


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
