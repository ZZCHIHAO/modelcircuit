"""Application frame and actual side-pin evidence for completed wires."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import dearpygui.dearpygui as dpg
from native.document import Document
from native.paths import resource_root
from native.render_acceptance import _capture, _frames
from native.ui import NativeWorkbench


def run(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    report = {'frames': [], 'port_nodes': [], 'native_link_ids': [], 'os_gestures': False,
              'preview': 'Native temporary connection preview remains visible; completed connections are orthogonal',
              'minimap_exclusion': 'Conservative configured-corner rectangle:25% editor size +16px; DPG has no exact public minimap rect'}
    with tempfile.TemporaryDirectory() as directory:
        ui = NativeWorkbench(Document(), None, directory, resource_root()); ui.build()
        try:
            def capture(name):
                _frames(ui, 4)
                wire_records = []
                for key, view in ui.canvas_host.views.items():
                    if not view.visible:
                        assert not view.edge_overlay.segments
                        continue
                    for identifier, segments in view.edge_overlay.segments.items():
                        assert identifier in view.link_targets
                        assert all(a[0] == b[0] or a[1] == b[1] for a, b in segments)
                        wire_records.append({'native_link_id': identifier, 'view_id': key, 'segments': segments})
                _capture(ui, output / (name + '.png'))
                report['frames'].append({'file': name + '.png', 'wires': wire_records,
                                         'viewport': [dpg.get_viewport_client_width(), dpg.get_viewport_client_height()]})
            capture('orthogonal-default-root')
            graph = deepcopy(ui.document.graph)
            definition = graph['definitions'][0]
            definition['output_mode'] = 'named'
            definition['graph']['inputs'].append({'id': 'extra', 'shape': [None, 4], 'dtype': 'float32'})
            definition['graph']['outputs'] = [{'id': 'features', 'source': 'call:activate'}, {'id': 'projected', 'source': 'call:project'}]
            for call in graph['calls']:
                if call['id'] in ('encode_a', 'encode_b'):
                    call['inputs']['extra'] = 'input:b'
            graph['calls'][2]['inputs'] = {'a': 'call:encode_a:features', 'b': 'call:encode_b:projected'}
            ui.document.replace_graph(graph); ui._reset_canvas_document(); _frames(ui, 4)
            view = ui.canvas_host.get('root')
            node = view.ref_nodes['call:encode_a']
            input_pins = [(value[2], ui._socket_position(attribute, output=False)) for attribute, value in view.target_attributes.items() if value[:2] == ('call', 'encode_a')]
            output_pins = [(value.rsplit(':', 1)[1], ui._socket_position(attribute, output=True)) for attribute, value in view.source_attributes.items() if value.startswith('call:encode_a:')]
            assert {p[0] for p in input_pins} == {'x', 'extra'}
            assert {p[0] for p in output_pins} == {'features', 'projected'}
            assert all(p[1][0] < output_pins[0][1][0] for p in input_pins)
            report['port_nodes'].append({'node': 'call:encode_a', 'inputs': input_pins, 'outputs': output_pins, 'node_rect': ui._node_rectangle(node)})
            capture('orthogonal-multiport-root')
            ui.catalog_panel.open_config('primitive.Linear'); _frames(ui, 3)
            assert not any(v.edge_overlay.segments for v in ui.canvas_host.views.values())
            capture('orthogonal-config-modal')
            ui.catalog_panel.cancel_config()
            _frames(ui, 2)
            ui.catalog_panel.place({'version': 1, 'component_id': 'fusion.weighted_sum', 'config': {}}, ui.canvas_host.token('root'))
            _frames(ui, 3)
            assert not any(v.edge_overlay.segments for v in ui.canvas_host.views.values()), ui.messages[-3:]
            capture('orthogonal-policy-modal')
            ui.catalog_panel.cancel_policy()
            key = ui.workbench_panel.open_call(['encode_a'])
            for width, height, prefix in ((1380, 860, 'default'), (1100, 760, 'minimum')):
                dpg.set_viewport_width(width); dpg.set_viewport_height(height); _frames(ui, 4)
                for layout in ('columns', 'rows'):
                    ui.workbench_panel.set_layout(layout)
                    ui.workbench.move_to_pane(key, 'secondary'); ui._render_graph()
                    capture('orthogonal-' + prefix + '-' + layout)
            # Native editor create-link callback is still the authoritative
            # contract; invoking it updates the same scope and redraws wires.
            ui.workbench_panel.set_layout('single')
            ui.workbench_panel.activate_view('root'); _frames(ui, 3)
            view = ui.canvas_host.get('root')
            source = next(a for a, value in view.source_attributes.items() if value == 'input:a')
            target = next(a for a, value in view.target_attributes.items() if value == ('call', 'fuse', 'a'))
            callback = dpg.get_item_callback(view.editor_tag)
            callback(view.editor_tag, [source, target], ui.canvas_host.token('root'))
            assert next(c for c in ui.document.graph['calls'] if c['id'] == 'fuse')['inputs']['a'] == 'input:a'
            _frames(ui, 4)
            view = ui.canvas_host.get('root')
            link, segments = next((key, value) for key, value in view.edge_overlay.segments.items() if value)
            a, b = max(segments, key=lambda segment: abs(segment[0][0] - segment[1][0]) + abs(segment[0][1] - segment[1][1]))
            point = [(a[i] + b[i]) / 2 for i in (0, 1)]
            assert ui._select_orthogonal_wire(ui.canvas_host.token('root'), point) == link
            capture('orthogonal-selected-wire')
            ui._delete_selected(); _frames(ui, 3)
            report['native_link_ids'].append(link)
            report['status'] = 'passed'
            (output / 'orthogonal-frame-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({'status': 'passed', 'frames': len(report['frames']), 'port_nodes': len(report['port_nodes'])}))
        finally:
            ui.shutdown()


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
