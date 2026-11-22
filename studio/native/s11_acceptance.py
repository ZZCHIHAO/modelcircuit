"""Acceptance via native controls, real workers and small generated data."""
from pathlib import Path
from time import monotonic
from unittest.mock import patch


def check_s11(manager, directory, resources, output, device='cpu'):
    import dearpygui.dearpygui as dpg
    from PIL import Image

    from studio_core.experiment_spec import default_classification_config, model_digest
    from studio_core.project_examples import example_graph as image_graph
    from studio_core.python_io import export_python, import_python
    from .document import Document
    from .render_acceptance import _capture, _frames
    from .ui import NativeWorkbench

    output = Path(output)
    ui = NativeWorkbench(Document(), manager, directory, resources)
    try:
        ui.build()

        def wait(condition, label):
            deadline = monotonic() + 50
            while not condition() and monotonic() < deadline:
                _frames(ui, 1)
            assert condition(), (label, ui.messages[-3:])

        def invoke(tag, data=None):
            return dpg.get_item_callback(tag)(tag, data, dpg.get_item_user_data(tag))

        def wire(source, target):
            view = ui.canvas_host.get('root')
            from_attr = next(attribute for attribute, ref in view.source_attributes.items() if ref == source)
            to_attr = next(attribute for attribute, ref in view.target_attributes.items() if ref == ('call', *target))
            invoke(view.editor_tag, [from_attr, to_attr])

        def probe(source, view_id='root'):
            view = ui.canvas_host.get(view_id)
            attribute = next(attr for attr, ref in view.source_attributes.items() if ref == source)
            button = dpg.get_item_children(attribute, 1)[0]
            invoke(button)

        def run(mode):
            before = (ui.debug_panel.record or {}).get('run_id')
            invoke('native_debug_' + mode)
            assert ui.debug_panel.record and ui.debug_panel.record['run_id'] != before, (mode, ui.messages[-3:])
            wait(lambda: ui.debug_panel.record['status'] in {'succeeded', 'failed', 'cancelled'}, mode)
            record = ui.debug_panel.record
            assert record['status'] == 'succeeded' and record['profile'] == 'visual-debug-v1', record.get('error')
            assert record['result']['device'] == device
            assert (manager.directory / (record['run_id'] + '.json')).is_file()
            return record['result']

        wait(lambda: device in ui.debug_panel.devices, 'device available')
        dpg.set_value('native_debug_device', device)
        before = ui.document.semantic_key()
        wire('input:b', ('fuse', 'b'))
        assert ui.document.semantic_key() == before
        assert 'fuse.b' in dpg.get_value('native_status') and '第 2 维' in dpg.get_value('native_status')
        wire('call:encode_b:out', ('fuse', 'b'))
        assert ui.document.validation()['valid']
        ui.document.set_frozen('head', True)
        ui._refresh()
        probe('call:encode_a:out')
        probe('call:encode_b:out')
        view_key = ui.workbench_panel.open_call(['encode_a'])
        _frames(ui, 2)
        probe('call:project:out', view_key)
        ui.workbench_panel.activate_view('root')
        assert ui.debug_panel.sources == ['call:encode_a:out', 'call:encode_b:out', 'call:encode_a/call:project:out']
        baseline = ui.document.semantic_key()
        forward = run('forward')
        assert forward['optimizer_steps'] == 0 and forward['probes']['call:encode_a/call:project:out']['activation']
        step = run('step')
        a, b = (step['probes'][f'call:encode_{name}:out'] for name in ('a', 'b'))
        assert a['instance'] == b['instance'] == 'encoder'
        assert a['activation']['values'] != b['activation']['values']
        assert a['gradient'] and b['gradient'] and step['optimizer_steps'] == 1
        assert any(row['update_norm'] > 0 for row in step['parameters'] if row['trainable'])
        assert all(row['update_norm'] == 0 for row in step['parameters'] if not row['trainable'])
        assert ui.document.semantic_key() == baseline
        restored = import_python(export_python(ui.document.graph))
        assert model_digest(restored) == baseline
        _capture(ui, output / 's11-probes.png')

        dpg.set_value('native_search', '相加融合')
        invoke('native_search', '相加融合')
        assert ui.catalog_panel.visible_ids == ['primitive.Add']
        view = ui.canvas_host.get('root')
        _frames(ui, 2)
        point = dpg.get_item_rect_min(view.ref_nodes['input:a'])
        with patch('native.catalog_panel.dpg.get_mouse_pos', return_value=(point[0] + 110, point[1] + 136)):
            dpg.get_item_configuration(view.container_tag)['drop_callback'](
                view.container_tag, ui.catalog_panel.payload('primitive.Add'), ui.canvas_host.token('root'))
        assert ui.document.selected_instance['kind'] == 'Add'
        ui._undo()
        assert ui.document.semantic_key() == baseline
        for query, identifier in [('Linear', 'primitive.Linear'), ('拼接融合', 'primitive.Concat'),
                                  ('ResNet18', 'backbone.resnet18'), ('ViT', 'vit.small_classifier'),
                                  ('SVM', 'primitive.SVC'), ('编码芯片', 'project:encoder_chip')]:
            dpg.set_value('native_search', query)
            invoke('native_search', query)
            assert identifier in ui.catalog_panel.visible_ids, (query, ui.catalog_panel.visible_ids)
        dpg.set_value('native_search', '')
        invoke('native_search', '')

        dataset = output / 'images'
        for index, split in enumerate(('train', 'val', 'test')):
            for label, value in [('dark', 20), ('bright', 160)]:
                folder = dataset / split / label
                folder.mkdir(parents=True)
                for image in range(2):
                    Image.new('L', (8, 8), value + index * 6 + image).save(folder / f'{image}.png')
        graph = image_graph()
        graph['instances'].append({'id': 'gain', 'kind': 'Scale', 'params': {'factor': 1.0}})
        graph['calls'].append({'id': 'gain', 'instance_id': 'gain', 'inputs': {'x': 'call:classes'},
                               'position': {'x': 890, 'y': 280}})
        graph['outputs'][0].update(source='call:gain:out', position={'x': 1100, 'y': 280})
        experiment = default_classification_config('image_folder')
        experiment['data'].update(path=str(dataset), split_mode='directories', image_size=[8, 8], channels=1)
        ui._install_portable_project({'graph': graph, 'workbench': None}, experiment, None)
        _frames(ui, 2)
        probe('call:gain:out')
        dpg.set_value('native_debug_data', '图像训练集')
        image_step = run('step')
        assert image_step['batch_kind'] == 'image_folder' and image_step['optimizer_steps'] == 1
        assert image_step['sample_count'] == 2 and image_step['dataset_digest']
        invoke('native_debug_select_sweep')
        dpg.set_value('native_debug_values', '1.0, 2.0')
        sweep = run('sweep')
        assert sweep['batch_digest'] == image_step['batch_digest']
        assert [item['value'] for item in sweep['sweep']] == [1.0, 2.0]
        assert all(row['loss'] is not None for row in sweep['sweep'])
        _capture(ui, output / 's11-sweep.png')
        return {'status': 'passed', 'device': device, 'rejected_mismatched_wire': True, 'selected_nested_probe': True,
                'forward_updates': forward['optimizer_steps'], 'train_updates': step['optimizer_steps'],
                'frozen_unchanged': True, 'shared_calls': 2, 'python_round_trip': True,
                'catalog_search_and_drop': True, 'image_step_updates': image_step['optimizer_steps'],
                'image_step_samples': image_step['sample_count'], 'sweep_trials': len(sweep['sweep'])}
    finally:
        ui.shutdown()
