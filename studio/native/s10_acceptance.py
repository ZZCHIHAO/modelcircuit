"""Native S10 workflow through DearPyGui callbacks and a real worker."""
from pathlib import Path
import shutil
import sys
from time import monotonic


def check_s10(manager, directory, resources, output, device='cpu'):
    import dearpygui.dearpygui as dpg

    from .document import Document
    from .render_acceptance import _capture, _frames
    from .s8_acceptance import _fixture
    from .ui import NativeWorkbench
    from studio_core.classification_data import inspect_dataset
    from studio_core.experiment_spec import default_segmentation_config
    from studio_core.project_runner import compare_project_runs, read_project_run
    from studio_core.research_project import read_project
    from studio_core.segmentation_templates import segmentation_graph

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    examples = output / '示例'
    examples.mkdir()
    ui = NativeWorkbench(Document(), manager, directory, resources)
    try:
        ui.build()
        dpg.set_value('native_right_tabs', dpg.get_alias_id('native_project_tab'))

        def invoke(name, app_data=None):
            tag = 'native_project_' + name
            return dpg.get_item_callback(tag)(tag, app_data, dpg.get_item_user_data(tag))

        def wait(predicate, message, timeout=85):
            deadline = monotonic() + timeout
            while not predicate() and monotonic() < deadline:
                _frames(ui, 1)
            if not predicate():
                raise AssertionError((message, ui.messages[-5:]))

        def run():
            dpg.set_value('native_project_device', device)
            invoke('run')
            wait(lambda: ui.project_panel.record and ui.project_panel.record['status'] in {'succeeded', 'failed', 'cancelled'}, 'run project')
            record = ui.project_panel.record
            assert record['status'] == 'succeeded', record.get('error')
            return read_project_run(record['result']['artifacts']['project_run'])

        def open_bind(project, data):
            invoke('picker_project', {'file_path_name': str(project)})
            invoke('open')
            wait(lambda: ui.project_panel.project and ui.project_panel.project['path'] == project, 'open project')
            tag = 'picker_data_directory' if data.is_dir() else 'picker_data_csv'
            invoke(tag, {'file_path_name': str(data)})
            invoke('bind')
            wait(lambda: ui.project_panel.binding is not None, 'rebind data')

        wait(lambda: device in ui.available_devices, 'available device')
        invoke('picker_example', {'file_path_name': str(examples)})
        wait(lambda: ui.project_panel.binding is not None, 'create and bind example')
        classification = run()
        assert classification['metrics']['test']['count'] == 4 and classification['device'] == device
        assert sum(call['instance_id'] == 'shared' for call in ui.document.graph['calls']) == 2
        saved = ui.project_panel.project['path']
        copied = output / '迁移示例'
        shutil.copytree(saved.parent.parent, copied)
        open_bind(copied / 'project' / 'project.json', copied / 'example-data')
        repeated = run()
        assert repeated['dataset_digest'] == classification['dataset_digest']
        assert repeated['local_dataset_digest'] != classification['local_dataset_digest']
        assert compare_project_runs(classification, repeated)['comparable']
        invoke('picker_right', {'file_path_name': str(classification['path'])})
        wait(lambda: ui.project_panel._reports['left'] and ui.project_panel._reports['right'], 'reopen comparable results')
        assert '一致' in dpg.get_value('native_project_compare_status')
        _capture(ui, output / 's10-run-history.png')

        volume_data = output / '原始影像'
        volume_manifest = _fixture(volume_data, True)
        import csv
        with volume_manifest.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        for row in rows:
            for key in ('image_path', 'mask_path'):
                row[key] = str((volume_data / row[key]).resolve())
        with volume_manifest.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        config = default_segmentation_config()
        config.update(epochs=1, batch_size=2, device=device, lr=.02)
        config['data'].update(path=str(volume_manifest), format='nifti', depth=5, image_size=[8, 8],
                              split_column='split', mask_values=[0, 255])
        manifest = inspect_dataset(config['data'], config['seed'])
        graph = segmentation_graph(manifest)
        ui._install_portable_project({'graph': graph, 'workbench': None}, config, manifest)
        project_dir = output / '分割工程'
        dpg.set_value('native_project_export_path', str(project_dir))
        dpg.set_value('native_project_include_weights', False)
        invoke('export')
        wait(lambda: not ui.project_panel._operation and (project_dir / 'project.json').is_file(), 'export segmentation project')
        project = read_project(project_dir)
        assert project['config']['profile'] == 'segmentation-v1' and project['config']['data']['path'] == ''
        assert (project_dir / 'run.py').is_file() and (project_dir / 'runtime' / 'studio_core' / 'project_runner.py').is_file()
        model = project['model_path']
        original_code = model.read_text(encoding='utf-8')
        edited_code = original_code.replace('kernel_size=3, padding=1', 'kernel_size=1, padding=0', 1)
        if edited_code == original_code:
            edited_code = original_code.replace('kernel_size=3, stride=1, padding=1', 'kernel_size=1, stride=1, padding=0', 1)
        assert edited_code != original_code, 'expected editable encoder convolution in exported Python'
        model.write_text(edited_code, encoding='utf-8')
        edited = read_project(project_dir)
        assert not edited['graph_error'] and edited['model_digest'] != edited['initial_model_digest']
        moved_volume = output / '影像新位置'
        shutil.copytree(volume_data, moved_volume)
        (moved_volume / volume_manifest.name).unlink()
        open_bind(project_dir / 'project.json', moved_volume)
        from studio_core.experiment_spec import model_digest
        assert model_digest(ui.document.graph) == edited['model_digest']
        segmentation = run()
        assert segmentation['profile'] == 'segmentation-v1' and segmentation['device'] == device
        assert segmentation['metrics']['test']['count'] == 2
        assert len(segmentation['output_sha256']) > 0
        assert all(path.is_file() for path in segmentation['artifacts'].values())
        assert not compare_project_runs(segmentation, repeated)['comparable']
        return {'status': 'passed', 'frozen': bool(getattr(sys, 'frozen', False)), 'device': device,
                'classification_test_samples': repeated['metrics']['test']['count'],
                'segmentation_test_samples': segmentation['metrics']['test']['count'],
                'relocated_data': True, 'relocated_absolute_csv_pairs': True, 'source_edit_restored': True, 'shared_layer': True,
                'run_reports_compared': True, 'standalone_source_exported': True}
    finally:
        ui.shutdown()
