"""Exercise the S14B desktop controls with real workers and small CSV files."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from time import monotonic


def check_s14b(manager, directory, resources, output, device):
    import dearpygui.dearpygui as dpg

    from studio_core.project_runner import read_project_run
    from .data_panel import MODES
    from .document import Document
    from .render_acceptance import _frames, _wait_ui
    from .ui import NativeWorkbench

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ui = NativeWorkbench(Document(), manager, directory, resources)
    try:
        ui.build()
        _frames(ui)

        def invoke(tag, value=None):
            dpg.get_item_callback(tag)(tag, value, dpg.get_item_user_data(tag))

        def choose(tag, value):
            dpg.set_value(tag, value)
            invoke(tag, value)

        def until(condition, label):
            deadline = monotonic() + 75
            while not condition() and monotonic() < deadline:
                _frames(ui, 1)
            assert condition(), (label, dpg.get_value('native_data_error'), ui.messages[-8:])

        def configure(path, kind):
            choose('native_task', next(label for label, mode in MODES.items() if mode == kind))
            invoke('native_data_open', {'file_path_name': str(path)})
            until(lambda: dpg.get_value('native_data_error').startswith('已读取'), 'CSV columns')
            if kind == 'csv':
                choose('native_csv_split', 'split')
            choose('native_standardize', False)
            invoke('native_data_inspect')
            until(lambda: ui.manifest is not None or not ui.inspecting, 'inspection')
            assert ui.manifest and ui.manifest['kind'] == kind, dpg.get_value('native_data_error')
            invoke('native_starter')
            invoke('native_starter_confirm')
            assert ui.document.validation()['valid']

        def run_probe():
            ui.debug_panel.add_probe('input:x')
            invoke('native_debug_step')
            until(lambda: ui.debug_panel.record and ui.debug_panel.record['status'] in
                  ('succeeded', 'failed', 'cancelled'), 'selected-sample step')
            assert ui.debug_panel.result, (dpg.get_value('native_debug_status'), ui.messages[-8:])
            return ui.debug_panel.result

        def move_project(name, dataset, run_device):
            folder = output / name
            dpg.set_value('native_project_export_path', str(folder))
            dpg.set_value('native_project_include_data', True)
            invoke('native_project_export')
            until(lambda: (folder / 'project.json').exists() and ui.project_panel._operation is None, 'export')
            moved = output / (name + '-moved')
            shutil.copytree(folder, moved)
            dataset.unlink()
            dpg.set_value('native_project_path', str(moved / 'project.json'))
            invoke('native_project_open')
            until(lambda: ui.project_panel.binding is not None, 'automatic data binding')
            assert ui.manifest and Path(ui._configuration()['data']['path']).is_relative_to(moved)
            dpg.set_value('native_project_device', run_device)
            invoke('native_project_run')
            until(lambda: ui.project_panel.record and ui.project_panel.record['status'] in
                  ('succeeded', 'failed', 'cancelled'), 'portable project run')
            record = ui.project_panel.record
            assert record['status'] == 'succeeded', record.get('error')
            return read_project_run(record['result']['artifacts']['project_run'], verify=True)

        classification = output / 'numeric.csv'
        classification.write_text('x,y,label,split\n' + ''.join(
            f"{i},{20+i},{i%2},{('train', 'val', 'test')[i//4]}\n" for i in range(12)), encoding='utf-8')
        configure(classification, 'csv')
        until(lambda: device in ui.available_devices, 'requested device')
        dpg.set_value('native_epochs', 1)
        choose('native_debug_data', '实验数据')
        dpg.set_value('native_debug_device', device)
        for index in (4, 2):
            dpg.set_value('native_debug_sample_index', index)
            invoke('native_debug_add_sample')
        observed = run_probe()
        assert observed['probes']['input:x']['activation']['values'] == [3.0, 23.0, 1.0, 21.0]
        invoke('native_debug_baseline')
        changed = deepcopy(ui.document.graph)
        activation = next(item for item in changed['instances'] if item['kind'] == 'ReLU')
        activation['kind'] = 'Identity'
        ui.document.replace_graph(changed)
        ui._reset_canvas_document()
        ui._refresh()
        run_probe()
        assert '同一批输入' in dpg.get_value('native_debug_comparison'), ui.messages[-8:]
        classification_report = move_project('classification', classification, device)
        assert classification_report['metrics']['test']['count'] == 4
        assert not classification_report['missing_artifacts'] and not classification_report['changed_artifacts']

        points = output / 'points.csv'
        points.write_text('u,v\n-4,0\n-3,1\n-5,0.5\n4,0\n3,-1\n5,-0.5\n', encoding='utf-8')
        configure(points, 'unsupervised_csv')
        invoke('native_run')
        record = _wait_ui(ui)
        assert record['status'] == 'succeeded', (record.get('error'), ui.messages[-8:])
        assert sorted(record['result']['analysis']['cluster_counts']) == [3, 3]
        assert dpg.get_item_children('native_analysis', 1)
        analysis_report = move_project('analysis', points, 'cpu')
        assert analysis_report['analysis']['cluster_counts'] == record['result']['analysis']['cluster_counts']
        assert (analysis_report['artifacts']['projection'].read_bytes()
                == Path(record['result']['artifacts']['projection']).read_bytes())
        return {'status': 'passed', 'frozen': bool(getattr(sys, 'frozen', False)), 'device': device,
                'probe_rows': observed['sample_ids'], 'portable_csv': 'passed', 'unsupervised': 'passed',
                'projection_reproduced': True}
    finally:
        ui.shutdown()
