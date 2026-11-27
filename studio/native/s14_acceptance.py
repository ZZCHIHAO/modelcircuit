"""Exercise S14 through desktop callbacks and real local workers."""

import csv
import json
from pathlib import Path
import shutil
import sys
from time import monotonic


def check_s14(manager, directory, resources, output, device):
    import dearpygui.dearpygui as dpg

    from studio_core.cohort_examples import table_examples, volume_examples
    from studio_core.experiment_spec import model_digest
    from studio_core.project_runner import compare_project_runs, read_project_run
    from studio_core.research_project import bind_project, read_project
    from .cohort_panel import IMAGE_SOURCES, TABLE_TASKS, VOLUME_FORMATS
    from .data_panel import MODES
    from .document import Document
    from .render_acceptance import _capture, _frames, _wait_ui
    from .ui import NativeWorkbench

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ui = NativeWorkbench(Document(), manager, directory, resources)
    runs = []
    try:
        ui.build()
        _frames(ui)

        def invoke(tag, value=None):
            dpg.get_item_callback(tag)(tag, value, dpg.get_item_user_data(tag))

        def choose(tag, value):
            dpg.set_value(tag, value)
            invoke(tag, value)

        def update(values):
            for key, value in values.items():
                dpg.set_value('native_cohort_' + key, value)

        def until(condition, label, timeout=75):
            deadline = monotonic() + timeout
            while not condition() and monotonic() < deadline:
                _frames(ui, 1)
            assert condition(), (label, dpg.get_value('native_data_error'), ui.messages[-8:])

        def task(kind):
            choose('native_task', next(label for label, mode in MODES.items() if mode == kind))
            until(lambda: ui.capabilities is not None and device in ui.available_devices, 'available devices')

        def select_data(path):
            invoke('native_data_settings')
            invoke('native_data_open', {'file_path_name': str(path)})
            until(lambda: dpg.get_value('native_data_error').startswith('已读取'), 'CSV columns')
            update({'split_column': 'split'})

        def inspect_and_create(*, inspect=True):
            if inspect:
                invoke('native_data_inspect')
                until(lambda: ui.manifest is not None or not ui.inspecting, 'data inspection')
                assert ui.manifest is not None, dpg.get_value('native_data_error')
            invoke('native_data_apply')
            _frames(ui)
            invoke('native_starter')
            assert ui.pending_starter is not None, ui.messages[-5:]
            invoke('native_starter_confirm')
            assert ui.document.validation()['valid'], ui.document.validation()

        def train(mode):
            choose('native_device', mode)
            dpg.set_value('native_epochs', 1)
            dpg.set_value('native_batch', 2)
            invoke('native_run')
            record = _wait_ui(ui)
            assert record['status'] == 'succeeded', (record.get('error'), ui.messages[-6:])
            assert record['result']['device'] == mode
            return record['result']

        def export_and_run(scenario, csv_file, mode, result, *, keep_weights=False):
            folder = output / (scenario + '-project')
            dpg.set_value('native_project_export_path', str(folder))
            dpg.set_value('native_project_include_weights', keep_weights)
            invoke('native_project_export')
            until(lambda: (folder / 'project.json').is_file() and ui.project_panel._operation is None, 'export')
            portable = read_project(folder)
            assert portable['model_digest'] == model_digest(ui.document.graph)
            moved = output / (scenario + '-moved')
            shutil.copytree(csv_file.parent, moved)
            dpg.set_value('native_project_path', str(portable['path']))
            invoke('native_project_open')
            until(lambda: ui.project_panel.project is not None, 'open exported project')
            dpg.set_value('native_project_data', str(moved / csv_file.name))
            invoke('native_project_bind')
            until(lambda: ui.project_panel.binding is not None, 'bind moved data')
            config, manifest = bind_project(portable, moved / csv_file.name)
            assert manifest['counts'] == result['dataset']['counts']
            if config['data']['kind'] == 'multimodal_csv':
                assert config['input_ports'] == result['probe_context']['input_ports']
            dpg.set_value('native_project_device', mode)
            invoke('native_project_run')
            until(lambda: ui.project_panel.record and ui.project_panel.record['status'] in
                  {'succeeded', 'failed', 'cancelled'}, 'train moved project')
            record = ui.project_panel.record
            assert record['status'] == 'succeeded', (record.get('error'), ui.messages[-6:])
            report = read_project_run(record['result']['artifacts']['project_run'], verify=True)
            assert not report['changed_artifacts'] and not report['missing_artifacts']
            actual = report['artifacts']['predictions'].read_bytes()
            expected = Path(result['artifacts']['predictions']).read_bytes()
            assert actual == expected, (scenario, mode, 'predictions differ after move')
            if 'cv_folds' in report['artifacts']:
                assert (report['artifacts']['cv_folds'].read_bytes()
                        == Path(result['artifacts']['cv_folds']).read_bytes())
            return report

        dpg.set_value('native_right_tabs', dpg.get_alias_id('native_experiment_tab'))
        if device == 'cpu':
            table = table_examples(directory / 'tables')
            task('table_csv')
            for task_label, table_task in TABLE_TASKS.items():
                select_data(table)
                update({'task': task_label, 'target_column': 'target' if table_task == 'regression' else 'label',
                        'folds': 3, 'numeric_columns': 'age, score', 'categorical_columns': 'site'})
                inspect_and_create()
                result = train('cpu')
                evidence = json.loads(Path(result['artifacts']['cv_folds']).read_text('utf-8'))
                with table.open(encoding='utf-8', newline='') as stream:
                    rows = {row['sample_id']: row for row in csv.DictReader(stream)}
                for fold in evidence['folds']:
                    training = {rows[sample]['patient_id'] for sample in fold['train_ids']}
                    validation = {rows[sample]['patient_id'] for sample in fold['val_ids']}
                    assert training and validation and not training & validation
                    assert 'test_only' not in fold['preprocessing']['categorical_vocabulary']['site']
                assert result['algorithm'].startswith('RandomForest') and result['metrics']['test']['count'] == 4
                assert '分组交叉验证' in dpg.get_value('native_final_metrics')
                _capture(ui, output / (table_task + '-results.png'))
                report = export_and_run('table-' + table_task, table, 'cpu', result)
                runs.append({'scenario': 'table-' + table_task, 'device': 'cpu', 'folds': report['folds'],
                             'test_count': result['metrics']['test']['count'], 'rebound_predictions': True})

        task('multimodal_csv')
        ct = volume_examples(directory / 'ct')
        mri = volume_examples(directory / 'mri', mri=True)
        for scenario, path in (('ct-tabular', ct), ('mri-tabular', mri), ('ct-slice-volume', ct)):
            is_mri = path == mri
            is_slice = scenario == 'ct-slice-volume'
            select_data(path)
            update({'has_volume': True, 'has_image': is_slice, 'has_tabular': not is_slice,
                    'port_volume': 'volume', 'port_image': 'image', 'port_tabular': 'clinical',
                    'volume_size_D': 4, 'volume_size_H': 8, 'volume_size_W': 8, 'spacing_D': 2.,
                    'image_size_H': 8, 'image_size_W': 8, 'feature_columns': 'age, score',
                    'volume_format': next(label for label, fmt in VOLUME_FORMATS.items() if (fmt == 'mri_nifti') == is_mri),
                    'image_source': next(label for label, source in IMAGE_SOURCES.items()
                                         if source == ('center_slice' if is_slice else 'raster')),
                    'sequence_2': 'flair' if is_mri else '（不使用）', 'registered': is_mri})
            inspect_and_create()
            result = train(device)
            expected = {'volume', 'image' if is_slice else 'tabular'}
            assert set(result['probe_context']['input_ports']) == expected
            assert result['dataset']['input_shapes']['volume'] == [None, 2 if is_mri else 1, 4, 8, 8]
            _capture(ui, output / (scenario + '-canvas.png'))
            report = export_and_run(scenario, path, device, result, keep_weights=True)
            comparison = None
            if scenario == 'ct-tabular':
                invoke('native_data_settings')
                update({'port_volume': ''})
                inspect_and_create(inspect=False)
                baseline = train(device)
                assert set(baseline['probe_context']['input_ports']) == {'tabular'}
                comparator = export_and_run('ct-tabular-baseline', path, device, baseline)
                comparison = compare_project_runs(report, comparator)
                assert comparison['comparable'], comparison
            runs.append({'scenario': scenario, 'device': device, 'modalities': sorted(expected),
                         'test_count': result['metrics']['test']['count'], 'rebound_predictions': True,
                         **({'single_and_fusion_compare': True} if comparison else {})})

        if device == 'cpu':
            # Exercise the example chooser as it appears in the product.
            dpg.set_value('native_project_example_kind', 'MRI 与表格融合')
            invoke('native_project_picker_example', {'file_path_name': str(output)})
            until(lambda: ui.project_panel.binding is not None and ui.project_panel._operation is None
                  and ui._configuration()['data'].get('volume_format') == 'mri_nifti', 'example creation')
            assert ui.document.validation()['valid']
        return {'status': 'passed', 'frozen': bool(getattr(sys, 'frozen', False)), 'device': device,
                'synthetic_data': True, 'workflows': runs}
    finally:
        ui.shutdown()
