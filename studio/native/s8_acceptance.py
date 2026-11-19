"""Run the segmentation workflow through the native workbench and real worker."""
import csv
from pathlib import Path
import sys
from time import monotonic


def _fixture(directory, volume):
    import nibabel as nib
    import numpy as np
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / 'pairs.csv'
    shape = (13, 11, 5) if volume else (11, 13)
    affine = np.array([[0, .75, 0, 41], [.5, 0, 0, -7], [0, 0, -2, 13], [0, 0, 0, 1.]])
    suffix = '.nii.gz' if volume else '.png'
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('sample_id', 'patient_id', 'image_path', 'mask_path', 'split'))
        for index, split in enumerate(('train', 'train', 'val', 'test', 'test')):
            mask = np.zeros(shape, dtype=np.uint8)
            mask[2 + index % 3:9, 3:9] = 255
            image = np.random.default_rng(index).integers(0, 40, shape, dtype=np.uint8) + (mask // 2)
            names = [f'{kind}_{index}{suffix}' for kind in ('image', 'mask')]
            for name, data in zip(names, (image, mask)):
                if volume:
                    sample = nib.Nifti1Image(data, affine)
                    sample.header.set_xyzt_units('mm')
                    nib.save(sample, directory / name)
                else:
                    Image.fromarray(data).save(directory / name)
            writer.writerow((f'sample_{index}', f'group_{index}', *names, split))
    return csv_path


def check_s8(manager, directory, resources, output, device='cpu'):
    import dearpygui.dearpygui as dpg
    import nibabel as nib
    import numpy as np
    from PIL import Image

    from .data_panel import MODES, SEGMENTATION_FORMATS
    from .document import Document
    from .render_acceptance import _capture, _frames, _wait_ui
    from .ui import NativeWorkbench
    from studio_core.experiment_spec import model_digest
    from studio_core.python_io import export_python, import_python

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    export = output / '掩码导出'
    export.mkdir(exist_ok=True)
    ui = NativeWorkbench(Document(), manager, directory, resources)
    evidence = []
    try:
        ui.build()
        _frames(ui)

        def invoke(tag, data=None):
            callback = dpg.get_item_callback(tag)
            return callback(tag, data, dpg.get_item_user_data(tag))

        def until(condition, label):
            deadline = monotonic() + 25
            while not condition() and monotonic() < deadline:
                _frames(ui, 1)
            assert condition(), (label, ui.messages[-5:])

        def choose(tag, value):
            dpg.set_value(tag, value)
            invoke(tag, value)

        dpg.set_value('native_right_tabs', dpg.get_alias_id('native_experiment_tab'))
        mode = next(label for label, kind in MODES.items() if kind == 'segmentation_csv')
        choose('native_task', mode)
        until(lambda: ui.capabilities is not None and device in ui.available_devices, 'device query')
        for volume in (False, True):
            format_name = 'nifti' if volume else 'image'
            path = _fixture(directory / ('示例体' if volume else '示例图'), volume)
            invoke('native_data_settings')
            invoke('native_data_open', {'file_path_name': str(path)})
            until(lambda: dpg.get_value('native_data_error').startswith('已读取'), 'CSV columns')
            choose('native_segmentation_format', next(label for label, fmt in SEGMENTATION_FORMATS.items() if fmt == format_name))
            choose('native_seg_split_column', 'split')
            dpg.set_value('native_segmentation_values', '0, 255')
            dpg.set_value('native_image_height', 9)
            dpg.set_value('native_image_width', 8)
            if volume:
                dpg.set_value('native_segmentation_depth', 4)
            invoke('native_data_inspect')
            until(lambda: ui.manifest is not None, 'data inspection')
            assert ui.manifest['counts'] == {'train': 2, 'val': 1, 'test': 2}
            assert ui.manifest['input_shape'] == ([None, 1, 4, 9, 8] if volume else [None, 1, 9, 8])
            if volume:
                _capture(ui, output / 's8-data-settings.png')
            invoke('native_data_apply')
            invoke('native_starter')
            invoke('native_starter_confirm')
            assert ui.document.validation()['valid']
            graph = ui.document.graph
            assert model_digest(graph) == model_digest(import_python(export_python(graph)))
            choose('native_device', device)
            dpg.set_value('native_epochs', 1)
            dpg.set_value('native_batch', 2)
            invoke('native_run')
            record = _wait_ui(ui)
            assert record['status'] == 'succeeded', (record.get('error'), ui.messages[-5:])
            result = record['result']
            test = result['metrics']['test']
            assert result['device'] == device and test['count'] == 2 and result['completed_epochs'] == 1
            assert any(parameter['update_norm'] > 0 for parameter in result['parameters'])
            assert 0 <= test['foreground_dice'] <= 1 and 0 <= test['foreground_iou'] <= 1
            assert dpg.does_item_exist('native_segmentation_image'), dpg.get_value('native_segmentation_result')
            assert len(ui.segmentation_panel.rows) == 2
            invoke('native_segmentation_next')
            assert ui.segmentation_panel.index == 1 and 'sample_4' in dpg.get_value('native_segmentation_result')
            invoke('native_segmentation_export')
            invoke('native_segmentation_export_dialog', {'file_path_name': str(export)})
            copied = next(export.glob('*000002' + ('.nii.gz' if volume else '.png')))
            if volume:
                source = nib.load(path.parent / 'image_4.nii.gz')
                prediction = nib.load(copied)
                assert prediction.shape == source.shape
                np.testing.assert_allclose(prediction.affine, source.affine)
                assert prediction.header.get_xyzt_units()[0] == 'mm'
                assert set(np.unique(np.asarray(prediction.dataobj))) <= {0, 255}
            else:
                with Image.open(copied) as prediction:
                    assert prediction.size == (13, 11) and set(np.unique(prediction)) <= {0, 255}
            _capture(ui, output / (format_name + '-segmentation.png'))
            workspace = output / (format_name + '-workspace.json')
            invoke('native_json_save', {'file_path_name': str(workspace)})
            invoke('native_json_open', {'file_path_name': str(workspace)})
            configuration = ui._configuration()
            assert configuration['profile'] == 'segmentation-v1' and configuration['data']['mask_values'] == [0, 255]
            assert configuration['data']['format'] == format_name and configuration['device'] == device
            evidence.append({'format': format_name, 'model_grid': result['dataset']['input_shape'], 'test': test,
                             'python_round_trip': True, 'original_grid_export': True, 'workspace_reopen': True,
                             'measured_updates': True, 'visual_preview': True, 'device': result['device']})
        return {'status': 'passed', 'frozen': bool(getattr(sys, 'frozen', False)), 'device': device, 'cases': evidence}
    finally:
        ui.shutdown()
