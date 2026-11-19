"""Exercise saved-run inference through native callbacks and worker processes."""
import json
from pathlib import Path
import shutil
import sys
from time import monotonic


def check_s9(manager, directory, resources, output, device='cpu'):
    import dearpygui.dearpygui as dpg
    import nibabel as nib
    import numpy as np
    from PIL import Image

    from .document import Document
    from .inference_panel import MODES
    from .render_acceptance import _capture, _frames
    from .s8_acceptance import _fixture
    from .ui import NativeWorkbench
    from studio_core.classification_data import inspect_dataset
    from studio_core.experiment_spec import default_segmentation_config, model_digest
    from studio_core.segmentation_templates import segmentation_graph

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    exports = output / '原始掩码'
    exports.mkdir(exist_ok=True)
    ui = NativeWorkbench(Document(), manager, directory, resources)
    canvas = model_digest(ui.document.graph)
    cases = []
    try:
        ui.build()
        _frames(ui)
        dpg.set_value('native_right_tabs', dpg.get_alias_id('native_inference_tab'))

        def invoke(tag, data=None):
            callback = dpg.get_item_callback(tag)
            return callback(tag, data, dpg.get_item_user_data(tag))

        def wait(predicate, label, seconds=70):
            deadline = monotonic() + seconds
            while not predicate() and monotonic() < deadline:
                _frames(ui, 1)
            if not predicate():
                raise AssertionError((label, ui.messages[-5:]))

        def choose(tag, value):
            dpg.set_value(tag, value)
            invoke(tag, value)

        wait(lambda: device in ui.available_devices, 'device availability')
        for volume in (False, True):
            fmt = 'nifti' if volume else 'image'
            train_root = directory / ('训练' + fmt)
            inputs = directory / ('新影像' + fmt)
            inputs.mkdir(parents=True, exist_ok=True)
            manifest_path = _fixture(train_root, volume)
            cfg = default_segmentation_config()
            cfg.update(epochs=1, batch_size=2, lr=.02, device=device)
            cfg['data'].update(path=str(manifest_path), format=fmt, depth=8 if volume else 1,
                               image_size=[16, 16], split_column='split', mask_values=[0, 255])
            manifest = inspect_dataset(cfg['data'], cfg['seed'])
            graph = segmentation_graph(manifest)
            training_id = manager.submit(graph, cfg)['run_id']
            wait(lambda: manager.get(training_id)['status'] in {'succeeded', 'failed'}, 'prepare a checkpoint')
            completed = manager.get(training_id)
            assert completed['status'] == 'succeeded', completed.get('error')
            original = completed['result']['artifacts']
            # Train and infer through separate workers without modifying the canvas.
            manifest_path.unlink()
            if volume:
                shape = (43, 37, 15)
                affine = np.array([[0., -.75, 0., 83.], [.5, 0., 0., -11.], [0., 0., 2., 37.], [0., 0., 0., 1.]])
                for index in (1, 2):
                    coordinates = np.indices(shape, dtype=np.float32)
                    values = (coordinates[0] + 3 * coordinates[1] + index * coordinates[2]).astype(np.int16)
                    image = nib.Nifti1Image(values, affine)
                    image.header.set_xyzt_units('mm')
                    nib.save(image, inputs / f'影像{index}.nii.gz')
                source = inputs
            else:
                source = inputs / '未标注.png'
                shutil.copy2(train_root / 'image_3.png', source)
            invoke('infer_choose_run')
            invoke('infer_picker_run', {'file_path_name': original['run_spec']})
            invoke('infer_choose_folder' if volume else 'infer_choose_image')
            invoke('infer_picker_folder' if volume else 'infer_picker_image', {'file_path_name': str(source)})
            dpg.set_value('infer_checkpoint', '')
            mode = next(name for name, value in MODES.items() if value == ('sliding' if volume else 'fixed'))
            choose('infer_mode', mode)
            dpg.set_value('infer_device', device)
            invoke('infer_start')
            wait(lambda: ui.inference_panel.record and ui.inference_panel.record['status'] in {'succeeded', 'failed', 'cancelled'}, 'inference')
            record = ui.inference_panel.record
            assert record['status'] == 'succeeded', (record.get('error'), ui.messages[-5:])
            assert record['result']['device'] == device and model_digest(ui.document.graph) == canvas
            report = ui.inference_panel.report
            report_path = ui.inference_panel.path
            assert len(report['cases']) == (2 if volume else 1)
            assert dpg.does_item_exist('infer_image')
            if volume:
                target = report['cases'][1]
                choose('infer_case', '2. 影像2.nii.gz')
                choose('infer_slice', 12)
                assert ui.inference_panel.slice == 11 and ui.inference_panel.case == 1
                assert '原始第 12/15 层' in dpg.get_value('infer_summary')
                assert target['tile_count'] > 1
            else:
                target = report['cases'][0]
                produced = report_path.parent / target['mask_file']
                assert produced.read_bytes() == (Path(original['predictions']).parent / 'masks/000001.png').read_bytes()
            invoke('infer_export')
            invoke('infer_picker_export', {'file_path_name': str(exports)})
            copied = exports / ('seg-' + report_path.parent.name + '-' + Path(target['mask_file']).name)
            assert copied.is_file()
            if volume:
                saved = nib.load(copied)
                assert saved.shape == shape and saved.header.get_xyzt_units()[0] == 'mm'
                np.testing.assert_allclose(saved.affine, affine)
                assert set(np.unique(np.asanyarray(saved.dataobj))) <= {0, 255}
            else:
                with Image.open(copied) as saved:
                    assert saved.size == (13, 11)
            _capture(ui, output / (fmt + '-inference.png'))
            invoke('infer_choose_report')
            invoke('infer_picker_report', {'file_path_name': str(report_path)})
            assert ui.inference_panel.case == ui.inference_panel.slice == 0 and dpg.does_item_exist('infer_image')
            cases.append({'format': fmt, 'mode': report['mode'], 'inputs': len(report['cases']),
                          'tiles': target['tile_count'], 'source_shape': target['source_shape'], 'export_exact_grid': True,
                          'history_reopened': True, 'same_as_training_test': True if not volume else None})
        invoke('infer_start')
        invoke('infer_cancel')
        wait(lambda: ui.inference_panel.record['status'] in {'cancelled', 'failed', 'succeeded'}, 'cancel inference')
        assert ui.inference_panel.record['status'] == 'cancelled'
        assert not (manager.directory / ui.inference_panel.record['run_id'] / 'inference.json').exists()
        return {'status': 'passed', 'frozen': bool(getattr(sys, 'frozen', False)), 'device': device,
                'cases': cases, 'cancelled': True, 'canvas_preserved': True}
    finally:
        ui.shutdown()
