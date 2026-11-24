"""Exercise medical segmentation with synthetic CT and MRI using desktop callbacks."""
import csv
import sys
from pathlib import Path
from time import monotonic


def _fixture(directory, fmt):
    import nibabel as nib
    import numpy as np

    directory.mkdir(parents=True)
    ct = fmt == 'dicom_ct'
    affine = np.diag([-.75 if ct else .75, -1.25 if ct else 1.25, 2.5, 1.])
    affine[:3, 3] = [-12. if ct else 12., 3., 40.]
    shape = (13, 11, 5)

    def save(name, value):
        image = nib.Nifti1Image(value, affine)
        image.header.set_xyzt_units('mm')
        nib.save(image, directory / name)

    pairs = directory / 'pairs.csv'
    with pairs.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'patient_id', 'image_path', 'mask_path', 'split', *([] if ct else ['flair'])])
        for index, split in enumerate(('train', 'val', 'test')):
            labels = np.zeros(shape, dtype=np.uint16)
            labels[2 + index:7 + index, 3:9, 1:4] = 255
            image = np.random.default_rng(index).integers(30, 75, shape, dtype=np.int16) + (labels > 0) * 500
            source = f'ct-{index}' if ct else f't1-{index}.nii.gz'
            annotation = f'label-{index}.nii.gz'
            save(annotation, labels)
            sequences = []
            if ct:
                _write_ct(directory / source, image)
            else:
                save(source, image.astype(np.float32))
                sequences = [f'flair-{index}.nii.gz']
                save(sequences[0], image.astype(np.float32) ** 1.5)
            writer.writerow([f'case-{index}', f'patient-{index}', source, annotation, split, *sequences])
    return pairs, shape, affine


def _write_ct(folder, image):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    folder.mkdir()
    series_uid, frame_uid = generate_uid(), generate_uid()
    depth = image.shape[-1]
    for z in range(depth):
        path = folder / f'slice-{depth - z}.dcm'
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        item = FileDataset(path, {}, file_meta=meta, preamble=b'\0' * 128)
        item.SOPClassUID, item.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        item.Modality, item.SeriesInstanceUID, item.FrameOfReferenceUID = 'CT', series_uid, frame_uid
        item.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
        item.Rows, item.Columns = image.shape[1], image.shape[0]
        item.SamplesPerPixel, item.PhotometricInterpretation = 1, 'MONOCHROME2'
        item.PixelRepresentation, item.BitsAllocated, item.BitsStored, item.HighBit = 1, 16, 16, 15
        item.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        item.ImagePositionPatient = [12., -3., 40. + z * 2.5]
        item.PixelSpacing, item.SliceThickness = [1.25, .75], 2.5
        item.RescaleSlope, item.RescaleIntercept = 2, -1000
        item.PixelData = image[:, :, z].T.astype('<i2').tobytes()
        item.save_as(path, enforce_file_format=True)


def check_s13(manager, directory, resources, output, device):
    import dearpygui.dearpygui as dpg
    import nibabel as nib
    import numpy as np

    from studio_core.experiment_spec import model_digest
    from studio_core.research_project import bind_project, read_project
    from .data_panel import MODES, SEGMENTATION_FORMATS
    from .document import Document
    from .render_acceptance import _capture, _frames, _wait_ui
    from .ui import NativeWorkbench

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ui = NativeWorkbench(Document(), manager, directory, resources)
    evidence = []
    try:
        ui.build()
        _frames(ui)

        def invoke(tag, value=None):
            dpg.get_item_callback(tag)(tag, value, dpg.get_item_user_data(tag))

        def choose(tag, value):
            dpg.set_value(tag, value)
            invoke(tag, value)

        def until(condition, label, timeout=35):
            deadline = monotonic() + timeout
            while not condition() and monotonic() < deadline:
                _frames(ui, 1)
            assert condition(), (label, ui.messages[-6:])

        dpg.set_value('native_right_tabs', dpg.get_alias_id('native_experiment_tab'))
        choose('native_task', next(label for label, kind in MODES.items() if kind == 'segmentation_csv'))
        until(lambda: ui.capabilities is not None and device in ui.available_devices, 'device query')
        for fmt in ('dicom_ct', 'mri_nifti'):
            ct = fmt == 'dicom_ct'
            pairs, shape, affine = _fixture(directory / fmt, fmt)
            invoke('native_data_settings')
            choose('native_segmentation_format', next(label for label, value in SEGMENTATION_FORMATS.items() if value == fmt))
            invoke('native_data_open', {'file_path_name': str(pairs)})
            until(lambda: dpg.get_value('native_data_error').startswith('已读取'), 'CSV columns')
            choose('native_seg_split_column', 'split')
            for key, value in {'segmentation_values': '0, 255', 'segmentation_depth': 4, 'image_height': 8,
                               'image_width': 8, 'medical_spacing_D': 1.5, 'medical_patches': 2,
                               'medical_overlap': .5}.items():
                dpg.set_value('native_' + key, value)
            if not ct:
                dpg.set_value('native_medical_registered', True)
                dpg.set_value('native_medical_sequence_2', 'flair')
            invoke('native_data_inspect')
            until(lambda: ui.manifest is not None, 'volume inspection')
            assert ui.manifest['input_shape'] == [None, 1 if ct else 2, 4, 8, 8]
            assert ui.manifest['counts'] == {'train': 1, 'val': 1, 'test': 1}
            invoke('native_data_apply')
            _frames(ui)
            invoke('native_medical_patch_open')
            assert dpg.is_item_shown('medical_patch_dialog'), ui.messages[-6:]
            until(lambda: ui.medical_patch_panel.metadata is not None or
                  dpg.get_value('medical_patch_status') != '正在按训练配置采样…', 'patch preview')
            assert ui.medical_patch_panel.metadata is not None, dpg.get_value('medical_patch_status')
            preview = ui.medical_patch_panel.metadata.copy()
            assert preview['case_id'] == 'case-0' and preview['patch_shape_dhw'] == [4, 8, 8]
            assert dpg.does_item_exist('medical_patch_image')
            _capture(ui, output / (fmt + '-patch.png'))
            ui.medical_patch_panel.close()
            invoke('native_starter')
            invoke('native_starter_confirm')
            choose('native_device', device)
            dpg.set_value('native_epochs', 1)
            dpg.set_value('native_batch', 2)
            invoke('native_run')
            record = _wait_ui(ui)
            assert record['status'] == 'succeeded', (record.get('error'), ui.messages[-6:])
            result = record['result']
            assert result['device'] == device and result['global_step'] == 1
            assert result['metrics']['test']['space'] == result['metrics']['val']['space'] == 'original'
            assert result['metrics']['test']['voxels'] == int(np.prod(shape))
            files = result['artifacts']
            native_mask = nib.load(Path(files['predictions']).parent / 'masks' / '000001.nii.gz')
            np.testing.assert_allclose(native_mask.affine, affine, rtol=0, atol=1e-7)
            assert native_mask.shape == shape
            prediction = np.asarray(native_mask.dataobj)
            labels = np.asarray(nib.load(pairs.parent / 'label-2.nii.gz').dataobj)
            foreground = prediction == 255
            target = labels == 255
            dice = 2 * (foreground & target).sum() / (foreground.sum() + target.sum())
            assert abs(result['metrics']['test']['foreground_dice'] - dice) < 1e-12
            assert set(np.unique(prediction)) <= {0, 255} and Path(files['case_metrics']).is_file()
            assert '原始网格' in dpg.get_value('native_segmentation_result')
            assert 'HD95' in dpg.get_value('native_segmentation_case_metrics')
            dpg.set_value('native_bottom_tabs', dpg.get_alias_id('native_segmentation_tab'))
            _capture(ui, output / (fmt + '-evaluation.png'))

            invoke('infer_use_best')
            dpg.set_value('infer_device', device)
            invoke('infer_picker_folder' if ct else 'infer_picker_image',
                   {'file_path_name': str(pairs.parent / ('ct-2' if ct else 't1-2.nii.gz'))})
            if not ct:
                assert dpg.is_item_shown('infer_sequence_2_row')
                invoke('infer_picker_sequence_2', {'file_path_name': str(pairs.parent / 'flair-2.nii.gz')})
            invoke('infer_start')
            until(lambda: ui.inference_panel.record and ui.inference_panel.record['status'] in
                  {'succeeded', 'failed', 'cancelled'}, 'sliding inference', timeout=80)
            inferred = ui.inference_panel.record
            assert inferred['status'] == 'succeeded', (inferred.get('error'), ui.messages[-6:])
            report = ui.inference_panel.report
            assert report['mode'] == 'sliding' and report['overlap'] == .5
            case = report['cases'][0]
            assert case['space'] == 'original' and case['tile_count'] > 1
            inferred_mask = nib.load(ui.inference_panel.path.parent / case['mask_file'])
            np.testing.assert_array_equal(np.asarray(inferred_mask.dataobj), prediction)
            np.testing.assert_allclose(inferred_mask.affine, affine, rtol=0, atol=1e-7)

            project_folder = output / (fmt + '-project')
            dpg.set_value('native_project_export_path', str(project_folder))
            dpg.set_value('native_project_include_weights', True)
            invoke('native_project_export')
            until(lambda: (project_folder / 'project.json').is_file() and not ui.project_panel._operation, 'export project')
            project = read_project(project_folder)
            _, rebound = bind_project(project, pairs.parent)
            assert project['weights'] and model_digest(project['graph']) == model_digest(ui.document.graph)
            assert rebound['preprocessing'] == ui.manifest['preprocessing']
            evidence.append({'format': fmt, 'input_shape': rebound['input_shape'], 'patch': preview,
                             'cases': rebound['counts'], 'original_shape_xyz': list(shape), 'grid_and_mask_match': True,
                             'foreground_dice_recomputed': float(dice), 'windows': case['tile_count'], 'steps': result['global_step'],
                             'project_rebinding': 'passed'})
        return {'status': 'passed', 'frozen': bool(getattr(sys, 'frozen', False)), 'device': device,
                'synthetic_data': True, 'workflows': evidence}
    finally:
        ui.shutdown()
