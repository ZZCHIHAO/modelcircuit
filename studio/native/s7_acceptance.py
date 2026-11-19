"""End-to-end native FitNets acceptance with a grouped synthetic 3D dataset."""
import csv
from pathlib import Path
from time import monotonic


def check_s7(manager, directory, resources, output, device):
    import dearpygui.dearpygui as dpg
    import nibabel as nib
    import numpy as np
    from .document import Document
    from .render_acceptance import _capture, _frames, _wait_ui
    from .ui import NativeWorkbench
    from studio_core.experiment_spec import default_classification_config
    from studio_core.python_io import export_python, import_python
    from studio_core.fitnets import validate_fitnets_graph

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = directory / 'fixture.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('sample_id','patient_id','path','label','split'))
        for i, split in enumerate(('train','val','test')):
            for j in range(4):
                name = f'volume_{i}_{j}.nii.gz'
                array = np.random.default_rng(i*4+j).normal(size=(10, 12, 5)).astype(np.float32)
                nib.save(nib.Nifti1Image(array, np.diag([.5, .5, 2.,1.])), directory/name)
                writer.writerow((f's{i}{j}',f'g{i}{j}',name,f'class_{j % 2}',split))
    config = default_classification_config('research_csv')
    config.update(epochs=1,batch_size=2,device=device,lr=.02)
    config['data'].update(path=str(path),format='nifti',depth=4,image_size=[8,8],split_column='split')
    ui = NativeWorkbench(Document(), manager, directory, resources, experiment=config)

    def invoke(tag, data=None):
        fn = dpg.get_item_callback(tag)
        return fn(tag,data,dpg.get_item_user_data(tag))

    def run():
        invoke('native_run')
        record = _wait_ui(ui)
        if record['status'] != 'succeeded':
            raise AssertionError(record.get('error') or ui.messages[-5:])
        return record['result']

    try:
        ui.build()
        _frames(ui,6)
        invoke('native_inspect')
        deadline = monotonic()+30
        while (ui.manifest is None or device not in ui.available_devices) and monotonic()<deadline:
            _frames(ui,1)
        assert ui.manifest is not None, ui.messages[-5:]
        assert ui.manifest['input_shape'] == [None,1,4,8,8]
        dpg.set_value('native_right_tabs',dpg.get_alias_id('native_experiment_tab'))
        invoke('native_teacher_starter')
        invoke('native_starter_confirm')
        teacher = run()
        invoke('native_fitnets_starter')
        invoke('native_starter_confirm')
        dpg.set_value('native_teacher_checkpoint',teacher['artifacts']['best_checkpoint'])
        dpg.set_value('native_hint_epochs',1)
        validate_fitnets_graph(import_python(export_python(ui.document.graph)))
        result = run()
        assert result['device'] == device and result['phase_summary']['teacher_unchanged']
        assert 'head' not in result['phase_summary']['hint_changed_instances']['student']
        assert result['phase_summary']['hint_changed_instances']['projector']
        assert 'head' in result['phase_summary']['distill_changed_instances']['student']
        assert not result['phase_summary']['distill_changed_instances']['projector']
        assert result['metrics']['test']['count'] == 4
        workspace = output/'fitnets-workspace.json'
        invoke('native_json_save',{'file_path_name':str(workspace)})
        assert workspace.exists()
        _capture(ui,output/'fitnets-workbench.png')
        invoke('native_json_open',{'file_path_name':str(workspace)})
        assert ui._configuration()['profile'] == 'fitnets-v1'
        assert ui._configuration()['data']['depth'] == 4
        return {'status':'passed','device':device,'sample_shape':[1,4,8,8],
                'phases':result['phase_summary'],'history':result['history'],
                'teacher_run':teacher['artifacts']['run_spec'],'fitnets_run':result['artifacts']['run_spec'],
                'student_checkpoint':result['artifacts']['student_checkpoint']}
    finally:
        ui.shutdown()
