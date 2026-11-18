"""Native callback and real worker acceptance for CSV regression."""
import csv
import json
import math
from pathlib import Path
from time import monotonic


def _fixture(directory):
    path = directory / 'regression-fixture.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['length', 'mass', 'response', 'group', 'split'])
        for i in range(36):
            length, mass = i / 18, (i % 7) / 5
            split = 'train' if i < 24 else 'val' if i < 30 else 'test'
            writer.writerow([length, mass, 1.4 * length - 0.3 * mass + 0.2, i // 2, split])
    return path


def _verify_predictions(result, graph, device):
    import torch
    from studio_core.runtime import build_model
    torch.set_num_threads(1)
    paths = {name: Path(path) for name, path in result['artifacts'].items()}
    manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))
    checkpoint = torch.load(paths['best_checkpoint'], map_location='cpu', weights_only=True)
    assert checkpoint['adapter'] == {'id': 'regression', 'version': 1}
    assert checkpoint['completed_epochs'] == result['best_epoch']
    assert result['best_epoch'] == min(result['history'], key=lambda row: row['val_loss'])['epoch']
    config = checkpoint['config']
    model = build_model(graph, device=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    samples = [row for row in manifest['records'] if row['split'] == 'test']
    expected = [row['label'] for row in samples]
    mean, std = (manifest['preprocessing'][key] for key in ('mean', 'std'))
    features = torch.tensor([[(value-mean[i])/std[i] for i, value in enumerate(row['features'])]
                             for row in samples], dtype=torch.float32)
    with torch.no_grad():
        predictions = model(**{config['input_port']: features.to(device)})['prediction'].flatten().cpu()
    with paths['predictions'].open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert [row['sample_id'] for row in rows] == [row['id'] for row in samples]
    assert [float(row['target']) for row in rows] == expected
    torch.testing.assert_close(predictions, torch.tensor([float(row['prediction']) for row in rows]), rtol=1e-6, atol=1e-6)
    residuals = [float(row['prediction']) - target for row, target in zip(rows, expected)]
    n = len(residuals)
    ss = math.fsum(value**2 for value in residuals)
    avg = math.fsum(expected) / n
    reference = {'mae': math.fsum(map(abs, residuals)) / n, 'rmse': math.sqrt(ss / n),
                 'r2': 1 - ss / math.fsum((target - avg)**2 for target in expected)}
    metrics = result['metrics']['test']
    assert metrics['count'] == n
    assert all(math.isclose(metrics[key], value, rel_tol=1e-9, abs_tol=1e-12) for key, value in reference.items())
    assert all(math.isclose(float(row['residual']), error) for row, error in zip(rows, residuals))
    return metrics


def check_s6(manager, directory, resources, output_directory, device='cpu'):
    import dearpygui.dearpygui as dpg
    from studio_core.experiment_spec import model_digest
    from studio_core.python_io import export_python, import_python
    from .document import Document
    from .render_acceptance import _capture, _frames, _wait_ui
    from .ui import NativeWorkbench
    from .workspace_state import read_workspace_full

    path = _fixture(directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    callbacks = []
    ui = NativeWorkbench(Document(), manager, directory, resources)

    def invoke(tag, data=None):
        callback = dpg.get_item_callback(tag)
        assert callable(callback), tag
        callbacks.append(tag)
        callback(tag, data, dpg.get_item_user_data(tag))

    def wait_for(predicate):
        deadline = monotonic() + 35
        while monotonic() < deadline:
            _frames(ui, 1)
            if predicate():
                return
        raise TimeoutError(f'UI query did not settle: {ui.messages[-5:]}')

    def train():
        invoke('native_run')
        record = _wait_ui(ui)
        assert record['status'] == 'succeeded', record.get('error', ui.messages[-5:])
        assert record['result']['device'] == device
        return record['result']

    try:
        ui.build()
        _frames(ui, 5)
        label = 'CSV 数值回归'
        dpg.set_value('native_task', label)
        invoke('native_task', label)
        invoke('native_data_settings')
        invoke('native_data_open', {'file_path_name': str(path)})
        wait_for(lambda: 'response' in ui.csv_header and device in ui.available_devices)
        assert not dpg.get_value('native_csv_target'), 'Regression requires explicit target selection'
        for tag, value in [('native_csv_target', 'response'), ('native_csv_group', 'group'), ('native_csv_split', 'split')]:
            dpg.set_value(tag, value)
            invoke(tag, value)
        dpg.set_value('native_device', device)
        invoke('native_device', device)
        dpg.set_value('native_epochs', 3)
        dpg.set_value('native_batch', 4)
        dpg.set_value('native_lr', 0.05)
        invoke('native_data_inspect')
        wait_for(lambda: ui.manifest is not None)
        assert ui.manifest['counts'] == {'train': 24, 'val': 6, 'test': 6}
        _capture(ui, output / 'regression-data.png')
        invoke('native_data_apply')
        invoke('native_starter')
        invoke('native_starter_confirm')
        graph = ui.document.graph
        assert graph['outputs'][0]['id'] == 'prediction'
        assert not dpg.get_value('native_data_compatibility')
        assert model_digest(import_python(export_python(graph))) == model_digest(graph)
        workspace = output / 'regression-workspace.json'
        invoke('native_json_save', {'file_path_name': str(workspace)})
        saved, experiment, _ = read_workspace_full(workspace)
        assert saved.semantic_key() == ui.document.semantic_key()
        assert experiment['profile'] == 'regression-v1' and experiment['data']['target_column'] == 'response'
        invoke('native_json_open', {'file_path_name': str(workspace)})
        assert dpg.get_value('native_task') == label
        invoke('native_inspect')
        wait_for(lambda: ui.manifest is not None)
        first = train()
        metrics = _verify_predictions(first, graph, device)
        assert any(item['grad_norm'] and item['update_norm'] > 0 for item in first['parameters'])
        assert first['probes']['call:prediction']['shape'] == [4, 1]
        assert all(text in dpg.get_value('native_final_metrics') for text in ('MAE', 'RMSE', 'R²'))
        _capture(ui, output / 'regression-result.png')
        invoke('native_data_settings')
        dpg.set_value('native_resume_checkpoint', first['artifacts']['last_checkpoint'])
        invoke('native_resume_checkpoint')
        invoke('native_data_apply')
        dpg.set_value('native_epochs', 5)
        invoke('native_epochs', 5)
        resumed = train()
        assert resumed['completed_epochs'] == 5 and resumed['global_step'] == 30
        assert resumed['history'][:3] == first['history']
        dpg.set_value('native_epochs', 1000)
        invoke('native_epochs', 1000)
        invoke('native_run')
        invoke('native_cancel')
        assert _wait_ui(ui)['status'] == 'cancelled'
        invoke('native_catalog_category', '传统机器学习模型库')
        panel = ui.catalog_panel
        assert panel.visible_ids == ['primitive.SVC', 'primitive.SVR']
        from .catalog_panel import screen_rect
        view = ui.canvas_host.get('root')
        token = ui.canvas_host.token('root')
        origin, size = screen_rect(view.container_tag)
        point = [origin[i] + size[i] / 2 for i in range(2)]
        assert panel.place(panel.payload('primitive.SVR'), token, screen_point=point)
        assert ui.document.selected_instance['kind'] == 'SVR'
        dpg.set_value('native_right_tabs', dpg.get_alias_id('native_experiment_tab'))
        invoke('native_svm_starter')
        invoke('native_starter_confirm')
        assert ui.document.validation()['valid']
        assert not dpg.get_item_configuration('native_epoch_controls')['show']
        graph = ui.document.graph
        source = export_python(graph)
        edited = import_python(source.replace('C=1.0', 'C=2.0'))
        assert edited['instances'][0]['params']['C'] == 2.0
        assert model_digest(import_python(source)) == model_digest(graph)
        invoke('native_run')
        classical = _wait_ui(ui)
        assert classical['status'] == 'succeeded', classical.get('error', ui.messages[-5:])
        result = classical['result']
        assert result['algorithm'] == 'SVR' and result['device'] == 'cpu'
        assert result['history'] == [] and result['metrics']['test']['count'] == 6
        assert 'SVR' in dpg.get_value('native_final_metrics')
        _capture(ui, output / 'svr-library-and-result.png')
        return {'status': 'passed', 'device': device, 'callbacks': callbacks, 'metrics': metrics,
                'svr_metrics': result['metrics']['test'], 'svm_drag_placed': True,
                'resume_epochs': resumed['completed_epochs'], 'python_round_trip': True, 'cancelled': True,
                'workspace_reopened': True, 'frames': ['regression-data.png', 'regression-result.png', 'svr-library-and-result.png']}
    finally:
        ui.shutdown()
