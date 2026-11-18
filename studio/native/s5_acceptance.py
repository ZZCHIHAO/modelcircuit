"""Exercise S5 through the real local worker with generated images."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from .main import close_workers
from .s4_acceptance import _image_fixture


def fixture_config(directory, device):
    return {'profile': 'classification-v2', 'data': _image_fixture(directory), 'device': device,
            'epochs': 1, 'batch_size': 2, 'optimizer': 'adamw', 'lr': 0.001, 'seed': 77,
            'max_runtime_seconds': 120, 'input_port': 'x'}


def fixture_graph():
    from studio_core.transformer_templates import make_small_vit
    bundle = make_small_vit(dropout=.1)
    definition = bundle.definitions[0]
    return deepcopy({**{key: value for key, value in bundle.requirements.items() if key != 'operator_set'},
                     'id': 's5_fixture', 'revision': 1, 'inputs': definition['graph']['inputs'],
                     'definitions': bundle.definitions, 'instances': [bundle.entry],
                     'calls': [{'id': 'vit', 'instance_id': bundle.entry['id'], 'inputs': {'x': 'input:x'}}],
                     'outputs': [{'id': 'logits', 'source': 'call:vit:logits'}]})


def check_runtime(manager, directory, device):
    import torch
    from studio_core.python_io import export_python, import_python
    from studio_core.runtime import build_model
    from studio_core.classification_data import build_datasets, inspect_dataset
    from .classification_acceptance import _wait

    graph = fixture_graph()
    config = fixture_config(directory, device)
    source = 'call:vit/call:encoder/call:block0/call:self_attention/call:attention:out'
    request = {'source': source, 'sample_index': 0, 'head_index': 1, 'query_range': [0, 4],
               'key_range': [0, 5], 'statistics': ['mean', 'max']}
    config['attention_probe_requests'] = [request]
    _, first = _wait(manager, graph, config)
    _, resumed = _wait(manager, graph, {**config, 'epochs': 2, 'resume_checkpoint': first['artifacts']['last_checkpoint']})
    _, continuous = _wait(manager, graph, {**config, 'epochs': 2})

    def checkpoint(result, key):
        return torch.load(result['artifacts'][key], map_location='cpu', weights_only=True)

    left, right = (checkpoint(run, 'last_checkpoint') for run in (resumed, continuous))
    if left['version'] != 3 or left['adapter'] != {'id': 'classification', 'version': 1}:
        raise AssertionError('Missing typed checkpoint identity')
    if left['global_step'] != 2 or left['checkpoint_signature'] != right['checkpoint_signature']:
        raise AssertionError('Resume did not preserve the run identity and optimizer progress')
    if device == 'cpu':
        torch.testing.assert_close(left['model_state'], right['model_state'], rtol=0, atol=0)
        torch.testing.assert_close(left['optimizer_state'], right['optimizer_state'], rtol=0, atol=0)
        if left['history'] != right['history']:
            raise AssertionError('Resumed CPU history differs')
    observed = continuous['probes'][source]
    if observed['omitted'] or observed['captured_bytes'] != 4 * 5 * 4 or observed['shape'] != [2, 4, 65, 65]:
        raise AssertionError(f'Unexpected bounded attention measurement: {observed}')

    code = export_python(graph)
    plain = '\n'.join(line for line in code.splitlines() if not line.startswith('STUDIO_VISUAL_METADATA = '))
    restored = import_python(plain)
    model = build_model(restored, device=device, capture=True, capture_policy={'requests': [request]})
    model.load_state_dict(checkpoint(continuous, 'best_checkpoint')['model_state'], strict=True)
    model.eval()
    datasets = build_datasets(inspect_dataset(config['data'], seed=config['seed']))
    batch, _, _ = next(iter(torch.utils.data.DataLoader(datasets['val'], batch_size=2)))
    with torch.no_grad():
        model(x=batch.to(device))
    torch.testing.assert_close(torch.tensor(model.last_probes[source]['values']), torch.tensor(observed['values']), atol=1e-6, rtol=1e-5)

    cancel = manager.submit(graph, {**config, 'epochs': 20})
    manager.cancel(cancel['run_id'])
    from time import monotonic, sleep
    deadline = monotonic() + 20
    state = manager.get(cancel['run_id'])['status']
    while state not in ('cancelled', 'succeeded', 'failed') and monotonic() < deadline:
        sleep(.05)
        state = manager.get(cancel['run_id'])['status']
    if state != 'cancelled':
        raise AssertionError(f'Worker cancellation ended in {state}')
    return {'device': device, 'parameters': first['parameter_count'], 'steps': left['global_step'],
            'checkpoint_version': left['version'], 'cpu_resume_bit_exact': device == 'cpu',
            'probe_shape': observed['shape'], 'probe_bytes': observed['captured_bytes'],
            'checkpoint_probe_replayed_after_python_import': True, 'cancelled': True,
            'torch': str(torch.__version__), 'cuda': torch.version.cuda}


def main():
    from server import RunManager
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    directory = args.output.parent / (args.output.stem + '-data')
    manager = RunManager(directory / 'runs')
    try:
        report = check_runtime(manager, directory, args.device)
    finally:
        close_workers(manager)
    report.update(status='passed', workers_stopped=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    # multiprocessing spawn requires the guard on Windows.
    main()
