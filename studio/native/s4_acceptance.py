"""Source/frozen S4 diagnostics using generated data and production workers.

RunManager ownership stays with the entry point. Temporary image fixtures are
removed on success and failure; worker checkpoints and diagnostic evidence stay
in the caller's data directory. No model weights or data are downloaded.
"""
import gc
import json
from pathlib import Path
import random
import sys
import tempfile
import time

from .classification_acceptance import _best_inference, _wait


def _require(condition, message):
    if not condition:
        raise RuntimeError('S4 acceptance: ' + message)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


def _cleanup_fixture(fixture):
    """Allow short Windows scanner locks without hiding persistent failures."""
    for attempt in range(6):
        try:
            fixture.cleanup()
            return
        except PermissionError:
            if attempt == 5:
                raise
            gc.collect()
            time.sleep(.05 * (2 ** attempt))


def check_catalog():
    """Check the shipped finite catalog without importing Torch or DPG."""
    from studio_core.component_catalog import list_components
    from studio_core.model_registry import list_registered_models
    catalog = [item for item in list_components() if item.category != 'classical' and item.id not in {'primitive.Conv3d', 'primitive.MaxPool3d', 'primitive.AdaptiveAvgPool3d'}]
    models = list_registered_models()
    _require(len(catalog) == 36 and len(models) == 14, 'expected 36 catalog entries and 14 model entries')
    _require(len({item.id for item in catalog}) == 36, 'duplicate catalog identifiers')
    _require(all(item.capabilities['expandable'] and item.capabilities['weights'] == 'none' for item in models),
             'model capabilities do not describe expandable random initialization')
    return {'component_count': len(catalog), 'model_count': len(models), 'component_ids': [item.id for item in catalog],
            'model_ids': [item.id for item in models], 'weights': 'none'}


def empty_graph(image=True):
    inputs = [{'id': 'x', 'shape': [None, 3, 32, 32], 'dtype': 'float32', 'position': {'x': 10, 'y': 80}}] if image else [
        {'id': name, 'shape': [None, 4], 'dtype': 'float32', 'position': {'x': 10, 'y': 80 + index * 180}}
        for index, name in enumerate(('a', 'b'))]
    return {'schema_version': 1, 'id': 's4_diagnostic', 'revision': 1, 'inputs': inputs,
            'instances': [], 'calls': [], 'definitions': [],
            'outputs': [{'id': 'logits' if image else 'features', 'position': {'x': 880, 'y': 140}}]}


def build_s4_document():
    from .document import Document
    from studio_core.component_catalog import build_component
    document = Document(empty_graph())
    bundle = build_component('model.resnet50_classifier', {'input_shape': [None, 3, 32, 32], 'num_classes': 2,
                                                          'resource_batch_bound': 2, 'resource_profile': 'hardware-v5'})
    reference = document.add_component(bundle, apply_policy=True)
    document.set_input(reference, 'x', 'input:x')
    document.set_output('logits', reference + ':logits')
    validated = document.validation()
    _require(validated['valid'], 'ResNet50 graph is invalid: ' + str(validated['errors']))
    _require(validated['parameter_count'] > 20_000_000, 'ResNet50 does not cross the legacy parameter limit')
    return document


def _image_fixture(directory):
    from PIL import Image
    root = Path(directory) / 'images'
    for split_index, split in enumerate(('train', 'val', 'test')):
        for label in range(2):
            path = root / split / ('class_' + str(label)) / 'sample.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            # Spatial texture avoids degenerate tiny-batch BatchNorm statistics.
            pixels = random.Random(77 + label + split_index * 2).randbytes(32 * 32 * 3)
            with Image.frombytes('RGB', (32, 32), pixels) as image:
                image.save(path)
    return {'kind': 'image_folder', 'path': str(root), 'split_mode': 'directories', 'image_size': [32, 32], 'channels': 3}


def _execution_contract(graph):
    """Compare resolved execution, excluding metadata-only names/versions/layout.

    Metadata-free Python naturally creates fresh document IDs and definition
    versions. Those are part of checkpoint identity, so model_digest equality
    would incorrectly reject an otherwise identical execution graph.
    """
    from studio_core.operator_rules import normalize_operator
    from studio_core.schema import prepare_graph
    resolved, _, count = prepare_graph(graph)
    def scope_contract(scope):
        instances = {item['id']: item for item in scope['instances']}
        aliases = {}
        for call in scope['calls']:
            instance = instances[call['instance_id']]
            ports = instance['_graph']['outputs'] if instance['kind'] == 'chip' else [{'id': 'out'}]
            if len(ports) == 1:
                aliases['call:' + call['id'] + ':' + ports[0]['id']] = 'call:' + call['id']
        return {'inputs': [{'id': item['id'], 'shape': item['shape'], 'dtype': item.get('dtype', 'float32')} for item in scope['inputs']],
            'instances': [{'id': item['id'], 'kind': item['kind'], 'frozen': item.get('frozen', False),
                           'execution': scope_contract(item['_graph']) if item['kind'] == 'chip' else normalize_operator(item['kind'], item.get('params', {}))}
                          for item in sorted(scope['instances'], key=lambda item: item['id'])],
            'calls': [{'id': item['id'], 'instance_id': item['instance_id'],
                       'inputs': {key: aliases.get(value, value) for key, value in item['inputs'].items()}}
                      for item in sorted(scope['calls'], key=lambda item: item['id'])],
            'outputs': [{'id': item['id'], 'source': aliases.get(item['source'], item['source'])} for item in scope['outputs']]}
    return {'resource_profile': graph.get('resource_profile'), 'resource_batch_bound': graph.get('resource_batch_bound'),
            'parameter_count': count, 'execution': scope_contract(resolved)}


def _python_round_trip(document, result, output, device):
    """Compare one candidate at a time, bounding simultaneous model storage."""
    import torch
    from studio_core.experiment_spec import model_digest
    from studio_core.python_io import import_python, python_profile
    from studio_core.runtime import build_model
    source = document.export_python()
    clean = '\n'.join(line for line in source.splitlines() if not line.startswith('STUDIO_VISUAL_METADATA = ')) + '\n'
    sources = {'metadata': source, 'no_metadata': clean}
    state = torch.load(result['artifacts']['best_checkpoint'], weights_only=True, map_location='cpu')['model_state']
    inputs = torch.arange(2 * 3 * 32 * 32, dtype=torch.float32, device=device).reshape(2, 3, 32, 32) / 6144
    model = build_model(document.graph, device=device, capture=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        expected = model(x=inputs)['logits'].detach().cpu()
    del model
    gc.collect()
    comparisons, artifacts = {}, {}
    for label, code in sources.items():
        path = output / ('resnet50-' + label + '.py')
        path.write_text(code, encoding='utf-8')
        artifacts['python_' + label] = str(path)
        graph = import_python(code)
        _require(_execution_contract(graph) == _execution_contract(document.graph), label + ' resolved execution contract changed')
        graph_path = output / ('resnet50-' + label + '.json')
        _write_json(graph_path, graph)
        artifacts['graph_' + label] = str(graph_path)
        candidate = build_model(graph, device=device, capture=False)
        candidate.load_state_dict(state, strict=True)
        candidate.eval()
        with torch.no_grad():
            actual = candidate(x=inputs)['logits'].detach().cpu()
        comparisons[label] = _compare(expected, actual, device, label)
        del candidate
        gc.collect()
    namespace = {'__name__': 'generated_s4_diagnostic'}
    # Only execute our own fixed generated model. Production user imports stay static.
    exec(compile(source, str(output / 'resnet50-metadata.py'), 'exec'), namespace)
    standalone = namespace['Model']().to(device)
    standalone.load_state_dict(state, strict=True)
    standalone.eval()
    with torch.no_grad():
        actual = standalone(x=inputs)['logits'].detach().cpu()
    comparisons['standalone'] = _compare(expected, actual, device, 'standalone')
    return {'profile': python_profile(document.graph), 'device': device, 'execution_contract_equal': True,
            'metadata_model_digest_equal': model_digest(import_python(source)) == model_digest(document.graph),
            'metadata_free_identity_notice': 'Fresh document/definition identity; resolved operations, parameters, wiring, ports and policy are equal',
            'input_shape': list(inputs.shape), 'output_shape': list(expected.shape), 'comparisons': comparisons,
            'simultaneous_candidate_models': 1, 'comparison': 'bit_exact' if device == 'cpu' else 'allclose'}, artifacts


def _compare(expected, actual, device, label):
    import torch
    exact = torch.equal(expected, actual)
    close = bool(torch.allclose(expected, actual, rtol=1e-5, atol=1e-6))
    _require(exact if device == 'cpu' else close, label + ' independent forward differs with identical checkpoint weights')
    return {'bit_equal': exact, 'close': close, 'maximum_absolute_error': float((expected - actual).abs().max())}


def check_s4_runtime(manager, directory, device='cpu'):
    """One actual large-model classification worker plus independent forwards."""
    import torch
    from studio_core.classification_data import inspect_dataset
    from studio_core.devices import device_capabilities, resolve_device
    from studio_core.experiment_spec import validate_classification_config
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / 's4-acceptance.json'
    report = {'status': 'running', 'kind': 'native-s4-runtime', 'device': device, 'profile': 'hardware-v5',
              'test_fixtures': True, 'fixture_notice': 'Generated six PNGs only; no scientific performance claim',
              'frozen': bool(getattr(sys, 'frozen', False)), 'worker_cleanup': 'caller', 'fixtures_cleaned': False,
              'gpu_execution': {'executed': False}, 'report_path': str(report_path)}
    previous_threads = torch.get_num_threads()
    fixture = None
    try:
        resolve_device(device)
        torch.set_num_threads(1)
        report['capabilities'] = device_capabilities()
        report['catalog'] = check_catalog()
        document = build_s4_document()
        fixture = tempfile.TemporaryDirectory(prefix='s4-images-', dir=directory)
        report['fixture_directory'] = fixture.name
        spec = _image_fixture(fixture.name)
        manifest = inspect_dataset(spec, seed=77)
        config = validate_classification_config({'profile': 'classification-v2', 'data': spec, 'epochs': 1, 'batch_size': 2,
            'optimizer': 'adamw', 'lr': 0.000001, 'seed': 77, 'device': device, 'max_runtime_seconds': 120,
            'expected_dataset_digest': manifest['dataset_digest']})
        run_id, result = _wait(manager, document.graph, config)
        _require(result['parameter_count'] > 20_000_000, 'worker did not train the large model')
        _require(result['global_step'] == 1, 'generated fixture did not perform exactly one optimizer step')
        memory = result.get('memory_report', {})
        _require(memory.get('policy') == 'hardware-v5' and memory.get('device') == device, 'worker memory admission missing/wrong device')
        _require(memory['estimate']['training'] and memory['estimate']['batch_size'] == 2, 'worker memory estimate is not the actual training batch')
        saved_memory = json.loads(Path(result['artifacts']['memory_report']).read_text(encoding='utf-8'))
        _require(saved_memory == memory, 'saved memory report differs from worker result')
        updated = sum(item['update_norm'] > 0 for item in result['parameters'])
        _require(updated > 0, 'worker did not update any measured parameter')
        report['training'] = {'run_id': run_id, 'model': 'model.resnet50_classifier', 'completed_epochs': result['completed_epochs'],
            'best_epoch': result['best_epoch'], 'global_step': result['global_step'], 'parameter_count': result['parameter_count'],
            'metrics': result['metrics'], 'dataset': result['dataset'], 'probe_context': result['probe_context'],
            'updated_parameter_tensors': updated, 'memory_report': memory}
        report['checkpoint_inference'] = _best_inference(document.graph, result, device)
        python_report, python_artifacts = _python_round_trip(document, result, directory, device)
        report['python_round_trip'] = python_report
        graph_path = directory / 'resnet50-graph.json'
        _write_json(graph_path, document.graph)
        report['artifacts'] = {**result['artifacts'], **python_artifacts, 'graph': str(graph_path)}
        report['gpu_execution'] = {'executed': device != 'cpu', 'device': result['device']}
        report['status'] = 'passed'
        return report
    except Exception as error:
        report['status'], report['error'] = 'failed', str(error)
        raise
    finally:
        try:
            if fixture is not None:
                _cleanup_fixture(fixture)
            report['fixtures_cleaned'] = True
        except Exception as error:
            report['status'] = 'failed'
            report['cleanup_error'] = str(error)
            report.setdefault('error', 'Fixture cleanup failed: ' + str(error))
            raise
        finally:
            torch.set_num_threads(previous_threads)
            _write_json(report_path, report)
