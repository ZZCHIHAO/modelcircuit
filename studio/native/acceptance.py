"""Explicit diagnostic run, also used to verify the frozen worker runtime."""
import math
import time

from .document import Document


def wait_terminal(manager, run_id, timeout=100):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = manager.get(run_id)
        if run['status'] in {'succeeded', 'failed', 'cancelled'}:
            return run
        time.sleep(0.05)
    manager.cancel(run_id)
    raise TimeoutError('Native acceptance worker did not finish within its deadline')


def check_runtime(manager):
    document = Document()
    assert document.validation()['valid']
    document.set_param('encoder', 'hidden_features', 10)
    document.set_param('head', 'in_features', 10)
    code = document.export_python()
    imported = Document()
    imported.import_python(code)
    assert imported.validation()['valid']
    assert next(x for x in imported.graph['instances'] if x['id'] == 'encoder')['params']['hidden_features'] == 10
    original = Document()
    run = manager.submit(original.graph, {'steps': 20, 'seed': 42, 'lr': 0.05, 'batch_size': 16})
    finished = wait_terminal(manager, run['run_id'])
    if finished['status'] != 'succeeded':
        raise RuntimeError('Frozen worker failed: ' + str(finished.get('error')))
    result = finished['result']
    assert result['parameter_count'] == 58
    assert math.isfinite(result['loss_after']) and result['loss_after'] < result['loss_before']
    assert 'call:encode_b/call:project' in result['probes']
    assert any(item['instance_id'] == 'encoder' and len(item['call_ids']) == 2 for item in result['shared_instances'])
    pending = manager.submit(original.graph, {'steps': 200})
    manager.cancel(pending['run_id'])
    cancelled = wait_terminal(manager, pending['run_id'])
    assert cancelled['status'] == 'cancelled'
    return {
        'status': 'passed', 'kind': 'native-runtime', 'run_id': run['run_id'],
        'cancel_run_id': pending['run_id'], 'parameter_count': result['parameter_count'],
        'loss_before': result['loss_before'], 'loss_after': result['loss_after'],
        'same_batch_accuracy': result['accuracy'], 'python_round_trip': True,
        'concrete_call_probe': True, 'data_source': result['data_source'],
    }
