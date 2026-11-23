"""Native acceptance for Linux DDP through the actual OpenSSH controls."""
import json
from pathlib import Path
from time import monotonic


def check_s12(manager, directory, resources, output, settings_path):
    import dearpygui.dearpygui as dpg
    from studio_core.project_runner import read_project_run
    from studio_core.remote_client import batch_plan
    from .document import Document
    from .remote_panel import tag
    from .render_acceptance import _capture, _frames
    from .ui import NativeWorkbench

    settings = json.loads(Path(settings_path).read_text(encoding='utf-8-sig'))
    output = Path(output)
    ui = NativeWorkbench(Document(), manager, directory, resources)
    jobs = []

    def invoke(widget, data=None):
        callback = dpg.get_item_callback(widget)
        return callback(widget, data, dpg.get_item_user_data(widget))

    def wait(predicate, label, timeout=120):
        until = monotonic() + timeout
        while not predicate() and monotonic() < until:
            _frames(ui, 1)
            if ui.remote_panel.record and ui.remote_panel.record['status'] == 'failed':
                detail = dpg.get_value(tag('log'))[-3000:]
                raise RuntimeError(f'{label}: {dpg.get_value(tag("summary"))}\n{detail}')
        if not predicate():
            raise RuntimeError(f'{label}: {dpg.get_value(tag("summary"))}; {ui.messages[-3:]}')

    def connect():
        invoke(tag('connect'))
        wait(lambda: ui.remote_panel.client is not None, 'OpenSSH connection', timeout=30)
        assert ui.remote_panel.inventory['ddp'], ui.remote_panel.inventory
        known = {gpu['uuid']: gpu['index'] for gpu in ui.remote_panel.inventory['gpus']}
        for gpu in settings['gpu_uuids']:
            assert gpu in known
            dpg.set_value(tag('gpu_' + gpu), True)
            invoke(tag('gpu_' + gpu), True)
        return known

    def present(job):
        return bool(ui.remote_panel.record and ui.remote_panel.record['job_id'] == job)

    try:
        ui.build()
        dpg.set_value('native_right_tabs', dpg.get_alias_id('native_experiment_tab'))
        invoke('native_open_remote')
        for key in ('host', 'python', 'workspace'):
            dpg.set_value(tag(key), settings[key])
        inventory = connect()
        project = Path(settings['project']).resolve()
        invoke(tag('picker'), {'file_path_name': str(project)})
        dpg.set_value(tag('data'), settings['data_path'])
        for name, key in [('batch', 'per_device'), ('accumulation', 'accumulation'), ('epochs', 'epochs')]:
            dpg.set_value(tag(name), settings[key])
        invoke(tag('batch'), settings['per_device'])
        plan = batch_plan(settings['gpu_uuids'], settings['per_device'], settings['accumulation'])
        assert str(plan['effective']) in dpg.get_value(tag('batch_plan'))
        invoke(tag('submit'))
        first_job = ui.remote_panel.job_id
        assert first_job
        jobs.append((ui.remote_panel.client, first_job))
        wait(lambda: present(first_job) and ui.remote_panel.record['status'] in {'running', 'preparing'}, 'job started')
        ui.shutdown()
        ui = NativeWorkbench(Document(), manager, directory, resources)
        ui.build()
        invoke('native_open_remote')
        assert dpg.get_value(tag('job')) == first_job
        connect()
        wait(lambda: present(first_job) and ui.remote_panel.record.get('resume_available'), 'wait for completed checkpoint')
        assert ui.remote_panel.record['status'] not in {'succeeded', 'failed', 'cancelled'}, ui.remote_panel.record
        invoke(tag('cancel'))
        wait(lambda: present(first_job) and ui.remote_panel.record['status'] == 'cancelled', 'remote process stopped')
        first = ui.remote_panel.record
        assert first['resume_available'] and first['progress']['epoch'] >= 1, first
        epoch = first['progress']['epoch']
        updates = first['progress']['updates']
        _capture(ui, output / 's12-cancelled.png')
        dpg.set_value(tag('epochs'), epoch + 1)
        invoke(tag('resume'))
        resumed_job = ui.remote_panel.job_id
        assert resumed_job != first_job
        jobs.append((ui.remote_panel.client, resumed_job))
        wait(lambda: present(resumed_job) and ui.remote_panel.record['status'] == 'succeeded', 'DDP resume')
        progress = ui.remote_panel.record['progress']
        assert progress['epoch'] == epoch + 1 and progress['updates'] > updates, progress
        invoke(tag('fetch'))
        wait(lambda: ui.project_panel._reports['left'] is not None, 'download and open project record')
        path = ui.project_panel._reports['left']['path']
        report = read_project_run(path, verify=True)
        assert report['status'] == 'succeeded' and report['metrics']['test']['count'] == settings['test_count'], report
        assert report['history'][-1]['epoch'] == epoch + 1
        execution = report['execution']
        assert execution['strategy'] == 'ddp' and execution['backend'] == 'nccl'
        assert execution['world_size'] == len(settings['gpu_uuids'])
        assert execution['effective_global_batch'] == plan['effective']
        assert [row['device_selector'] for row in execution['devices']] == settings['gpu_uuids']
        assert all(row['train_count'] == execution['train_samples_available'] - execution['train_samples_dropped_per_epoch']
                   for row in report['history'])
        assert not report['changed_artifacts'] and not report['missing_artifacts']
        _capture(ui, output / 's12-project-results.png')
        return {'status': 'passed', 'remote_strategy': 'Linux CUDA DDP', 'gpu_count': len(settings['gpu_uuids']),
                'gpu_indices': [inventory[gpu] for gpu in settings['gpu_uuids']],
                'effective_batch': plan['effective'], 'first_job': first_job, 'resumed_job': resumed_job,
                'checkpoint_epoch': epoch, 'resumed_epoch': epoch + 1, 'committed_updates': progress['updates'],
                'test_samples': report['metrics']['test']['count'], 'reconnected': True,
                'train_samples_per_rank': execution['train_samples_per_rank'],
                'train_samples_dropped_per_epoch': execution['train_samples_dropped_per_epoch'],
                'artifact_checksums_verified': True}
    finally:
        ui.shutdown()
        # Scope cleanup to handles created by the diagnostic. This is useful
        # when a renderer error happens before the cancel action is reached.
        for client, job in jobs:
            try:
                if client.status(job)['status'] not in {'succeeded', 'failed', 'cancelled'}:
                    client.cancel(job)
            except (OSError, ValueError, TimeoutError):
                pass
