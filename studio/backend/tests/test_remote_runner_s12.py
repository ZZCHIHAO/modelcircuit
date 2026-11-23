import importlib.util
import io
import json
from pathlib import Path
import os
import sys
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import zipfile


GPU0 = 'GPU-00000000-1111-2222-3333-444444444444'
GPU1 = 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
JOB = '01' * 16
FILES = {'project.json': '{}', 'dataset.json': '{}', 'model.py': '# syntax-only model'}


class RemoteRunnerTests(TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('studio_core.remote_runner'), 'remote job runner is missing')
        from studio_core import remote_runner
        self.runner = remote_runner
        fixture = TemporaryDirectory()
        self.addCleanup(fixture.cleanup)
        self.root = Path(fixture.name) / 'jobs'

    def request(self, **options):
        return {'files': FILES, 'data_path': '/research/data/fixture', 'gpu_uuids': [GPU1, GPU0],
                'per_device_batch_size': 2, 'accumulation_steps': 3, 'epochs': 2, 'request_id': JOB, **options}

    def job(self, state='succeeded'):
        job = self.root / JOB
        job.mkdir(parents=True)
        (job / 'run').mkdir()
        self.runner._write_json(job / 'state.json', {'job_id': JOB, 'status': state})
        return job

    def test_probe_parses_csv_without_initializing_cuda(self):
        listing = f'0, {GPU0}, "NVIDIA, Test A", 14, 200, 7\n1, {GPU1}, Card B, 0, 256, 0\n'
        self.assertEqual(self.runner._parse_gpus(listing), [
            {'index': 0, 'uuid': GPU0, 'name': 'NVIDIA, Test A', 'memory_used_mib': 14,
             'memory_total_mib': 200, 'utilization': 7},
            {'index': 1, 'uuid': GPU1, 'name': 'Card B', 'memory_used_mib': 0,
             'memory_total_mib': 256, 'utilization': 0},
        ])

    def test_submission_passes_only_validated_explicit_order_and_uses_idempotent_id(self):
        with (patch.object(self.runner, '_probe_gpus', return_value=[{'uuid': GPU0}, {'uuid': GPU1}]),
              patch.object(self.runner, '_ddp_available', return_value=True),
              patch.object(self.runner, '_read_project_for_job', return_value={'graph': {}}),
              patch.object(self.runner, '_launch_supervisor') as launch):
            accepted = self.runner.submit(self.root, self.request())
            repeated = self.runner.submit(self.root, self.request())
            self.assertEqual(accepted['job_id'], JOB)
            self.assertEqual(repeated['job_id'], JOB)
            launch.assert_called_once()
            with self.assertRaisesRegex(ValueError, 'different request'):
                self.runner.submit(self.root, self.request(epochs=3))
        spec = json.loads((self.root / JOB / 'spec.json').read_text(encoding='utf-8'))
        command, env = self.runner._worker_command(self.root / JOB, spec)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], f'{GPU1},{GPU0}')
        self.assertIn('--nproc-per-node=2', command)
        self.assertEqual(command[command.index('--data') + 1], '/research/data/fixture')
        self.assertEqual(accepted['effective_batch_size'], 12)

    def test_invalid_or_duplicate_gpu_selection_never_creates_job(self):
        for selection in ([GPU0], [GPU0, GPU0], [GPU0, 'GPU-temp'], [GPU0, GPU1 + ',all']):
            with self.assertRaises(ValueError):
                self.runner.submit(self.root, self.request(gpu_uuids=selection))
        self.assertFalse(self.root.exists())

    def test_rejected_dataset_paths_never_create_job(self):
        for invalid in ('relative/path', '/root/../dataset', '/r\x00oot', '/data\nnow', '//hostname/data'):
            with self.assertRaises(ValueError):
                self.runner.submit(self.root, self.request(data_path=invalid))
        self.assertFalse(self.root.exists())

    def test_unavailable_gpu_never_creates_job(self):
        with (patch.object(self.runner, '_probe_gpus', return_value=[{'uuid': GPU0}]),
              patch.object(self.runner, '_ddp_available', return_value=True)):
            with self.assertRaisesRegex(ValueError, 'Unavailable GPU'):
                self.runner.submit(self.root, self.request())
        self.assertFalse((self.root / JOB).exists())

    def test_cancel_writes_request_and_continues_showing_checkpoint_after_terminal_status(self):
        job = self.job('running')
        self.runner._write_json(job / 'progress.json', {'epoch': 1, 'epochs': 2, 'updates': 2,
                                                         'resume_available': True})
        self.assertFalse(self.runner.status(self.root, JOB)['resume_available'])
        (job / 'run' / 'last.pt').write_bytes(b'checkpoint')
        record = self.runner.cancel(self.root, JOB)
        self.assertTrue((job / 'cancel').is_file())
        self.assertEqual(record['status'], 'cancel_requested')
        self.assertTrue(record['resume_available'])
        self.assertTrue(record['progress']['resume_available'])
        self.runner._write_json(job / 'state.json', {'job_id': JOB, 'status': 'cancelled'})
        self.assertEqual(self.runner.status(self.root, JOB)['status'], 'cancelled')

    def test_status_rejects_paths_and_unknown_jobs(self):
        self.job()
        for job in ('../../another', '02' * 16, str(self.root / JOB)):
            with self.assertRaises(ValueError):
                self.runner.status(self.root, job)

    def test_fetch_packs_only_verified_artifacts_without_datasets(self):
        import hashlib
        job = self.job()
        run = job / 'run'
        metrics = {'test': {'loss': 0.25, 'count': 4}}
        raw = json.dumps({'metrics': metrics}).encode()
        (run / 'metrics.json').write_bytes(raw)
        (run / 'training-input.bin').write_bytes(b'do not fetch this')
        report = {'format': 'modelcircuit-project-run', 'version': 1, 'operation': 'train',
                  'run_id': JOB, 'status': 'succeeded', 'metrics': metrics,
                  'artifacts': {'metrics': 'metrics.json'}, 'artifact_sha256': {'metrics': hashlib.sha256(raw).hexdigest()}}
        self.runner._write_json(run / 'project-run.json', report)
        stream = io.BytesIO()
        self.runner.fetch(self.root, JOB, stream)
        with zipfile.ZipFile(stream) as archive:
            self.assertEqual(set(archive.namelist()), {'project-run.json', 'metrics.json'})
            self.assertEqual(archive.read('metrics.json'), raw)
        (run / 'metrics.json').write_bytes(b'altered')
        with self.assertRaisesRegex(ValueError, 'changed or missing'):
            self.runner.fetch(self.root, JOB, io.BytesIO())

    def test_fetch_rejects_incomplete_jobs(self):
        self.job('running')
        with self.assertRaisesRegex(ValueError, 'Only succeeded'):
            self.runner.fetch(self.root, JOB, io.BytesIO())


    def test_gpu_leases_reject_intersection_and_release_partial_reservations(self):
        third = 'GPU-01234567-89ab-cdef-0123-456789abcdef'
        self.root.mkdir()
        held = {}

        def lock(handle, _flags):
            identity = os.fstat(handle).st_ino
            if identity in held:
                raise BlockingIOError('already locked')
            held[identity] = handle

        os_close = os.close

        def close(handle):
            identity = os.fstat(handle).st_ino
            if held.get(identity) == handle:
                del held[identity]
            os_close(handle)

        fake_fcntl = SimpleNamespace(flock=lock, LOCK_EX=2, LOCK_NB=4)
        with (patch.dict(sys.modules, {'fcntl': fake_fcntl}), patch.object(self.runner.os, 'close', side_effect=close),
              patch.object(self.runner.os, 'O_NOFOLLOW', 0, create=True)):
            with self.runner._reserve_gpus(self.root, [GPU0, GPU1]):
                with self.assertRaisesRegex(ValueError, 'already reserved'):
                    with self.runner._reserve_gpus(self.root, [third, GPU1]):
                        self.fail('Intersecting job acquired a held GPU')
                self.assertEqual(len(held), 2)
                with self.runner._reserve_gpus(self.root, [third]):
                    self.assertEqual(len(held), 3)
            self.assertFalse(held)
            with self.runner._reserve_gpus(self.root, [GPU0, GPU1, third]):
                self.assertEqual(len(held), 3)

    def test_only_selected_active_cards_are_rejected(self):
        row = f'{GPU1}, 12345\n'
        with patch.object(self.runner.subprocess, 'run', return_value=SimpleNamespace(stdout=row)):
            self.runner._check_selected_gpu_activity([GPU0])
            with self.assertRaisesRegex(ValueError, 'active compute process'):
                self.runner._check_selected_gpu_activity([GPU0, GPU1])
