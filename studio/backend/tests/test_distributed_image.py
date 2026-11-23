import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest


class DistributedImagePlanTests(unittest.TestCase):
    def test_epoch_shards_have_complete_updates_and_no_duplicate_samples(self):
        from studio_core.distributed_image import epoch_shard

        participants = [epoch_shard(19, 24, 1, rank, 2, 2, 2) for rank in range(2)]
        self.assertEqual([len(indices) for indices in participants], [8, 8])
        self.assertEqual(len(set(participants[0] + participants[1])), 16)
        independent_order = epoch_shard(19, 24, 1, 0, 1, 4, 2)
        for start in range(0, len(independent_order), 8):
            index = start // 2
            self.assertCountEqual(participants[0][index:index + 4] + participants[1][index:index + 4],
                                  independent_order[start:start + 8])
        later = [epoch_shard(19, 24, 2, rank, 2, 2, 2) for rank in range(2)]
        self.assertNotEqual(participants, later)
        with self.assertRaisesRegex(ValueError, 'effective global batch'):
            epoch_shard(7, 24, 1, 0, 2, 2, 2)


class DistributedImageRuntimeTests(unittest.TestCase):
    def _run_workers(self, root, name, project, data, *, epochs, world=2, batch=1, resume=None):
        # Windows' vendored Torch lacks libuv support in the torchrun launcher.
        # Start the exact worker entry point with the same rank environment,
        # using a loopback store that supports the installed Gloo build.
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = str(listener.getsockname()[1])
        command = [sys.executable, '-B', '-m', 'studio_core.distributed_image', '--device', 'cpu',
                   '--project', str(project), '--data', str(data), '--out', str(root / name),
                   '--progress', str(root / (name + '.json')), '--epochs', str(epochs),
                   '--per-device-batch', str(batch), '--accumulation', '2']
        if resume:
            command.extend(('--resume', str(root / resume)))
        logs = []
        workers = []
        try:
            for rank in range(world):
                environment = {**os.environ, 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': port,
                               'USE_LIBUV': '0', 'LOCAL_RANK': str(rank), 'RANK': str(rank),
                               'LOCAL_WORLD_SIZE': str(world), 'WORLD_SIZE': str(world)}
                log = (root / f'{name}-rank-{rank}.log').open('wb')
                logs.append(log)
                workers.append(subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT))
            for rank, process in enumerate(workers):
                self.assertEqual(process.wait(timeout=50), 0,
                                 (root / f'{name}-rank-{rank}.log').read_text(encoding='utf-8', errors='replace'))
        finally:
            for process in workers:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
            for log in logs:
                log.close()
        return root / name

    def test_two_ranks_match_one_global_batch_and_resume_at_an_epoch_boundary(self):
        import torch
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest('PyTorch Gloo backend unavailable')
        from studio_core.project_examples import create_example_project
        from studio_core.project_runner import compare_project_runs, predict_project, read_project_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project, data, _ = create_example_project(root / 'fixture')
            one = self._run_workers(root, 'one-rank', project, data, epochs=2, world=1, batch=2)
            two = self._run_workers(root, 'two-ranks', project, data, epochs=2)
            first = self._run_workers(root, 'first-epoch', project, data, epochs=1)
            resumed = self._run_workers(root, 'resumed', project, data, epochs=2, resume=first.name)
            reports = [read_project_run(directory, verify=True) for directory in (one, two, first, resumed)]
            for report in reports:
                self.assertEqual(report['status'], 'succeeded')
                self.assertEqual(report['changed_artifacts'], [])
                self.assertEqual(report['missing_artifacts'], [])
                self.assertEqual([report['metrics'][split]['count'] for split in ('train', 'val', 'test')], [4, 4, 4])
                self.assertEqual(report['execution']['effective_global_batch'], 4)
            self.assertEqual(reports[1]['execution']['world_size'], 2)
            self.assertEqual(reports[-1]['resumed_from_epoch'], 1)
            self.assertEqual(reports[1]['global_step'], 2)
            comparison = compare_project_runs(*reports[:2])
            self.assertTrue(comparison['comparable'])
            self.assertAlmostEqual(comparison['metrics']['loss']['delta'], 0, places=6)
            self.assertEqual(reports[1]['history'], reports[-1]['history'])
            states = [torch.load(directory / 'last.pt', map_location='cpu', weights_only=True)['model_state']
                      for directory in (one, two, resumed)]
            torch.testing.assert_close(states[0], states[1], rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(states[1], states[2], rtol=0, atol=0)
            progress = json.loads((root / 'resumed.json').read_text(encoding='utf-8'))
            self.assertEqual((progress['epoch'], progress['updates'], progress['resume_available']), (2, 2, True))
            prediction = predict_project(project, next((data / 'test' / 'dark').glob('*.png')), root / 'predict', weights=two)
            result = read_project_run(prediction['artifacts']['project_run'], verify=True)
            self.assertEqual(result['cases'], 1)
            self.assertEqual(result['changed_artifacts'], [])
