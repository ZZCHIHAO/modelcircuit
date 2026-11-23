import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


class RemoteExecutionTests(unittest.TestCase):
    def client(self):
        self.assertIsNotNone(importlib.util.find_spec('studio_core.remote_client'), 'S12 SSH client is missing')
        from studio_core import remote_client
        return remote_client

    def test_validated_gpu_uuids_describe_the_effective_batch(self):
        module = self.client()
        gpus = ['GPU-00000000-0000-0000-0000-000000000001', 'GPU-00000000-0000-0000-0000-000000000002']
        self.assertEqual(module.batch_plan(gpus, 8, 2), {'per_device': 8, 'per_step': 16, 'effective': 32})
        with self.assertRaises(ValueError):
            module.batch_plan([gpus[0], gpus[0]], 8, 2)

    def test_ssh_profile_rejects_options_and_quotes_paths_as_arguments(self):
        import shlex
        module = self.client()
        with self.assertRaises(ValueError):
            module.RemoteProfile('-oProxyCommand=untrusted', '/opt/python', '/work')
        root = "/user/work space's " + '$notes'
        profile = module.RemoteProfile('my-lab', '/opt/tools/python3', root.strip())
        client = module.RemoteClient(profile, Path('.'))
        command = client.ssh_command(['arg with spaces', root])
        self.assertEqual(shlex.split(command[-1]), ['arg with spaces', root])
        self.assertEqual(command[-2], 'my-lab')
        self.assertIn('StrictHostKeyChecking=yes', command)

    def test_received_report_is_validated_and_published_atomically(self):
        module = self.client()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'result.zip'
            metrics = {'test': {'count': 4, 'accuracy': 0.5}}
            raw = json.dumps({'metrics': metrics}).encode()
            report = {'format': 'modelcircuit-project-run', 'version': 1, 'operation': 'train', 'run_id': 'test',
                      'status': 'succeeded', 'metrics': metrics, 'artifacts': {'metrics': 'metrics.json'},
                      'artifact_sha256': {'metrics': hashlib.sha256(raw).hexdigest()}}
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('project-run.json', json.dumps(report))
                stream.writestr('metrics.json', raw)
            destination = root / 'complete'
            self.assertEqual(module.unpack_result(archive, destination), destination / 'project-run.json')
            self.assertTrue((destination / 'metrics.json').is_file())
            with zipfile.ZipFile(archive, 'a') as stream:
                stream.writestr('../escape.txt', b'should never be written')
            with self.assertRaises(ValueError):
                module.unpack_result(archive, root / 'rejected')
            self.assertFalse((root / 'rejected').exists())
            self.assertFalse((root / 'escape.txt').exists())
