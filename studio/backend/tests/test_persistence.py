import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from server import RunManager


class PersistenceTests(unittest.TestCase):
    def test_atomic_record_replace_retries_a_transient_windows_reader_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = RunManager(directory)
            record = {'run_id': 'read_lock', 'status': 'queued'}
            manager._save(record)
            original_replace = os.replace
            attempts = []
            def transient_replace(source, destination):
                attempts.append(str(destination))
                if len(attempts) <= 2:
                    raise PermissionError(5, 'Simulated Windows reader denies delete-sharing')
                return original_replace(source, destination)
            record['status'] = 'succeeded'
            with patch('server.os.replace', side_effect=transient_replace):
                manager._save(record)
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads((Path(directory) / 'read_lock.json').read_text(encoding='utf-8'))['status'], 'succeeded')
            self.assertFalse((Path(directory) / 'read_lock.tmp').exists())


if __name__ == '__main__':
    unittest.main()
