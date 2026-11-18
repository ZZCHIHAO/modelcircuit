import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from native.document import Document
from native.workspace_state import save_workspace,synthetic_config


class NativeStorageRetryTests(unittest.TestCase):
    def test_transient_windows_sharing_lock_does_not_lose_document_or_workspace(self):
        real_replace=os.replace
        for workspace in (False,True):
            with self.subTest(workspace=workspace),tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'saved.json';attempts=[]
                def transient(source,target):
                    attempts.append(1)
                    if len(attempts)<3:raise PermissionError('simulated Windows sharing lock')
                    return real_replace(source,target)
                with patch('os.replace',side_effect=transient):
                    if workspace:save_workspace(path,Document(),synthetic_config())
                    else:Document().save_json(path)
                self.assertEqual(len(attempts),3)
                self.assertTrue(json.loads(path.read_text(encoding='utf-8')))
                self.assertEqual([item.name for item in Path(directory).iterdir()],['saved.json'])

    def test_persistent_lock_is_bounded_and_preserves_the_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'saved.json';path.write_text('previous bytes',encoding='utf-8')
            with patch('os.replace',side_effect=PermissionError('persistent lock')) as replace:
                with self.assertRaises(PermissionError):Document().save_json(path)
            self.assertEqual(replace.call_count,5)
            self.assertEqual(path.read_text(),'previous bytes')
            self.assertEqual([item.name for item in Path(directory).iterdir()],['saved.json'])


if __name__=='__main__':unittest.main()
