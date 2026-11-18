from pathlib import Path
import tempfile
import unittest

from native.paths import resource_root


class S3RenderTests(unittest.TestCase):
    def test_paired_named_chip_callbacks_execute_real_training_and_roundtrips(self):
        from native.s3_render_acceptance import check_s3_render
        from server import RunManager
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manager = RunManager(directory / "runs")
            try:
                report = check_s3_render(manager, directory / "data", resource_root(), directory / "output", device="cpu")
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["device"], "cpu")
                self.assertTrue(report["workspace_roundtrip"])
                self.assertTrue(report["python_roundtrip"])
                self.assertTrue(report["qualified_probe"])
                for name in report["screenshots"]:
                    self.assertTrue((directory / "output" / name).is_file())
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
