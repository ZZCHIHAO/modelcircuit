from pathlib import Path
import tempfile
import unittest

from native.paths import resource_root


class ClassificationRenderTests(unittest.TestCase):
    def test_native_callbacks_train_csv_and_images_and_restore_workspace(self):
        from native.classification_render_acceptance import check_classification_render
        from server import RunManager
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = RunManager(root / "runs")
            try:
                report = check_classification_render(manager, root / "data", resource_root(), root / "render", device="cpu")
                self.assertEqual(report["status"], "passed")
                self.assertEqual(set(report["runs"]), {"csv", "image_folder"})
                self.assertTrue(report["workspace_experiment_restored"])
                for name in report["screenshots"]:
                    self.assertTrue((root / "render" / name).is_file())
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
