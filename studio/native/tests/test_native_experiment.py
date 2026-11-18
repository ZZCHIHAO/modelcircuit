from pathlib import Path
import tempfile
import unittest

from native.document import Document
from native.ui import NativeWorkbench


class NativeExperimentTests(unittest.TestCase):
    def test_automatic_workspace_contains_experiment_and_keeps_legacy_draft(self):
        with tempfile.TemporaryDirectory() as temporary:
            ui = NativeWorkbench(Document(), None, temporary, temporary)
            ui.experiment["lr"] = 0.2
            ui._save_draft()
            from native.workspace_state import read_workspace
            _, experiment = read_workspace(Path(temporary) / "workspace.json")
            self.assertEqual(experiment["lr"], 0.2)
            self.assertTrue((Path(temporary) / "draft.json").is_file())

    def test_model_only_python_change_preserves_experiment_and_stales_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            ui = NativeWorkbench(Document(), None, temporary, temporary)
            ui.experiment["seed"] = 123
            key = ui._experiment_identity()
            code = ui.document.export_python().replace("out_features=2", "out_features=3")
            ui.document.import_python(code)
            self.assertEqual(ui.experiment["seed"], 123)
            self.assertNotEqual(ui._experiment_identity(), key)


if __name__ == "__main__":
    unittest.main()
