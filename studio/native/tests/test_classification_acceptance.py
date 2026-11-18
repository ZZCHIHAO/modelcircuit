"""The source/frozen diagnostic exercises real manager-owned worker processes."""
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def acceptance():
    try:
        return importlib.import_module("native.classification_acceptance")
    except ModuleNotFoundError as exc:
        raise AssertionError("Headless native classification acceptance is not implemented") from exc


class ClassificationAcceptanceTests(unittest.TestCase):
    def test_module_import_does_not_require_dpg(self):
        script = "import sys; import native.classification_acceptance; assert not any(name.startswith('dearpygui') for name in sys.modules)"
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_manager_acceptance_writes_replayable_fixture_report(self):
        from server import RunManager
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manager = RunManager(directory / "runs")
            try:
                report = acceptance().check_classification_runtime(manager, directory / "acceptance", device="cpu")
                self.assertEqual(report["status"], "passed")
                self.assertEqual(report["kind"], "native-classification-runtime")
                self.assertTrue(report["test_fixtures"])
                self.assertEqual(report["device"], "cpu")
                self.assertFalse(report["gpu_execution"]["executed"])
                self.assertTrue(report["resume"]["bit_exact"])
                self.assertTrue(report["resume"]["bit_exact_required"])
                self.assertTrue(report["cnn_python_round_trip"]["forward_equal"])
                for kind in ("csv", "image_folder"):
                    task = report["tasks"][kind]
                    self.assertEqual(task["completed_epochs"], 3)
                    self.assertTrue(task["checkpoint_inference"]["predictions_equal"])
                    self.assertEqual(task["checkpoint_inference"]["count"], task["metrics"]["test"]["count"])
                    self.assertGreater(task["metrics"]["train"]["count"], 0)
                    self.assertTrue(all(Path(path).is_file() for path in task["artifacts"].values()))
                persisted = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
                self.assertEqual(persisted, report)
                self.assertEqual(len(manager.runs), 4)
                self.assertTrue(all(record["status"] == "succeeded" for record in manager.runs.values()))
                json.dumps(report, allow_nan=False)
            finally:
                manager.close()

    def test_cuda_report_policy_checks_continuation_without_claiming_bit_identity(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first_path, resumed_path = directory / "first.pt", directory / "resumed.pt"
            history = [{"epoch": 1, "train_loss": 0.5}]
            first = {"device": "cuda:0", "completed_epochs": 1, "global_step": 3, "history": history,
                     "artifacts": {"last_checkpoint": str(first_path)}}
            resumed = {"device": "cuda:0", "completed_epochs": 3, "global_step": 9, "history": history + [{"epoch": 2}, {"epoch": 3}],
                       "artifacts": {"last_checkpoint": str(resumed_path)}}
            # CPU tensors stand in only for on-disk report-policy inputs. This
            # test does not perform or claim any CUDA execution.
            torch.save({"config": {"device": "cuda:0"}, "completed_epochs": 1, "global_step": 3, "history": history}, first_path)
            torch.save({"config": {"device": "cuda:0", "resume_checkpoint": str(first_path)}, "completed_epochs": 3, "global_step": 9,
                        "history": resumed["history"], "model_state": {"weight": torch.ones(1)}}, resumed_path)
            policy = acceptance()._resume_check({}, resumed, first, "partial", "continued", device="cuda:0")
            self.assertTrue(policy["continuation_succeeded"])
            self.assertFalse(policy["bit_exact_required"])
            self.assertIsNone(policy["bit_exact"])
            self.assertEqual(policy["origin_device"], "cuda:0")


if __name__ == "__main__":
    unittest.main()
