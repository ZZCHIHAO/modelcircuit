import threading
from pathlib import Path
import tempfile
import time
import unittest


class ExperimentPanelTests(unittest.TestCase):
    def test_background_query_never_blocks_the_caller_and_reports_errors(self):
        from native.experiment_panel import BackgroundQueries
        gate = threading.Event()
        query = BackgroundQueries()
        caller = threading.get_ident()
        def slow(cancelled):
            gate.wait(2)
            return threading.get_ident()
        try:
            started = time.monotonic()
            query.submit("inspect", slow)
            self.assertLess(time.monotonic() - started, 0.1)
            self.assertEqual(query.poll(), [])
            gate.set()
            result = self._wait(query)
            self.assertNotEqual(result[0]["value"], caller)
            query.submit("devices", lambda cancelled: 1 / 0)
            self.assertIn("division", self._wait(query)[0]["error"])
        finally:
            gate.set(); query.close()

    def test_superseded_query_cannot_restore_stale_inspection(self):
        from native.experiment_panel import BackgroundQueries
        gate = threading.Event(); query = BackgroundQueries()
        try:
            query.submit("inspect", lambda cancelled: (gate.wait(2), "old")[-1])
            query.submit("inspect", lambda cancelled: "new")
            result = self._wait(query)
            self.assertEqual(result[0]["value"], "new")
            gate.set(); time.sleep(0.03)
            self.assertEqual(query.poll(), [])
        finally:
            gate.set(); query.close()

    def test_real_history_has_distinct_train_validation_and_final_test(self):
        from native.experiment_panel import result_view
        result = {"profile": "classification-v2", "device": "cuda:0", "history": [{"epoch": 1, "train_loss": 0.8, "val_loss": 1.1}], "metrics": {"test": {"loss": 0.9, "accuracy": 0.5, "count": 10}}, "best_epoch": 1, "artifacts": {"best_checkpoint": "D:/best.pt"}, "probe_context": {"split": "val", "epoch": 1, "mode": "eval", "sample_ids": ["s1"], "gradient_source": "best epoch"}}
        view = result_view(result)
        self.assertEqual(view["train"], [[1], [0.8]])
        self.assertEqual(view["val"], [[1], [1.1]])
        self.assertIn("test", view["summary"])
        self.assertIn("cuda:0", view["summary"])
        self.assertIn("val", view["probe"])
        self.assertIn("best.pt", view["artifacts"])

    def test_live_real_curves_consume_completed_epoch_events_not_latest_batch(self):
        from native.experiment_panel import progress_points
        epoch = {"seq": 1, "type": "progress", "phase": "epoch", "epoch": 1, "step": 3, "train_loss": 0.8, "val_loss": 1.1}
        record = {"events": [epoch, {"seq": 2, "type": "progress", "phase": "train", "epoch": 2, "step": 4, "loss": 0.5}], "progress": {"phase": "train", "epoch": 2, "step": 4, "loss": 0.5}}
        self.assertEqual(progress_points(record, 0, real=True), [epoch])
        self.assertEqual(progress_points(record, 1, real=True), [])
        self.assertEqual(progress_points(record, 0, real=False), [record["progress"]])

    def test_dataset_summary_discloses_ignored_sidecars(self):
        from native.experiment_panel import manifest_summary
        summary = manifest_summary({"kind": "image_folder", "input_shape": [None, 1, 16, 18], "class_names": ["a", "b"], "counts": {"train": 12, "val": 4, "test": 4}, "ignored_files": ["a/x.png.baiduyun.uploading.cfg"], "warnings": ["Ignored known sidecar"]})
        self.assertIn("忽略", summary)
        self.assertIn("1", summary)

    def test_paired_summary_identifies_both_input_shapes(self):
        from native.experiment_panel import manifest_summary
        summary = manifest_summary({"kind": "paired_csv", "input_shapes": {"image": [None, 1, 8, 10], "tabular": [None, 2]}, "class_names": ["a", "b"], "counts": {"train": 12, "val": 4, "test": 4}})
        self.assertIn("image", summary)
        self.assertIn("tabular", summary)
        self.assertIn("8", summary)

    def test_header_reader_accepts_512_features_and_five_paired_metadata_columns(self):
        from native.experiment_panel import csv_columns
        columns = ["f" + str(index) for index in range(512)] + ["label", "patient_id", "split", "sample_id", "image_path"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "paired.csv"
            path.write_text(",".join(columns) + "\n", encoding="utf-8")
            self.assertEqual(csv_columns(path), columns)

    @staticmethod
    def _wait(query):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            result = query.poll()
            if result:
                return result
            time.sleep(0.005)
        raise AssertionError("query did not finish")


if __name__ == "__main__":
    unittest.main()
