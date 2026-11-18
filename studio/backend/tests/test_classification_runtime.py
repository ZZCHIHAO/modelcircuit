"""Tiny real-file epoch training, artifact inference, and CPU resume tests."""
import csv
import importlib
import json
import tempfile
import unittest
from pathlib import Path


def training():
    try:
        return importlib.import_module("studio_core.classification_runtime")
    except ModuleNotFoundError as exc:
        raise AssertionError("real classification training is not implemented") from exc


def classifier(dropout=True):
    instances = [{"id": "hidden", "kind": "Linear", "params": {"in_features": 2, "out_features": 6}},
                 {"id": "relu", "kind": "ReLU", "params": {}},
                 {"id": "head", "kind": "Linear", "params": {"in_features": 6, "out_features": 2}}]
    calls = [{"id": "hidden_call", "instance_id": "hidden", "inputs": {"x": "input:x"}},
             {"id": "relu_call", "instance_id": "relu", "inputs": {"x": "call:hidden_call"}}]
    source = "call:relu_call"
    if dropout:
        instances.append({"id": "drop", "kind": "Dropout", "params": {"p": 0.25}})
        calls.append({"id": "drop_call", "instance_id": "drop", "inputs": {"x": source}})
        source = "call:drop_call"
    calls.append({"id": "head_call", "instance_id": "head", "inputs": {"x": source}})
    return {"schema_version": 1, "id": "csv_classifier", "revision": 1, "inputs": [{"id": "x", "shape": [None, 2], "dtype": "float32"}],
            "instances": instances, "calls": calls, "outputs": [{"id": "logits", "source": "call:head_call"}], "definitions": []}


class ClassificationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        path = self.root / "samples.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["x", "y", "label", "split"])
            for index in range(36):
                writer.writerow([index / 20, (index % 7) / 5, "positive" if index % 2 else "negative", "train" if index < 20 else "val" if index < 28 else "test"])
        self.config = {"profile": "classification-v2", "data": {"kind": "csv", "path": str(path), "feature_columns": ["x", "y"], "target_column": "label", "split_column": "split"},
                       "epochs": 4, "batch_size": 5, "lr": 0.02, "optimizer": "adamw", "seed": 77, "device": "cpu"}

    def run_training(self, name, **updates):
        return training().run_classification(classifier(), dict(self.config, **updates), self.root / name)

    def test_real_epochs_write_artifacts_and_validation_selected_provenance(self):
        events = []
        result = training().run_classification(classifier(), self.config, self.root / "run", progress=events.append)
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["profile"], "classification-v2")
        self.assertEqual(result["data_source"], "csv")
        self.assertEqual(result["completed_epochs"], 4)
        self.assertEqual(result["global_step"], 16)
        self.assertEqual(len(result["history"]), 4)
        expected = min(result["history"], key=lambda entry: entry["val_loss"])
        self.assertEqual(result["best_epoch"], expected["epoch"])
        self.assertEqual(result["metrics"]["test"]["count"], 8)
        self.assertEqual(result["probe_context"]["split"], "val")
        self.assertEqual(result["probe_context"]["epoch"], result["best_epoch"])
        self.assertIn("training", result["probe_context"]["gradient_source"])
        self.assertEqual(result["probes"]["input:x"]["shape"], [5, 2])
        self.assertTrue(any(measurement["update_norm"] > 0 for measurement in result["parameters"]))
        self.assertTrue(all(Path(path).is_file() for path in result["artifacts"].values()))
        run_spec = json.loads(Path(result["artifacts"]["run_spec"]).read_text(encoding="utf-8"))
        self.assertEqual(run_spec["environment"]["torch"], str(self.torch.__version__))
        self.assertTrue(run_spec["environment"]["python"])
        self.assertTrue(run_spec["environment"]["pillow"])
        self.assertTrue(run_spec["environment"]["platform"])
        self.assertIsNone(run_spec["environment"]["cuda"])
        self.assertIsNone(run_spec["environment"]["cudnn"])
        with Path(result["artifacts"]["predictions"]).open(newline="") as stream:
            predictions = list(csv.DictReader(stream))
        self.assertEqual(len(predictions), 8)
        self.assertEqual(len({row["sample_id"] for row in predictions}), 8)
        self.assertTrue(any(event["phase"] == "train" for event in events))
        epoch_events = [event for event in events if event["phase"] == "epoch"]
        self.assertEqual(len(epoch_events), 4)
        self.assertEqual([event["train_loss"] for event in epoch_events], [entry["train_loss"] for entry in result["history"]])
        self.assertEqual([event["val_loss"] for event in epoch_events], [entry["val_loss"] for entry in result["history"]])
        self.assertEqual(sum(event["phase"] == "test" and event["batch"] == 1 for event in events), 1)
        self.assertLess(len(json.dumps(result, allow_nan=False).encode()), 2_000_000)

    def test_best_checkpoint_reproduces_saved_test_inference(self):
        result = self.run_training("run")
        from studio_core.classification_data import build_datasets
        from studio_core.runtime import build_model
        checkpoint = self.torch.load(result["artifacts"]["best_checkpoint"], weights_only=True, map_location="cpu")
        model = build_model(classifier(), device="cpu", capture=False)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        manifest = json.loads(Path(result["artifacts"]["manifest"]).read_text(encoding="utf-8"))
        dataset = build_datasets(manifest)["test"]
        with Path(result["artifacts"]["predictions"]).open(newline="") as stream:
            predictions = list(csv.DictReader(stream))
        with self.torch.no_grad():
            for index in range(len(dataset)):
                features, label, identifier = dataset[index]
                actual = model(x=features.unsqueeze(0))["logits"].argmax(1).item()
                self.assertEqual(int(predictions[index]["prediction"]), actual)
                self.assertEqual(predictions[index]["sample_id"], identifier)

    def test_cpu_epoch_boundary_resume_is_bit_exact_with_dropout_and_adamw(self):
        continuous = self.run_training("continuous")
        partial = self.run_training("partial", epochs=2)
        resumed = self.run_training("resumed", resume_checkpoint=partial["artifacts"]["last_checkpoint"])
        load = lambda path: self.torch.load(path, weights_only=True, map_location="cpu")
        expected = load(continuous["artifacts"]["last_checkpoint"])
        actual = load(resumed["artifacts"]["last_checkpoint"])
        for name, tensor in expected["model_state"].items():
            self.assertTrue(self.torch.equal(actual["model_state"][name], tensor), name)
        self.assertEqual(resumed["history"], continuous["history"])
        self.assertEqual(resumed["metrics"], continuous["metrics"])
        self.assertEqual(resumed["parameters"], continuous["parameters"])

    def test_changed_dataset_optimizer_or_graph_rejects_resume(self):
        partial = self.run_training("partial", epochs=1)
        checkpoint = partial["artifacts"]["last_checkpoint"]
        for updates in ({"lr": 0.03}, {"batch_size": 4}, {"optimizer": "sgd"}):
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, "checkpoint.*identity|identity.*checkpoint"):
                self.run_training("bad", resume_checkpoint=checkpoint, **updates)
        graph = classifier()
        graph["instances"][-1]["params"]["p"] = 0.5
        with self.assertRaisesRegex(ValueError, "checkpoint.*identity|identity.*checkpoint"):
            training().run_classification(graph, dict(self.config, resume_checkpoint=checkpoint), self.root / "bad_graph")
        path = Path(self.config["data"]["path"])
        path.write_text(path.read_text().replace("0.05,0.2", "0.06,0.2"))
        with self.assertRaisesRegex(ValueError, "checkpoint.*identity|identity.*checkpoint"):
            self.run_training("bad_data", resume_checkpoint=checkpoint)

    def test_cancel_mid_epoch_preserves_last_complete_checkpoint(self):
        cancelled = [False]
        def progress(event):
            if event["phase"] == "train" and event["epoch"] == 2 and event["batch"] == 1:
                cancelled[0] = True
        with self.assertRaises(InterruptedError):
            training().run_classification(classifier(), self.config, self.root / "cancelled", progress=progress, cancelled=lambda: cancelled[0])
        checkpoint = self.torch.load(self.root / "cancelled" / "last.pt", weights_only=True, map_location="cpu")
        self.assertEqual(checkpoint["completed_epochs"], 1)
        self.assertEqual(checkpoint["global_step"], 4)
        self.assertFalse((self.root / "cancelled" / "predictions.csv").exists())

    def test_expected_digest_shape_and_multiple_root_inputs_fail_before_training(self):
        with self.assertRaisesRegex(ValueError, "digest"):
            self.run_training("digest", expected_dataset_digest="0" * 64)
        graph = classifier()
        graph["inputs"].append({"id": "extra", "shape": [None, 2], "dtype": "float32"})
        with self.assertRaisesRegex(ValueError, "one.*input|single.*input"):
            training().run_classification(graph, self.config, self.root / "multi")
        graph = classifier()
        graph["instances"][2]["params"]["out_features"] = 3
        with self.assertRaisesRegex(ValueError, "logits"):
            training().run_classification(graph, self.config, self.root / "shape")

    def test_png_imagefolder_trains_a_real_cnn_with_sgd(self):
        from PIL import Image
        image_root = self.root / "images"
        for s, split in enumerate(("train", "val", "test")):
            for label in range(2):
                for index in range(4):
                    target = image_root / split / str(label) / (str(index) + ".png")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (10, 11), (label * 100 + s * 10 + index, label * 90, index * 30)).save(target)
        graph = {"schema_version": 1, "resource_profile": "classification-v2", "id": "image_classifier", "revision": 1,
                 "inputs": [{"id": "x", "shape": [None, 3, 8, 8], "dtype": "float32"}],
                 "instances": [{"id": "conv", "kind": "Conv2d", "params": {"in_channels": 3, "out_channels": 4, "kernel_size": 3}},
                               {"id": "pool", "kind": "AdaptiveAvgPool2d", "params": {"output_size": [1, 1]}},
                               {"id": "flat", "kind": "Flatten", "params": {}},
                               {"id": "head", "kind": "Linear", "params": {"in_features": 4, "out_features": 2}}],
                 "calls": [{"id": "conv_call", "instance_id": "conv", "inputs": {"x": "input:x"}},
                           {"id": "pool_call", "instance_id": "pool", "inputs": {"x": "call:conv_call"}},
                           {"id": "flat_call", "instance_id": "flat", "inputs": {"x": "call:pool_call"}},
                           {"id": "head_call", "instance_id": "head", "inputs": {"x": "call:flat_call"}}],
                 "outputs": [{"id": "logits", "source": "call:head_call"}], "definitions": []}
        config = dict(self.config, epochs=2, optimizer="sgd", batch_size=3,
                      data={"kind": "image_folder", "path": str(image_root), "split_mode": "directories", "image_size": [8, 8], "channels": 3})
        result = training().run_classification(graph, config, self.root / "cnn")
        self.assertEqual(result["data_source"], "image_folder")
        self.assertEqual(result["global_step"], 6)
        self.assertEqual(result["metrics"]["test"]["count"], 8)
        self.assertEqual(result["probes"]["call:conv_call"]["shape"], [3, 4, 6, 6])

    def test_resume_after_cancellation_matches_uninterrupted_run(self):
        reference = self.run_training("reference")
        flag = [False]
        def progress(event):
            if event["phase"] == "train" and event["epoch"] == 2 and event["batch"] == 2:
                flag[0] = True
        with self.assertRaises(InterruptedError):
            training().run_classification(classifier(), self.config, self.root / "interrupted", progress=progress, cancelled=lambda: flag[0])
        resumed = self.run_training("restored", resume_checkpoint=str(self.root / "interrupted" / "last.pt"))
        self.assertEqual(reference["history"], resumed["history"])
        self.assertEqual(reference["metrics"], resumed["metrics"])

    def test_inference_only_resume_and_ui_layout_change_keep_epoch_state(self):
        completed = self.run_training("first")
        graph = classifier()
        graph["name"] = "renamed model"
        graph["revision"] = 9
        graph["calls"][0]["position"] = {"x": 991, "y": 77}
        resumed = training().run_classification(graph, dict(self.config, resume_checkpoint=completed["artifacts"]["last_checkpoint"]), self.root / "inference")
        self.assertEqual(resumed["history"], completed["history"])
        self.assertEqual(resumed["metrics"], completed["metrics"])

    def test_run_restores_callers_cpu_and_python_random_states(self):
        import random
        random.seed(957)
        self.torch.manual_seed(867)
        expected_python = random.getstate()
        expected_torch = self.torch.get_rng_state().clone()
        self.run_training("rng", epochs=1)
        self.assertEqual(random.getstate(), expected_python)
        self.assertTrue(self.torch.equal(self.torch.get_rng_state(), expected_torch))

    def test_cpu_training_does_not_seed_unselected_cuda_devices(self):
        from unittest.mock import patch
        with patch.object(self.torch.cuda, "manual_seed_all", side_effect=AssertionError("Unselected CUDA RNG was changed")):
            result = self.run_training("cpu_only", epochs=1)
        self.assertEqual(result["device"], "cpu")

    def test_large_valid_output_names_keep_control_summary_below_two_megabytes(self):
        graph = classifier(False)
        graph["outputs"].extend({"id": "alias_" + str(index) + "x" * 3990, "source": "input:x"} for index in range(248))
        result = training().run_classification(graph, dict(self.config, epochs=1), self.root / "bounded")
        self.assertLess(len(json.dumps(result, allow_nan=False).encode("utf-8")), 2_000_000)
        self.assertIn("logits", result["outputs"])
        self.assertTrue(result["warnings"])

    def test_validation_can_select_an_earlier_epoch_than_last(self):
        path = Path(self.config["data"]["path"])
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["x", "y", "label", "split"])
            for index in range(36):
                positive = bool(index % 2)
                split = "train" if index < 20 else "val" if index < 28 else "test"
                # Held-out validation reverses the train relationship so later
                # fitting gets worse there. Test labels retain the relationship.
                label = positive if split != "val" else not positive
                writer.writerow([(1 if positive else -1) + index / 1000, index / 100, "positive" if label else "negative", split])
        result = training().run_classification(classifier(False), dict(self.config, optimizer="sgd", lr=0.3), self.root / "early_best")
        self.assertLess(result["best_epoch"], result["completed_epochs"])
        best = self.torch.load(result["artifacts"]["best_checkpoint"], weights_only=True, map_location="cpu")
        last = self.torch.load(result["artifacts"]["last_checkpoint"], weights_only=True, map_location="cpu")
        self.assertEqual(best["completed_epochs"], result["best_epoch"])
        self.assertEqual(result["metrics"]["val"]["loss"], min(entry["val_loss"] for entry in result["history"]))
        self.assertTrue(any(not self.torch.equal(best["model_state"][name], tensor) for name, tensor in last["model_state"].items()))


if __name__ == "__main__":
    unittest.main()
