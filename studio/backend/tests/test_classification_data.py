"""Real local-file classification data contracts (no external datasets)."""
import csv
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def data_module():
    try:
        return importlib.import_module("studio_core.classification_data")
    except ModuleNotFoundError as exc:
        raise AssertionError("classification dataset inspection is not implemented") from exc


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def csv(self, rows=None):
        path = self.root / "features.csv"
        rows = rows or [[i, 1, "a" if i % 2 == 0 else "b", "g" + str(i), "train" if i < 8 else "val" if i < 12 else "test"] for i in range(16)]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["f", "constant", "label", "group", "split"])
            writer.writerows(rows)
        return {"kind": "csv", "path": str(path), "target_column": "label", "feature_columns": ["f", "constant"], "split_column": "split", "group_column": "group", "standardize": True, "validation_fraction": 0.2, "test_fraction": 0.2}

    def test_csv_freezes_contents_and_fits_scaling_on_train_only(self):
        module = data_module()
        spec = self.csv()
        manifest = module.inspect_dataset(spec)
        self.assertEqual(manifest["version"], 2)
        self.assertEqual(manifest["input_shape"], [None, 2])
        self.assertEqual(manifest["counts"], {"train": 8, "val": 4, "test": 4})
        self.assertEqual(manifest["class_names"], ["a", "b"])
        self.assertEqual(manifest["preprocessing"]["mean"], [3.5, 1.0])
        self.assertEqual(manifest["preprocessing"]["std"][1], 1.0)
        self.assertEqual(manifest, module.inspect_dataset(spec))
        json.dumps(manifest, allow_nan=False)

    def test_inspection_has_no_torch_dependency(self):
        spec = self.csv()
        script = "import sys,json; from studio_core.classification_data import inspect_dataset; inspect_dataset(json.loads(sys.argv[1])); assert 'torch' not in sys.modules"
        completed = subprocess.run([sys.executable, "-c", script, json.dumps(spec)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_stratified_split_is_seeded_and_preserves_all_classes(self):
        spec = self.csv()
        spec.update(group_column=None, split_column=None)
        first = data_module().inspect_dataset(spec, seed=7)
        self.assertEqual(first, data_module().inspect_dataset(spec, seed=7))
        second = data_module().inspect_dataset(spec, seed=8)
        self.assertNotEqual(first["dataset_digest"], second["dataset_digest"])
        self.assertTrue(all(all(count > 0 for count in counts.values()) for counts in first["class_counts"].values()))

    def test_group_split_never_leaks_a_group(self):
        rows = [[i, 1, "ab"[i % 2], "person" + str(i // 2), "unused"] for i in range(24)]
        spec = self.csv(rows)
        spec["split_column"] = None
        manifest = data_module().inspect_dataset(spec, seed=12)
        seen = {}
        for record in manifest["records"]:
            self.assertEqual(seen.setdefault(record["group"], record["split"]), record["split"])
        self.assertTrue(all(all(count > 0 for count in counts.values()) for counts in manifest["class_counts"].values()))

    def test_nonfinite_missing_and_explicit_group_leak_fail_with_source(self):
        for bad in ("nan", "inf", "", "abc"):
            spec = self.csv([[bad, 1, "a", "p", "train"]])
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "features.csv.*row 2"):
                data_module().inspect_dataset(spec)
        spec = self.csv()
        text = Path(spec["path"]).read_text(encoding="utf-8").replace("g8,val", "g0,val")
        Path(spec["path"]).write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "group.*g0.*features.csv"):
            data_module().inspect_dataset(spec)

    def test_duplicate_feature_contents_across_splits_fail(self):
        spec = self.csv()
        text = Path(spec["path"]).read_text(encoding="utf-8").replace("8,1,a,g8,val", "0,1,a,g8,val")
        Path(spec["path"]).write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content.*features.csv.*row"):
            data_module().inspect_dataset(spec)

    def test_float32_rounding_cannot_hide_cross_split_feature_leakage(self):
        rows = [[10_000_000_000 + i, 1, "ab"[i % 2], "g" + str(i), "train" if i < 8 else "val" if i < 12 else "test"] for i in range(16)]
        spec = self.csv(rows)
        spec["standardize"] = False
        with self.assertRaisesRegex(ValueError, "content.*features.csv.*row"):
            data_module().inspect_dataset(spec)

    def test_standardized_float32_rounding_is_checked_after_train_only_fit(self):
        rows = [[i if i < 8 else 10_000_000_000 + i, 1, "ab"[i % 2], "g" + str(i), "train" if i < 8 else "val" if i < 12 else "test"] for i in range(16)]
        with self.assertRaisesRegex(ValueError, "content.*features.csv.*row"):
            data_module().inspect_dataset(self.csv(rows))

    def test_oversized_label_is_rejected_at_its_source_before_training(self):
        rows = [[i, 1, "a" if i % 2 == 0 else "b", "g" + str(i), "train" if i < 8 else "val" if i < 12 else "test"] for i in range(16)]
        rows[0][2] = "x" * 257
        with self.assertRaisesRegex(ValueError, "label.*limit.*features.csv.*row 2"):
            data_module().inspect_dataset(self.csv(rows))

    def test_source_change_is_rejected_before_dataset_construction(self):
        spec = self.csv()
        manifest = data_module().inspect_dataset(spec)
        Path(spec["path"]).write_text(Path(spec["path"]).read_text() + "17,1,a,g17,test\n")
        with self.assertRaisesRegex(ValueError, "changed.*features.csv"):
            data_module().build_datasets(manifest)

    def test_inspection_cancellation_is_observed(self):
        with self.assertRaises(InterruptedError):
            data_module().inspect_dataset(self.csv(), cancelled=lambda: True)

    def test_second_source_scan_can_be_cancelled_before_all_images_are_rehashed(self):
        manifest = data_module().inspect_dataset(self.images())
        callbacks = [0]
        def cancelled():
            callbacks[0] += 1
            return callbacks[0] >= 3
        with self.assertRaises(InterruptedError):
            data_module().build_datasets(manifest, cancelled=cancelled)
        self.assertEqual(callbacks[0], 3)

    def test_inspected_manifest_does_not_alias_editable_data_settings(self):
        spec = self.csv()
        manifest = data_module().inspect_dataset(spec)
        frozen = json.dumps(manifest, sort_keys=True)
        spec["feature_columns"][0] = "edited_after_inspection"
        self.assertEqual(json.dumps(manifest, sort_keys=True), frozen)

    def images(self, directories=True):
        from PIL import Image
        root = self.root / "images"
        for s, split in enumerate(("train", "val", "test") if directories else ("",)):
            for c, name in enumerate(("cat", "dog")):
                for index in range(3):
                    target = root / split / name / (str(index) + ".png")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    image = Image.new("RGB", (10, 9), (20 + s * 50 + index, c * 100, index * 30))
                    image.save(target)
        return {"kind": "image_folder", "path": str(root), "split_mode": "directories" if directories else "stratified", "image_size": [8, 12], "channels": 3, "validation_fraction": 0.2, "test_fraction": 0.2}

    def test_image_directories_decode_and_honor_splits(self):
        spec = self.images()
        manifest = data_module().inspect_dataset(spec)
        self.assertEqual(manifest["input_shape"], [None, 3, 8, 12])
        self.assertEqual(manifest["counts"], {"train": 6, "val": 6, "test": 6})
        self.assertEqual(manifest["class_names"], ["cat", "dog"])

    def test_known_sync_and_system_sidecars_are_audited_without_changing_data_identity(self):
        module = data_module()
        spec = self.images()
        root = Path(spec["path"])
        before = module.inspect_dataset(spec)
        names = [".DS_Store", "desktop.ini", "0.png.baiduyun.uploading.cfg", "train/Thumbs.db", "val/.DS_Store",
                 "train/cat/0.png.baiduyun.uploading.cfg", "test/dog/desktop.ini", "train/cat/nested/Thumbs.db"]
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Operating system/sync test metadata", encoding="utf-8")
        after = module.inspect_dataset(spec)
        self.assertEqual(after["counts"], before["counts"])
        self.assertEqual(after["records"], before["records"])
        self.assertEqual(after["dataset_digest"], before["dataset_digest"])
        self.assertEqual({item["relative_path"] for item in after["ignored_files"]}, set(names))
        self.assertTrue(all(item["reason"] for item in after["ignored_files"]))
        self.assertTrue(any("sidecar" in warning for warning in after["warnings"]))
        json.dumps(after, allow_nan=False)
        for name in names:
            (root / name).write_text("Sync metadata can change without changing model input", encoding="utf-8")
        self.assertEqual(module.inspect_dataset(spec)["dataset_digest"], before["dataset_digest"])
        self.assertEqual(len(module.build_datasets(after)["train"]), 6)

    def test_known_sidecars_also_work_in_stratified_imagefolder(self):
        spec = self.images(directories=False)
        root = Path(spec["path"])
        (root / ".DS_Store").write_text("metadata", encoding="utf-8")
        (root / "cat" / "0.png.baiduyun.uploading.cfg").write_text("metadata", encoding="utf-8")
        manifest = data_module().inspect_dataset(spec)
        self.assertEqual(len(manifest["ignored_files"]), 2)
        self.assertEqual(manifest["counts"], {"train": 2, "val": 2, "test": 2})

    def test_unknown_files_and_damaged_images_are_not_ignored_as_sidecars(self):
        spec = self.images()
        root = Path(spec["path"])
        known = root / "train" / "cat" / "0.png.baiduyun.uploading.cfg"
        known.write_text("known metadata", encoding="utf-8")
        for name in ("notes.cfg", "0.png.baiduyun.uploading.cfg.bak", "Thumbs.db.png"):
            bad = root / "val" / "dog" / name
            bad.write_bytes(b"This is not a supported image")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name.replace(".", r"\.")):
                data_module().inspect_dataset(spec)
            bad.unlink()
        bad = root / "train" / "cat" / "0.png"
        bad.write_bytes(b"Broken original PNG")
        with self.assertRaisesRegex(ValueError, "Cannot decode.*0.png"):
            data_module().inspect_dataset(spec)

    def test_bad_and_duplicated_images_fail_with_paths(self):
        spec = self.images()
        bad = Path(spec["path"]) / "val" / "cat" / "0.png"
        bad.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "0.png"):
            data_module().inspect_dataset(spec)
        spec = self.images()
        source = Path(spec["path"]) / "train" / "cat" / "0.png"
        bad.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "content.*0.png"):
            data_module().inspect_dataset(spec)

    def test_images_that_collapse_to_same_training_input_cannot_cross_splits(self):
        from PIL import Image
        spec = self.images()
        target = Path(spec["path"]) / "val" / "cat" / "0.png"
        # Same effective input despite different source dimensions/PNG bytes.
        Image.new("RGB", (20, 18), (20, 0, 0)).save(target)
        with self.assertRaisesRegex(ValueError, "content.*0.png"):
            data_module().inspect_dataset(spec)

    def test_lazy_datasets_return_float32_int_and_id(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available in this interpreter")
        module = data_module()
        csv_datasets = module.build_datasets(module.inspect_dataset(self.csv()))
        x, label, identifier = csv_datasets["train"][0]
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(label, 0)
        self.assertIsInstance(identifier, str)
        image_manifest = module.inspect_dataset(self.images())
        image_datasets = module.build_datasets(image_manifest)
        x, label, identifier = image_datasets["val"][0]
        self.assertEqual(list(x.shape), [3, 8, 12])
        self.assertGreaterEqual(float(x.min()), 0)
        self.assertLessEqual(float(x.max()), 1)
        path = Path(next(record for record in image_manifest["records"] if record["split"] == "train")["source_path"])
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            image_datasets["train"][0]


if __name__ == "__main__":
    unittest.main()
