import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from studio_core.classification_data import inspect_dataset, build_datasets


class ResearchDataTests(unittest.TestCase):
    def test_grouped_high_depth_round_trip_and_content_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "manifest.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sample_id", "patient_id", "path", "label", "split"])
                for i, split in enumerate(("train", "val", "test")):
                    for k in range(2):
                        name = f"slice_{i}_{k}.png"
                        Image.new("I;16", (8, 8), 300 + i * 600 + k * 250).save(root / name)
                        writer.writerow([f"s{i}{k}", f"p{i}{k}", name, f"class_{k}", split])
            spec = {"kind": "research_csv", "path": str(csv_path), "format": "gray16", "image_size": [8, 8],
                    "channels": 1, "depth": 1, "intensity_min": 0, "intensity_max": 2000, "split_column": "split"}
            manifest = inspect_dataset(spec)
            self.assertEqual(manifest["input_shape"], [None, 1, 8, 8])
            dataset = build_datasets(manifest)["train"]
            pixels, label, sample = dataset[0]
            self.assertEqual((sample, label), ("s00", 0))
            self.assertAlmostEqual(float(pixels.mean()), 300 / 2000, places=6)
            Image.new("I;16", (8, 8), 800).save(root / "slice_0_0.png")
            with self.assertRaisesRegex(ValueError, "changed"):
                dataset[0]
