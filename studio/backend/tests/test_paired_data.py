import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from paired_fixture import paired_fixture, write_rows
from studio_core.classification_data import build_datasets, inspect_dataset


class PairedDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.spec, self.rows = paired_fixture(self.root)

    def test_manifest_keeps_numeric_repeats_but_fits_only_train(self):
        self.rows[-1]['f'] = 1000
        write_rows(self.spec, self.rows)
        manifest = inspect_dataset(self.spec)
        self.assertEqual(manifest['version'], 3)
        self.assertEqual(manifest['input_shapes'], {'image': [None, 3, 8, 8], 'tabular': [None, 2]})
        self.assertNotIn('input_shape', manifest)
        self.assertEqual(manifest['counts'], {'train': 12, 'val': 6, 'test': 6})
        self.assertEqual(manifest['preprocessing']['tabular']['mean'], [1.0, 1.0])
        self.assertAlmostEqual(manifest['preprocessing']['tabular']['std'][0], (2 / 3) ** 0.5)
        self.assertEqual(manifest['preprocessing']['tabular']['std'][1], 1.0)
        self.assertEqual([record['id'] for record in manifest['records']], [row['sample_id'] for row in self.rows])
        self.assertEqual(manifest, inspect_dataset(self.spec))
        json.dumps(manifest, allow_nan=False)

    def test_inspection_needs_no_torch(self):
        script = "import json,sys; from studio_core.classification_data import inspect_dataset; inspect_dataset(json.loads(sys.argv[1])); assert 'torch' not in sys.modules"
        completed = subprocess.run([sys.executable, '-c', script, json.dumps(self.spec)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_nonempty_unique_ids_and_patient_groups_are_mandatory(self):
        for key, value in [('sample_id', ''), ('patient_id', ''), ('sample_id', 'sample_001')]:
            rows = [dict(row) for row in self.rows]; rows[0][key] = value
            write_rows(self.spec, rows)
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'paired.csv.*row 2|sample_001.*paired.csv'):
                inspect_dataset(self.spec)

    def test_patient_and_image_path_content_leakage_fail_independently(self):
        for key, value in [('patient_id', 'patient_0'), ('image_path', self.rows[0]['image_path'])]:
            rows = [dict(row) for row in self.rows]; rows[12][key] = value
            write_rows(self.spec, rows)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'leakage.*paired.csv.*row'):
                inspect_dataset(self.spec)
        write_rows(self.spec, self.rows)
        (self.root / self.rows[12]['image_path']).write_bytes((self.root / self.rows[0]['image_path']).read_bytes())
        with self.assertRaisesRegex(ValueError, 'image.*content.*leakage.*paired.csv'):
            inspect_dataset(self.spec)

    def test_group_split_is_deterministic_and_keeps_patients_together(self):
        self.spec['split_column'] = None
        manifest = inspect_dataset(self.spec, seed=7)
        self.assertEqual(manifest, inspect_dataset(self.spec, seed=7))
        seen = {}
        for record in manifest['records']:
            self.assertEqual(seen.setdefault(record['group'], record['split']), record['split'])
        self.assertTrue(all(all(count > 0 for count in values.values()) for values in manifest['class_counts'].values()))

    def test_missing_bad_or_outside_root_images_report_csv_row_and_path(self):
        for reference in ('images/missing.png', '../outside.png', 'images/metadata.cfg'):
            rows = [dict(row) for row in self.rows]; rows[0]['image_path'] = reference
            write_rows(self.spec, rows)
            with self.subTest(reference=reference), self.assertRaisesRegex(ValueError, 'paired.csv.*row 2'):
                inspect_dataset(self.spec)
        write_rows(self.spec, self.rows)
        (self.root / self.rows[0]['image_path']).write_bytes(b'bad PNG')
        with self.assertRaisesRegex(ValueError, 'paired.csv.*row 2.*sample_000.png'):
            inspect_dataset(self.spec)

    def test_frozen_sources_and_lazy_images_reject_changes_and_cancel(self):
        manifest = inspect_dataset(self.spec)
        datasets = build_datasets(manifest)
        image = self.root / self.rows[0]['image_path']
        image.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed.*sample_000.png'):
            datasets['train'][0]
        with self.assertRaisesRegex(ValueError, 'changed.*sample_000.png'):
            build_datasets(manifest)
        with self.assertRaises(InterruptedError):
            build_datasets(manifest, cancelled=lambda: True)

    def test_csv_and_manifest_changes_are_rejected_before_loading(self):
        import copy
        manifest = inspect_dataset(self.spec)
        tampered = copy.deepcopy(manifest); tampered['records'][0]['features'][0] = 0.5
        with self.assertRaisesRegex(ValueError, 'manifest digest'):
            build_datasets(tampered)
        self.rows[0]['f'] = 0.5; write_rows(self.spec, self.rows)
        with self.assertRaisesRegex(ValueError, 'changed.*paired.csv'):
            build_datasets(manifest)

    def test_symlink_image_cannot_escape_configured_root(self):
        import os
        with tempfile.TemporaryDirectory() as outside:
            from PIL import Image
            external = Path(outside) / 'external.png'
            Image.new('RGB', (8, 8), (80, 90, 100)).save(external)
            link = self.root / 'images' / 'escape.png'
            try:
                os.symlink(external, link)
            except OSError as exc:
                self.skipTest('Symlink creation unavailable: ' + str(exc))
            self.rows[0]['image_path'] = 'images/escape.png'; write_rows(self.spec, self.rows)
            with self.assertRaisesRegex(ValueError, 'paired.csv.*row 2.*escapes image_root'):
                inspect_dataset(self.spec)

    def test_joint_shuffle_preserves_image_numeric_label_and_sample_alignment(self):
        import torch
        from torch.utils.data import DataLoader
        manifest = inspect_dataset(self.spec)
        datasets = build_datasets(manifest)
        expected = {row['sample_id']: row for row in self.rows}
        order = []
        for inputs, labels, identifiers in DataLoader(datasets['train'], batch_size=5, shuffle=True, generator=torch.Generator().manual_seed(19), num_workers=0):
            self.assertEqual(set(inputs), {'image', 'tabular'})
            self.assertEqual(len(inputs['image']), len(labels))
            self.assertEqual(len(inputs['tabular']), len(labels))
            for index, identifier in enumerate(identifiers):
                row = expected[identifier]; source_index = int(identifier[-3:])
                self.assertAlmostEqual(float(inputs['image'][index, 0, 0, 0]), (20 + source_index * 5) / 255, places=6)
                self.assertAlmostEqual(float(inputs['tabular'][index, 0]), (row['f'] - 1) / manifest['preprocessing']['tabular']['std'][0], places=6)
                self.assertEqual(int(labels[index]), source_index % 2)
            order.extend(identifiers)
        self.assertEqual(set(order), {row['sample_id'] for row in self.rows[:12]})
        self.assertNotEqual(order, [row['sample_id'] for row in self.rows[:12]])


if __name__ == '__main__': unittest.main()
