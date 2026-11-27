from pathlib import Path
import tempfile
import unittest

from native.document import Document
from native.paths import resource_root
from native.ui import NativeWorkbench
from studio_core.experiment_spec import default_classification_config


class CohortControlsTests(unittest.TestCase):
    def test_restoring_controls_keeps_exact_column_names_and_inactive_fields(self):
        table = default_classification_config('table_csv')
        table['data'].update(numeric_columns=['[BMI]'], categorical_columns=['site '])
        multi = default_classification_config('multimodal_csv')
        multi['data'].update(modalities=['image'], volume_column=None, feature_columns=['external, feature'])
        multi['input_ports'] = {'image': 'view'}
        with tempfile.TemporaryDirectory() as root:
            ui = NativeWorkbench(Document(), None, Path(root), resource_root())
            try:
                ui.build()
                for config in (table, multi):
                    with self.subTest(kind=config['data']['kind']):
                        ui._install_experiment(config)
                        ui._sync_experiment_widgets()
                        restored = ui._configuration()
                        self.assertEqual(restored['data'], config['data'])
                        self.assertEqual(restored['input_ports'], config['input_ports'])
            finally:
                ui.shutdown()
