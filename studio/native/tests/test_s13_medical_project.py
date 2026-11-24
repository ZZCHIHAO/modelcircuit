"""Exercise the normal supervised, portable project, and inference entry points."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

from studio_core.classification_data import inspect_dataset
from studio_core.experiment_spec import default_segmentation_config
from studio_core.project_runner import read_project_run, train_project
from studio_core.research_project import bind_project, export_project, read_project
from studio_core.segmentation_templates import segmentation_graph


class MedicalProjectTests(unittest.TestCase):
    def test_registered_mri_project_trains_and_recovers_original_grid(self):
        import torch
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            affine = np.array([[1., 0., 0., 8.], [0., 1., 0., -6.], [0., 0., 2., 4.], [0., 0., 0., 1.]])
            lines = ['sample_id,patient_id,image_path,mask_path,split,flair']
            for i, split in enumerate(('train', 'val', 'test')):
                field = np.indices((12, 10, 6)).sum(axis=0).astype(np.float32)
                field[i, 2, 1] += 50.
                mask = np.zeros(field.shape, dtype=np.uint16)
                mask[2+i:6+i, 2:6, 1:5] = 255
                for name, array in [('t1', field), ('flair', field * field), ('mask', mask)]:
                    image = nib.Nifti1Image(array, affine)
                    image.header.set_xyzt_units('mm')
                    nib.save(image, root / f'{name}-{i}.nii.gz')
                lines.append(f'a{i},p{i},t1-{i}.nii.gz,mask-{i}.nii.gz,{split},flair-{i}.nii.gz')
            csv_file = root / 'pairs.csv'
            csv_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            config = default_segmentation_config()
            config.update(batch_size=2, epochs=1)
            config['data'].update(path=str(csv_file), format='mri_nifti', channels=2, depth=4, image_size=[8, 8],
                                  sequence_columns=['flair'], registered=True, split_column='split', mask_values=[0, 255],
                                  spacing_mm=[1.5, 1., 1.], patches_per_case=2, overlap=.25)
            manifest = inspect_dataset(config['data'], config['seed'])
            self.assertEqual(manifest['input_shape'], [None, 2, 4, 8, 8])
            project = read_project(export_project(root / 'project', segmentation_graph(manifest), config))
            _, rebound = bind_project(project, root)
            self.assertEqual(rebound['counts'], {'train': 1, 'val': 1, 'test': 1})
            result = train_project(project['path'], root, root / 'run')
            report = read_project_run(result['artifacts']['project_run'], verify=True)
            self.assertEqual(report['status'], 'succeeded')
            self.assertEqual(report['evaluation'], 'segmentation-native-test-v1')
            self.assertEqual(report['metrics']['test']['space'], 'original')
            self.assertEqual(report['metrics']['test']['voxels'], 12 * 10 * 6)
            self.assertEqual(report['global_step'], 1)
            self.assertFalse(report['changed_artifacts'] or report['changed_outputs'])
            self.assertTrue(report['artifacts']['case_metrics'].is_file())
            mask = nib.load(root / 'run' / 'masks' / '000001.nii.gz')
            self.assertEqual(mask.shape, (12, 10, 6))
            np.testing.assert_allclose(mask.affine, affine)
            self.assertLessEqual(set(np.unique(mask.dataobj)), {0, 255})
            # Exercise the bundled standalone runtime without workspace imports.
            environment = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
            predicted = subprocess.run(
                [sys.executable, 'run.py', 'predict', '--source', str(root / 't1-2.nii.gz'),
                 '--sequence', str(root / 'flair-2.nii.gz'), '--weights', str(root / 'run'),
                 '--out', str(root / 'predictions')], cwd=root / 'project', env=environment,
                capture_output=True, text=True, encoding='utf-8', timeout=60,
            )
            self.assertEqual(predicted.returncode, 0, predicted.stderr)
            inference_path = root / 'predictions' / 'inference.json'
            inference = json.loads(inference_path.read_text('utf-8'))
            self.assertEqual(inference['mode'], 'sliding')
            case = inference['cases'][0]
            self.assertEqual(case['space'], 'original')
            self.assertEqual(len(case['sequence_sha256']), 1)
            predicted_mask = nib.load(inference_path.parent / case['mask_file'])
            np.testing.assert_array_equal(np.asanyarray(mask.dataobj), np.asanyarray(predicted_mask.dataobj))
            np.testing.assert_allclose(predicted_mask.affine, affine)
