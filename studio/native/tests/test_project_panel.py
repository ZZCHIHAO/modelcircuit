from pathlib import Path
import os
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import dearpygui.dearpygui as dpg

from native.project_panel import ProjectPanel


class ProjectPanelTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.app = SimpleNamespace(
            guarded=lambda action, *args: lambda s=None, a=None, u=None: action(*args, sender=s, app_data=a, user_data=u),
            data_directory=self.root, _refresh_devices=Mock(),
            inference_panel=SimpleNamespace(show_report=Mock()),
            segmentation_panel=SimpleNamespace(sync=Mock()),
        )
        dpg.create_context()
        self.addCleanup(dpg.destroy_context)
        self.panel = ProjectPanel(self.app)
        self.addCleanup(self.panel.close)
        with dpg.window():
            self.panel.build_controls()
            with dpg.tab_bar(tag='native_bottom_tabs'):
                self.panel.build_results()
                dpg.add_tab(label='Predictions', tag='native_segmentation_tab')
        self.panel.build_resources()

    def _report(self, name, **changes):
        folder = self.root / name
        folder.mkdir()
        report = dict(path=folder / 'project-run.json', status='succeeded', operation='train', profile='segmentation-v1',
                      dataset_digest='a' * 64, evaluation='segmentation-v1-test-v1',
                      metrics={'test': {'foreground_dice': .8, 'count': 2}},
                      history=[{'epoch': 2, 'train_loss': .6, 'val_loss': .7},
                               {'epoch': 1, 'train_loss': .8, 'val_loss': .9}], artifacts={})
        return {**report, **changes}

    @staticmethod
    def _invoke(name, app_data=None):
        tag = 'native_project_' + name
        dpg.get_item_callback(tag)(tag, app_data)

    def test_same_protocol_plots_loss_and_data_mismatch_removes_chart(self):
        left = self._report('first')
        right = self._report('second', history=[{'epoch': 1, 'train_loss': .4, 'val_loss': .5}])
        self.panel._reports.update(left=left, right=right)
        self.panel._render_reports()
        self.assertEqual(dpg.get_value('native_project_loss_left_train')[:2], [[1.0, 2.0], [.8, .6]])
        self.assertEqual(dpg.get_value('native_project_loss_right_val')[:2], [[1.0], [.5]])
        right['dataset_digest'] = 'b' * 64
        self.panel._render_reports()
        self.assertFalse(dpg.does_item_exist('native_project_loss_comparison'))

    def test_report_actions_open_only_its_directory_or_the_matching_internal_viewer(self):
        inference = self.root / 'inference' / 'inference.json'
        predictions = self.root / 'training' / 'predictions.csv'
        self.panel._reports.update(
            left=self._report('inference', operation='predict', artifacts={'inference': inference}),
            right=self._report('training', artifacts={'predictions': predictions}),
        )
        predictions.touch()
        self.panel._render_reports()
        with patch('native.project_panel.os.startfile', create=True) as open_directory:
            self._invoke('open_directory_left')
        open_directory.assert_called_once_with(str(inference.parent.resolve()), 'explore')
        self._invoke('view_output_left')
        self.app.inference_panel.show_report.assert_called_once_with(inference)
        self._invoke('view_output_right')
        self.app.segmentation_panel.sync.assert_called_once_with(
            {'profile': 'segmentation-v1', 'artifacts': {'predictions': predictions}}, True, True)

    def test_weight_candidates_follow_completion_time_without_duplicates(self):
        from native.project_panel import _export_checkpoint

        def create(name, seconds):
            folder = self.root / name
            folder.mkdir()
            files = {key: folder / item for key, item in (
                ('run_spec','run_spec.json'), ('best_checkpoint','best.pt'), ('metrics','metrics.json'))}
            for key, path in files.items():
                path.touch()
                os.utime(path, (seconds, seconds))
            return files

        classic = create('classic', 100)
        project = create('project', 200)
        os.utime(classic['best_checkpoint'], (300, 300))
        def exported(*candidates):
            return _export_checkpoint(candidates, [], None, lambda: False)

        self.assertEqual(exported(classic, project, classic), (project['run_spec'], classic['run_spec']))
        old_report = {**project, 'finished_at': '1970-01-01T00:00:50+00:00'}
        self.assertEqual(exported(classic, old_report), (classic['run_spec'], project['run_spec']))
        project['best_checkpoint'].unlink()
        self.assertEqual(exported(classic, project), (classic['run_spec'],))

    def test_bundled_pair_directory_fills_data_and_clears_previous_csv_root(self):
        project = {'path': self.root / 'project.json', 'project_id': 'demo',
                   'config': {'data': {'kind': 'segmentation_csv', 'format': 'nifti'}},
                   'paired_csv': self.root / 'dataset-pairs.csv'}
        self.panel._set_project(project, root=self.root / 'old-csv-root')
        target = self.root / 'relocated-images'
        self._invoke('picker_pair_directory', {'file_path_name': str(target)})
        self.assertEqual(self.panel._binding_fields(), (str(target), None))
        self.assertTrue(dpg.is_item_shown('native_project_choose_pair_directory'))
        self.panel._set_project({**project, 'paired_csv': None})
        self.assertFalse(dpg.is_item_shown('native_project_choose_pair_directory'))

    def test_report_read_checks_artifacts_and_broken_previews_are_hidden(self):
        from native.project_panel import _read_report

        report = self._report('run', artifacts={'predictions': self.root / 'run' / 'predictions.csv'},
                              missing_outputs=['masks/item.nii.gz'], changed_artifacts=['metrics'])
        cancelled = Mock(return_value=False)
        with patch('studio_core.project_runner.read_project_run', return_value=report) as read:
            self.panel._reports['left'] = _read_report(report['path'], cancelled)
        read.assert_called_once_with(report['path'], verify=True, cancelled=cancelled)
        self.panel._render_reports()
        self.assertFalse(dpg.does_item_exist('native_project_view_output_left'))
        self.assertTrue(dpg.does_item_exist('native_project_open_directory_left'))
        integrity = dpg.get_value('native_project_integrity_left')
        self.assertIn('1', integrity)
