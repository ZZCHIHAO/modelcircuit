"""Native model-library and local extension paths use registered callbacks."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import dearpygui.dearpygui as dpg

from native.chip_library import save_bundle
from native.document import Document
from native.paths import resource_root
from native.ui import NativeWorkbench
from studio_core.module_templates import make_module


def save_local(directory):
    bundle = make_module('fusion.weighted_sum', {'features': 4})
    return save_bundle({'definitions': bundle.definitions, 'resource_profile': 'vision-v4',
                        'resource_batch_bound': 2}, bundle.entry['definition_id'], '1.0.0', directory)


class ModelCatalogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = save_local(Path(self.directory.name) / 'chips')
        self.ui = NativeWorkbench(Document(), None, self.directory.name, resource_root())
        self.ui.build()

    def tearDown(self):
        self.ui.shutdown()
        self.directory.cleanup()

    def invoke(self, tag, data=None):
        return dpg.get_item_callback(tag)(tag, data, dpg.get_item_user_data(tag))

    def test_local_search_configuration_and_policy_insertion_are_atomic(self):
        self.assertIn('本地芯片', dpg.get_item_configuration('native_catalog_category')['items'])
        self.invoke('native_catalog_category', '本地芯片')
        panel = self.ui.catalog_panel
        self.assertEqual(len(panel.visible_ids), 1)
        identifier = panel.visible_ids[0]
        self.assertTrue(identifier.startswith('local:'))
        self.invoke('native_catalog_config_' + identifier)
        self.assertEqual(set(panel.config_widgets), {'alpha'})
        dpg.set_value(panel.config_widgets['alpha'], '0.75')
        before = deepcopy(self.ui.document.graph)
        self.invoke('native_catalog_apply')
        self.assertEqual(before, self.ui.document.graph)
        self.invoke('native_catalog_policy_apply')
        self.assertEqual(self.ui.document.selected_instance['params'], {'alpha': .75})
        self.assertEqual(self.ui.document.selected_instance['definition_id'], 'WeightedSum')
        self.assertEqual(self.ui.document.graph['revision'], before['revision'] + 1)
        self.assertEqual(len(self.ui.document._undo), 1)
        self.assertEqual(len(self.ui.document.selected_call['inputs']), 0)
        view = self.ui.canvas_host.get('root')
        call = self.ui.document.selected_call['id']
        self.assertEqual({port for kind, identifier, port in view.target_attributes.values()
                          if kind == 'call' and identifier == call}, {'a', 'b'})
        self.assertIn('call:' + call + ':features', view.source_attributes.values())

    def test_refresh_rejects_old_local_config_and_keeps_valid_results_with_damage(self):
        panel = self.ui.catalog_panel
        self.invoke('native_catalog_category', '本地芯片')
        self.assertEqual(len(panel.visible_ids), 1, 'Local saved architecture must be indexed')
        identifier = panel.visible_ids[0]
        self.invoke('native_catalog_config_' + identifier)
        old_apply = dpg.get_item_callback('native_catalog_apply')
        old_data = dpg.get_item_user_data('native_catalog_apply')
        before = deepcopy(self.ui.document.graph)
        self.path.unlink()
        (self.path.parent / 'broken.json').write_text('{oops', encoding='utf-8')
        self.invoke('native_catalog_refresh_local')
        old_apply('native_catalog_apply', None, old_data)
        self.assertEqual(before, self.ui.document.graph)
        self.assertEqual(panel.visible_ids, [])
        self.assertTrue(any('broken.json' in issue for issue in panel.local_index.diagnostics))
        self.assertFalse(dpg.is_item_shown('native_catalog_config'))
        save_local(self.path.parent)
        self.invoke('native_catalog_refresh_local')
        self.assertEqual(len(panel.visible_ids), 1)
        self.assertTrue(any('broken.json' in issue for issue in panel.local_index.diagnostics))

    def test_model_categories_are_visible_without_hiding_primitive_categories(self):
        values = dpg.get_item_configuration('native_catalog_category')['items']
        for label in ('视觉主干', '完整模型', '基础层', '功能芯片', '工程芯片', '本地芯片'):
            self.assertIn(label, values)

    def test_saving_chip_version_refreshes_component_library_immediately(self):
        self.invoke('native_catalog_category', '本地芯片')
        before = set(self.ui.catalog_panel.visible_ids)
        self.ui.edit_definition = 'encoder_chip'
        dpg.set_value('native_chip_version', '1.2.3')
        self.ui._library_save()
        self.assertEqual(len(set(self.ui.catalog_panel.visible_ids) - before), 1)
        specs = self.ui.catalog_panel.local_index.list_components('encoder_chip')
        self.assertEqual(len(specs), 1)
        self.assertIn('1.2.3', specs[0].label)

    def test_registered_model_drag_payloads_have_correct_categories_and_pins(self):
        panel = self.ui.catalog_panel
        self.invoke('native_catalog_category', '视觉主干')
        self.assertEqual(len(panel.visible_ids), 9)
        self.assertIn('backbone.vgg16', panel.visible_ids)
        self.assertIn('backbone.vgg19', panel.visible_ids)
        self.assertIn('backbone.resnet50', panel.visible_ids)
        self.invoke('native_catalog_category', '完整模型')
        self.assertEqual(len(panel.visible_ids), 5)
        identifier = 'model.squeezenet1_1_classifier'
        self.assertIn(identifier, panel.visible_ids)
        self.invoke('native_catalog_config_' + identifier)
        dpg.set_value(panel.config_widgets['input_shape'], '[null,3,32,32]')
        dpg.set_value(panel.config_widgets['num_classes'], '2')
        self.invoke('native_catalog_apply')
        self.invoke('native_catalog_policy_apply')
        instance = self.ui.document.selected_instance
        self.assertEqual(instance['kind'], 'chip')
        definition = self.ui._definition(instance['definition_id'])
        self.assertEqual([port['id'] for port in definition['graph']['inputs']], ['x'])
        self.assertEqual([port['id'] for port in definition['graph']['outputs']], ['logits'])
        key = self.ui.workbench_panel.open_call([self.ui.document.selected_call['id']])
        self.assertEqual(self.ui.workbench.views[key].target.definition_id, instance['definition_id'])
        self.assertTrue(any('Fire' in item['id'] for item in self.ui.document.graph['definitions']))

    def test_model_and_local_button_text_fits_and_selected_category_is_visible(self):
        for category in ('视觉主干', '完整模型', '本地芯片'):
            self.invoke('native_catalog_category', category)
            self.assertEqual(dpg.get_value('native_catalog_category'), category)
            for identifier in self.ui.catalog_panel.visible_ids:
                tag = 'native_catalog_add_' + identifier
                config = dpg.get_item_configuration(tag)
                width = dpg.get_text_size(config['label'])[0]
                self.assertLessEqual(width, config['width'] - 12, identifier)

    def test_legacy_functional_chip_keeps_hardware_model_policy_without_modal_downgrade(self):
        graph = deepcopy(self.ui.document.graph)
        graph.update(resource_profile='hardware-v5', resource_batch_bound=2)
        self.ui.document.replace_graph(graph)
        self.ui._reset_canvas_document()
        self.ui._refresh()
        panel = self.ui.catalog_panel
        self.assertTrue(panel.place({'version': 1, 'component_id': 'fusion.weighted_sum', 'config': {}},
                                    self.ui.canvas_host.token('root')))
        self.assertFalse(dpg.is_item_shown('native_catalog_policy'))
        self.assertEqual(self.ui.document.graph['resource_profile'], 'hardware-v5')
        description = panel.policy_description({'resource_profile': 'hardware-v5', 'resource_batch_bound': 2})
        self.assertIn('硬件', description)
        self.assertIn('无固定参数量上限', description)

    def test_existing_project_hardware_upgrade_preserves_configuration_and_rejects_stale_apply(self):
        self.assertTrue(dpg.does_item_exist('native_catalog_hardware_policy'), 'Hardware policy upgrade must be discoverable')
        config = deepcopy(self.ui._configuration())
        before = deepcopy(self.ui.document.graph)
        self.invoke('native_catalog_hardware_policy')
        old_apply = dpg.get_item_callback('native_catalog_policy_apply')
        old_data = dpg.get_item_user_data('native_catalog_policy_apply')
        self.invoke('native_catalog_policy_cancel')
        self.assertEqual(self.ui.document.graph, before)
        self.invoke('native_catalog_hardware_policy')
        self.invoke('native_catalog_policy_apply')
        self.assertEqual(self.ui.document.graph['resource_profile'], 'hardware-v5')
        self.assertEqual(self.ui.document.graph['resource_batch_bound'], 64)
        self.assertEqual(self.ui._configuration(), config)
        upgraded = deepcopy(self.ui.document.graph)
        old_apply('native_catalog_policy_apply', None, old_data)
        self.assertEqual(self.ui.document.graph, upgraded)
        self.ui._undo()
        expected = deepcopy(before)
        expected['revision'] = self.ui.document.graph['revision']
        self.assertEqual(self.ui.document.graph, expected)

    def test_native_factory_registry_extension_reaches_desktop_without_ui_rewrite(self):
        from studio_core.backbone_templates import make_resnet18
        from studio_core.model_registry import DEFAULT_MODEL_REGISTRY
        spec = deepcopy(DEFAULT_MODEL_REGISTRY.list_models()[0])
        spec.id = 'backbone.lab_resnet'
        spec.label = '实验室自定义主干'
        registry = DEFAULT_MODEL_REGISTRY.with_model(spec, lambda config: make_resnet18(**config))
        self.ui.shutdown()
        self.ui = NativeWorkbench(Document(), None, self.directory.name, resource_root(), model_registry=registry)
        self.ui.build()
        self.invoke('native_catalog_category', '视觉主干')
        self.assertEqual(len(self.ui.catalog_panel.visible_ids), 10)
        self.assertIn(spec.id, self.ui.catalog_panel.visible_ids)
        self.invoke('native_catalog_config_' + spec.id)
        dpg.set_value(self.ui.catalog_panel.config_widgets['input_shape'], '[null,3,32,32]')
        self.invoke('native_catalog_apply')
        self.invoke('native_catalog_policy_apply')
        self.assertEqual(self.ui.document.selected_instance['kind'], 'chip')
        self.assertEqual(self.ui.document.graph['resource_profile'], 'hardware-v5')
        self.assertEqual(len(DEFAULT_MODEL_REGISTRY.list_models()), 14)

    def test_policy_confirmation_footer_matches_action_when_dialog_is_reused(self):
        self.invoke('native_catalog_hardware_policy')
        self.assertTrue(dpg.does_item_exist('native_catalog_policy_footer'), 'Confirmation text must describe the actual operation')
        text = dpg.get_value('native_catalog_policy_footer')
        self.assertIn('只修改执行策略', text)
        self.assertNotIn('组件', text)
        self.invoke('native_catalog_policy_cancel')
        self.ui.catalog_panel.place({'version': 1, 'component_id': 'fusion.weighted_sum', 'config': {}},
                                   self.ui.canvas_host.token('root'))
        text = dpg.get_value('native_catalog_policy_footer')
        self.assertIn('策略和组件', text)
        self.assertNotIn('只修改执行策略', text)


if __name__ == '__main__':
    unittest.main()
