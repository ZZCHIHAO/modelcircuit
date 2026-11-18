from copy import deepcopy
from tempfile import TemporaryDirectory
import unittest

from native.document import Document
from test_tensor_component_insert import chip


class TensorPanelTests(unittest.TestCase):
    def test_incompatible_chips_do_not_take_existing_connection(self):
        doc = Document()
        call = doc.add_component(chip(), apply_policy=True)
        original = doc.graph['inputs'][0]
        name = original['id']
        doc.set_input(call, 'x', 'input:' + name)
        doc.add_input('mask', [None, 4], dtype='bool', axes=['batch', 'feature'], batch_axis=0)
        before = deepcopy(doc.graph)
        history = len(doc._undo)
        with self.assertRaisesRegex(ValueError, 'dtype'):
            doc.set_input(call, 'x', 'input:mask')
        self.assertEqual(doc.graph, before)
        self.assertEqual(len(doc._undo), history)

    def test_contract_edit_is_atomic(self):
        doc = Document()
        doc.add_component(chip(), apply_policy=True)
        doc.add_input('mask', [4], dtype='bool', axes=['key'], batch_axis=None)
        before = deepcopy(doc.graph)
        invalid = {'shape': [None], 'dtype': 'bool', 'axes': ['key'], 'batch_axis': None}
        with self.assertRaisesRegex(ValueError, 'batch axis'):
            doc.set_tensor_input('mask', invalid)
        self.assertEqual(doc.graph, before)
        doc.set_tensor_input('mask', {'shape': [1, 4], 'dtype': 'bool',
                                     'axes': ['batch', 'key'], 'batch_axis': 0})
        self.assertEqual(doc.graph['inputs'][-1]['batch_axis'], 0)
        doc.undo()
        self.assertEqual(doc.graph['inputs'][-1]['axes'], ['key'])

    def test_copy_creates_independent_instance_while_share_reuses_it(self):
        doc = Document()
        node = doc.add_component(chip(), apply_policy=True)
        doc.set_input(node, 'x', 'input:' + doc.graph['inputs'][0]['id'])
        original = deepcopy(doc.selected_instance)
        doc.share(node)
        self.assertEqual(doc.selected_instance['id'], original['id'])
        doc.copy_call(node)
        self.assertNotEqual(doc.selected_instance['id'], original['id'])
        self.assertEqual(doc.selected_instance['definition_id'], original['definition_id'])
        self.assertEqual(doc.selected_call['inputs'], doc._call(node)['inputs'])

    def test_chip_output_dtype_is_checked_when_wiring_another_chip(self):
        doc = Document()
        target = doc.add_component(chip(), apply_policy=True)
        mask = chip()
        mask.definitions[0]['graph']['inputs'][0]['dtype'] = 'bool'
        source = doc.add_component(mask)
        doc.add_input('mask', [None, 4], dtype='bool', axes=['batch', 'feature'], batch_axis=0)
        doc.set_input(source, 'x', 'input:mask')
        with self.assertRaisesRegex(ValueError, 'dtype'):
            doc.set_input(target, 'x', source + ':y')

    def test_native_catalog_upgrade_and_invalid_input_draft(self):
        import dearpygui.dearpygui as dpg
        from native.paths import resource_root
        from native.ui import NativeWorkbench

        with TemporaryDirectory() as directory:
            ui = NativeWorkbench(Document(), None, directory, resource_root())
            self.addCleanup(ui.shutdown)
            ui.build()

            def invoke(tag, data=None):
                return dpg.get_item_callback(tag)(tag, data, dpg.get_item_user_data(tag))

            invoke('native_search', 'vit')
            self.assertIn('vit.small_classifier', ui.catalog_panel.visible_ids)
            invoke('native_catalog_config_vit.small_classifier')
            patch = ui.catalog_panel.config_widgets['patch_size']
            dpg.set_value(patch, '7')
            invoke('native_catalog_apply')
            self.assertIn('patch_size', dpg.get_value('native_catalog_error'))
            self.assertEqual(dpg.get_value(patch), '7')
            dpg.set_value(patch, '4')
            invoke('native_catalog_apply')
            self.assertEqual(ui.document.graph['schema_version'], 1)
            invoke('native_catalog_policy_apply')
            self.assertEqual(ui.document.graph['schema_version'], 2)
            self.assertTrue(any(item['kind'] == 'chip' and item['definition_id'] == 'SmallVisionTransformer'
                                for item in ui.document.graph['instances']))
            ui.shutdown()
