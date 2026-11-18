"""Explicit definition/call cursors over one shared Document and history."""
import copy
import unittest

from native.document import Document
from studio_core.examples import example_graph


def view(definition_id=None, path=(), selected=()):
    from native.workbench_state import ScopeTarget, ViewState
    return ViewState('test', ScopeTarget(definition_id), list(path), list(selected), None, 'primary')


def scoped(doc, current):
    from native.view_document import view_scope
    return view_scope(doc, current)


class ViewDocumentTests(unittest.TestCase):
    def test_new_registered_operators_have_real_defaults_scoped_calls_and_atomic_undo(self):
        from studio_core.operators import operator_defaults
        for kind in ('BatchNorm2d', 'Identity', 'Sigmoid', 'Multiply', 'Scale', 'OneMinus'):
            with self.subTest(kind=kind):
                doc = Document(); before = copy.deepcopy(doc.graph)
                current = view('encoder_chip')
                with scoped(doc, current):
                    ref = doc.add(kind)
                    self.assertEqual(doc.selected, ref)
                    self.assertEqual(doc.selected_instance['kind'], kind)
                    self.assertEqual(doc.selected_instance['params'], operator_defaults(kind))
                    self.assertEqual(doc.selected_call['instance_id'], doc.selected_instance['id'])
                    self.assertEqual(doc.selected_call['inputs'], {})
                    self.assertEqual(doc.scope_id, 'encoder_chip')
                    self.assertEqual(len(doc.scope['calls']), len(before['definitions'][0]['graph']['calls']) + 1)
                self.assertEqual(doc.graph['calls'], before['calls'])
                self.assertEqual(doc.graph['revision'], before['revision'] + 1)
                self.assertEqual(len(doc._undo), 1)
                self.assertTrue(doc.undo())
                before['revision'] = doc.graph['revision']
                self.assertEqual(doc.graph, before)

    def test_unbound_definition_can_rename_interface_without_fake_probe_or_navigation_edits(self):
        doc = Document(); graph = doc.graph; revision = graph['revision']; key = doc.semantic_key()
        with scoped(doc, view('encoder_chip', selected=['call:project'])):
            self.assertEqual(doc.scope_id, 'encoder_chip')
            self.assertIs(doc.scope, graph['definitions'][0]['graph'])
            self.assertEqual(doc.trail, [])
            self.assertIsNone(doc.probe_source())
        self.assertIs(doc.graph, graph)
        self.assertEqual((doc.graph['revision'], doc.semantic_key()), (revision, key))
        self.assertFalse(doc.can_undo)
        with scoped(doc, view('encoder_chip')): doc.rename_input('x', 'features')
        self.assertEqual(doc.graph['definitions'][0]['graph']['inputs'][0]['id'], 'features')
        self.assertEqual(doc.graph['inputs'][0]['id'], 'a')
        self.assertIsNone(doc.scope_id)
        self.assertTrue(doc.can_undo)

    def test_nested_root_override_and_selection_restore_chip_cursor(self):
        doc = Document(); doc.enter('encode_a'); doc.select('call:project')
        before = copy.deepcopy(doc.trail)
        with scoped(doc, view()):
            self.assertIsNone(doc.scope_id)
            self.assertIs(doc.scope, doc.graph)
            with scoped(doc, view('encoder_chip', ['encode_b'], ['call:activate'])):
                self.assertEqual(doc.probe_source(), 'call:encode_b/call:activate')
            self.assertIsNone(doc.scope_id)
            self.assertEqual(doc.trail, [])
        self.assertEqual(doc.trail, before)
        self.assertEqual(doc.selected, 'call:project')
        self.assertEqual(doc.scope_id, 'encoder_chip')

    def test_command_failure_and_rollback_restore_original_cursor_and_exception(self):
        doc = Document(); doc.select('input:a'); original = copy.deepcopy(doc.graph)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            with scoped(doc, view('encoder_chip', ['encode_a'])):
                doc.remove_many(['call:project', 'call:missing'])
        self.assertEqual(doc.graph, original)
        self.assertIsNone(doc.scope_id)
        self.assertEqual(doc.trail, [])
        self.assertEqual(doc.selected, 'input:a')
        self.assertFalse(doc.can_undo)

    def test_same_named_root_and_definition_calls_do_not_cross_edit(self):
        graph = example_graph(); graph['calls'][0]['id'] = 'project'
        doc = Document(graph)
        with scoped(doc, view('encoder_chip')): doc.move('call:project', 12, 34)
        self.assertEqual(doc.graph['definitions'][0]['graph']['calls'][0]['position'], {'x': 12, 'y': 34})
        self.assertEqual(doc.graph['calls'][0]['position'], {'x': 250, 'y': 90})

    def test_invalid_or_forged_paths_fail_before_any_cursor_change(self):
        doc = Document(); doc.select('input:b')
        for current in (view('missing'), view('encoder_chip', ['classify']),
                        view('encoder_chip', ['encode_a', 'project']),
                        view(None, ['encode_a']), view('encoder_chip', ['absent'])):
            with self.assertRaises(ValueError):
                with scoped(doc, current): self.fail('Invalid scope was entered')
            self.assertIsNone(doc.scope_id)
            self.assertEqual(doc.trail, [])
            self.assertEqual(doc.selected, 'input:b')

    def test_deep_real_call_path_keeps_shared_calls_distinct(self):
        graph = example_graph()
        inner = copy.deepcopy(graph['definitions'][0])
        inner['id'] = 'wrapper'
        inner['graph'] = {'inputs': [{'id': 'x'}],
                          'instances': [{'id': 'inner', 'kind': 'chip', 'definition_id': 'encoder_chip', 'params': {}}],
                          'calls': [{'id': 'inside', 'instance_id': 'inner', 'inputs': {}}],
                          'outputs': [{'id': 'out', 'source': 'call:inside'}]}
        graph['definitions'].append(inner)
        graph['instances'][0]['definition_id'] = 'wrapper'
        doc = Document(graph)
        for call in ('encode_a', 'encode_b'):
            with scoped(doc, view('encoder_chip', [call, 'inside'], ['call:project'])):
                self.assertEqual(doc.probe_source(), 'call:' + call + '/call:inside/call:project')

    def test_navigation_from_unbound_definition_does_not_invent_root_call_trail(self):
        graph = example_graph()
        leaf = copy.deepcopy(graph['definitions'][0]); leaf['id'] = 'leaf'
        graph['definitions'].append(leaf)
        graph['definitions'][0]['graph']['instances'].append(
            {'id': 'child', 'kind': 'chip', 'definition_id': 'leaf', 'params': {}})
        graph['definitions'][0]['graph']['calls'].append(
            {'id': 'nested', 'instance_id': 'child', 'inputs': {}})
        doc = Document(graph)
        with scoped(doc, view('encoder_chip')):
            doc.enter('nested')
            self.assertEqual(doc.scope_id, 'leaf')
            self.assertEqual(doc.trail, [])
            self.assertIsNone(doc.probe_source('input:x'))
            doc.root()
            self.assertIsNone(doc.scope_id)
        with scoped(doc, view()):
            doc.enter('encode_a')
            self.assertEqual(doc.probe_source('input:x'), 'call:encode_a/input:x')
            self.assertTrue(doc.back())
            self.assertIsNone(doc.scope_id)

    def test_definition_defaults_and_distinct_instance_overrides_remain_separate(self):
        doc = Document(); second = doc.add('chip', 'encoder_chip'); second_id = doc.selected_instance['id']
        with scoped(doc, view(None, selected=['call:encode_b'])):
            self.assertEqual(doc.selected_instance['id'], 'encoder')
            doc.set_param(doc.selected_instance['id'], 'hidden_features', 10)
        with scoped(doc, view(None, selected=[second])):
            self.assertEqual(doc.selected_instance['id'], second_id)
            self.assertEqual(doc.selected_instance['params']['hidden_features'], 8)
        with scoped(doc, view('encoder_chip')):
            doc.set_param('dense', 'out_features', 12)
        self.assertEqual(doc.graph['instances'][0]['params']['hidden_features'], 10)
        self.assertEqual(doc.graph['definitions'][0]['public_parameters'][1]['default'], 8)

    def test_undo_replaces_graph_and_later_commands_stay_on_current_graph(self):
        doc = Document(); doc.set_param('encoder', 'hidden_features', 10)
        with scoped(doc, view('encoder_chip', ['encode_a'])):
            old_graph = doc.graph
            doc.undo()
            self.assertIsNot(doc.graph, old_graph)
            self.assertIs(doc.scope, doc.graph)
        with scoped(doc, view('encoder_chip')): doc.rename_input('x', 'features')
        self.assertEqual(doc.graph['definitions'][0]['graph']['inputs'][0]['id'], 'features')
        self.assertEqual(old_graph['definitions'][0]['graph']['inputs'][0]['id'], 'x')

    def test_import_replaces_graph_but_restores_valid_original_cursor_by_id(self):
        doc = Document(); doc.enter('encode_b'); doc.select('call:project')
        code = doc.export_python()
        with scoped(doc, view()): doc.import_python(code)
        self.assertEqual(doc.scope_id, 'encoder_chip')
        self.assertEqual(doc.probe_source(), 'call:encode_b/call:project')
        self.assertIs(doc.scope, doc.graph['definitions'][0]['graph'])

    def test_missing_original_target_on_replace_restores_root_without_hiding_error(self):
        doc = Document(); doc.enter('encode_a'); doc.select('call:project')
        replacement = example_graph(); replacement['definitions'] = []
        with self.assertRaisesRegex(RuntimeError, 'original error'):
            with scoped(doc, view()):
                doc.replace_graph(replacement)
                raise RuntimeError('original error')
        self.assertIsNone(doc.scope_id)
        self.assertEqual(doc.trail, [])
        self.assertIsNone(doc.selected)

    def test_removed_original_selection_cannot_reappear_after_scope_restore(self):
        doc = Document(); doc.select('call:encode_a')
        with scoped(doc, view()): doc.remove('call:encode_a')
        self.assertIsNone(doc.selected)


if __name__ == '__main__': unittest.main()
