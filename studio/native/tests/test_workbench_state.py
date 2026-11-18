"""Pure workbench state contracts; no native GUI context."""
import copy
import json
import unittest

from native.document import Document


def state():
    from native.workbench_state import WorkbenchState
    return WorkbenchState()


class WorkbenchStateTests(unittest.TestCase):
    def test_reopening_definition_changes_context_epoch_preserves_pane_and_model(self):
        doc, workbench = Document(), state()
        before = copy.deepcopy(doc.graph)
        workbench.set_layout('columns')
        first = workbench.open_definition('encoder_chip', ['encode_a'], pane='secondary')
        epoch = workbench.views[first].context_epoch
        second = workbench.open_definition('encoder_chip', ['encode_b'])
        self.assertEqual(first, second)
        self.assertEqual(workbench.views[first].pane, 'secondary')
        self.assertEqual(workbench.views[first].call_path, ['encode_b'])
        self.assertEqual(workbench.views[first].context_epoch, epoch + 1)
        workbench.open_definition('encoder_chip', ['encode_b'])
        self.assertEqual(workbench.views[first].context_epoch, epoch + 1)
        self.assertEqual(len(workbench.views), 2)
        self.assertEqual(doc.graph, before)
        self.assertFalse(doc.can_undo)

    def test_root_is_fixed_and_navigation_is_atomic_on_bad_arguments(self):
        workbench = state()
        self.assertEqual(workbench.focused_view_id, 'root')
        before = workbench.to_payload()
        for action in (lambda: workbench.close('root'),
                       lambda: workbench.move_to_pane('root', 'secondary'),
                       lambda: workbench.activate('missing'),
                       lambda: workbench.set_layout('grid'),
                       lambda: workbench.open_definition('ok', ['a'] * 7),
                       lambda: workbench.open_definition('ok', [float('nan')]),
                       lambda: workbench.open_definition('ok', pane='third')):
            with self.assertRaises(ValueError): action()
            self.assertEqual(workbench.to_payload(), before)
        self.assertEqual(workbench.open_root(), 'root')

    def test_sixteenth_tab_is_allowed_and_seventeenth_is_atomically_rejected(self):
        workbench = state()
        for index in range(15): workbench.open_definition('chip_' + str(index))
        before = workbench.to_payload()
        with self.assertRaises(ValueError): workbench.open_definition('overflow')
        self.assertEqual(workbench.to_payload(), before)
        self.assertEqual(len(workbench.views), 16)
        workbench.open_definition('chip_0', ['new_context'])

    def test_closing_last_secondary_falls_back_to_primary_active(self):
        workbench = state()
        primary = workbench.open_definition('primary_chip')
        workbench.set_layout('columns')
        secondary = workbench.open_definition('secondary_chip', pane='secondary')
        workbench.close(secondary)
        self.assertEqual(workbench.focused_view_id, primary)
        self.assertEqual(workbench.focused_pane, 'primary')
        self.assertIsNone(workbench.active_by_pane['secondary'])

    def test_single_layout_falls_back_and_activating_hidden_tab_moves_it(self):
        workbench = state()
        workbench.set_layout('rows')
        chip = workbench.open_definition('encoder_chip', pane='secondary')
        workbench.set_layout('single')
        self.assertEqual(workbench.focused_view_id, 'root')
        workbench.activate(chip)
        self.assertEqual(workbench.views[chip].pane, 'primary')
        self.assertEqual(workbench.focused_view_id, chip)
        self.assertEqual(workbench.views['root'].pane, 'primary')

    def test_move_of_active_tab_leaves_valid_focus_in_both_layouts(self):
        workbench = state()
        chip = workbench.open_definition('encoder_chip')
        workbench.move_to_pane(chip, 'secondary')
        self.assertEqual(workbench.focused_view_id, 'root')
        workbench.set_layout('rows')
        workbench.activate(chip)
        workbench.move_to_pane(chip, 'primary')
        self.assertEqual(workbench.focused_pane, 'primary')
        self.assertEqual(workbench.focused_view_id, chip)

    def test_payload_roundtrip_uses_ordered_views_and_excludes_transient_state(self):
        from native.workbench_state import WorkbenchState
        doc, workbench = Document(), state()
        workbench.set_layout('columns')
        chip = workbench.open_definition('encoder_chip', ['encode_a'], pane='secondary')
        workbench.views[chip].selected_refs = ['call:project']
        workbench.views[chip].probe_port = 'out'
        workbench.open_definition('encoder_chip', ['encode_b'])
        workbench.warnings.append('old warning')
        payload = workbench.to_payload()
        self.assertEqual([view['view_id'] for view in payload['views']], ['root', chip])
        self.assertEqual(set(payload['views'][1]), {'view_id', 'target', 'call_path', 'selected_refs', 'probe_port', 'pane'})
        restored = WorkbenchState.from_payload(json.loads(json.dumps(payload)), doc)
        self.assertEqual(restored.to_payload(), payload)
        self.assertEqual(restored.views[chip].context_epoch, 0)
        self.assertEqual(restored.warnings, [])
        payload['views'][1]['call_path'].clear()
        self.assertEqual(restored.views[chip].call_path, ['encode_b'])

    def test_malformed_payloads_reset_whole_workbench_and_report_warning(self):
        from native.workbench_state import WorkbenchState
        clean = state().to_payload()
        malformed = [None, [], {'layout': 'single'}, dict(clean, unexpected='code')]
        for mutate in (lambda p: p['views'].append(copy.deepcopy(p['views'][0])),
                       lambda p: p['views'][0].update(call_path=['encode_a']),
                       lambda p: p['views'][0].update(probe_port=float('inf')),
                       lambda p: p['views'][0].update(extra='handle'),
                       lambda p: p.update(focused_pane='other')):
            candidate = copy.deepcopy(clean); mutate(candidate); malformed.append(candidate)
        for payload in malformed:
            with self.subTest(payload=payload):
                restored = WorkbenchState.from_payload(payload, Document())
                self.assertEqual(restored.to_payload(), clean)
                self.assertTrue(restored.warnings)
                self.assertTrue(all(type(item) is str for item in restored.warnings))

    def test_reconcile_clears_lost_path_but_keeps_definition_and_selection(self):
        doc, workbench = Document(), state()
        chip = workbench.open_definition('encoder_chip', ['encode_a'])
        workbench.views[chip].selected_refs = ['call:project', 'call:gone']
        workbench.views[chip].probe_port = 'gone'
        epoch = workbench.views[chip].context_epoch
        doc.remove('call:encode_a')
        messages = workbench.reconcile(doc)
        self.assertEqual(workbench.views[chip].call_path, [])
        self.assertEqual(workbench.views[chip].selected_refs, ['call:project'])
        self.assertIsNone(workbench.views[chip].probe_port)
        self.assertEqual(workbench.views[chip].context_epoch, epoch + 1)
        self.assertEqual(workbench.warnings, messages)
        self.assertEqual(workbench.reconcile(doc), [])

    def test_reconcile_deleted_focused_definition_falls_back_and_replace_resets(self):
        doc, workbench = Document(), state()
        workbench.set_layout('columns')
        chip = workbench.open_definition('encoder_chip', ['encode_a'], pane='secondary')
        doc.graph['definitions'].clear()
        self.assertTrue(workbench.reconcile(doc))
        self.assertNotIn(chip, workbench.views)
        self.assertEqual(workbench.focused_view_id, 'root')
        workbench.open_definition('new')
        workbench.reconcile(doc, model_replaced=True)
        self.assertEqual(list(workbench.views), ['root'])
        self.assertEqual(workbench.layout, 'single')

    def test_reconcile_interface_rename_and_undo_clean_invalid_refs(self):
        from native.view_document import view_scope
        doc, workbench = Document(), state()
        chip = workbench.open_definition('encoder_chip')
        workbench.views[chip].selected_refs = ['input:x']
        with view_scope(doc, workbench.views[chip]): doc.rename_input('x', 'features')
        self.assertTrue(workbench.reconcile(doc))
        self.assertEqual(workbench.views[chip].selected_refs, [])
        workbench.views[chip].selected_refs = ['input:features']
        doc.undo()
        self.assertTrue(workbench.reconcile(doc))
        self.assertEqual(workbench.views[chip].selected_refs, [])

    def test_payload_stale_references_are_sanitized_without_losing_valid_tab(self):
        from native.workbench_state import WorkbenchState
        workbench = state()
        chip = workbench.open_definition('encoder_chip', ['missing'])
        workbench.views[chip].selected_refs = ['call:missing']
        restored = WorkbenchState.from_payload(workbench.to_payload(), Document())
        self.assertIn(chip, restored.views)
        self.assertEqual(restored.views[chip].call_path, [])
        self.assertEqual(restored.views[chip].selected_refs, [])
        self.assertTrue(restored.warnings)

    def test_stale_active_view_reference_does_not_discard_valid_definition_tabs(self):
        from native.workbench_state import WorkbenchState
        workbench = state()
        chip = workbench.open_definition('encoder_chip')
        payload = workbench.to_payload()
        payload['active_by_pane']['primary'] = 'deleted_view'
        restored = WorkbenchState.from_payload(payload, Document())
        self.assertIn(chip, restored.views)
        self.assertEqual(restored.focused_view_id, 'root')
        self.assertTrue(restored.warnings)

    def test_oversized_open_is_rejected_without_partially_creating_a_tab(self):
        workbench = state(); before = workbench.to_payload()
        with self.assertRaises(ValueError): workbench.open_definition('a' * (8 * 1024 * 1024))
        self.assertEqual(workbench.to_payload(), before)

    def test_serialization_rejects_public_state_containing_objects_or_wrong_collection_types(self):
        workbench = state(); chip = workbench.open_definition('encoder_chip')
        workbench.views[chip].call_path = {'encode_a': 'graph payload'}
        with self.assertRaises(ValueError): workbench.to_payload()
        workbench.views[chip].call_path = []
        workbench.views[chip].selected_refs = [object()]
        with self.assertRaises(ValueError): workbench.to_payload()

    def test_ambiguous_definition_in_editable_draft_closes_only_affected_view(self):
        doc, workbench = Document(), state()
        chip = workbench.open_definition('encoder_chip')
        doc.graph['definitions'].append(copy.deepcopy(doc.graph['definitions'][0]))
        messages = workbench.reconcile(doc)
        self.assertNotIn(chip, workbench.views)
        self.assertEqual(workbench.focused_view_id, 'root')
        self.assertTrue(messages)

    def test_duplicate_kind_local_node_refs_are_cleared_without_reattributing_probe(self):
        from native.workbench_state import WorkbenchState
        from native.view_document import view_scope
        for collection, reference in (('inputs', 'input:a'), ('calls', 'call:encode_a'),
                                      ('outputs', 'output:logits')):
            with self.subTest(collection=collection):
                doc, workbench = Document(), state()
                chip = workbench.open_definition('encoder_chip')
                doc.graph[collection].append(copy.deepcopy(doc.graph[collection][0]))
                before = copy.deepcopy(doc.graph)
                workbench.views['root'].selected_refs = [reference, 'call:classify']
                workbench.views['root'].probe_port = 'out'
                restored = WorkbenchState.from_payload(workbench.to_payload(), doc)
                self.assertEqual(restored.views['root'].selected_refs, ['call:classify'])
                self.assertIsNone(restored.views['root'].probe_port)
                self.assertIn(chip, restored.views)
                self.assertEqual(doc.graph, before)
                self.assertTrue(restored.warnings)
                with view_scope(doc, restored.views['root']):
                    self.assertEqual(doc.selected, 'call:classify')

    def test_duplicate_instance_probe_source_is_cleared_but_call_selection_is_retained(self):
        doc, workbench = Document(), state()
        chip = workbench.open_definition('encoder_chip')
        doc.graph['instances'].append(copy.deepcopy(doc.graph['instances'][0]))
        before = copy.deepcopy(doc.graph)
        workbench.views['root'].selected_refs = ['call:encode_a']
        workbench.views['root'].probe_port = 'out'
        messages = workbench.reconcile(doc)
        self.assertEqual(workbench.views['root'].selected_refs, ['call:encode_a'])
        self.assertIsNone(workbench.views['root'].probe_port)
        self.assertIn(chip, workbench.views)
        self.assertEqual(doc.graph, before)
        self.assertTrue(messages)

    def test_ambiguous_chip_definition_or_output_port_cannot_supply_probe_port(self):
        for ambiguity in ('definition', 'output'):
            with self.subTest(ambiguity=ambiguity):
                doc, workbench = Document(), state()
                if ambiguity == 'definition':
                    doc.graph['definitions'].append(copy.deepcopy(doc.graph['definitions'][0]))
                else:
                    outputs = doc.graph['definitions'][0]['graph']['outputs']
                    outputs.append(copy.deepcopy(outputs[0]))
                before = copy.deepcopy(doc.graph)
                workbench.views['root'].selected_refs = ['call:encode_a']
                workbench.views['root'].probe_port = 'out'
                messages = workbench.reconcile(doc)
                self.assertEqual(workbench.views['root'].selected_refs, ['call:encode_a'])
                self.assertIsNone(workbench.views['root'].probe_port)
                self.assertEqual(doc.graph, before)
                self.assertTrue(messages)

    def test_same_identifier_in_different_node_kinds_remains_unambiguous(self):
        doc, workbench = Document(), state()
        doc.graph['inputs'][0]['id'] = 'encode_a'
        workbench.views['root'].selected_refs = ['call:encode_a', 'input:encode_a']
        workbench.views['root'].probe_port = 'out'
        self.assertEqual(workbench.reconcile(doc), [])
        self.assertEqual(workbench.views['root'].selected_refs, ['call:encode_a', 'input:encode_a'])
        self.assertEqual(workbench.views['root'].probe_port, 'out')


if __name__ == '__main__': unittest.main()
