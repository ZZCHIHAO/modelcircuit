"""Atomic insertion keeps editable drafts usable and package policy explicit."""
from copy import deepcopy
import unittest

from native.document import Document
from studio_core.component_types import ComponentBundle
from studio_core.module_templates import make_module
from studio_core.operators import operator_kinds, operator_defaults


def insert(doc, bundle, position=None, **kwargs):
    if not hasattr(doc, 'add_component'):
        raise AssertionError('Document atomic component insertion is missing')
    return doc.add_component(bundle, position, **kwargs)


def primitive(kind='Identity', params=None):
    return ComponentBundle({'kind': kind, 'params': operator_defaults(kind) if params is None else params}, [], {'operator_set': 'vision-ops-v4'})


def snapshot(doc):
    return deepcopy((doc.graph, doc.trail, doc.selected, doc._undo, doc._redo))


class ComponentInsertTests(unittest.TestCase):
    def test_all_17_primitives_place_unconnected_in_draft_with_one_undo(self):
        for kind in operator_kinds():
            with self.subTest(kind=kind):
                doc = Document(); before = deepcopy(doc.graph)
                doc.graph['outputs'][0].pop('source', None)
                draft = deepcopy(doc.graph)
                ref = insert(doc, primitive(kind), {'x': -30, 'y': -10.5})
                self.assertEqual(doc.selected, ref)
                self.assertEqual(doc.selected_instance['kind'], kind)
                self.assertEqual(doc.selected_call['inputs'], {})
                self.assertEqual(doc.selected_call['position'], {'x': -30, 'y': -10.5})
                self.assertEqual(doc.graph['revision'], before['revision'] + 1)
                self.assertEqual(len(doc._undo), 1)
                self.assertEqual(doc.graph.get('resource_profile'), before.get('resource_profile'))
                self.assertTrue(doc.undo())
                expected = deepcopy(draft); expected['revision'] = doc.graph['revision']
                self.assertEqual(doc.graph, expected)
                self.assertTrue(doc.redo())
                self.assertEqual(doc.graph['calls'][-1]['id'], ref[5:])

    def test_policy_requires_opt_in_and_adoption_undoes_with_add(self):
        doc = Document(); bundle = make_module('fusion.weighted_sum', {'features': 4, 'alpha': .7})
        before = snapshot(doc)
        with self.assertRaises(ValueError): insert(doc, bundle)
        self.assertEqual(snapshot(doc), before)
        ref = insert(doc, bundle, apply_policy=True)
        self.assertEqual(doc.graph['resource_profile'], 'vision-v4')
        self.assertEqual(doc.graph['resource_batch_bound'], 2)
        self.assertEqual(doc.selected_instance['params'], {'alpha': .7})
        self.assertEqual(len(doc._undo), 1)
        doc.undo()
        self.assertEqual(doc.graph.get('resource_profile'), before[0].get('resource_profile'))
        self.assertEqual(doc.graph['calls'], before[0]['calls'])
        doc.redo(); self.assertEqual(doc.graph['calls'][-1]['id'], ref[5:])

    def test_factory_definitions_reuse_but_instances_are_independent(self):
        for identifier, config in [('head.mlp', {'in_features': 4, 'hidden_features': 8, 'num_classes': 2}), ('fusion.concat_projection', {'a_features': 4, 'b_features': 4, 'out_features': 8}), ('fusion.weighted_sum', {'features': 4}), ('fusion.gated', {'features': 4}), ('block.residual_basic', {'input_shape': [None, 8, 8, 8], 'in_channels': 8, 'out_channels': 8})]:
            with self.subTest(identifier=identifier):
                doc = Document(); bundle = make_module(identifier, config)
                insert(doc, bundle, apply_policy=True); first = deepcopy(doc.selected_instance)
                inserted = deepcopy(doc.graph['definitions'][-1])
                insert(doc, bundle); second = doc.selected_instance
                self.assertNotEqual(first['id'], second['id'])
                self.assertEqual(first['definition_id'], second['definition_id'])
                self.assertEqual(sum(item['id'] == inserted['id'] for item in doc.graph['definitions']), 1)
                self.assertEqual(inserted, bundle.definitions[0])
                self.assertEqual(len(doc._undo), 2)

    def test_insertion_targets_current_definition(self):
        doc = Document(); doc.enter('encode_a')
        root_count = len(doc.graph['calls']); count = len(doc.scope['calls'])
        insert(doc, primitive('Scale'))
        self.assertEqual(len(doc.graph['calls']), root_count)
        self.assertEqual(len(doc.scope['calls']), count + 1)
        self.assertEqual(doc.scope_id, 'encoder_chip')

    def test_invalid_bundle_position_and_policy_are_atomic(self):
        for bundle, position, kwargs in [(primitive(), {'x': float('nan'), 'y': 2}, {}), (primitive(), {'x': True, 'y': 2}, {}), (primitive(), {'x': 1, 'y': 2, 'z': 3}, {}), (primitive(), [1, 2], {}), (primitive('Scale', {'factor': float('inf')}), None, {}), (primitive(), None, {'apply_policy': 1}), (ComponentBundle({'kind': 'Identity', 'bogus': 1}, [], {'operator_set': 'vision-ops-v4'}), None, {}), (ComponentBundle({'kind': 'Identity', 'params': {}}, [], {'operator_set': 'unknown'}), None, {})]:
            with self.subTest(position=position, kwargs=kwargs):
                doc = Document(); doc.select('call:encode_a'); before = snapshot(doc)
                with self.assertRaises(ValueError): insert(doc, bundle, position, **kwargs)
                self.assertEqual(snapshot(doc), before)

    def test_old_add_supports_new_registered_operators(self):
        for kind in ('BatchNorm2d', 'Identity', 'Sigmoid', 'Multiply', 'Scale', 'OneMinus'):
            doc = Document()
            try: ref = doc.add(kind)
            except ValueError as error: self.fail(str(error))
            self.assertEqual(doc.selected_instance['kind'], kind)
            self.assertEqual(doc.selected, ref)

    def test_nested_definition_collision_remaps_references_and_preserves_defaults(self):
        doc = Document()
        child = deepcopy(doc.graph['definitions'][0]); child['version'] = '1.0.0'
        child['graph']['instances'][0]['params']['out_features'] = 6
        child['public_parameters'][1]['default'] = 6
        wrapper = {'id': 'wrapper', 'version': '1.0.0', 'public_parameters': [], 'bindings': {},
            'graph': {'inputs': [{'id': 'x', 'shape': [None, 4], 'dtype': 'float32'}],
                'instances': [{'id': 'nested', 'kind': 'chip', 'definition_id': child['id'], 'params': {}}],
                'calls': [{'id': 'nested_call', 'instance_id': 'nested', 'inputs': {'x': 'input:x'}}],
                'outputs': [{'id': 'out', 'source': 'call:nested_call'}]}}
        bundle = ComponentBundle({'kind': 'chip', 'definition_id': 'wrapper', 'params': {}}, [wrapper, child],
            {'operator_set': 'vision-ops-v4', 'resource_profile': doc.graph.get('resource_profile')})
        original = deepcopy(doc.graph['definitions'][0])
        insert(doc, bundle)
        self.assertEqual(doc.graph['definitions'][0], original)
        added = next(item for item in doc.graph['definitions'] if item['id'] == 'wrapper')
        self.assertEqual(added['graph']['instances'][0]['definition_id'], 'encoder_chip_2')
        mapped = next(item for item in doc.graph['definitions'] if item['id'] == 'encoder_chip_2')
        self.assertEqual(mapped['public_parameters'][1]['default'], 6)

    def test_current_definition_may_be_an_unrunnable_draft_when_adding_factory(self):
        doc = Document(); doc.enter('encode_a')
        doc.scope['outputs'][0].pop('source')
        insert(doc, make_module('fusion.weighted_sum', {'features': 4}), apply_policy=True)
        self.assertEqual(doc.selected_instance['definition_id'], 'WeightedSum')
        self.assertEqual(doc.scope_id, 'encoder_chip')

    def test_limit_failure_after_policy_adoption_rolls_back_redo_and_selection(self):
        doc = Document(); doc.add('ReLU'); doc.undo(); doc.select('call:encode_a')
        doc.graph['calls'].extend({'id': 'draft_' + str(index), 'instance_id': 'encoder', 'inputs': {}} for index in range(512 - len(doc.graph['calls'])))
        before = snapshot(doc)
        with self.assertRaises(ValueError):
            insert(doc, make_module('fusion.weighted_sum', {'features': 4}), apply_policy=True)
        self.assertEqual(snapshot(doc), before)

    def test_invalid_closure_defaults_and_override_are_atomic(self):
        for damage in ('default', 'override', 'missing', 'unrelated'):
            doc = Document(); bundle = make_module('fusion.weighted_sum', {'features': 4})
            if damage == 'default': bundle.definitions[0]['public_parameters'][0]['default'] = True
            if damage == 'override': bundle.entry['params']['alpha'] = True
            if damage == 'missing': bundle.definitions = []
            if damage == 'unrelated': bundle.definitions += deepcopy(doc.graph['definitions'])
            before = snapshot(doc)
            with self.subTest(damage=damage), self.assertRaises(ValueError): insert(doc, bundle, apply_policy=True)
            self.assertEqual(snapshot(doc), before)

    def test_current_project_definition_cannot_be_inserted_recursively(self):
        doc = Document(); doc.enter('encode_a'); definition = deepcopy(doc.graph['definitions'][0])
        bundle = ComponentBundle({'kind': 'chip', 'definition_id': definition['id'], 'params': {}}, [definition],
            {'operator_set': 'vision-ops-v4', 'resource_profile': doc.graph.get('resource_profile')})
        before = snapshot(doc)
        with self.assertRaises(ValueError): insert(doc, bundle)
        self.assertEqual(snapshot(doc), before)

    def test_primitive_cannot_smuggle_resource_policy_adoption(self):
        doc = Document(); bundle = primitive()
        bundle.requirements.update(resource_profile='vision-v4', resource_batch_bound=2)
        before = snapshot(doc)
        with self.assertRaises(ValueError): insert(doc, bundle, apply_policy=True)
        self.assertEqual(snapshot(doc), before)

    def test_chip_requires_explicit_policy_even_with_adoption_opt_in_and_preserves_history(self):
        for requirements in ({'operator_set': 'vision-ops-v4'},
                {'operator_set': 'vision-ops-v4', 'resource_batch_bound': 2},
                {'operator_set': 'vision-ops-v4', 'resource_profile': 'vision-v4'}):
            for adopt in (False, True):
                with self.subTest(requirements=requirements, apply_policy=adopt):
                    doc = Document(); doc.add('ReLU'); doc.add('Sigmoid'); doc.undo()
                    doc.enter('encode_a'); doc.select('call:project')
                    bundle = make_module('fusion.weighted_sum', {'features': 4})
                    bundle.requirements = deepcopy(requirements)
                    before = snapshot(doc)
                    with self.assertRaises(ValueError): insert(doc, bundle, apply_policy=adopt)
                    self.assertEqual(snapshot(doc), before)

    def test_chip_explicit_null_policy_preserves_legacy_project_and_undo(self):
        doc = Document(); doc.graph.pop('resource_profile', None)
        bundle = make_module('fusion.weighted_sum', {'features': 4})
        bundle.requirements = {'operator_set': 'vision-ops-v4', 'resource_profile': None}
        before = deepcopy(doc.graph)
        insert(doc, bundle)
        self.assertNotIn('resource_profile', doc.graph)
        self.assertNotIn('resource_batch_bound', doc.graph)
        self.assertEqual(doc.selected_instance['definition_id'], 'WeightedSum')
        self.assertEqual(len(doc._undo), 1)
        doc.undo(); before['revision'] = doc.graph['revision']
        self.assertEqual(doc.graph, before)

    def test_unbound_view_scope_places_only_in_target_definition(self):
        from native.workbench_state import ScopeTarget, ViewState
        from native.view_document import view_scope
        doc = Document(); doc.select('input:a'); root_calls = deepcopy(doc.graph['calls'])
        view = ViewState('insertion', ScopeTarget('encoder_chip'), [], [], None, 'primary')
        with view_scope(doc, view):
            ref = insert(doc, primitive('Sigmoid'))
            self.assertEqual(doc.selected, ref)
            self.assertEqual(doc.selected_instance['kind'], 'Sigmoid')
        self.assertEqual(doc.graph['calls'], root_calls)
        self.assertIsNone(doc.scope_id)
        self.assertEqual(doc.selected, 'input:a')
        self.assertEqual(len(doc._undo), 1)

    def test_inserted_reused_definition_has_independent_runtime_weights(self):
        import torch
        from studio_core.runtime import build_model
        graph = {'schema_version': 1, 'id': 'tiny', 'revision': 1,
            'inputs': [{'id': 'x', 'shape': [None, 3], 'dtype': 'float32'}],
            'instances': [], 'calls': [], 'outputs': [{'id': 'first'}], 'definitions': []}
        doc = Document(graph)
        bundle = make_module('head.mlp', {'in_features': 3, 'hidden_features': 4, 'num_classes': 2})
        first = insert(doc, bundle, apply_policy=True); first_instance = doc.selected_instance['id']
        doc.set_input(first, 'x', 'input:x'); doc.set_output('first', first + ':logits')
        second = insert(doc, bundle); second_instance = doc.selected_instance['id']
        doc.set_input(second, 'x', 'input:x'); doc.add_output('second', second + ':logits')
        model = build_model(doc.graph)
        left, right = model.components[first_instance], model.components[second_instance]
        right.load_state_dict(left.state_dict())
        self.assertNotEqual(left.components['linear1'].weight.data_ptr(), right.components['linear1'].weight.data_ptr())
        before = right.components['linear1'].weight.detach().clone()
        outputs = model(x=torch.randn(2, 3))
        torch.testing.assert_close(outputs['first'], outputs['second'])
        outputs['first'].square().mean().backward()
        torch.optim.SGD(model.parameters(), lr=.01).step()
        self.assertIsNone(right.components['linear1'].weight.grad)
        torch.testing.assert_close(right.components['linear1'].weight, before)

    def test_insertion_cannot_extend_existing_call_path_beyond_depth_limit(self):
        graph = {'schema_version': 1, 'id': 'depth', 'revision': 1,
            'inputs': [{'id': 'x', 'shape': [None, 4], 'dtype': 'float32'}],
            'instances': [{'id': 'top', 'kind': 'chip', 'definition_id': 'level_0', 'params': {}}],
            'calls': [{'id': 'top', 'instance_id': 'top', 'inputs': {'x': 'input:x'}}],
            'outputs': [{'id': 'out', 'source': 'call:top'}], 'definitions': []}
        for index in range(6):
            instance = {'id': 'inner', 'kind': 'Identity', 'params': {}} if index == 5 else {
                'id': 'inner', 'kind': 'chip', 'definition_id': 'level_' + str(index + 1), 'params': {}}
            graph['definitions'].append({'id': 'level_' + str(index), 'version': '1.0.0',
                'public_parameters': [], 'bindings': {}, 'graph': {
                    'inputs': [{'id': 'x', 'shape': [None, 4], 'dtype': 'float32'}], 'instances': [instance],
                    'calls': [{'id': 'inside', 'instance_id': 'inner', 'inputs': {'x': 'input:x'}}],
                    'outputs': [{'id': 'out', 'source': 'call:inside'}]}})
        doc = Document(graph); doc.enter('top')
        for _ in range(5): doc.enter('inside')
        before = snapshot(doc)
        with self.assertRaises(ValueError):
            insert(doc, make_module('fusion.weighted_sum', {'features': 4}), apply_policy=True)
        self.assertEqual(snapshot(doc), before)


if __name__ == '__main__': unittest.main()
