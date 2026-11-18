import copy
import unittest

from studio_core.experiment_spec import (checkpoint_signature, default_classification_config,
                                        experiment_signature, model_digest, validate_classification_config)
from studio_core.examples import example_graph


class ExperimentSpecTests(unittest.TestCase):
    def valid_config(self):
        config = default_classification_config()
        config['data'].update(path='D:/fixtures/features.csv', feature_columns=['f1','f2'])
        return config

    def test_defaults_are_fresh_and_validation_is_finite_and_explicit(self):
        config = self.valid_config()
        checked = validate_classification_config(config)
        self.assertEqual(checked['device'], 'cpu')
        self.assertEqual(checked['profile'], 'classification-v2')
        checked['data']['feature_columns'].append('f3')
        self.assertEqual(config['data']['feature_columns'], ['f1','f2'])
        for field, value in [('epochs',True),('lr',float('nan')),('batch_size',65),('device','cuda'),('device','auto')]:
            invalid=copy.deepcopy(config); invalid[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError): validate_classification_config(invalid)
        invalid=copy.deepcopy(config); invalid['gpu_ids']=[0,1]
        with self.assertRaises(ValueError): validate_classification_config(invalid)

    def test_label_group_and_split_columns_cannot_be_features(self):
        for key in ['target_column','group_column','split_column']:
            config=self.valid_config(); config['data'][key]='f1'
            with self.subTest(key=key),self.assertRaises(ValueError): validate_classification_config(config)
        config=self.valid_config(); config['data']['validation_fraction']=0.6;config['data']['test_fraction']=0.5
        with self.assertRaises(ValueError): validate_classification_config(config)

    def test_huge_finite_integers_are_rejected_as_validation_errors(self):
        for field in ('lr','validation_fraction'):
            config=self.valid_config()
            if field=='lr': config[field]=10**400
            else: config['data'][field]=10**400
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_classification_config(config)

    def test_image_contract_and_single_named_input(self):
        config=default_classification_config('image_folder');config['data']['path']='D:/fixtures/images'
        self.assertEqual(validate_classification_config(config)['data']['image_size'],[32,32])
        for key,value in [('image_size',[0,32]),('channels',2),('split_mode','unknown')]:
            invalid=copy.deepcopy(config);invalid['data'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_classification_config(invalid)

    def test_run_identity_tracks_settings_and_dataset_but_not_node_movement(self):
        graph=example_graph();config=self.valid_config()
        initial=experiment_signature(graph,config,'a'*64)
        graph['calls'][0]['position']['x']+=100;graph['revision']+=1
        self.assertEqual(initial,experiment_signature(graph,config,'a'*64))
        self.assertNotEqual(initial,experiment_signature(graph,config,'b'*64))
        config['device']='cuda:0'
        self.assertNotEqual(initial,experiment_signature(graph,config,'a'*64))

    def test_checkpoint_identity_allows_more_epochs_and_device_change_but_not_new_data_or_lr(self):
        graph=example_graph();config=self.valid_config()
        signature=checkpoint_signature(graph,config,'a'*64)
        graph['name']='Renamed only';config['epochs']+=5;config['device']='cuda:0';config['resume_checkpoint']='last.pt'
        self.assertEqual(signature,checkpoint_signature(graph,config,'a'*64))
        self.assertNotEqual(signature,checkpoint_signature(graph,config,'b'*64))
        config['lr']*=2
        self.assertNotEqual(signature,checkpoint_signature(graph,config,'a'*64))
        self.assertEqual(model_digest(graph),model_digest(copy.deepcopy(graph)))

    def test_parameter_names_matching_metadata_are_semantic(self):
        for key in ('name', 'position', 'revision'):
            graph = example_graph()
            graph['instances'][0]['params'][key] = 0.2
            before = model_digest(graph)
            graph['instances'][0]['params'][key] = 0.7
            with self.subTest(key=key):
                self.assertNotEqual(before, model_digest(graph))
        graph = example_graph()
        graph['definitions'][0]['bindings']['name'] = ['activation.p']
        before = model_digest(graph)
        graph['definitions'][0]['bindings']['name'] = ['dense.out_features']
        self.assertNotEqual(before, model_digest(graph))

    def test_real_chip_binding_named_name_remains_semantic_with_unchanged_state_shapes(self):
        from studio_core.schema import validate_graph
        graph = example_graph()
        definition = graph['definitions'][0]
        definition['graph']['instances'][1] = {'id': 'activation', 'kind': 'Dropout', 'params': {'p': 0.2}}
        definition['public_parameters'].append({'id': 'name', 'name': 'Drop probability', 'default': 0.2})
        definition['bindings']['name'] = ['activation.p']
        graph['instances'][0]['params']['name'] = 0.2
        self.assertTrue(validate_graph(graph)['valid'])
        before = model_digest(graph)
        graph['instances'][0]['params']['name'] = 0.7
        self.assertTrue(validate_graph(graph)['valid'])
        self.assertNotEqual(before, model_digest(graph))

    def test_s3_bound_shadow_values_do_not_change_identity_but_public_defaults_do(self):
        graph = example_graph(); graph['definitions'][0]['output_mode'] = 'named'
        before = model_digest(graph)
        shadow = copy.deepcopy(graph)
        shadow['definitions'][0]['graph']['instances'][0]['params'] = {}
        self.assertEqual(before, model_digest(shadow))
        shadow['definitions'][0]['graph']['instances'][0]['params'] = {'in_features': 999, 'out_features': 333}
        self.assertEqual(before, model_digest(shadow))
        shadow['definitions'][0]['public_parameters'][1]['default'] = 10
        self.assertNotEqual(before, model_digest(shadow))
        explicit = copy.deepcopy(graph)
        explicit['instances'][0]['params'].pop('hidden_features')
        self.assertNotEqual(before, model_digest(explicit), 'unbound explicit/implicit constructor overrides have distinct future default behavior')

    def test_s3_single_output_aliases_and_effective_modes_are_canonical(self):
        graph = example_graph()
        graph['calls'][2]['inputs']['a'] = 'call:encode_a:out'
        alternate = copy.deepcopy(graph)
        alternate['definitions'][0]['output_mode'] = 'tensor'
        alternate['calls'][2]['inputs']['b'] = 'call:encode_b:out'
        alternate['outputs'][0]['source'] = 'call:classify:out'
        self.assertEqual(model_digest(graph), model_digest(alternate))
        meaningful = example_graph()
        meaningful['definitions'][0]['graph']['outputs'][0]['id'] = 'features'
        explicit_named = copy.deepcopy(meaningful)
        explicit_named['definitions'][0]['output_mode'] = 'named'
        self.assertEqual(model_digest(meaningful), model_digest(explicit_named))

    def test_qualified_primitive_only_roundtrip_does_not_change_identity_version(self):
        from studio_core.python_io import export_python, import_python
        graph = {'schema_version': 1, 'id': 'qualified_only', 'revision': 1,
                 'inputs': [{'id': 'x', 'shape': [None, 4], 'dtype': 'float32'}],
                 'instances': [{'id': 'head', 'kind': 'Linear', 'params': {'in_features': 4, 'out_features': 2}}],
                 'calls': [{'id': 'project', 'instance_id': 'head', 'inputs': {'x': 'input:x'}}],
                 'outputs': [{'id': 'logits', 'source': 'call:project:out'}], 'definitions': []}
        source = export_python(graph)
        restored = import_python(source)
        self.assertEqual(model_digest(graph), model_digest(restored))
        self.assertEqual(checkpoint_signature(graph, self.valid_config(), 'a' * 64), checkpoint_signature(restored, self.valid_config(), 'a' * 64))

    def test_s3_execution_order_is_canonical_but_independent_stochastic_order_is_semantic(self):
        graph = example_graph(); graph['definitions'][0]['output_mode'] = 'named'
        graph['definitions'][0]['graph']['instances'][1] = {'id': 'activation', 'kind': 'Dropout', 'params': {'p': 0.5}}
        same = copy.deepcopy(graph)
        same['calls'].insert(0, same['calls'].pop())
        self.assertEqual(model_digest(graph), model_digest(same), 'moving a blocked dependent call does not change execution')
        changed = copy.deepcopy(graph)
        changed['calls'][0], changed['calls'][1] = changed['calls'][1], changed['calls'][0]
        self.assertNotEqual(model_digest(graph), model_digest(changed), 'independent dropout invocations retain their declaration order')

    def test_s3_digest_remains_available_for_incomplete_or_invalid_drafts(self):
        for mutation in ('empty_outputs', 'unknown_instance', 'unknown_definition', 'unknown_reference'):
            graph = example_graph(); graph['definitions'][0]['output_mode'] = 'named'
            if mutation == 'empty_outputs': graph['definitions'][0]['graph']['outputs'] = []
            elif mutation == 'unknown_instance': graph['calls'][0]['instance_id'] = 'missing'
            elif mutation == 'unknown_definition': graph['instances'][0]['definition_id'] = 'missing'
            else: graph['calls'][2]['inputs']['a'] = 'call:missing:unknown'
            with self.subTest(mutation=mutation):
                before = copy.deepcopy(graph)
                self.assertEqual(len(model_digest(graph)), 64)
                self.assertEqual(graph, before)

    def test_nested_defaults_preserve_inheritance_explicit_override_and_binding(self):
        from studio_core.python_io import export_python, import_python
        from studio_core.schema import prepare_graph
        leaf = {'id': 'leaf', 'version': '1.0.0', 'output_mode': 'named',
                'public_parameters': [{'id': 'probability', 'type': 'float', 'default': 0.2}], 'bindings': {'probability': ['drop.p']},
                'graph': {'inputs': [{'id': 'x', 'shape': [None, 4], 'dtype': 'float32'}],
                          'instances': [{'id': 'drop', 'kind': 'Dropout', 'params': {'p': 0.2}}],
                          'calls': [{'id': 'mask', 'instance_id': 'drop', 'inputs': {'x': 'input:x'}}],
                          'outputs': [{'id': 'out', 'source': 'call:mask'}]}}
        outer = {'id': 'outer', 'version': '1.0.0', 'output_mode': 'named',
                 'public_parameters': [{'id': 'rate', 'type': 'float', 'default': 0.2}], 'bindings': {'rate': ['bound.probability']},
                 'graph': {'inputs': [{'id': 'x', 'shape': [None, 4], 'dtype': 'float32'}],
                           'instances': [{'id': name, 'kind': 'chip', 'definition_id': 'leaf', 'params': {'probability': 0.2} if name == 'explicit' else {}} for name in ('bound', 'implicit', 'explicit')],
                           'calls': [{'id': name + '_call', 'instance_id': name, 'inputs': {'x': 'input:x'}} for name in ('bound', 'implicit', 'explicit')],
                           'outputs': [{'id': name, 'source': 'call:' + name + '_call:out'} for name in ('bound', 'implicit', 'explicit')]}}
        graph = {'schema_version': 1, 'id': 'nested_defaults', 'revision': 1,
                 'inputs': [{'id': 'x', 'shape': [None, 4], 'dtype': 'float32'}],
                 'instances': [{'id': name, 'kind': 'chip', 'definition_id': 'outer', 'params': {'rate': 0.2} if name == 'explicit_parent' else {}} for name in ('default_parent', 'explicit_parent')],
                 'calls': [{'id': name + '_call', 'instance_id': name, 'inputs': {'x': 'input:x'}} for name in ('default_parent', 'explicit_parent')],
                 'outputs': [{'id': 'features', 'source': 'call:default_parent_call:bound'}], 'definitions': [leaf, outer]}
        source = export_python(graph)
        restored = import_python(source)
        self.assertEqual(model_digest(graph), model_digest(restored))
        self.assertEqual(checkpoint_signature(graph, self.valid_config(), 'a' * 64), checkpoint_signature(restored, self.valid_config(), 'a' * 64))
        def rates(candidate):
            resolved, _, _ = prepare_graph(candidate)
            return [[child['_graph']['instances'][0]['params']['p'] for child in parent['_graph']['instances']] for parent in resolved['instances']]
        self.assertEqual(rates(restored), [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]])
        changed = copy.deepcopy(restored); changed['definitions'][0]['public_parameters'][0]['default'] = 0.6
        self.assertEqual(rates(changed), [[0.2, 0.6, 0.2], [0.2, 0.6, 0.2]])
        changed = copy.deepcopy(restored); changed['definitions'][1]['public_parameters'][0]['default'] = 0.7
        self.assertEqual(rates(changed), [[0.7, 0.2, 0.2], [0.2, 0.2, 0.2]])
        self.assertIn("probability: 'float'=0.2", source)
        edited_leaf = import_python(source.replace("probability: 'float'=0.2", "probability: 'float'=0.6"))
        self.assertEqual(rates(edited_leaf), [[0.2, 0.6, 0.2], [0.2, 0.6, 0.2]])
        self.assertNotEqual(model_digest(graph), model_digest(edited_leaf))
        self.assertIn("rate: 'float'=0.2", source)
        edited_parent = import_python(source.replace("rate: 'float'=0.2", "rate: 'float'=0.7"))
        self.assertEqual(rates(edited_parent), [[0.7, 0.2, 0.2], [0.2, 0.2, 0.2]])

    def test_paired_defaults_validate_named_mappings_and_reserved_columns(self):
        config = default_classification_config('paired_csv')
        config['data'].update(path='D:/fixtures/paired.csv', feature_columns=['f1', 'f2'])
        checked = validate_classification_config(config)
        self.assertEqual(checked['input_ports'], {'image': 'image', 'tabular': 'clinical'})
        self.assertEqual(checked['data']['id_column'], 'sample_id')
        self.assertEqual(checked['data']['group_column'], 'patient_id')
        for mapping in ({'image': 'image'}, {'image': 'x', 'tabular': 'x'}, {'image': 'image', 'tabular': 'bad-id'}, {'image': 'image', 'tabular': 'clinical', 'extra': 'x'}, None):
            invalid = copy.deepcopy(config); invalid['input_ports'] = mapping
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                validate_classification_config(invalid)
        for field, value in [('group_column', None), ('id_column', ''), ('image_column', 'label'), ('image_root', 5)]:
            invalid = copy.deepcopy(config); invalid['data'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_classification_config(invalid)
        invalid = copy.deepcopy(config); invalid['input_port'] = 'explicit_ambiguous'
        with self.assertRaises(ValueError): validate_classification_config(invalid)
        invalid = copy.deepcopy(config); invalid['data']['feature_columns'] = ['sample_id']
        with self.assertRaises(ValueError): validate_classification_config(invalid)

    def test_paired_signature_binds_mapping_but_s2_none_mapping_is_compatible(self):
        legacy = self.valid_config(); graph = example_graph()
        before = checkpoint_signature(graph, legacy, 'a' * 64)
        self.assertEqual(before, '5b7729a5734169d588336b258655309098fb5e5d1d159319c8f864362a9c0eee')
        self.assertEqual(model_digest(graph), '4a6eaa25608c861572720647cbb21353767593ab87c3e287efec439970409854')
        legacy['input_ports'] = None
        self.assertEqual(before, checkpoint_signature(graph, validate_classification_config(legacy), 'a' * 64))
        invalid = copy.deepcopy(legacy); invalid['input_ports'] = {'image': 'x', 'tabular': 'clinical'}
        with self.assertRaises(ValueError): validate_classification_config(invalid)
        paired = default_classification_config('paired_csv')
        paired['data'].update(path='D:/fixtures/paired.csv', feature_columns=['f1'])
        before = checkpoint_signature(graph, paired, 'a' * 64)
        paired['input_ports']['tabular'] = 'metadata'
        self.assertNotEqual(before, checkpoint_signature(graph, paired, 'a' * 64))


if __name__=='__main__':unittest.main()
