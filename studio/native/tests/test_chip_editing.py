import copy
import json
from pathlib import Path
import tempfile
import unittest

from native.document import Document


class ChipEditingTests(unittest.TestCase):
    def test_display_names_do_not_invalidate_model_measurements(self):
        doc=Document();before=doc.semantic_key()
        doc.graph['name']='New display title'
        doc.graph['inputs'][0]['name']='Image branch title'
        doc.graph['definitions'][0]['public_parameters'][0]['name']='Changed label'
        self.assertEqual(doc.semantic_key(),before)

    def test_semantic_undo_redo_is_atomic_bounded_and_has_monotonic_revision(self):
        doc=Document();initial=copy.deepcopy(doc.graph)
        doc.set_param('encoder','hidden_features',9)
        changed=copy.deepcopy(doc.graph)
        self.assertTrue(doc.can_undo)
        self.assertTrue(doc.undo())
        self.assertEqual(doc.graph['instances'],initial['instances'])
        self.assertGreater(doc.graph['revision'],changed['revision'])
        self.assertTrue(doc.can_redo)
        revision=doc.graph['revision'];self.assertTrue(doc.redo())
        self.assertEqual(doc.graph['instances'],changed['instances'])
        self.assertGreater(doc.graph['revision'],revision)
        doc.undo();doc.set_frozen('encoder',True)
        self.assertFalse(doc.can_redo)
        for index in range(80):doc.set_param('encoder','hidden_features',index+2)
        count=0
        while doc.undo():count+=1
        self.assertLessEqual(count,32)

    def test_interface_renames_update_all_calls_and_sources(self):
        doc=Document();doc.enter('encode_a')
        doc.rename_input('x','features')
        self.assertEqual(doc.scope['calls'][0]['inputs']['x'],'input:features')
        self.assertTrue(all('features' in call['inputs'] for call in doc.graph['calls'] if call['instance_id']=='encoder'))
        doc.set_input_shape('features',[None,4])
        doc.add_input('unused',[None,2]);doc.remove_input('unused')
        before=copy.deepcopy(doc.graph)
        with self.assertRaises(ValueError):doc.remove_input('features')
        self.assertEqual(doc.graph,before)
        self.assertTrue(doc.validation()['valid'],doc.validation())

    def test_named_output_expansion_migrates_aliases_rename_and_delete_safely(self):
        doc=Document();doc.enter('encode_a')
        doc.add_output('projection','call:project')
        self.assertEqual(doc.graph['calls'][2]['inputs'],{'a':'call:encode_a:out','b':'call:encode_b:out'})
        doc.rename_output('out','activated')
        self.assertEqual(doc.graph['calls'][2]['inputs']['a'],'call:encode_a:activated')
        self.assertTrue(doc.validation()['valid'],doc.validation())
        self.assertEqual(doc.output_ports('activate'),['out'])
        doc.root()
        self.assertIn('call:encode_a:projection',doc.source_options())
        self.assertNotIn('call:encode_a',doc.source_options())
        doc.set_output('features','call:encode_a:projection')
        self.assertEqual(doc.probe_source('output:features'),'call:encode_a:projection')
        doc.enter('encode_a');before=copy.deepcopy(doc.graph)
        with self.assertRaises(ValueError):doc.remove_output('projection')
        self.assertEqual(doc.graph,before)
        doc.root();doc.set_output('features',None);doc.enter('encode_a');doc.remove_output('projection')
        self.assertTrue(doc.validation()['valid'] is False) # disconnected root output remains an editable draft

    def test_public_defaults_and_bindings_are_transactional(self):
        doc=Document()
        parameters=[{'id':'width','name':'Width','type':'int','default':8}]
        doc.set_public_parameters('encoder_chip',parameters,{'width':['dense.out_features']})
        definition=doc.graph['definitions'][0]
        self.assertEqual(definition['public_parameters'],parameters)
        self.assertEqual(definition['bindings'],{'width':['dense.out_features']})
        self.assertEqual(doc.graph['instances'][0]['params'],{})
        self.assertTrue(doc.validation()['valid'],doc.validation())
        before=copy.deepcopy(doc.graph)
        for binding,value in [({'width':['missing.out_features']},8),({'width':['dense.out_features']},True)]:
            invalid=copy.deepcopy(parameters);invalid[0]['default']=value
            with self.assertRaises(ValueError):doc.set_public_parameters('encoder_chip',invalid,binding)
            self.assertEqual(doc.graph,before)

    def test_reapplying_public_interface_is_a_noop_and_unexportable_names_are_rejected(self):
        doc=Document();definition=doc.graph['definitions'][0]
        revision=doc.graph['revision']
        doc.set_public_parameters(definition['id'],definition['public_parameters'],definition['bindings'])
        self.assertEqual(doc.graph['revision'],revision)
        self.assertFalse(doc.can_undo)
        for name in ('self','class','call_conflict'):
            before=copy.deepcopy(doc.graph)
            with self.assertRaises(ValueError):doc.add_input(name,[None,4])
            self.assertEqual(doc.graph,before)
        with self.assertRaises(ValueError):
            doc.set_public_parameters('encoder_chip',[{'id':'self','default':8}],{'self':['dense.out_features']})

    def test_malformed_bundle_internals_are_value_errors_and_leave_history_unchanged(self):
        from native.chip_library import validate_bundle
        payload={'chip_format_version':1,'definition_id':'bad','definitions':[{'id':'bad','version':'1.0.0',
                 'graph':{'inputs':[],'instances':['not-an-instance'],'calls':[],'outputs':[]}}]}
        with self.assertRaises(ValueError):validate_bundle(payload)

    def test_nested_public_type_and_parent_binding_deletion_are_atomic(self):
        doc=Document();doc.encapsulate(['encode_a'],'Outer','outer_chip','encode_a')
        doc.set_public_parameters('encoder_chip',[{'id':'width','type':'int','default':8}],{'width':['dense.out_features']})
        before=copy.deepcopy(doc.graph)
        with self.assertRaises(ValueError):
            doc.set_public_parameters('outer_chip',[{'id':'width','type':'float','default':8.5}],{'width':['encoder.width']})
        self.assertEqual(doc.graph,before)
        doc.set_public_parameters('outer_chip',[{'id':'width','type':'int','default':8}],{'width':['encoder.width']})
        before=copy.deepcopy(doc.graph)
        with self.assertRaises(ValueError):doc.set_public_parameters('encoder_chip',[],{})
        self.assertEqual(doc.graph,before)

    def test_library_import_rejects_merged_definition_limit_before_commit(self):
        from native.chip_library import validate_bundle
        leaves=[{'id':'leaf_'+str(index),'version':'1.0.0','output_mode':'named','public_parameters':[],'bindings':{},
                 'graph':{'inputs':[{'id':'x','shape':[None,2],'dtype':'float32'}],'instances':[],'calls':[],
                          'outputs':[{'id':'out','source':'input:x'}]}} for index in range(31)]
        root={'id':'many','version':'1.0.0','output_mode':'named','public_parameters':[],'bindings':{},
              'graph':{'inputs':[{'id':'x','shape':[None,2],'dtype':'float32'}],
                       'instances':[{'id':'instance_'+str(i),'kind':'chip','definition_id':item['id'],'params':{}} for i,item in enumerate(leaves)],
                       'calls':[{'id':'call_'+str(i),'instance_id':'instance_'+str(i),'inputs':{'x':'input:x'}} for i in range(31)],
                       'outputs':[{'id':'out','source':'call:call_30:out'}]}}
        payload={'chip_format_version':1,'definition_id':'many','definitions':leaves+[root]}
        validate_bundle(payload)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'many.json';path.write_text(json.dumps(payload),encoding='utf-8')
            doc=Document();before=copy.deepcopy(doc.graph)
            with self.assertRaises(ValueError):doc.import_chip(path)
            self.assertEqual(doc.graph,before);self.assertFalse(doc.can_undo)

    def test_multi_delete_is_one_atomic_semantic_command(self):
        doc=Document();doc.add_input('unused',[None,2]);before=copy.deepcopy(doc.graph)
        with self.assertRaises(ValueError):doc.remove_many(['input:unused','input:a'])
        self.assertEqual(doc.graph,before)
        extra=doc.add('ReLU');before=copy.deepcopy(doc.graph)
        doc.remove_many(['input:unused',extra])
        self.assertNotIn('unused',[port['id'] for port in doc.graph['inputs']])
        doc.undo()
        self.assertEqual(doc.graph['inputs'],before['inputs'])
        self.assertEqual(doc.graph['calls'],before['calls'])

    def test_encapsulation_can_have_named_outputs_in_nested_scope(self):
        doc=Document();doc.enter('encode_a')
        ref=doc.encapsulate(['project','activate'],'Two signals','signals',outputs={'raw':'call:project','activated':'call:activate'})
        self.assertEqual(doc.scope_id,'encoder_chip')
        self.assertEqual(doc.output_ports(ref),['raw','activated'])
        self.assertTrue(doc.validation()['valid'],doc.validation())
        self.assertEqual(doc.graph['definitions'][-1]['output_mode'],'named')
        self.assertTrue(doc.undo())
        self.assertNotIn('signals',[item['id'] for item in doc.graph['definitions']])

    def test_versioned_library_closure_collision_and_failed_import(self):
        doc=Document();doc.encapsulate(['encode_a','encode_b','fuse'],'Fusion','fusion_chip','fuse')
        with tempfile.TemporaryDirectory() as temporary:
            path=doc.save_chip('fusion_chip','1.0.0',temporary)
            payload=json.loads(Path(path).read_text(encoding='utf-8'))
            self.assertEqual(payload['chip_format_version'],1)
            self.assertEqual({d['id'] for d in payload['definitions']},{'encoder_chip','fusion_chip'})
            self.assertEqual(doc.save_chip('fusion_chip','1.0.0',temporary),path)
            target=Document();target.set_param('encoder','hidden_features',7)
            # Change definition contents, not just an instance override, to force a true ID collision.
            target.graph['definitions'][0]['name']='A distinct local definition'
            imported=target.import_chip(path)
            self.assertEqual(imported,'fusion_chip')
            imported_definition=next(d for d in target.graph['definitions'] if d['id']==imported)
            nested=next(i for i in imported_definition['graph']['instances'] if i['kind']=='chip')
            self.assertNotEqual(nested['definition_id'],'encoder_chip')
            before=copy.deepcopy(target.graph)
            bad=Path(temporary)/'bad.json';bad.write_text('{"chip_format_version":999}')
            with self.assertRaises(ValueError):target.import_chip(bad)
            self.assertEqual(target.graph,before)
            doc.enter('encode_a');doc.set_param('dense','out_features',9)
            with self.assertRaises(ValueError):doc.save_chip('fusion_chip','1.0.0',temporary)


if __name__=='__main__':unittest.main()
