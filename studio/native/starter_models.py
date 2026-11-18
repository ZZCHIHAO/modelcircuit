"""Small editable model examples, without GUI or training dependencies."""


def paired_starter_graph(manifest):
    shapes=manifest.get('input_shapes',{});classes=manifest.get('class_names',[])
    image=shapes.get('image');tabular=shapes.get('tabular')
    if not(isinstance(image,list) and len(image)==4 and isinstance(tabular,list) and len(tabular)==2 and len(classes)>=2):
        raise ValueError('Inspect paired image/tabular data before creating a starter model')
    image_graph={'inputs':[{'id':'x','shape':image[:],'dtype':'float32','position':{'x':-40,'y':50}}],
                 'instances':[{'id':'conv','kind':'Conv2d','params':{'in_channels':image[1],'out_channels':8,'kernel_size':3,'stride':2,'padding':1}},
                              {'id':'activation','kind':'ReLU','params':{}},
                              {'id':'pool','kind':'AdaptiveAvgPool2d','params':{'output_size':[1,1]}},
                              {'id':'flatten','kind':'Flatten','params':{}}],
                 'calls':[{'id':'conv','instance_id':'conv','inputs':{'x':'input:x'},'position':{'x':220,'y':50}},
                          {'id':'activation','instance_id':'activation','inputs':{'x':'call:conv'},'position':{'x':520,'y':50}},
                          {'id':'pool','instance_id':'pool','inputs':{'x':'call:activation'},'position':{'x':220,'y':250}},
                          {'id':'flatten','instance_id':'flatten','inputs':{'x':'call:pool'},'position':{'x':520,'y':250}}],
                 'outputs':[{'id':'features','source':'call:flatten','position':{'x':820,'y':250}}]}
    table_graph={'inputs':[{'id':'x','shape':tabular[:],'dtype':'float32','position':{'x':-40,'y':100}}],
                 'instances':[{'id':'dense','kind':'Linear','params':{'in_features':tabular[1],'out_features':8}},
                              {'id':'activation','kind':'ReLU','params':{}}],
                 'calls':[{'id':'dense','instance_id':'dense','inputs':{'x':'input:x'},'position':{'x':250,'y':100}},
                          {'id':'activation','instance_id':'activation','inputs':{'x':'call:dense'},'position':{'x':550,'y':100}}],
                 'outputs':[{'id':'features','source':'call:activation','position':{'x':850,'y':100}}]}
    return {'schema_version':1,'resource_profile':'classification-v2','id':'paired_classifier','name':'影像 + 表格融合分类器','revision':1,
            'inputs':[{'id':'image','name':'二维影像','shape':image[:],'dtype':'float32','position':{'x':-40,'y':105}},
                      {'id':'clinical','name':'数值表格','shape':tabular[:],'dtype':'float32','position':{'x':-40,'y':270}}],
            'instances':[{'id':'image_encoder','kind':'chip','definition_id':'paired_image_encoder','name':'影像编码芯片','params':{}},
                         {'id':'table_encoder','kind':'chip','definition_id':'paired_table_encoder','name':'表格编码芯片','params':{}},
                         {'id':'fusion','kind':'Concat','params':{'dim':1}},
                         {'id':'head','kind':'Linear','params':{'in_features':16,'out_features':len(classes)}}],
            'calls':[{'id':'encode_image','instance_id':'image_encoder','inputs':{'x':'input:image'},'position':{'x':240,'y':50}},
                     {'id':'encode_table','instance_id':'table_encoder','inputs':{'x':'input:clinical'},'position':{'x':240,'y':270}},
                     {'id':'fuse','instance_id':'fusion','inputs':{'a':'call:encode_image:features','b':'call:encode_table:features'},'position':{'x':540,'y':160}},
                     {'id':'classify','instance_id':'head','inputs':{'x':'call:fuse'},'position':{'x':830,'y':160}}],
            'outputs':[{'id':'logits','source':'call:classify','position':{'x':1065,'y':90}},
                       {'id':'features','source':'call:fuse','position':{'x':1065,'y':300}}],
            'definitions':[{'id':'paired_image_encoder','name':'影像编码芯片','version':'1.0.0','output_mode':'named',
                            'public_parameters':[{'id':'width','name':'输出特征维数','type':'int','default':8}],
                            'bindings':{'width':['conv.out_channels']},'graph':image_graph},
                           {'id':'paired_table_encoder','name':'表格编码芯片','version':'1.0.0','output_mode':'named',
                            'public_parameters':[{'id':'width','name':'输出特征维数','type':'int','default':8}],
                            'bindings':{'width':['dense.out_features']},'graph':table_graph}]}
