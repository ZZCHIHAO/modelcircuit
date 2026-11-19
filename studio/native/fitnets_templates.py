"""Teacher, student and projector starter graphs for native FitNets setup."""
from copy import deepcopy


def _port(name, shape):
    return {'id': name, 'shape': list(shape), 'dtype': 'float32'}


def _scope(shape, layers, outputs):
    instances, calls = [], []
    previous = 'input:x'
    for index, (name, kind, params) in enumerate(layers):
        instances.append({'id': name, 'kind': kind, 'params': params})
        calls.append({'id': name, 'instance_id': name, 'inputs': {'x': previous},
                      'position': {'x': 220 + 190 * (index % 3), 'y': 50 + 210 * (index // 3)}})
        previous = 'call:' + name
    return {'inputs': [_port('x', shape)], 'instances': instances, 'calls': calls,
            'outputs': [{'id': name, 'source': 'call:' + source} for name, source in outputs.items()]}


def starter(manifest, *, teacher=False):
    shape = manifest['input_shape']
    classes = len(manifest['class_names'])
    spatial_rank = len(shape)-2
    if spatial_rank not in (0, 2, 3) or classes < 2:
        raise ValueError('FitNets starter expects tabular, image, volume or video classification data')

    def layer(name, incoming, outgoing, *, pointwise=False):
        if not spatial_rank:
            return name, 'Linear', {'in_features': incoming, 'out_features': outgoing}
        return name, f'Conv{spatial_rank}d', {'in_channels': incoming, 'out_channels': outgoing,
                                             'kernel_size': 1 if pointwise else 3, 'padding': 0 if pointwise else 1}

    def head(width):
        layers = []
        if spatial_rank:
            layers.extend([('pool', f'AdaptiveAvgPool{spatial_rank}d', {'output_size':[1]*spatial_rank}),
                           ('flatten', 'Flatten', {})])
        return [*layers, ('head','Linear', {'in_features':width, 'out_features':classes})]

    # Same instance keys in the standalone teacher and in the teacher chip.
    t = _scope(shape, [layer('encoder', shape[1], 8), ('hint','ReLU',{}), *head(8)],
               {'logits':'head', 'hint':'hint'})
    graph = {'schema_version':1, 'resource_profile':'classification-v2', 'revision':1, 'definitions':[]}
    if teacher:
        standalone = deepcopy(t)
        standalone['outputs'] = [standalone['outputs'][0]]
        return {**graph, **standalone, 'id':'fitnets_teacher', 'name':'FitNets 教师预训练'}
    s = _scope(shape, [layer('encoder',shape[1],3), ('relu','ReLU',{}), layer('prefix',3,4), ('guided','ReLU',{}), *head(4)],
               {'logits':'head','guided':'guided'})
    p = _scope([None,4,*shape[2:]], [layer('projection',4,8,pointwise=True)], {'out':'projection'})
    graph.update(id='fitnets_starter', name='FitNets 教师与学生', inputs=[_port('x',shape)],
                 instances=[], calls=[], outputs=[])
    for role, scope, source, position in (
            ('teacher',t,'input:x',(240,55)), ('student',s,'input:x',(240,290)),
            ('projector',p,'call:student:guided',(510,390))):
        graph['definitions'].append({'id':role+'_chip','graph':scope})
        graph['instances'].append({'id':role,'kind':'chip','definition_id':role+'_chip','params':{},'frozen':role=='teacher'})
        graph['calls'].append({'id':role,'instance_id':role,'inputs':{'x':source},'position':dict(zip(('x','y'),position))})
    for name, source, position in (
            ('teacher_logits','teacher:logits',(520,30)), ('teacher_hint','teacher:hint',(520,145)),
            ('logits','student:logits',(520,255)), ('student_guided','student:guided',(775,285)),
            ('student_projection','projector',(775,420))):
        graph['outputs'].append({'id':name, 'source':'call:'+source, 'position':dict(zip(('x','y'),position))})
    return graph
