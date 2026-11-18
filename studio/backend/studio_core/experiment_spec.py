"""Finite, serializable task settings and reproducibility identities; no torch."""
import copy
import hashlib
import json
import math
import re


def default_classification_config(data_kind='csv'):
    if data_kind == 'csv':
        data = {'kind':'csv','path':'','target_column':'label','feature_columns':[],
                'group_column':None,'split_column':None,'standardize':True,
                'validation_fraction':0.2,'test_fraction':0.2}
    elif data_kind == 'image_folder':
        data = {'kind':'image_folder','path':'','split_mode':'stratified','image_size':[32,32],
                'channels':3,'validation_fraction':0.2,'test_fraction':0.2}
    elif data_kind == 'paired_csv':
        data = {'kind':'paired_csv','path':'','image_root':'','image_column':'image_path','id_column':'sample_id',
                'target_column':'label','feature_columns':[],'group_column':'patient_id','split_column':None,
                'standardize':True,'image_size':[32,32],'channels':3,'validation_fraction':0.2,'test_fraction':0.2}
    else:
        raise ValueError('Unsupported classification data kind: '+str(data_kind))
    return {'profile':'classification-v2','data':data,'input_port':'x',
            'input_ports':{'image':'image','tabular':'clinical'} if data_kind=='paired_csv' else None,'epochs':10,'batch_size':16,
            'lr':0.001,'optimizer':'adamw','seed':42,'device':'cpu','max_runtime_seconds':3600,
            'resume_checkpoint':None,'expected_dataset_digest':None}


def _require(condition, message):
    if not condition: raise ValueError(message)


def _number(value):
    if type(value) not in (int,float): return False
    try: return math.isfinite(value)
    except OverflowError: return False


def _hash(value):
    try: raw=json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    except (ValueError,TypeError) as error: raise ValueError('Experiment must contain finite JSON values') from error
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def validate_classification_config(config):
    _require(isinstance(config,dict),'Experiment config must be an object')
    _require(config.get('profile')=='classification-v2','Expected classification-v2 profile')
    data=config.get('data')
    _require(isinstance(data,dict),'data must be an object')
    defaults=default_classification_config(data.get('kind'))
    _require(config.keys()<=defaults.keys(),'Unsupported experiment option (S2 supports a single CPU/CUDA device)')
    result=copy.deepcopy(defaults)
    result.update(copy.deepcopy(config))
    _require(data.keys()<=defaults['data'].keys(),'Unsupported data option')
    effective_data=copy.deepcopy(defaults['data']);effective_data.update(copy.deepcopy(data));result['data']=effective_data
    _hash(result)
    for field,low,high in [('epochs',1,1000),('batch_size',1,64),('seed',0,2**31-1),('max_runtime_seconds',10,604800)]:
        _require(type(result[field]) is int and low<=result[field]<=high,field+' is outside the supported integer range')
    _require(_number(result['lr']) and 1e-6<=result['lr']<=1,'lr must be between 0.000001 and 1')
    _require(result['optimizer'] in ('sgd','adamw'),'optimizer must be sgd or adamw')
    _require(isinstance(result['device'],str) and re.fullmatch(r'cpu|cuda:(0|[1-9][0-9]*)',result['device']),
             'device must be cpu or one explicit cuda:N device; multi-GPU is not implemented')
    _require(isinstance(result['input_port'],str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',result['input_port']), 'Invalid graph input port')
    if data.get('kind')=='paired_csv':
        _require(result['input_port']=='x','paired_csv uses input_ports; legacy input_port must remain x')
        mapping=result['input_ports']
        _require(isinstance(mapping,dict) and set(mapping)=={'image','tabular'},'input_ports must map image and tabular exactly once')
        _require(all(isinstance(port,str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',port) for port in mapping.values()),'Invalid named input_ports')
        _require(len(set(mapping.values()))==2,'Paired graph input ports must be distinct')
    else:
        _require(result['input_ports'] is None,'input_ports is only supported for paired_csv')
    for key in ('resume_checkpoint','expected_dataset_digest'):
        _require(result[key] is None or isinstance(result[key],str) and bool(result[key].strip()), key+' must be null or a nonempty string')
    if result['expected_dataset_digest'] is not None:
        _require(re.fullmatch(r'[0-9a-f]{64}',result['expected_dataset_digest']) is not None,'Invalid expected_dataset_digest')
    _require(isinstance(effective_data['path'],str) and bool(effective_data['path'].strip()),'Select a dataset path')
    _require(len(effective_data['path'])<=4096,'Dataset path is too long')
    for key in ('validation_fraction','test_fraction'):
        _require(_number(effective_data[key]) and 0<effective_data[key]<1,key+' must be between 0 and 1')
    _require(effective_data['validation_fraction']+effective_data['test_fraction']<1,'Train fraction must be positive')
    if effective_data['kind'] in ('csv','paired_csv'):
        target=effective_data['target_column'];features=effective_data['feature_columns']
        _require(isinstance(target,str) and bool(target.strip()),'Select a CSV target column')
        _require(isinstance(features,list) and 0<len(features)<=512 and all(isinstance(x,str) and bool(x.strip()) for x in features),
                 'Select 1..512 explicit numeric feature columns')
        _require(len(set(features))==len(features),'Feature columns must be unique')
        for key in ('group_column','split_column'):
            _require(effective_data[key] is None or isinstance(effective_data[key],str) and bool(effective_data[key].strip()),key+' must be null or a column name')
        metadata=[target,effective_data['group_column'],effective_data['split_column']]
        if effective_data['kind']=='paired_csv':
            for key in ('image_column','id_column','group_column'):
                _require(isinstance(effective_data[key],str) and bool(effective_data[key].strip()),'paired_csv requires '+key)
            _require(isinstance(effective_data['image_root'],str) and len(effective_data['image_root'])<=4096,'image_root must be a path string or empty')
            metadata += [effective_data['image_column'],effective_data['id_column']]
        reserved={x for x in metadata if x is not None}
        _require(not(set(features)&reserved),'Target, group and split columns cannot be input features')
        _require(len(reserved)==len([x for x in metadata if x is not None]),
                 'Target/group/split must use different columns')
        _require(type(effective_data['standardize']) is bool,'standardize must be a boolean')
    if effective_data['kind'] in ('image_folder','paired_csv'):
        size=effective_data['image_size']
        _require(isinstance(size,list) and len(size)==2 and all(type(x) is int and 8<=x<=512 for x in size),'image_size must be [height,width], each 8..512')
        _require(type(effective_data['channels']) is int and effective_data['channels'] in (1,3),'Image channels must be 1 or 3')
        if effective_data['kind']=='image_folder':
            _require(effective_data['split_mode'] in ('stratified','directories'),'Unknown image split_mode')
    return result


def _model_semantics(value, context='graph'):
    if context=='literal':
        return copy.deepcopy(value)
    if isinstance(value,dict):
        result={}
        for key,child in value.items():
            if key in {'position','revision','name'}: continue
            if context=='node' and key in {'params','bindings','default'}:
                child_context='literal'
            elif key=='graph': child_context='graph'
            elif key in {'inputs','outputs','instances','calls','definitions','public_parameters'} and context in {'graph','node'}:
                child_context='node'
            else: child_context='metadata'
            result[key]=_model_semantics(child,child_context)
        return result
    if isinstance(value,list): return [_model_semantics(child,context) for child in value]
    return value


def _object_list(value, key):
    items=value.get(key,[]) if isinstance(value,dict) else []
    return [item for item in items if isinstance(item,dict)] if isinstance(items,list) else []


def _uses_s3_ports(graph):
    """Detect new semantics that survive Python roundtrip, tolerating drafts.

    Qualified primitive/single-output references alone do not activate a new
    identity representation: Python naturally imports these as legacy aliases.
    """
    definitions=_object_list(graph,'definitions')
    for definition in definitions:
        ports=[item.get('id') for item in _object_list(definition.get('graph'),'outputs')]
        if definition.get('output_mode')=='named' or (ports and ports!=['out']):return True
        if any('type' in item for item in _object_list(definition,'public_parameters')):return True
    return False


def _canonical_s3_graph(graph):
    """Remove only representation differences that cannot affect execution.

    Bound raw target values are overridden on every invocation. Unbound chip
    overrides remain literal, even when currently equal to a public default:
    that distinction determines behavior after the declaration is edited.
    """
    from .python_io import _ordered_calls
    result=copy.deepcopy(graph)
    definitions=_object_list(result,'definitions')
    registry={item['id']:item for item in definitions if isinstance(item.get('id'),str)}
    for definition in definitions:
        definition.setdefault('version','0.1.0')
        scope=definition.get('graph')
        ports=[item.get('id') for item in _object_list(scope,'outputs')]
        mode=definition.get('output_mode')
        if ports and all(isinstance(port,str) for port in ports) and ('output_mode' not in definition or mode in ('tensor','named')) and not(mode=='tensor' and len(ports)!=1):
            if mode=='named' or ports!=['out']:definition['output_mode']='named'
            else:definition.pop('output_mode',None)
        children={item['id']:item for item in _object_list(scope,'instances') if isinstance(item.get('id'),str)}
        public={item.get('id') for item in _object_list(definition,'public_parameters') if isinstance(item.get('id'),str)}
        bindings=definition.get('bindings',{})
        if not isinstance(bindings,dict):continue
        for parameter, paths in bindings.items():
            if parameter not in public or not isinstance(paths,list):continue
            for path in paths:
                if not isinstance(path,str) or len(path.split('.'))!=2:continue
                child_id,field=path.split('.')
                child=children.get(child_id)
                if child is None:continue
                params=child.get('params',{})
                if not isinstance(params,dict):continue
                params.pop(field,None)
                child['params']=params
    for scope in [result]+[definition.get('graph') for definition in definitions]:
        for instance in _object_list(scope,'instances'):
            if instance.get('frozen') is False:instance.pop('frozen')
        if isinstance(scope,dict):
            try:scope['calls']=_ordered_calls(scope,registry)
            except (ValueError,KeyError,TypeError,AttributeError):pass
    return result


def _canonical_source_aliases(graph):
    """Use the historical bare alias for every legal single-output call."""
    from .ports import output_port_ids
    from .schema import PRIMITIVES
    result=copy.deepcopy(graph)
    definitions=_object_list(result,'definitions')
    registry={item['id']:item for item in definitions if isinstance(item.get('id'),str)}
    for definition in definitions:
        ports=[item.get('id') for item in _object_list(definition.get('graph'),'outputs')]
        if definition.get('output_mode')=='tensor' and ports==['out']:
            definition.pop('output_mode')
    for scope in [result]+[definition.get('graph') for definition in definitions]:
        instances={item['id']:item for item in _object_list(scope,'instances') if isinstance(item.get('id'),str)}
        aliases={}
        for call in _object_list(scope,'calls'):
            identifier,instance_id=call.get('id'),call.get('instance_id')
            if not isinstance(identifier,str) or not isinstance(instance_id,str):continue
            instance=instances.get(instance_id)
            if instance is None or instance.get('kind') not in PRIMITIVES|{'chip'}:continue
            try:
                ports=output_port_ids({key:value for key,value in instance.items() if key!='_graph'},registry)
            except (ValueError,KeyError,TypeError):continue
            if len(ports)==1:aliases['call:'+identifier+':'+ports[0]]='call:'+identifier
        for call in _object_list(scope,'calls'):
            inputs=call.get('inputs')
            if isinstance(inputs,dict):
                for key,value in list(inputs.items()):
                    if isinstance(value,str):inputs[key]=aliases.get(value,value)
        for output in _object_list(scope,'outputs'):
            source=output.get('source')
            if isinstance(source,str):output['source']=aliases.get(source,source)
    return result


def model_digest(graph):
    aliases=_canonical_source_aliases(graph)
    from .vision_python import VISION_KINDS
    scopes=[aliases]+[definition.get('graph') for definition in _object_list(aliases,'definitions')]
    uses_v4=graph.get('resource_profile') in ('vision-v4', 'hardware-v5') or any(instance.get('kind') in VISION_KINDS for scope in scopes for instance in _object_list(scope,'instances'))
    if uses_v4:
        # v4 source spells legacy-none explicitly and imports it as absence.
        # Keep that representation neutral only on the new identity path;
        # historical v1-v3 digests retain their original byte representation.
        if aliases.get('resource_profile') is None:
            aliases.pop('resource_profile', None)
        from .operator_rules import normalize_operator
        from .schema import PRIMITIVES
        for scope in scopes:
            for instance in _object_list(scope,'instances'):
                if instance.get('kind') in PRIMITIVES:
                    try: instance['params']=normalize_operator(instance['kind'],instance.get('params',{}))
                    except (ValueError,TypeError,KeyError,OverflowError): pass
    semantic_graph=_canonical_s3_graph(aliases) if uses_v4 or _uses_s3_ports(graph) else aliases
    return _hash(_model_semantics(semantic_graph))


def experiment_signature(graph,config,dataset_digest=None):
    return _hash({'model':model_digest(graph),'config':config,'dataset_digest':dataset_digest})


def checkpoint_signature(graph,config,dataset_digest):
    binding_key='input_ports' if config.get('data',{}).get('kind')=='paired_csv' else 'input_port'
    settings={key:config[key] for key in (binding_key,'optimizer','lr','batch_size','seed')}
    return _hash({'model':model_digest(graph),'training':settings,'dataset_digest':dataset_digest})
