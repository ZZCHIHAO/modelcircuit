"""Atomic semantic commands and bounded history, independent of the GUI."""
from copy import deepcopy
from functools import wraps
import json
import keyword
import math
import re

from studio_core.ports import output_port_ids, parse_source
from studio_core.tensor_specs import normalize_tensor_spec

IDENTIFIER=re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def require(condition,message):
    if not condition:raise ValueError(message)


def identifier(value):
    require(type(value) is str and bool(IDENTIFIER.fullmatch(value)),'Invalid interface identifier: '+str(value))
    return value


def encoded(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def shape_value(shape):
    require(type(shape) is list and 0<len(shape)<=8,'Shape must be a list with 1..8 axes')
    require(all(type(value) is int and value>0 or index==0 and value is None for index,value in enumerate(shape)),
            'Only the first shape axis may be null; other axes must be positive integers')
    return list(shape)


def _public_value(parameter,value):
    kind=parameter.get('type','auto')
    require(kind in ('auto','int','float','bool','shape'),'Unsupported public parameter type')
    if kind=='int':require(type(value) is int,'Integer public value required')
    if kind=='bool':require(type(value) is bool,'Boolean public value required')
    if kind=='float':
        try:valid=type(value) in (int,float) and math.isfinite(value)
        except OverflowError:valid=False
        require(valid,'Finite numeric public value required')
    if kind=='shape':
        dimensions=value if type(value) is list else [value]
        require(0<len(dimensions)<=8 and all(type(axis) is int and axis>0 for axis in dimensions),'Positive shape value required')


def _input_id(value):
    identifier(value)
    require(not keyword.iskeyword(value) and value!='self' and not value.startswith('call_'),
            'Input name conflicts with standalone Python arguments')


def command(function):
    @wraps(function)
    def invoke(self,*args,**kwargs):
        outer=self._edit_depth==0
        if outer:
            before=deepcopy(self.graph);trail=deepcopy(self.trail);selected=self.selected
        self._edit_depth+=1
        try:
            result=function(self,*args,**kwargs)
            if outer:
                # Storage validation also bounds work before history is retained.
                from .document import _structure
                _structure(self.graph)
        except Exception:
            if outer:self.graph,self.trail,self.selected=before,trail,selected
            raise
        finally:self._edit_depth-=1
        if outer and encoded(before)!=encoded(self.graph):
            self._undo.append((before,len(encoded(before).encode('utf-8'))))
            self._redo.clear();self._trim_history()
        return result
    return invoke


class EditableCommands:
    def _init_edits(self):
        self._undo=[];self._redo=[];self._edit_depth=0

    def _trim_history(self):
        while len(self._undo)+len(self._redo)>32 or sum(size for _,size in self._undo+self._redo)>32*1024*1024:
            (self._undo if self._undo else self._redo).pop(0)

    @property
    def can_undo(self):return bool(self._undo)

    @property
    def can_redo(self):return bool(self._redo)

    def _restore_history(self,source,target):
        if not source:return False
        old=deepcopy(self.graph);revision=self.graph['revision']
        graph,_=source.pop();target.append((old,len(encoded(old).encode('utf-8'))))
        self.graph=deepcopy(graph);self.graph['revision']=max(revision,self.graph['revision'])+1
        self.root();self._trim_history();return True

    def undo(self):return self._restore_history(self._undo,self._redo)
    def redo(self):return self._restore_history(self._redo,self._undo)

    @command
    def replace_graph(self,graph):
        from .document import _structure
        _structure(graph)
        revision=self.graph['revision'];self.graph=deepcopy(graph)
        self.graph['revision']=revision+1;self.root()

    def _scopes(self):return [self.graph]+[definition['graph'] for definition in self.graph.get('definitions',[])]

    def _definition_callers(self,definition_id):
        for scope in self._scopes():
            instances={item['id']:item for item in scope['instances']}
            for call in scope['calls']:
                instance=instances.get(call['instance_id'],{})
                if instance.get('kind')=='chip' and instance.get('definition_id')==definition_id:
                    yield scope,call,instance

    @staticmethod
    def _rewrite_sources(scope,replacements):
        for call in scope['calls']:
            call['inputs']={port:replacements.get(source,source) for port,source in call['inputs'].items()}
        for output in scope['outputs']:
            if 'source' in output:output['source']=replacements.get(output['source'],output['source'])

    def output_ports(self,call_id):
        call=self._call(call_id);instance=self._instance(call['instance_id'])
        return output_port_ids(instance,self.graph.get('definitions',[]))

    def source_options(self):
        result=['input:'+item['id'] for item in self.scope['inputs']]
        for call in self.scope['calls']:
            ports=self.output_ports(call['id'])
            if len(ports)==1:result.append('call:'+call['id'])
            result.extend('call:'+call['id']+':'+port for port in ports)
        return result

    @command
    def remove_many(self,refs):
        require(type(refs) in (list,tuple) and all(type(ref) is str for ref in refs),'Selected references must be a list')
        refs=list(dict.fromkeys(refs))
        for ref in refs:self._node(ref)
        # Selected calls are removed first so their consumed boundary inputs
        # can be removed in the same transaction. Any later failure rolls all
        # commands back and creates no history entry.
        for ref in sorted(refs,key=lambda ref:0 if ref.startswith('call:') else 1):
            kind,identifier_value=ref.split(':',1)
            if kind=='input':self.remove_input(identifier_value)
            elif kind=='output':self.remove_output(identifier_value)
            else:self.remove(ref)

    @command
    def add_input(self,identifier_value,shape,name=None,dtype='float32',*,axes=None,batch_axis=None):
        _input_id(identifier_value)
        if self.graph['schema_version']==2:
            from .chip_library import legacy_tensor_port
            defaults=legacy_tensor_port({'shape':shape_value(shape)}) if axes is None else {'axes':axes,'batch_axis':batch_axis}
            contract=self._tensor_contract({'shape':shape,'dtype':dtype,**defaults})
        else:
            require(dtype=='float32' and axes is None and batch_axis is None,'Only float32 tensor inputs are supported in schema 1')
            contract={'shape':shape_value(shape),'dtype':dtype}
        require(not any(item['id']==identifier_value for item in self.scope['inputs']),'Input already exists')
        self.scope['inputs'].append({'id':identifier_value,'name':name or identifier_value,**contract})
        self._changed();return 'input:'+identifier_value

    def _tensor_contract(self,spec):
        require(self.graph['schema_version']==2,'Tensor input contracts require schema 2')
        contract=normalize_tensor_spec(spec)
        if contract['batch_axis']==0 and contract['shape'][0] is not None:
            require(contract['shape'][0]<=self.graph['resource_batch_bound'],'Fixed batch exceeds resource_batch_bound')
        return contract

    @command
    def set_tensor_input(self,identifier_value,spec):
        contract=self._tensor_contract(spec)
        port=self._node('input:'+identifier_value)
        if any(port.get(key)!=value for key,value in contract.items()):
            port.update(contract);self._changed()

    @command
    def rename_input(self,old_id,new_id):
        _input_id(new_id)
        port=next((item for item in self.scope['inputs'] if item['id']==old_id),None)
        require(port is not None,'Unknown input: '+old_id)
        if old_id==new_id:return
        require(not any(item['id']==new_id for item in self.scope['inputs']),'Input already exists')
        port['id']=new_id
        self._rewrite_sources(self.scope,{'input:'+old_id:'input:'+new_id})
        if self.scope_id:
            for _,call,_ in self._definition_callers(self.scope_id):
                if old_id in call['inputs']:call['inputs'][new_id]=call['inputs'].pop(old_id)
        if self.selected=='input:'+old_id:self.selected='input:'+new_id
        self._changed()

    @command
    def set_input_shape(self,identifier_value,shape):
        shape=shape_value(shape)
        port=next((item for item in self.scope['inputs'] if item['id']==identifier_value),None)
        require(port is not None,'Unknown input: '+identifier_value)
        if self.graph['schema_version']==2:
            return self.set_tensor_input(identifier_value, {'shape':shape,'dtype':port['dtype'],
                'axes':port['axes'],'batch_axis':port['batch_axis']})
        if port.get('shape')!=shape:port['shape']=shape;self._changed()

    @command
    def remove_input(self,identifier_value):
        require(len(self.scope['inputs'])>1,'At least one input must remain')
        source='input:'+identifier_value
        require(not any(source in call['inputs'].values() for call in self.scope['calls']) and
                not any(output.get('source')==source for output in self.scope['outputs']),
                'Disconnect the consumed input before removing it')
        self.remove(source)
        if self.scope_id:
            for _,call,_ in self._definition_callers(self.scope_id):call['inputs'].pop(identifier_value,None)

    @command
    def add_output(self,identifier_value,source=None,name=None):
        identifier(identifier_value)
        require(not any(item['id']==identifier_value for item in self.scope['outputs']),'Output already exists')
        if source is not None:parse_source(source)
        if self.scope_id:
            if len(self.scope['outputs'])==1:
                previous=self.scope['outputs'][0]['id']
                for scope,call,_ in self._definition_callers(self.scope_id):
                    self._rewrite_sources(scope,{'call:'+call['id']:'call:'+call['id']+':'+previous})
            next(item for item in self.graph['definitions'] if item['id']==self.scope_id)['output_mode']='named'
        output={'id':identifier_value,'name':name or identifier_value}
        if source is not None:output['source']=source
        self.scope['outputs'].append(output);self._changed();return 'output:'+identifier_value

    @command
    def rename_output(self,old_id,new_id):
        identifier(new_id)
        port=next((item for item in self.scope['outputs'] if item['id']==old_id),None)
        require(port is not None,'Unknown output: '+old_id)
        if old_id==new_id:return
        require(not any(item['id']==new_id for item in self.scope['outputs']),'Output already exists')
        port['id']=new_id
        if self.scope_id:
            definition=next(item for item in self.graph['definitions'] if item['id']==self.scope_id)
            definition['output_mode']='named'
            for scope,call,_ in self._definition_callers(self.scope_id):
                self._rewrite_sources(scope,{'call:'+call['id']+':'+old_id:'call:'+call['id']+':'+new_id})
        if self.selected=='output:'+old_id:self.selected='output:'+new_id
        self._changed()

    @command
    def remove_output(self,identifier_value):
        require(len(self.scope['outputs'])>1,'At least one output must remain')
        if self.scope_id:
            for scope,call,_ in self._definition_callers(self.scope_id):
                source='call:'+call['id']+':'+identifier_value
                require(not any(source in other['inputs'].values() for other in scope['calls']) and
                        not any(output.get('source')==source for output in scope['outputs']),
                        'Disconnect the consumed output before removing it')
        self.remove('output:'+identifier_value)

    @command
    def set_public_parameters(self,definition_id,parameters,bindings):
        from studio_core.schema import _primitive,_resolve_scope,_Budget,resource_limits
        from studio_core.python_io import _PUBLIC_RESERVED
        definition=next((item for item in self.graph['definitions'] if item['id']==definition_id),None)
        require(definition is not None,'Unknown chip definition')
        require(type(parameters) is list and type(bindings) is dict,'Parameters/bindings must be a list/object')
        encoded(parameters);encoded(bindings)
        declarations={};targets=set()
        children={item['id']:deepcopy(item) for item in definition['graph']['instances']}
        for parameter in parameters:
            require(type(parameter) is dict and 'default' in parameter,'Public parameter requires a default')
            key=identifier(parameter.get('id'));require(key not in declarations,'Duplicate public parameter')
            require(not keyword.iskeyword(key) and key not in _PUBLIC_RESERVED|{'Chip_'+item['id'] for item in self.graph['definitions']},
                    'Public parameter name conflicts with standalone Python')
            _public_value(parameter,parameter['default'])
            declarations[key]=parameter
        require(bindings.keys()<=declarations.keys(),'Binding names must be declared public parameters')
        for key,paths in bindings.items():
            require(type(paths) is list and paths and all(type(path) is str for path in paths),'Each binding needs a nonempty path list')
            for path in paths:
                parts=path.split('.')
                require(len(parts)==2 and all(IDENTIFIER.fullmatch(part) for part in parts),'Binding must be instance.parameter')
                require(path not in targets,'A target cannot have multiple public bindings');targets.add(path)
                child_id,param_id=parts;require(child_id in children,'Unknown binding target: '+path)
                child=children[child_id]
                if child['kind']=='chip':
                    nested=next(item for item in self.graph['definitions'] if item['id']==child['definition_id'])
                    require(param_id in {item['id'] for item in nested.get('public_parameters',[])},'Unknown nested public parameter')
                child.setdefault('params',{})[param_id]=deepcopy(declarations[key]['default'])
        for child in children.values():
            if child['kind']=='chip':continue
            if self.graph['schema_version']==2:
                from studio_core.tensor_ops import normalize_tensor_operator
                normalize_tensor_operator(child['kind'],child.get('params',{}))
            else:_primitive(child)
        if encoded(definition.get('public_parameters',[]))==encoded(parameters) and encoded(definition.get('bindings',{}))==encoded(bindings):
            return
        removed={item['id'] for item in definition.get('public_parameters',[])}-declarations.keys()
        for parent in self.graph['definitions']:
            affected={item['id'] for item in parent['graph']['instances'] if item.get('definition_id')==definition_id}
            forbidden={child+'.'+key for child in affected for key in removed}
            require(not any(path in forbidden for paths in parent.get('bindings',{}).values() for path in paths),
                    'Disconnect parent public bindings before removing or renaming this parameter')
        definition['public_parameters']=deepcopy(parameters);definition['bindings']=deepcopy(bindings)
        for _,_,instance in self._definition_callers(definition_id):
            instance['params']={key:value for key,value in instance.get('params',{}).items() if key in declarations}
        definitions={item['id']:item for item in self.graph['definitions']}
        scope=definition['graph']
        fake={'inputs':deepcopy(scope['inputs']),'instances':[{'id':'binding_validation','kind':'chip','definition_id':definition_id,'params':{}}],
              'calls':[],'outputs':[{'id':'value','source':'input:'+scope['inputs'][0]['id']}]}
        from studio_core.schema import graph_resource_limits
        resolved=_resolve_scope(fake,definitions,budget=_Budget(graph_resource_limits(self.graph),
                                                                  tensor_contracts=self.graph['schema_version']==2))
        def validate_nested(current):
            for instance in current['instances']:
                if instance['kind']=='chip':
                    child=definitions[instance['definition_id']]
                    for parameter in child.get('public_parameters',[]):
                        _public_value(parameter,instance.get('params',{}).get(parameter['id'],parameter['default']))
                    validate_nested(instance['_graph'])
        validate_nested(resolved)
        self._changed()

    @command
    def import_chip(self,path):
        from .chip_library import import_bundle
        policy={key:self.graph[key] for key in ('resource_profile','resource_batch_bound') if key in self.graph}
        definitions,definition_id=import_bundle(path,self.graph.get('definitions',[]),resource_policy=policy,
                                                tensor_project=self.graph['schema_version']==2)
        if definitions!=self.graph.get('definitions',[]):self.graph['definitions']=definitions;self._changed()
        return definition_id

    @command
    def save_chip(self,definition_id,version,library_directory):
        from .chip_library import save_bundle
        path=save_bundle(self.graph,definition_id,version,library_directory)
        definition=next(item for item in self.graph['definitions'] if item['id']==definition_id)
        if definition.get('version')!=version:definition['version']=version;self._changed()
        return path
