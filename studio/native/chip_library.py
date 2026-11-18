"""Finite, versioned chip bundles with explicit dependency closure."""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import tempfile

from studio_core.schema import validate_graph,MAX_DEFINITIONS,_preflight
from studio_core.component_types import tensor_requirements, validate_tensor_requirements

VERSION=re.compile(r'^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-[A-Za-z0-9.-]+)?$')
IDENTIFIER=re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
MAX_BYTES=4*1024*1024
VISION_OPERATORS={'BatchNorm2d','Identity','Sigmoid','Multiply','Scale','OneMinus'}


def _policy(value):
    """Validate a finite explicit resource policy without changing a project."""
    _require(type(value) is dict,'Chip import requires an explicit resource policy')
    profile=value.get('resource_profile')
    _require(profile in (None,'classification-v2','vision-v4','hardware-v5'),'Unsupported chip resource policy')
    expected={'resource_profile','resource_batch_bound'} if profile in ('vision-v4','hardware-v5') else {'resource_profile'}
    _require(set(value)<=expected,'Unsupported chip resource policy fields')
    if profile in ('vision-v4','hardware-v5'):
        bound=value.get('resource_batch_bound')
        _require(type(bound) is int and 1<=bound<=64,'Chip policy resource_batch_bound must be an integer in 1..64')
        return {'resource_profile':profile,'resource_batch_bound':bound}
    _require('resource_batch_bound' not in value,'resource_batch_bound requires vision-v4')
    return {'resource_profile':profile}


def policy_satisfies(current, required):
    """Hardware execution admits legacy operators without downgrading its policy."""
    current, required = _policy(current), _policy(required)
    return current == required or (current['resource_profile'] == 'hardware-v5' and (
        required['resource_profile'] in (None, 'classification-v2') or (
        required['resource_profile'] == 'vision-v4'
        and current['resource_batch_bound'] == required['resource_batch_bound'])))


def _requirements(payload):
    version=payload.get('chip_format_version')
    _require(type(version) is int and version in (1,2,3),'Unsupported chip_format_version')
    expected={'chip_format_version','definition_id','definitions'}|({'requirements'} if version!=1 else set())
    _require(set(payload)==expected,'Invalid chip bundle fields')
    if version==1:return None
    requirements=payload['requirements']
    if version==3:return validate_tensor_requirements(requirements)
    _require(type(requirements) is dict and requirements.get('operator_set')=='vision-ops-v4',
             'Unsupported chip operator_set')
    profile=requirements.get('resource_profile')
    fields={'operator_set','resource_profile'}|({'resource_batch_bound'} if profile in ('vision-v4','hardware-v5') else set())
    _require(set(requirements)==fields,'Invalid chip requirements fields')
    return _policy({key:value for key,value in requirements.items() if key!='operator_set'})


def resource_policy(graph):
    return _policy({key: graph[key] for key in ('resource_profile', 'resource_batch_bound') if key in graph})


def project_requirements(graph):
    policy = resource_policy(graph)
    if graph.get('schema_version') == 2:
        return tensor_requirements(policy['resource_batch_bound'])
    return {'operator_set': 'vision-ops-v4', **policy}


def requires_adoption(graph, requirements):
    current = resource_policy(graph)
    required = resource_policy(requirements) if 'resource_profile' in requirements else None
    if requirements.get('schema_version') == 2:
        validate_tensor_requirements(requirements)
        return graph.get('schema_version') != 2 or current != required
    return required is not None and not policy_satisfies(current, required)


def graph_settings(requirements):
    settings = {'schema_version': requirements.get('schema_version', 1), **resource_policy(requirements)}
    if settings['schema_version'] == 2:
        settings['capability_profile'] = 'tensor-v1'
    return settings


def legacy_tensor_port(port):
    """Assign conventional axis names to a legacy float32 port in a project copy."""
    rank = len(port['shape'])
    trailing = {1: [], 2: ['feature'], 3: ['token', 'feature'], 4: ['channel', 'height', 'width']}
    return {'axes': ['batch', *trailing.get(rank, [f'axis_{index}' for index in range(1, rank)])],
            'batch_axis': 0}


def normalize_legacy_scope(scope):
    for port in scope['inputs']:
        if port.get('dtype') == 'float32' and type(port.get('shape')) is list:
            port.update(legacy_tensor_port(port))


def _uses_vision_operators(definitions):
    return any(instance.get('kind') in VISION_OPERATORS
               for definition in definitions for instance in definition['graph']['instances'])


def _require(condition,message):
    if not condition:raise ValueError(message)


def _json(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def _closure(definition_id,definitions):
    ordered=[];visited=set();active=set()
    def visit(key):
        _require(key in definitions,'Missing chip dependency: '+str(key))
        _require(key not in active,'Recursive chip dependency: '+str(key))
        if key in visited:return
        _require(len(active)<8,'Chip dependency depth limit exceeded')
        active.add(key)
        for instance in definitions[key]['graph']['instances']:
            if instance.get('kind')=='chip':visit(instance.get('definition_id'))
        active.remove(key);visited.add(key);ordered.append(key)
    visit(definition_id)
    return ordered


def validate_bundle(payload):
    _require(type(payload) is dict,'Invalid chip bundle fields')
    policy=_requirements(payload)
    _require(len(_json(payload).encode('utf-8'))<=MAX_BYTES,'Chip bundle exceeds 4 MiB')
    _require(type(payload['definition_id']) is str and IDENTIFIER.fullmatch(payload['definition_id']),'Invalid bundle definition_id')
    entries=payload['definitions'];_require(type(entries) is list and 0<len(entries)<=MAX_DEFINITIONS,'Chip bundle exceeds definition limit')
    from .document import _structure
    version=payload['chip_format_version']
    settings=graph_settings(policy) if policy is not None else {'schema_version':1,'resource_profile':'classification-v2'}
    _structure({**settings,'id':'bundle_storage','revision':1,'inputs':[],'instances':[],'calls':[],'outputs':[],'definitions':entries})
    definitions={}
    for definition in entries:
        _require(type(definition) is dict and type(definition.get('id')) is str and IDENTIFIER.fullmatch(definition['id']),'Invalid chip definition')
        _require(definition['id'] not in definitions,'Duplicate chip definition')
        _require(type(definition.get('version')) is str and VERSION.fullmatch(definition['version']),'Chip requires a semantic version')
        _require(type(definition.get('graph')) is dict and type(definition['graph'].get('instances')) is list,'Invalid chip graph')
        definitions[definition['id']]=definition
    ordered=_closure(payload['definition_id'],definitions)
    _require(set(ordered)==set(definitions),'Bundle contains unrelated definitions')
    _require(policy is not None or not _uses_vision_operators(entries),'New operators require chip_format_version 2')
    # Validate each definition's declared defaults without executing Python.
    for key in ordered:
        definition=definitions[key];scope=definition['graph']
        _require(type(scope.get('inputs')) is list and type(scope.get('outputs')) is list,'Chip requires interfaces')
        if version!=3:
            _require(all('axes' not in port and 'batch_axis' not in port for port in scope['inputs']),
                     'Tensor axes require chip_format_version 3')
        test_graph={**settings,'id':'library_validation','revision':1,
                    'inputs':deepcopy(scope['inputs']),
                    'instances':[{'id':'library_instance','kind':'chip','definition_id':key,'params':{}}],
                    'calls':[{'id':'library_call','instance_id':'library_instance','inputs':{port['id']:'input:'+port['id'] for port in scope['inputs']}}],
                    'outputs':[{'id':port['id'],'source':'call:library_call:'+port['id']} for port in scope['outputs']],
                    'definitions':deepcopy(entries)}
        report=validate_graph(test_graph)
        _require(report['valid'],'Invalid chip defaults/structure: '+ '; '.join(issue['message'] for issue in report['errors']))
    return definitions,ordered


def _read(path):
    path=Path(path)
    _require(path.stat().st_size<=MAX_BYTES,'Chip bundle exceeds 4 MiB')
    try:
        value=json.loads(path.read_text(encoding='utf-8-sig'),parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite chip JSON')))
    except (UnicodeError,json.JSONDecodeError) as error:raise ValueError('Invalid chip bundle JSON') from error
    validate_bundle(value);return value


def read_bundle(path):
    """Read and validate one architecture package without importing it into a project."""
    return _read(path)


def save_bundle(graph,definition_id,version,directory):
    _require(type(version) is str and VERSION.fullmatch(version),'Version must be semantic, e.g. 1.0.0')
    definitions={definition['id']:deepcopy(definition) for definition in graph.get('definitions',[])}
    _require(definition_id in definitions,'Unknown chip definition')
    definitions[definition_id]['version']=version
    ordered=_closure(definition_id,definitions)
    payload={'chip_format_version':1,'definition_id':definition_id,'definitions':[definitions[key] for key in ordered]}
    if graph.get('schema_version') == 2:
        payload.update(chip_format_version=3, requirements=project_requirements(graph))
    elif graph.get('resource_profile') in ('vision-v4','hardware-v5') or _uses_vision_operators(payload['definitions']):
        payload.update(chip_format_version=2,requirements=project_requirements(graph))
    validate_bundle(payload)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=directory/(definition_id+'__'+version+'.mlfchip.json')
    if path.exists():
        _require(_json(_read(path))==_json(payload),'This chip version already exists with different contents; use a new version')
        return path
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=directory,prefix='.chip-',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name);stream.write(json.dumps(payload,ensure_ascii=False,allow_nan=False,indent=2));stream.flush();os.fsync(stream.fileno())
        # Atomic create-if-absent: an immutable version is never overwritten.
        try:os.link(temporary,path)
        except FileExistsError:
            _require(_json(_read(path))==_json(payload),'This chip version already exists with different contents; use a new version')
        return path
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def import_bundle(path,existing,*,resource_policy=None,tensor_project=False):
    return merge_bundle(_read(path),existing,resource_policy=resource_policy,tensor_project=tensor_project)


def merge_bundle(payload,existing,*,resource_policy=None,tensor_project=False):
    """Validate and merge an in-memory bundle without touching the filesystem."""
    definitions,ordered=validate_bundle(payload)
    requirements=_requirements(payload)
    is_tensor=payload['chip_format_version']==3
    _require(not is_tensor or tensor_project,'Tensor chip requires a schema 2 project; upgrade explicitly before import')
    if requirements is not None:
        _require(resource_policy is not None,'Chip import requires an explicit matching resource policy; apply the project policy first')
        required=_policy({key:value for key,value in requirements.items() if key in ('resource_profile','resource_batch_bound')})
        _require(policy_satisfies(resource_policy, required),'Chip resource policy does not match the project; apply the policy explicitly before import')
    result=deepcopy(existing);by_id={definition['id']:definition for definition in result};mapping={}
    for original in ordered:
        candidate=deepcopy(definitions[original])
        if tensor_project and not is_tensor:
            normalize_legacy_scope(candidate['graph'])
        for instance in candidate['graph']['instances']:
            if instance.get('kind')=='chip':instance['definition_id']=mapping[instance['definition_id']]
        identifier=original;number=2
        while identifier in by_id:
            candidate['id']=identifier
            if _json(candidate)==_json(by_id[identifier]):break
            identifier=original+'_'+str(number);number+=1
        else:
            candidate['id']=identifier;result.append(candidate);by_id[identifier]=candidate
        mapping[original]=identifier
    _require(len(result)<=MAX_DEFINITIONS,'Merged chip library exceeds definition limit')
    _preflight({'definitions':result})
    return result,mapping[payload['definition_id']]
