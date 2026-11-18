"""Transactional component placement; no GUI, runtime or filesystem access."""
from copy import deepcopy
import math

from studio_core.component_types import ComponentBundle, validate_tensor_requirements
from studio_core.operator_rules import normalize_operator, _parameter_count, _buffer_shapes
from studio_core.schema import (PRIMITIVES, MAX_ITEMS, MAX_DEFINITIONS, MAX_CHIP_DEPTH, _Budget, _preflight,
    graph_resource_limits, prepare_graph)
from .document_editing import command, require, identifier
from .chip_library import (_policy, _closure, graph_settings, merge_bundle, normalize_legacy_scope,
                           requires_adoption, resource_policy)


def _entry(bundle):
    from .document import _json_values
    require(type(bundle) is ComponentBundle, 'Component must be a ComponentBundle')
    raw = {'entry': bundle.entry, 'definitions': bundle.definitions, 'requirements': bundle.requirements}
    _json_values(raw); _preflight(raw)
    require(type(bundle.entry) is dict and type(bundle.definitions) is list and type(bundle.requirements) is dict,
            'Invalid component fields')
    entry = deepcopy(bundle.entry)
    require(set(entry) <= {'id', 'kind', 'params', 'definition_id', 'name', 'frozen'}, 'Unknown component entry fields')
    if 'id' in entry: identifier(entry['id'])
    require(type(entry.get('kind')) is str and entry['kind'] in PRIMITIVES | {'chip'}, 'Unsupported component kind')
    require(type(entry.get('params', {})) is dict, 'Component params must be an object')
    require(type(entry.get('frozen', False)) is bool, 'Component frozen must be a boolean')
    if 'name' in entry: require(type(entry['name']) is str, 'Component name must be a string')
    if entry['kind'] == 'chip': identifier(entry.get('definition_id'))
    else: require('definition_id' not in entry and not bundle.definitions, 'Primitive cannot contain chip definitions')
    requirements = bundle.requirements
    if requirements.get('operator_set') == 'tensor-ops-v1':
        return entry, validate_tensor_requirements(requirements)
    require(requirements.get('operator_set') == 'vision-ops-v4', 'Unsupported component operator_set')
    if entry['kind'] != 'chip':
        require(set(requirements) == {'operator_set'}, 'Primitive requirements must contain only operator_set')
        policy = None
    else:
        require('resource_profile' in requirements, 'Chip resource policy must be explicit')
        policy = _policy({key: value for key, value in requirements.items() if key != 'operator_set'})
    return entry, policy


def _check_entry(definition, entry, definitions, requirements):
    """Validate actual overrides independently of any unfinished project wires."""
    scope = definition['graph']
    instance = {**entry, 'id': 'component'}
    prepare_graph({**graph_settings(requirements), 'id': 'component_validation', 'revision': 1,
        'inputs': deepcopy(scope['inputs']), 'instances': [instance],
        'calls': [{'id': 'component_call', 'instance_id': 'component',
                   'inputs': {port['id']: 'input:' + port['id'] for port in scope['inputs']}}],
        'outputs': [{'id': port['id'], 'source': 'call:component_call:' + port['id']} for port in scope['outputs']],
        'definitions': definitions})


def _check_expansion(graph):
    """Bound nesting/work without demanding complete draft tensor wiring."""
    registry = {item['id']: item for item in graph.get('definitions', [])}
    budget = _Budget()

    def visit(scope, stack=()):
        budget.resolve(scope)
        for instance in scope['instances']:
            if instance['kind'] != 'chip': continue
            key = instance.get('definition_id')
            # An existing unresolved reference is an editable draft error; the
            # new component itself has already had its complete closure checked.
            if key not in registry: continue
            require(key not in stack, 'Recursive chip dependency')
            require(len(stack) < MAX_CHIP_DEPTH, 'Chip nesting depth limit exceeded')
            visit(registry[key]['graph'], (*stack, key))

    visit(graph)
    for definition in registry.values():
        # prepare_graph checks every definition through a three-node synthetic
        # wrapper, including unused definitions; retain its work accounting.
        budget.resolve({'inputs': [None], 'instances': [None], 'calls': [], 'outputs': [None]})
        visit(definition['graph'], (definition['id'],))


class ComponentInsertion:
    @command
    def set_execution_policy(self, policy):
        """Explicit project-wide policy edit, including one-step undo/redo."""
        policy = _policy(policy)
        current = resource_policy(self.graph)
        require(self.graph['schema_version'] == 1 or policy['resource_profile'] == 'hardware-v5',
                'Tensor projects require the hardware-v5 policy')
        if current == policy:
            return False
        self.graph.pop('resource_profile', None)
        self.graph.pop('resource_batch_bound', None)
        if policy['resource_profile'] is not None:
            self.graph.update(policy)
        self._changed()
        return True

    @command
    def add_component(self, bundle, position=None, *, apply_policy=False):
        """Add one independent instance/call and closure in one undo command.

        Resource policy adoption is opt-in. Existing editable draft connections
        are never inferred or repaired as a prerequisite to placement.
        """
        from .document import _structure
        require(type(apply_policy) is bool, 'apply_policy must be a boolean')
        entry, requested = _entry(bundle)
        if position is not None:
            require(type(position) is dict and set(position) == {'x', 'y'}, 'Position must contain exactly x and y')
            # Existing move() owns the strict finite numeric coordinate contract.
            for axis in ('x', 'y'):
                value = position[axis]
                try: finite = type(value) in (int, float) and math.isfinite(value)
                except OverflowError: finite = False
                require(finite, 'Position coordinates must be finite numbers')
        current = resource_policy(self.graph)
        adoption = requires_adoption(self.graph, bundle.requirements)
        tensor_bundle = requested is not None and requested.get('schema_version') == 2
        requires_upgrade = tensor_bundle and self.graph['schema_version'] == 1
        require(not adoption or apply_policy,
                'Component requires an explicit schema 2 tensor upgrade and policy confirmation' if requires_upgrade else
                'Component resource policy does not match the project; explicitly apply policy before adding')
        policy = resource_policy(requested) if adoption else current
        tensor_project = self.graph['schema_version'] == 2 or tensor_bundle
        require(not tensor_project or policy['resource_profile'] == 'hardware-v5',
                'Cannot downgrade a tensor project to a legacy resource policy')
        limits = graph_resource_limits(policy)
        if requires_upgrade:
            for scope in self._scopes():
                normalize_legacy_scope(scope)
            self.graph.update(schema_version=2, capability_profile='tensor-v1')
        if entry['kind'] == 'chip':
            requirements = requested if tensor_bundle else {'operator_set': 'vision-ops-v4', **policy}
            payload = {'chip_format_version': 3 if tensor_bundle else 2, 'definition_id': entry['definition_id'],
                'definitions': bundle.definitions, 'requirements': requirements}
            definitions, definition_id = merge_bundle(payload, self.graph.get('definitions', []),
                                                       resource_policy=policy, tensor_project=tensor_project)
            original = next(item for item in bundle.definitions if item['id'] == entry['definition_id'])
            _check_entry(original, entry, bundle.definitions, requirements)
            entry['definition_id'] = definition_id
            closure = _closure(definition_id, {item['id']: item for item in definitions})
            require(self.scope_id not in closure, 'Component would create a recursive chip definition')
        else:
            entry['params'] = normalize_operator(entry['kind'], entry.get('params', {}))
            require(limits.max_parameters is None or _parameter_count(entry['kind'], entry['params']) <= limits.max_parameters, 'Component parameter budget exceeded')
            require(limits.max_tensor_elements is None or all(math.prod(shape) <= limits.max_tensor_elements for shape in _buffer_shapes(entry['kind'], entry['params'])),
                    'Component buffer budget exceeded')
            definitions = self.graph.get('definitions', [])
        # The decorator owns rollback if the combined finite storage boundary
        # rejects an otherwise individually valid closure or current scope.
        if adoption:
            self.graph.pop('resource_profile', None); self.graph.pop('resource_batch_bound', None)
            if policy['resource_profile'] is not None: self.graph.update(policy)
        if entry['kind'] == 'chip': self.graph['definitions'] = definitions
        ref = self._append(entry['kind'], entry.get('definition_id'))
        instance = self.selected_instance
        instance['params'].update(deepcopy(entry.get('params', {})))
        for key in ('name', 'frozen'):
            if key in entry: instance[key] = entry[key]
        if position is not None: self.selected_call['position'] = deepcopy(position)
        _structure(self.graph); _preflight(self.graph)
        require(len(self.graph.get('definitions', [])) <= MAX_DEFINITIONS, 'Definition limit exceeded')
        for scope in self._scopes():
            for key in ('inputs', 'instances', 'calls', 'outputs'):
                require(len(scope[key]) <= MAX_ITEMS, key + ' item limit exceeded')
        _check_expansion(self.graph)
        self._changed()
        return ref
