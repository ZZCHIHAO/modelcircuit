export const PRIMITIVES = [
  { kind: 'Linear', label: '线性层', group: '映射', params: { in_features: 8, out_features: 8 }, args: ['x'] },
  { kind: 'ReLU', label: 'ReLU 激活', group: '激活', params: {}, args: ['x'] },
  { kind: 'GELU', label: 'GELU 激活', group: '激活', params: {}, args: ['x'] },
  { kind: 'Add', label: '逐元素相加', group: '融合', params: {}, args: ['a', 'b'] },
  { kind: 'Concat', label: '特征拼接', group: '融合', params: { dim: -1 }, args: ['a', 'b'] },
  { kind: 'LayerNorm', label: '层归一化', group: '归一化', params: { normalized_shape: 8 }, args: ['x'] },
  { kind: 'Dropout', label: '随机失活', group: '正则化', params: { p: 0.1 }, args: ['x'] },
];

export const copy = (value) => JSON.parse(JSON.stringify(value));
export function assertEditableProject(project) {
  const record = (value) => Boolean(value && typeof value === 'object' && !Array.isArray(value));
  if (!record(project) || project.schema_version !== 1 || !Array.isArray(project.definitions)) throw new Error('需要 schema_version = 1 的模型项目与 definitions 数组。');
  function inspect(graph, name) {
    if (!record(graph) || !['inputs', 'instances', 'calls', 'outputs'].every((part) => Array.isArray(graph[part]))) throw new Error(`${name} 缺少 inputs、instances、calls 或 outputs 数组。`);
    for (const part of ['inputs', 'instances', 'calls', 'outputs']) if (!graph[part].every((item) => record(item) && typeof item.id === 'string')) throw new Error(`${name}.${part} 中的每一项都需要字符串 ID。`);
    if (!graph.instances.every((item) => typeof item.kind === 'string' && record(item.params))) throw new Error(`${name} 的实例需要 kind 和 params 对象。`);
    if (!graph.calls.every((item) => typeof item.instance_id === 'string' && record(item.inputs) && Object.values(item.inputs).every((value) => typeof value === 'string'))) throw new Error(`${name} 的调用需要 instance_id 和 inputs 来源映射。`);
    if (!graph.outputs.every((item) => typeof item.source === 'string')) throw new Error(`${name} 的输出需要 source 字符串。`);
  }
  inspect(project, '模型');
  for (const definition of project.definitions) {
    if (!record(definition) || typeof definition.id !== 'string') throw new Error('芯片定义需要字符串 ID。');
    inspect(definition.graph, definition.id);
    if (definition.public_parameters && (!Array.isArray(definition.public_parameters) || !definition.public_parameters.every((parameter) => record(parameter) && typeof parameter.id === 'string'))) throw new Error(`${definition.id} 的公开参数格式无效。`);
    if (definition.bindings && (!record(definition.bindings) || !Object.values(definition.bindings).every((targets) => Array.isArray(targets) && targets.every((target) => typeof target === 'string')))) throw new Error(`${definition.id} 的参数绑定需要目标数组。`);
  }
  return project;
}
export const getScope = (project, scopeId) => scopeId
  ? project?.definitions?.find((definition) => definition.id === scopeId)?.graph
  : project;
export function chipContextEntry(project, parentScopeId, definitionId, callId) {
  const parent = getScope(project, parentScopeId);
  const call = callId && parent?.calls.find((item) => item.id === callId);
  const instance = call && parent.instances.find((item) => item.id === call.instance_id);
  const valid = instance?.kind === 'chip' && instance.definition_id === definitionId;
  return { id: definitionId, callId: valid ? callId : null, name: project.definitions.find((item) => item.id === definitionId)?.name || definitionId };
}
export function probePrefixForTrail(trail) {
  return trail.length && trail.every((entry) => entry.callId) ? trail.map((entry) => `call:${entry.callId}`).join('/') : null;
}
export function chooseProbeSource(probes, { scopeId, probePrefix, localSource }) {
  if (scopeId && !probePrefix) return null;
  if (localSource) return scopeId ? `${probePrefix}/${localSource}` : localSource;
  return Object.keys(probes || {}).find((source) => scopeId
    ? source.startsWith(`${probePrefix}/`) && !source.slice(probePrefix.length + 1).includes('/')
    : !source.includes('/')) || null;
}
export function editGraph(project, scopeId, mutate, semantic = true) {
  const next = copy(project);
  const scope = getScope(next, scopeId);
  if (!scope) return project;
  mutate(scope, next);
  if (semantic) next.revision = (Number(project.revision) || 0) + 1;
  return next;
}
export const inputNodeId = (id) => `input:${id}`;
export const callNodeId = (id) => `call:${id}`;
export const outputNodeId = (id) => `output:${id}`;
export function moveNodes(project, scopeId, nodes) {
  return editGraph(project, scopeId, (scope) => {
    for (const node of nodes) {
      const items = node.id.startsWith('call:') ? scope.calls : node.id.startsWith('input:') ? scope.inputs : scope.outputs;
      const target = items.find((item) => item.id === node.id.slice(node.id.indexOf(':') + 1));
      if (target) target.position = { x: node.position.x, y: node.position.y };
    }
  }, false);
}
export function portNames(instance, project) {
  if (instance.kind === 'chip') {
    return project.definitions?.find((definition) => definition.id === instance.definition_id)?.graph.inputs?.map((input) => input.id) || ['x'];
  }
  return PRIMITIVES.find((primitive) => primitive.kind === instance.kind)?.args || ['x'];
}
export function uniqueId(prefix, used) {
  const names = new Set(used);
  let id = prefix;
  let index = 2;
  while (names.has(id)) id = `${prefix}_${index++}`;
  return id;
}
export function connectGraph(project, scopeId, connection) {
  return editGraph(project, scopeId, (scope) => {
    const source = connection.source;
    if (!source?.startsWith('input:') && !source?.startsWith('call:')) return;
    if (connection.target?.startsWith('output:')) {
      const output = scope.outputs.find((item) => item.id === connection.target.slice(7));
      if (output) output.source = source;
    } else if (connection.target?.startsWith('call:')) {
      const call = scope.calls.find((item) => item.id === connection.target.slice(5));
      if (call && connection.targetHandle) call.inputs[connection.targetHandle] = source;
    }
  });
}
export function disconnectEdges(project, scopeId, edges) {
  return editGraph(project, scopeId, (scope) => {
    for (const edge of edges) {
      if (edge.target.startsWith('output:')) {
        const output = scope.outputs.find((item) => item.id === edge.target.slice(7));
        if (output?.source === edge.source) output.source = '';
      } else {
        const call = scope.calls.find((item) => `call:${item.id}` === edge.target);
        if (call?.inputs[edge.targetHandle] === edge.source) delete call.inputs[edge.targetHandle];
      }
    }
  });
}
export function deleteNodes(project, scopeId, ids) {
  const selected = new Set(ids);
  return editGraph(project, scopeId, (scope, document) => {
    const affectedInstances = new Set(scope.calls.filter((item) => selected.has(callNodeId(item.id))).map((item) => item.instance_id));
    const removedPorts = scope.inputs.filter((item) => selected.has(inputNodeId(item.id))).map((item) => item.id);
    scope.inputs = scope.inputs.filter((item) => !selected.has(inputNodeId(item.id)));
    scope.calls = scope.calls.filter((item) => !selected.has(callNodeId(item.id)));
    scope.outputs = scope.outputs.filter((item) => !selected.has(outputNodeId(item.id)));
    const used = new Set(scope.calls.map((item) => item.instance_id));
    scope.instances = scope.instances.filter((item) => used.has(item.id) || !affectedInstances.has(item.id));
    for (const call of scope.calls) {
      for (const [port, source] of Object.entries(call.inputs)) {
        if (selected.has(source)) delete call.inputs[port];
      }
    }
    for (const output of scope.outputs) if (selected.has(output.source)) output.source = '';
    if (scopeId && removedPorts.length) for (const parent of [document, ...document.definitions.map((item) => item.graph)]) {
      const chips = new Set(parent.instances.filter((item) => item.definition_id === scopeId).map((item) => item.id));
      for (const call of parent.calls) if (chips.has(call.instance_id)) for (const port of removedPorts) delete call.inputs[port];
    }
  });
}
export function addComponent(project, scopeId, kind, definitionId) {
  let newId;
  const next = editGraph(project, scopeId, (scope) => {
    const primitive = PRIMITIVES.find((item) => item.kind === kind);
    const definition = project.definitions?.find((item) => item.id === definitionId);
    const prefix = kind === 'chip' ? 'encoder' : kind.toLowerCase();
    const instanceId = uniqueId(prefix, scope.instances.map((item) => item.id));
    newId = uniqueId(instanceId, scope.calls.map((item) => item.id));
    const instance = {
      id: instanceId, kind, name: definition?.name || primitive?.label || kind,
      params: definition ? Object.fromEntries((definition.public_parameters || []).map((param) => [param.id, param.default])) : copy(primitive?.params || {}),
      frozen: false,
    };
    if (definition) instance.definition_id = definition.id;
    scope.instances.push(instance);
    scope.calls.push({ id: newId, instance_id: instanceId, inputs: {}, position: { x: 270 + (scope.calls.length % 3) * 90, y: 100 + scope.calls.length * 90 } });
  });
  return { project: next, nodeId: callNodeId(newId) };
}
export function addSharedCall(project, scopeId, callId) {
  let newId;
  const next = editGraph(project, scopeId, (scope) => {
    const source = scope.calls.find((item) => item.id === callId);
    if (!source) return;
    newId = uniqueId(source.id, scope.calls.map((item) => item.id));
    scope.calls.push({ id: newId, instance_id: source.instance_id, inputs: {}, position: { x: (source.position?.x || 250), y: (source.position?.y || 100) + 220 } });
  });
  return { project: next, nodeId: callNodeId(newId) };
}
export function renameIO(project, scopeId, category, previousId, nextId) {
  nextId = nextId.trim();
  if (previousId === nextId) return { project, nodeId: `${category}:${previousId}` };
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(nextId)) throw new Error('端口 ID 只能使用英文字母、数字和下划线，且不能以数字开头。');
  const scope = getScope(project, scopeId);
  const collection = category === 'input' ? 'inputs' : 'outputs';
  if (scope[collection].some((item) => item.id === nextId)) throw new Error('该端口 ID 已存在。');
  const next = editGraph(project, scopeId, (graph, document) => {
    const port = graph[collection].find((item) => item.id === previousId);
    if (!port) throw new Error('找不到需要重命名的端口。');
    port.id = nextId;
    if (category !== 'input') return;
    for (const call of graph.calls) for (const [key, value] of Object.entries(call.inputs)) if (value === `input:${previousId}`) call.inputs[key] = `input:${nextId}`;
    for (const output of graph.outputs) if (output.source === `input:${previousId}`) output.source = `input:${nextId}`;
    if (scopeId) for (const parent of [document, ...document.definitions.map((item) => item.graph)]) {
      const instances = new Set(parent.instances.filter((item) => item.kind === 'chip' && item.definition_id === scopeId).map((item) => item.id));
      for (const call of parent.calls) if (instances.has(call.instance_id) && Object.hasOwn(call.inputs, previousId)) {
        if (Object.hasOwn(call.inputs, nextId)) throw new Error('调用已含同名参数，无法安全重命名。');
        call.inputs[nextId] = call.inputs[previousId]; delete call.inputs[previousId];
      }
    }
  });
  return { project: next, nodeId: `${category}:${nextId}` };
}
export function createChip(project, options) {
  const { id, name, mode = 'template', sourceScopeId, callIds = [], outputCallId, shapes = {}, seedInputs = {} } = options;
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(id)) throw new Error('芯片 ID 需要以字母或下划线开头，仅包含英文字母、数字和下划线。');
  if (!name.trim()) throw new Error('请输入芯片名称。');
  if (project.definitions.some((item) => item.id === id)) throw new Error('该芯片 ID 已存在。');
  let definition, callInputs = {};
  if (mode === 'copy') {
    const original = project.definitions.find((item) => item.id === sourceScopeId);
    if (!original) throw new Error('找不到源芯片定义。');
    if (original.graph.outputs.length !== 1) throw new Error('S1 芯片需要且只能有一个输出，请先调整内部图。');
    definition = { ...copy(original), id, name: name.trim(), version: '0.1.0' }; callInputs = copy(seedInputs);
  } else if (mode === 'selection') {
    const scope = getScope(project, sourceScopeId);
    const selected = new Set(callIds), calls = scope.calls.filter((call) => selected.has(call.id));
    if (!calls.length || !calls.some((call) => call.id === outputCallId)) throw new Error('请选择至少一个调用，并指定芯片的唯一输出节点。');
    const used = new Set(calls.map((call) => call.instance_id)), boundary = new Map(), inputs = [];
    for (const call of calls) for (const source of Object.values(call.inputs)) if (!source.startsWith('call:') || !selected.has(source.slice(5))) {
      if (boundary.has(source)) continue;
      const shape = shapes[source] || (source.startsWith('input:') && scope.inputs.find((input) => `input:${input.id}` === source)?.shape);
      if (!Array.isArray(shape)) throw new Error('边界张量形状未知，请先通过模型校验后再封装。');
      const port = uniqueId(source.slice(source.indexOf(':') + 1), inputs.map((input) => input.id));
      boundary.set(source, port); inputs.push({ id: port, name: port, shape: copy(shape), dtype: 'float32' });
      if (!sourceScopeId) callInputs[port] = source;
    }
    if (!inputs.length) throw new Error('芯片需要至少一个外部输入。');
    const clonedCalls = copy(calls), minX = Math.min(...calls.map((call) => call.position?.x || 0)), minY = Math.min(...calls.map((call) => call.position?.y || 0));
    for (const call of clonedCalls) { for (const [port, source] of Object.entries(call.inputs)) if (boundary.has(source)) call.inputs[port] = `input:${boundary.get(source)}`; call.position = { x: (call.position?.x || 0) - minX + 230, y: (call.position?.y || 0) - minY + 120 }; }
    definition = { id, name: name.trim(), version: '0.1.0', public_parameters: [], bindings: {}, graph: { inputs, instances: copy(scope.instances.filter((instance) => used.has(instance.id))), calls: clonedCalls, outputs: [{ id: 'out', source: `call:${outputCallId}` }] } };
  } else {
    const dimension = options.inputFeatures ?? 4;
    if (!Number.isInteger(dimension) || dimension < 1) throw new Error('输入维度需要是正整数。');
    definition = { id, name: name.trim(), version: '0.1.0', public_parameters: [{ id: 'in_features', name: '输入维度', default: dimension }, { id: 'hidden_features', name: '特征维度', default: 8 }], bindings: { in_features: ['dense.in_features'], hidden_features: ['dense.out_features'] }, graph: { inputs: [{ id: 'x', name: '特征输入', shape: [null, dimension], dtype: 'float32' }], instances: [{ id: 'dense', name: '线性映射', kind: 'Linear', params: { in_features: dimension, out_features: 8 } }, { id: 'activation', name: '激活', kind: 'ReLU', params: {} }], calls: [{ id: 'project', instance_id: 'dense', inputs: { x: 'input:x' }, position: { x: 220, y: 120 } }, { id: 'activate', instance_id: 'activation', inputs: { x: 'call:project' }, position: { x: 510, y: 120 } }], outputs: [{ id: 'out', source: 'call:activate' }] } };
    if (project.inputs[0]) callInputs.x = `input:${project.inputs[0].id}`;
  }
  let instanceId, callId;
  const next = editGraph(project, null, (root) => {
    root.definitions.push(definition);
    instanceId = uniqueId(`${id}_instance`, root.instances.map((item) => item.id)); callId = uniqueId(`${id}_call`, root.calls.map((item) => item.id));
    root.instances.push({ id: instanceId, kind: 'chip', definition_id: id, name: name.trim(), params: Object.fromEntries((definition.public_parameters || []).map((item) => [item.id, item.default])), frozen: false });
    root.calls.push({ id: callId, instance_id: instanceId, inputs: callInputs, position: { x: 410, y: Math.max(500, ...root.calls.map((call) => (call.position?.y || 0) + 230)) } });
  });
  return { project: next, nodeId: callNodeId(callId), definitionId: id, callId };
}
export function updatePublicParameter(project, definitionId, parameterId, changes) {
  const definition = project.definitions.find((item) => item.id === definitionId);
  if (!definition) throw new Error('找不到芯片定义。');
  const nextId = (changes.id ?? parameterId)?.trim();
  if (!changes.remove && !/^[A-Za-z_][A-Za-z0-9_]*$/.test(nextId)) throw new Error('公开参数 ID 只能使用英文字母、数字和下划线，且不能以数字开头。');
  if (nextId !== parameterId && (definition.public_parameters || []).some((item) => item.id === nextId)) throw new Error('该公开参数 ID 已存在。');
  return editGraph(project, definitionId, (_, document) => {
    const target = document.definitions.find((item) => item.id === definitionId);
    target.bindings ||= {}; target.public_parameters ||= [];
    let parameter = target.public_parameters.find((item) => item.id === parameterId);
    if (!parameter && !changes.remove) { parameter = { id: nextId, name: changes.name || nextId, default: changes.default ?? 8 }; target.public_parameters.push(parameter); }
    if (changes.remove) { target.public_parameters = target.public_parameters.filter((item) => item.id !== parameterId); delete target.bindings[parameterId]; }
    else {
      if (nextId !== parameterId && Object.hasOwn(target.bindings, parameterId)) { target.bindings[nextId] = target.bindings[parameterId]; delete target.bindings[parameterId]; }
      parameter.id = nextId;
      if (changes.name !== undefined) parameter.name = changes.name;
      if (changes.default !== undefined) parameter.default = changes.default;
      if (changes.targets !== undefined) { if (changes.targets.length) target.bindings[nextId] = copy(changes.targets); else delete target.bindings[nextId]; }
    }
    if (changes.remove || nextId !== parameterId) for (const scope of [document, ...document.definitions.map((item) => item.graph)]) for (const instance of scope.instances) if (instance.kind === 'chip' && instance.definition_id === definitionId && Object.hasOwn(instance.params || {}, parameterId)) {
      if (!changes.remove) instance.params[nextId] = instance.params[parameterId]; delete instance.params[parameterId];
    }
  });
}
export function semanticKey(project) {
  if (!project) return '';
  return JSON.stringify(project, (key, value) => key === 'position' ? undefined : value);
}
export function formatShape(shape) { return Array.isArray(shape) ? `[${shape.map((part) => part == null ? 'N' : part).join(', ')}]` : '待校验'; }
export function makeFlow(project, scopeId, validation, selectedId) {
  const scope = getScope(project, scopeId);
  if (!scope) return { nodes: [], edges: [] };
  const shapes = scopeId ? {} : validation?.shapes || {};
  const nodes = [];
  const edges = [];
  const selected = (id) => Array.isArray(selectedId) ? selectedId.includes(id) : selectedId === id;
  for (let i = 0; i < scope.inputs.length; i++) {
    const input = scope.inputs[i];
    nodes.push({ id: inputNodeId(input.id), type: 'tensor', position: input.position || { x: -40, y: 105 + i * 215 }, selected: selected(inputNodeId(input.id)), data: { category: 'input', input, title: input.name || input.id, subtitle: input.id, shape: shapes[inputNodeId(input.id)] || input.shape, ports: [] } });
  }
  for (const call of scope.calls) {
    const instance = scope.instances.find((item) => item.id === call.instance_id);
    if (!instance) continue;
    const definition = project.definitions?.find((item) => item.id === instance.definition_id);
    const sharedCalls = scope.calls.filter((item) => item.instance_id === instance.id);
    const nodeId = callNodeId(call.id);
    const error = validation?.errors?.find((item) => item.node_id === call.id || item.node_id === instance.id);
    nodes.push({ id: nodeId, type: 'component', position: call.position || { x: 240, y: 180 }, selected: selected(nodeId), data: { category: instance.kind === 'chip' ? 'chip' : 'primitive', title: instance.name || instance.kind, subtitle: call.id, instance, call, definition, sharedCalls, ports: portNames(instance, project), shape: shapes[nodeId], error } });
    for (const [port, source] of Object.entries(call.inputs || {})) {
      if (source) edges.push({ id: `${source}>${nodeId}:${port}`, source, target: nodeId, targetHandle: port, type: 'smoothstep', data: { shape: shapes[source] }, markerEnd: { type: 'arrowclosed', width: 10, height: 10, color: '#718998' }, style: { stroke: '#718998', strokeWidth: 1.4 }, selected: false });
    }
  }
  const rightEdge = Math.max(800, ...scope.calls.map((call) => (call.position?.x || 0) + 270));
  for (let i = 0; i < scope.outputs.length; i++) {
    const output = scope.outputs[i];
    const nodeId = outputNodeId(output.id);
    nodes.push({ id: nodeId, type: 'tensor', position: output.position || { x: i === 0 ? rightEdge : rightEdge - 270, y: 140 + i * 250 }, selected: selected(nodeId), data: { category: 'output', output, title: output.id, subtitle: '模型输出', shape: shapes[output.source], ports: ['value'] } });
    if (output.source) edges.push({ id: `${output.source}>${nodeId}`, source: output.source, target: nodeId, targetHandle: 'value', type: 'smoothstep', markerEnd: { type: 'arrowclosed', width: 10, height: 10, color: '#70dbc4' }, style: { stroke: '#70dbc4', strokeWidth: 1.4 } });
  }
  return { nodes, edges };
}
