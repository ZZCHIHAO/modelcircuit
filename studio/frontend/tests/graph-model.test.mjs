import test from 'node:test';
import assert from 'node:assert/strict';
import { editGraph, connectGraph, disconnectEdges, deleteNodes, addSharedCall, makeFlow, moveNodes, semanticKey, assertEditableProject, renameIO, createChip, updatePublicParameter } from '../src/graph.mjs';

const fixture = () => ({ revision: 4, inputs: [{ id: 'a', shape: [null, 4] }], instances: [{ id: 'encoder', kind: 'Linear', params: { in_features: 4, out_features: 8 } }], calls: [{ id: 'first', instance_id: 'encoder', inputs: { x: 'input:a' }, position: { x: 200, y: 100 } }, { id: 'second', instance_id: 'encoder', inputs: { x: 'call:first' }, position: { x: 400, y: 100 } }], outputs: [{ id: 'out', source: 'call:second' }], definitions: [{ id: 'chip', graph: { inputs: [], instances: [], calls: [], outputs: [] } }] });

test('named ports and output targets persist their actual source references', () => {
  const original = fixture();
  const connected = connectGraph(original, null, { source: 'input:a', target: 'call:second', targetHandle: 'x' });
  assert.equal(connected.calls[1].inputs.x, 'input:a');
  assert.equal(original.calls[1].inputs.x, 'call:first');
  assert.equal(connected.revision, 5);
  assert.equal(connectGraph(connected, null, { source: 'call:first', target: 'output:out' }).outputs[0].source, 'call:first');
});
test('deleting one call keeps the shared module and removes dependent references', () => {
  const next = deleteNodes(fixture(), null, ['call:first']);
  assert.equal(next.instances.length, 1);
  assert.deepEqual(next.calls[0].inputs, {});
  assert.equal(next.outputs[0].source, 'call:second');
  const empty = deleteNodes(next, null, ['call:second']);
  assert.equal(empty.instances.length, 0);
  assert.equal(empty.outputs[0].source, '');
});
test('shared call adds a use of the same identity without duplicating weights', () => {
  const { project, nodeId } = addSharedCall(fixture(), null, 'first');
  assert.equal(project.instances.length, 1);
  assert.equal(project.calls[2].instance_id, 'encoder');
  assert.equal(nodeId, 'call:first_2');
});
test('nested edits increment root revision and position edits preserve semantic key', () => {
  const original = fixture();
  const nested = editGraph(original, 'chip', (scope) => scope.inputs.push({ id: 'x' }));
  assert.equal(nested.revision, 5);
  assert.equal(original.definitions[0].graph.inputs.length, 0);
  const moved = editGraph(original, null, (scope) => { scope.calls[0].position.x = 500; }, false);
  assert.equal(moved.revision, 4);
  assert.equal(semanticKey(moved), semanticKey(original));
});
test('flow includes actual input, call and output ports and shared identities', () => {
  const { nodes, edges } = makeFlow(fixture(), null, { shapes: { 'call:second': [null, 8] } }, 'call:first');
  assert.equal(nodes.length, 4);
  assert.equal(nodes.find((node) => node.id === 'call:first').data.sharedCalls.length, 2);
  assert.ok(edges.some((edge) => edge.target === 'call:second' && edge.targetHandle === 'x'));
  assert.ok(edges.some((edge) => edge.target === 'output:out'));
});
test('broken file structures are rejected before replacing the editable draft', () => {
  assert.throws(() => assertEditableProject({ schema_version: 1, definitions: [], calls: [] }), /缺少/);
  const project = { ...fixture(), schema_version: 1 };
  assert.doesNotThrow(() => assertEditableProject(project));
  project.calls[0].inputs = null;
  assert.throws(() => assertEditableProject(project), /来源映射/);
});
test('disconnect removes only the matching source and preserves an intervening rewire', () => {
  const graph = fixture();
  const edge = { source: 'call:first', target: 'call:second', targetHandle: 'x' };
  assert.deepEqual(disconnectEdges(graph, null, [edge]).calls[1].inputs, {});
  graph.calls[1].inputs.x = 'input:a';
  assert.equal(disconnectEdges(graph, null, [edge]).calls[1].inputs.x, 'input:a');
});
test('IO rename changes source identities and rejects conflicts without mutating original', () => {
  const graph = fixture();
  const renamed = renameIO(graph, null, 'input', 'a', 'features');
  assert.equal(renamed.project.calls[0].inputs.x, 'input:features');
  assert.equal(renamed.nodeId, 'input:features');
  assert.equal(graph.inputs[0].id, 'a');
  assert.throws(() => renameIO(graph, null, 'input', 'a', '2bad'), /ID/);
  assert.equal(renameIO(graph, null, 'output', 'out', 'logits').project.outputs[0].id, 'logits');
});
test('chip input rename updates named arguments of every shared call', () => {
  const graph = fixture();
  graph.definitions[0].graph = { inputs: [{ id: 'x', shape: [null, 4] }], instances: [], calls: [], outputs: [{ id: 'out', source: 'input:x' }] };
  graph.instances[0] = { id: 'encoder', kind: 'chip', definition_id: 'chip', params: {} };
  const renamed = renameIO(graph, 'chip', 'input', 'x', 'tensor').project;
  assert.equal(renamed.calls[0].inputs.tensor, 'input:a');
  assert.equal(renamed.calls[1].inputs.tensor, 'call:first');
  assert.equal(renamed.definitions[0].graph.outputs[0].source, 'input:tensor');
  const deleted = deleteNodes(graph, 'chip', ['input:x']);
  assert.deepEqual(deleted.calls.map((call) => call.inputs), [{}, {}]);
});
test('encapsulation preserves internal sharing, boundary sources and original graph', () => {
  const original = fixture();
  const created = createChip(original, { mode: 'selection', name: '新芯片', id: 'wrapped', callIds: ['first', 'second'], outputCallId: 'second', shapes: { 'input:a': [null, 4] } });
  const definition = created.project.definitions.at(-1);
  assert.equal(definition.graph.instances.length, 1);
  assert.equal(definition.graph.calls[1].inputs.x, 'call:first');
  assert.equal(definition.graph.calls[0].inputs.x, 'input:a');
  assert.equal(definition.graph.outputs[0].source, 'call:second');
  assert.equal(created.project.calls.at(-1).inputs.a, 'input:a');
  assert.equal(created.project.instances.at(-1).definition_id, 'wrapped');
  assert.equal(created.project.revision, 5);
  assert.equal(original.definitions.length, 1);
  assert.equal(original.calls.length, 2);
});
test('partial encapsulation infers a call boundary and refuses unknown tensor shapes', () => {
  const graph = fixture();
  const options = { mode: 'selection', name: '尾部', id: 'tail', callIds: ['second'], outputCallId: 'second' };
  assert.throws(() => createChip(graph, options), /形状未知/);
  const created = createChip(graph, { ...options, shapes: { 'call:first': [null, 8] } });
  assert.deepEqual(created.project.definitions.at(-1).graph.inputs[0].shape, [null, 8]);
  assert.equal(created.project.calls.at(-1).inputs.first, 'call:first');
});
test('template and copied definitions preserve independent identity and public bindings', () => {
  const graph = fixture();
  const created = createChip(graph, { id: 'new_chip', name: '新芯片', inputFeatures: 4 });
  const copied = createChip(created.project, { mode: 'copy', sourceScopeId: 'new_chip', id: 'copy_chip', name: '副本', seedInputs: { x: 'input:a' } });
  assert.deepEqual(copied.project.definitions.at(-1).bindings, { in_features: ['dense.in_features'], hidden_features: ['dense.out_features'] });
  assert.equal(copied.project.calls.at(-1).inputs.x, 'input:a');
  copied.project.definitions.at(-1).graph.instances[0].params.out_features = 99;
  assert.equal(created.project.definitions.at(-1).graph.instances[0].params.out_features, 8);
  assert.throws(() => createChip(graph, { id: 'bad_chip', name: '非法维度', inputFeatures: 0 }), /正整数/);
});
test('flow multi-selection preserves selected call identities', () => {
  const graph = makeFlow(fixture(), null, null, ['call:first', 'call:second']);
  assert.equal(graph.nodes.filter((node) => node.selected).length, 2);
});
test('public parameter edits preserve instance overrides and keep binding identities aligned', () => {
  const graph = createChip(fixture(), { id: 'new_chip', name: '新芯片', inputFeatures: 4 }).project;
  const changedDefault = updatePublicParameter(graph, 'new_chip', 'hidden_features', { default: 16 });
  assert.equal(changedDefault.instances.at(-1).params.hidden_features, 8);
  assert.equal(changedDefault.definitions.at(-1).public_parameters[1].default, 16);
  const renamed = updatePublicParameter(graph, 'new_chip', 'hidden_features', { id: 'width' });
  assert.equal(renamed.instances.at(-1).params.width, 8);
  assert.equal(renamed.instances.at(-1).params.hidden_features, undefined);
  assert.deepEqual(renamed.definitions.at(-1).bindings.width, ['dense.out_features']);
  const removed = updatePublicParameter(renamed, 'new_chip', 'width', { remove: true });
  assert.equal(removed.instances.at(-1).params.width, undefined);
  assert.equal(removed.definitions.at(-1).bindings.width, undefined);
  assert.equal(graph.instances.at(-1).params.hidden_features, 8);
});
test('new public parameters support binding and clearing obsolete target paths', () => {
  const graph = createChip(fixture(), { id: 'new_chip', name: '新芯片', inputFeatures: 4 }).project;
  const added = updatePublicParameter(graph, 'new_chip', null, { id: 'new_width', default: 8, name: '宽度' });
  const bound = updatePublicParameter(added, 'new_chip', 'new_width', { targets: ['dense.out_features'] });
  assert.deepEqual(bound.definitions.at(-1).bindings.new_width, ['dense.out_features']);
  const cleared = updatePublicParameter(bound, 'new_chip', 'new_width', { targets: [] });
  assert.equal(cleared.definitions.at(-1).bindings.new_width, undefined);
  assert.throws(() => updatePublicParameter(bound, 'new_chip', 'new_width', { id: 'in_features' }), /已存在/);
});
test('dragging multiple selected nodes persists every position without semantic changes', () => {
  const original = fixture();
  const moved = moveNodes(original, null, [{ id: 'call:first', position: { x: 123, y: 456 } }, { id: 'call:second', position: { x: 789, y: 100 } }, { id: 'output:out', position: { x: 900, y: 200 } }]);
  assert.deepEqual(moved.calls.map((call) => call.position.x), [123, 789]);
  assert.deepEqual(moved.outputs[0].position, { x: 900, y: 200 });
  assert.equal(semanticKey(moved), semanticKey(original));
  assert.equal(moved.revision, original.revision);
});
