import test from 'node:test';
import assert from 'node:assert/strict';
import { applyNodeChanges } from '@xyflow/react';
import { selectedNodeIds, selectFlowNodes, reconcileFlowNodes, hierarchyInstanceTargets } from '../src/flow-state.mjs';

const initial = () => [{ id: 'a', selected: true, position: { x: 0, y: 0 }, data: { title: 'a' } }, { id: 'b', selected: false, position: { x: 200, y: 0 }, data: { title: 'b' } }];
test('React Flow selections update controlled nodes without rebuilding graph data', () => {
  const original = initial();
  const selected = applyNodeChanges([{ id: 'b', type: 'select', selected: true }], original);
  assert.deepEqual(selectedNodeIds(selected), ['a', 'b']);
  assert.equal(selected[0], original[0]);
  assert.equal(selected[1].data, original[1].data);
  assert.equal(selectFlowNodes(selected, ['a', 'b']), selected);
});
test('semantic graph rebuild preserves current selection and measured dimensions', () => {
  const previous = initial();
  previous[0].selected = false; previous[1].selected = true; previous[1].measured = { width: 192, height: 180 };
  const next = initial(); next[1].data = { title: 'updated parameter' };
  const rebuilt = reconcileFlowNodes(next, previous);
  assert.deepEqual(selectedNodeIds(rebuilt), ['b']);
  assert.equal(rebuilt[1].data.title, 'updated parameter');
  assert.deepEqual(rebuilt[1].measured, { width: 192, height: 180 });
});
test('external selection can target a newly added node and scope changes clear old selection', () => {
  const previous = initial();
  const next = [...initial(), { id: 'new', position: { x: 400, y: 0 }, data: {} }];
  assert.deepEqual(selectedNodeIds(reconcileFlowNodes(next, previous, ['new'])), ['new']);
  assert.deepEqual(selectedNodeIds(reconcileFlowNodes(next, previous, [])), []);
});
test('shared instance inspection selects its own first call while internal navigation remains explicit', () => {
  const calls = [{ id: 'other', instance_id: 'independent' }, { id: 'encode_a', instance_id: 'encoder' }, { id: 'encode_b', instance_id: 'encoder' }];
  assert.deepEqual(hierarchyInstanceTargets('encoder', calls), { inspectionNodeId: 'call:encode_a', internalCallId: null });
  assert.deepEqual(hierarchyInstanceTargets('independent', calls), { inspectionNodeId: 'call:other', internalCallId: 'other' });
});
