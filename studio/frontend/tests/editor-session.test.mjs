import test from 'node:test';
import assert from 'node:assert/strict';
import { initialPythonSession, pythonSessionReducer as reduce } from '../src/python-session.mjs';
import { chipContextEntry, probePrefixForTrail, chooseProbeSource } from '../src/graph.mjs';

test('pending Python requests lock editing and close, then accept only their own response', () => {
  let session = reduce(initialPythonSession(), { type: 'OPEN' });
  session = reduce(session, { type: 'EDIT', code: 'my draft' });
  session = reduce(session, { type: 'REQUEST', id: 1, kind: 'export' });
  assert.equal(reduce(session, { type: 'EDIT', code: 'newer draft' }).code, 'my draft');
  assert.equal(reduce(session, { type: 'CLOSE' }).open, true);
  const obsolete = reduce(session, { type: 'EXPORT_SUCCESS', id: 0, code: 'obsolete' });
  assert.equal(obsolete.code, 'my draft');
  assert.equal(obsolete.busy, true);
  session = reduce(session, { type: 'EXPORT_SUCCESS', id: 1, code: 'generated', sourceKey: 'r1' });
  assert.equal(session.code, 'generated');
  assert.equal(session.busy, false);
});
test('failed import drafts and diagnostics survive close and reopen', () => {
  let session = reduce(initialPythonSession(), { type: 'OPEN' });
  session = reduce(session, { type: 'EDIT', code: 'unsupported python' });
  session = reduce(session, { type: 'REQUEST', id: 1, kind: 'import' });
  session = reduce(session, { type: 'FAILURE', id: 1, message: 'unsupported syntax' });
  session = reduce(session, { type: 'CLOSE' });
  session = reduce(session, { type: 'OPEN' });
  assert.equal(session.code, 'unsupported python');
  assert.equal(session.error, 'unsupported syntax');
  assert.equal(session.dirty, true);
  assert.equal(session.hasDraft, true);
});
test('regeneration failures retain draft and obsolete completions cannot replace newer request', () => {
  let session = reduce(initialPythonSession(), { type: 'EDIT', code: 'retained' });
  session = reduce(session, { type: 'REQUEST', id: 1, kind: 'export' });
  session = reduce(session, { type: 'FAILURE', id: 1, message: 'offline' });
  assert.equal(session.code, 'retained');
  session = reduce(session, { type: 'REQUEST', id: 2, kind: 'export' });
  session = reduce(session, { type: 'EXPORT_SUCCESS', id: 1, code: 'old response' });
  assert.equal(session.code, 'retained');
  assert.equal(session.busy, true);
  session = reduce(session, { type: 'EXPORT_SUCCESS', id: 2, code: 'latest response' });
  assert.equal(session.code, 'latest response');
});
test('file reads use the same transaction guard and successful imports retain their code', () => {
  let session = reduce(initialPythonSession(), { type: 'REQUEST', id: 1, kind: 'file' });
  session = reduce(session, { type: 'FILE_SUCCESS', id: 1, code: 'file draft' });
  assert.equal(session.dirty, true);
  session = reduce(session, { type: 'REQUEST', id: 2, kind: 'import' });
  session = reduce(session, { type: 'IMPORT_SUCCESS', id: 2, sourceKey: 'r2' });
  assert.equal(session.code, 'file draft');
  assert.equal(session.open, false);
  assert.equal(session.dirty, false);
});
const project = {
  instances: [{ id: 'first', kind: 'chip', definition_id: 'shared' }, { id: 'second', kind: 'chip', definition_id: 'shared' }],
  calls: [{ id: 'call_first', instance_id: 'first' }, { id: 'call_second', instance_id: 'second' }],
  definitions: [{ id: 'shared', name: '共同定义', graph: { instances: [], calls: [] } }],
};
test('hierarchy navigation resolves the concrete independent instance call', () => {
  const first = chipContextEntry(project, null, 'shared', 'call_first');
  const second = chipContextEntry(project, null, 'shared', 'call_second');
  assert.equal(probePrefixForTrail([first]), 'call:call_first');
  assert.equal(probePrefixForTrail([second]), 'call:call_second');
  const probes = { 'call:call_first/call:dense': { mean: 1 }, 'call:call_second/call:dense': { mean: 2 } };
  assert.equal(chooseProbeSource(probes, { scopeId: 'shared', probePrefix: probePrefixForTrail([second]), localSource: 'call:dense' }), 'call:call_second/call:dense');
});
test('definition-only and invalid call navigation never guess the first measured probe', () => {
  const unknown = chipContextEntry(project, null, 'shared');
  const invalid = chipContextEntry(project, null, 'shared', 'missing_call');
  assert.equal(probePrefixForTrail([unknown]), null);
  assert.equal(probePrefixForTrail([invalid]), null);
  assert.equal(chooseProbeSource({ 'call:call_first/call:dense': {} }, { scopeId: 'shared', probePrefix: null, localSource: 'call:dense' }), null);
});
