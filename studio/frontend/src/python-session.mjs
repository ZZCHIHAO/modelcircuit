export function initialPythonSession() {
  return { open: false, code: '', dirty: false, hasDraft: false, busy: false, error: '', requestId: 0, requestKind: null, sourceKey: '', confirmRegenerate: false };
}
export function pythonSessionReducer(state, action) {
  if (action.type === 'OPEN') return { ...state, open: true };
  if (action.type === 'CLOSE') return state.busy ? state : { ...state, open: false, confirmRegenerate: false };
  if (action.type === 'EDIT') return state.busy ? state : { ...state, code: action.code, dirty: true, hasDraft: true, error: '', confirmRegenerate: false };
  if (action.type === 'CONFIRM_REGENERATE') return state.busy ? state : { ...state, confirmRegenerate: true };
  if (action.type === 'KEEP_DRAFT') return { ...state, confirmRegenerate: false };
  if (action.type === 'REQUEST') {
    if (action.id <= state.requestId) return state;
    return { ...state, requestId: action.id, requestKind: action.kind, busy: true, error: '', confirmRegenerate: false };
  }
  if (action.id !== state.requestId || !state.busy) return state;
  if (action.type === 'FAILURE') return { ...state, busy: false, error: action.message, requestKind: null };
  if (action.type === 'EXPORT_SUCCESS' && state.requestKind === 'export') return { ...state, code: action.code, dirty: false, hasDraft: true, busy: false, error: '', requestKind: null, sourceKey: action.sourceKey || '' };
  if (action.type === 'FILE_SUCCESS' && state.requestKind === 'file') return { ...state, code: action.code, dirty: true, hasDraft: true, busy: false, error: '', requestKind: null, sourceKey: '' };
  if (action.type === 'IMPORT_SUCCESS' && state.requestKind === 'import') return { ...state, open: false, dirty: false, busy: false, error: '', requestKind: null, sourceKey: action.sourceKey || '' };
  return state;
}
