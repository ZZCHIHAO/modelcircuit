export function hierarchyInstanceTargets(instanceId, calls) {
  const matches = calls.filter((call) => call.instance_id === instanceId);
  return { inspectionNodeId: matches[0] ? `call:${matches[0].id}` : null, internalCallId: matches.length === 1 ? matches[0].id : null };
}
export function selectedNodeIds(nodes) {
  return nodes.filter((node) => node.selected).map((node) => node.id);
}
export function selectFlowNodes(nodes, ids) {
  const selected = new Set(ids);
  let changed = false;
  const next = nodes.map((node) => {
    const value = selected.has(node.id);
    if (Boolean(node.selected) === value) return node;
    changed = true; return { ...node, selected: value };
  });
  return changed ? next : nodes;
}
export function reconcileFlowNodes(next, previous, explicitSelection = null) {
  const previousById = new Map(previous.map((node) => [node.id, node]));
  const selected = new Set(explicitSelection ?? selectedNodeIds(previous));
  return next.map((node) => {
    const old = previousById.get(node.id);
    return { ...node, ...(old?.measured ? { measured: old.measured } : {}), selected: selected.has(node.id) };
  });
}
export function reconcileFlowEdges(next, previous) {
  const selected = new Set(previous.filter((edge) => edge.selected).map((edge) => edge.id));
  return next.map((edge) => ({ ...edge, selected: selected.has(edge.id) }));
}
