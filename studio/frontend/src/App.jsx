import React, { useCallback, useEffect, useMemo, useRef, useReducer, useState } from 'react';
import { ReactFlow, Background, Controls, applyNodeChanges, applyEdgeChanges } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Cpu, Cube, FloppyDisk, FolderOpen, Code, Play, Stop, CheckCircle, WarningCircle, CaretRight, SlidersHorizontal, ArrowLeft, GitBranch, X, DownloadSimple, UploadSimple, CornersOut, Info, ArrowClockwise } from '@phosphor-icons/react';
import { api, download } from './api.js';
import { getScope, editGraph, makeFlow, moveNodes, connectGraph, disconnectEdges, deleteNodes, addComponent, addSharedCall, uniqueId, semanticKey, assertEditableProject, renameIO, createChip, updatePublicParameter, chipContextEntry, probePrefixForTrail } from './graph.mjs';
import { initialPythonSession, pythonSessionReducer } from './python-session.mjs';
import { selectedNodeIds as getSelectedNodeIds, selectFlowNodes, reconcileFlowNodes, reconcileFlowEdges } from './flow-state.mjs';
import { Button, Badge, Field, NumericField, Modal } from './components/common.jsx';
import { nodeTypes } from './components/Nodes.jsx';
import { Library } from './components/Library.jsx';
import { Inspector } from './components/Inspector.jsx';
import { ResultPanel } from './components/ResultPanel.jsx';

const DRAFT_KEY = 'ml-forge-s1-project';
const TERMINAL = new Set(['succeeded', 'failed', 'cancelled']);
const FIT_VIEW_OPTIONS = { padding: 0.14, maxZoom: 1.05 };
const PRO_OPTIONS = { hideAttribution: false };
const DELETE_KEYS = ['Backspace', 'Delete'];
function readDraft() { try { return assertEditableProject(JSON.parse(localStorage.getItem(DRAFT_KEY))); } catch { return null; } }
export function App() {
  const [project, setProject] = useState(readDraft);
  const [loading, setLoading] = useState(!project), [health, setHealth] = useState(null), [offline, setOffline] = useState(false), [notice, setNotice] = useState('');
  const [scopeId, setScopeId] = useState(null), [selectedId, setSelectedId] = useState('call:encode_a'), [trail, setTrail] = useState([]);
  const [chipDraft, setChipDraft] = useState(null), [chipError, setChipError] = useState('');
  const [validation, setValidation] = useState(null), [validating, setValidating] = useState(false), [validationKey, setValidationKey] = useState('');
  const [run, setRun] = useState(null), [runKey, setRunKey] = useState(''), [result, setResult] = useState(null), [resultRevision, setResultRevision] = useState(null), [resultKey, setResultKey] = useState('');
  const [config, setConfig] = useState({ steps: 20, lr: 0.05, seed: 42, batch_size: 16 }), [configOpen, setConfigOpen] = useState(false), [starting, setStarting] = useState(false), [tab, setTab] = useState('probes');
  const [pythonSession, dispatchPython] = useReducer(pythonSessionReducer, undefined, initialPythonSession);
  const { open: pythonOpen, code: pythonCode, dirty: pythonDirty, busy: pythonBusy, error: pythonError } = pythonSession;
  const pythonRequest = useRef({ id: 0, pending: false });
  const [saveState, setSaveState] = useState('已保存在本机'), [nodes, setNodes] = useState([]), [edges, setEdges] = useState([]);
  const selectionIntent = useRef(['call:encode_a']);
  const selectedNodeIds = useMemo(() => getSelectedNodeIds(nodes), [nodes]);
  const selectOnly = useCallback((id) => {
    const ids = id ? [id] : [];
    selectionIntent.current = ids;
    setSelectedId(id); setNodes((current) => selectFlowNodes(current, ids));
  }, []);
  const fileInput = useRef(null), pythonInput = useRef(null), flow = useRef(null), latestProject = useRef(project);
  const key = useMemo(() => semanticKey(project), [project]);
  const scope = getScope(project, scopeId), definition = project?.definitions?.find((item) => item.id === scopeId);
  const busy = starting || Boolean(run && !TERMINAL.has(run.status)), stale = Boolean(result && resultKey !== key);
  const currentValidation = validation && validationKey === key ? validation : null;
  const selectedCall = selectedId?.startsWith('call:') && scope?.calls.find((item) => `call:${item.id}` === selectedId);
  const selectedInstance = selectedCall && scope.instances.find((item) => item.id === selectedCall.instance_id);

  useEffect(() => { latestProject.current = project; }, [project]);
  const fetchHealth = useCallback(async () => { try { setHealth(await api('/health')); setOffline(false); } catch { setOffline(true); setHealth(null); } }, []);
  const loadExample = useCallback(async () => { setLoading(true); setNotice(''); try { const response = await api('/example'); setProject(response.graph); setScopeId(null); setTrail([]); selectOnly('call:encode_a'); setOffline(false); } catch (error) { setNotice(error.message); setOffline(true); } finally { setLoading(false); } }, [selectOnly]);
  useEffect(() => { fetchHealth(); if (!latestProject.current) loadExample(); }, [fetchHealth, loadExample]);
  useEffect(() => { if (!project) return; try { localStorage.setItem(DRAFT_KEY, JSON.stringify(project)); setSaveState('已保存在本机'); } catch { setSaveState('本机存储不可用 · 请保存 JSON'); } }, [project]);
  const validate = useCallback(async (document = latestProject.current) => {
    if (!document) return;
    const requestedKey = semanticKey(document); setValidating(true);
    try { const report = await api('/validate', { graph: document }); if (semanticKey(latestProject.current) === requestedKey) { setValidation(report); setValidationKey(requestedKey); } setOffline(false); return report; }
    catch (error) { setOffline(true); setNotice(error.message); }
    finally { setValidating(false); }
  }, []);
  useEffect(() => { if (!project) return; const timer = setTimeout(() => validate(latestProject.current), 650); return () => clearTimeout(timer); }, [key, validate]);
  useEffect(() => {
    const elements = makeFlow(project, scopeId, currentValidation, null);
    const explicitSelection = selectionIntent.current; selectionIntent.current = null;
    setNodes((current) => reconcileFlowNodes(elements.nodes, current, explicitSelection));
    setEdges((current) => reconcileFlowEdges(elements.edges, current));
  }, [project, scopeId, currentValidation]);
  const fit = useCallback(() => { setTimeout(() => flow.current?.fitView({ ...FIT_VIEW_OPTIONS, duration: 250 }), 80); }, []);
  useEffect(() => { const timer = setTimeout(() => flow.current?.fitView({ padding: 0.14, duration: 250, maxZoom: 1.05 }), 100); return () => clearTimeout(timer); }, [scopeId, Boolean(project)]);
  useEffect(() => {
    if (!run?.run_id || TERMINAL.has(run.status)) return;
    let stopped = false, timer;
    async function poll() {
      try { const update = await api(`/runs/${run.run_id}`); if (stopped) return; setRun(update); setOffline(false); if (update.result) { setResult(update.result); setResultRevision(update.revision); setResultKey(runKey); } if (TERMINAL.has(update.status)) { if (update.error) setNotice(update.error); return; } }
      catch (error) { if (!stopped) { setOffline(true); setNotice(error.message); } }
      if (!stopped) timer = setTimeout(poll, 700);
    }
    timer = setTimeout(poll, 180); return () => { stopped = true; clearTimeout(timer); };
  }, [run?.run_id, run?.status, runKey]);
  const change = useCallback((mutate, semantic = true) => setProject((current) => current ? editGraph(current, scopeId, mutate, semantic) : current), [scopeId]);
  const navigate = useCallback((nextScope, callId, fromRoot = false) => {
    if (!nextScope) { setScopeId(null); setTrail([]); }
    else {
      const entry = chipContextEntry(latestProject.current, fromRoot ? null : scopeId, nextScope, callId);
      setTrail((current) => fromRoot ? [entry] : [...current, entry]); setScopeId(nextScope);
    }
    selectOnly(null); setNotice('');
  }, [scopeId, selectOnly]);
  const handleFlowInit = useCallback((instance) => { flow.current = instance; }, []);
  const handleNodesChange = useCallback((changes) => {
    if (changes.some((change) => change.type === 'select' || change.type === 'remove')) selectionIntent.current = null;
    setNodes((current) => applyNodeChanges(changes, current));
  }, []);
  const handleEdgesChange = useCallback((changes) => setEdges((current) => applyEdgeChanges(changes, current)), []);
  const handleNodeClick = useCallback((_, node) => setSelectedId(node.id), []);
  const handlePaneClick = useCallback(() => selectOnly(null), [selectOnly]);
  const handleNodeDoubleClick = useCallback((_, node) => {
    if (node.data.instance?.kind === 'chip') navigate(node.data.instance.definition_id, node.data.call.id);
  }, [navigate]);
  const handleNodeDragStop = useCallback((_, node, moved) => setProject((current) => moveNodes(current, scopeId, moved?.length ? moved : [node])), [scopeId]);
  const handleConnect = useCallback((connection) => setProject((current) => connectGraph(current, scopeId, connection)), [scopeId]);
  const handleNodesDelete = useCallback((deleted) => {
    setProject((current) => deleteNodes(current, scopeId, deleted.map((node) => node.id))); selectOnly(null);
  }, [scopeId, selectOnly]);
  const handleEdgesDelete = useCallback((deleted) => setProject((current) => disconnectEdges(current, scopeId, deleted)), [scopeId]);
  function navigateCrumb(index) { const next = trail.slice(0, index + 1); setTrail(next); setScopeId(next.at(-1).id); selectOnly(null); }
  function addPrimitive(kind, definitionId) { if (!project) return; const added = addComponent(project, scopeId, kind, definitionId); setProject(added.project); selectOnly(added.nodeId); fit(); }
  function addShared(callId) { const added = addSharedCall(project, scopeId, callId); setProject(added.project); selectOnly(added.nodeId); fit(); }
  function deleteSelected() { if (!selectedId || !project) return; setProject(deleteNodes(project, scopeId, [selectedId])); selectOnly(null); }
  function addIO(category) { change((graph) => { const items = category === 'input' ? graph.inputs : graph.outputs; const id = uniqueId(category, items.map((item) => item.id)); items.push(category === 'input' ? { id, name: '新输入', shape: [null, 4], dtype: 'float32' } : { id, source: '' }); }); fit(); }
  function renamePort(category, oldId, newId) {
    try { const renamed = renameIO(project, scopeId, category, oldId, newId); setProject(renamed.project); selectOnly(renamed.nodeId); return true; }
    catch (error) { setNotice(error.message); return false; }
  }
  function openChip(mode) {
    const callIds = scope?.calls.filter((call) => selectedNodeIds.includes(`call:${call.id}`)).map((call) => call.id) || [];
    const prefix = mode === 'copy' ? `${scopeId}_copy` : 'custom_chip';
    setChipError(''); setChipDraft({ mode, name: mode === 'copy' ? `${definition?.name || '芯片'} 副本` : '自定义芯片', id: uniqueId(prefix, project.definitions.map((item) => item.id)), callIds, outputCallId: callIds.at(-1) || '', inputFeatures: project.inputs[0]?.shape?.at(-1) || 4 });
  }
  function saveChip() {
    try {
      const seedCall = trail.length === 1 && project.calls.find((call) => call.id === trail[0].callId);
      const created = createChip(project, { ...chipDraft, sourceScopeId: scopeId, shapes: currentValidation?.shapes || {}, seedInputs: seedCall?.inputs || {} });
      setProject(created.project); navigate(null); selectOnly(created.nodeId); setChipDraft(null); setNotice('芯片已保存到组件库，并在根图新增独立实例。'); fit();
    } catch (error) { setChipError(error.message); }
  }
  function saveJSON() { if (project) download(`${project.id || 'model'}.json`, JSON.stringify(project, null, 2)); }
  useEffect(() => { function handler(event) { if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); const document = latestProject.current; if (document) download(`${document.id || 'model'}.json`, JSON.stringify(document, null, 2)); } } window.addEventListener('keydown', handler); return () => window.removeEventListener('keydown', handler); }, []);
  async function openJSON(event) {
    const file = event.target.files?.[0]; if (!file) return;
    try { const document = assertEditableProject(JSON.parse(await file.text())); setProject(document); navigate(null); setNotice(`已打开 ${file.name}`); fit(); }
    catch (error) { setNotice(`无法打开项目：${error.message}`); }
    event.target.value = '';
  }
  function beginPythonRequest(kind) {
    if (pythonRequest.current.pending) return null;
    const id = pythonRequest.current.id + 1;
    pythonRequest.current = { id, pending: true };
    dispatchPython({ type: 'REQUEST', id, kind }); return id;
  }
  function pythonRequestActive(id) { return pythonRequest.current.pending && pythonRequest.current.id === id; }
  function finishPythonRequest(id) { if (pythonRequest.current.id === id) pythonRequest.current.pending = false; }
  function closePython() { if (!pythonRequest.current.pending) dispatchPython({ type: 'CLOSE' }); }
  function openPython() { dispatchPython({ type: 'OPEN' }); if (!pythonSession.hasDraft && !pythonRequest.current.pending) regeneratePython(); }
  function requestPythonRegeneration() {
    if (pythonRequest.current.pending) return;
    if (pythonSession.dirty) dispatchPython({ type: 'CONFIRM_REGENERATE' }); else regeneratePython();
  }
  async function regeneratePython() {
    const id = beginPythonRequest('export'); if (id === null) return;
    const document = latestProject.current;
    try { const exported = await api('/export', { graph: document }); if (!pythonRequestActive(id)) return; dispatchPython({ type: 'EXPORT_SUCCESS', id, code: exported.code, sourceKey: semanticKey(document) }); setOffline(false); }
    catch (error) { if (pythonRequestActive(id)) dispatchPython({ type: 'FAILURE', id, message: error.message }); }
    finally { finishPythonRequest(id); }
  }
  async function openPythonFile(event) {
    const file = event.target.files?.[0]; event.target.value = ''; if (!file) return;
    const id = beginPythonRequest('file'); if (id === null) return;
    try { const code = await file.text(); if (pythonRequestActive(id)) dispatchPython({ type: 'FILE_SUCCESS', id, code }); }
    catch (error) { if (pythonRequestActive(id)) dispatchPython({ type: 'FAILURE', id, message: error.message }); }
    finally { finishPythonRequest(id); }
  }
  async function importPython() {
    const id = beginPythonRequest('import'); if (id === null) return;
    try {
      const imported = await api('/import', { code: pythonCode }); if (!pythonRequestActive(id)) return;
      const document = assertEditableProject(imported.graph);
      document.revision = Math.max((latestProject.current?.revision || 0) + 1, document.revision || 1);
      setProject(document); navigate(null); dispatchPython({ type: 'IMPORT_SUCCESS', id, sourceKey: semanticKey(document) }); setOffline(false); setNotice('Python 已重新解析，画布已更新。'); fit();
    } catch (error) { if (pythonRequestActive(id)) dispatchPython({ type: 'FAILURE', id, message: error.message }); }
    finally { finishPythonRequest(id); }
  }
  async function startRun() {
    if (!project || busy) return; setStarting(true); setNotice('');
    try { const report = await validate(project); if (!report?.valid) { setTab('console'); if (report) setNotice('模型校验未通过，请先修复下方错误。'); return; } const response = await api('/runs', { graph: project, config }); setResult(null); setResultKey(''); setRunKey(key); setRun(response); setOffline(false); setTab('training'); }
    catch (error) { setNotice(error.message); } finally { setStarting(false); }
  }
  async function cancelRun() { try { const response = await api(`/runs/${run.run_id}/cancel`, {}); setRun((current) => ({ ...current, ...response })); } catch (error) { setNotice(error.message); } }

  return <div className="workbench">
    <input ref={fileInput} type="file" accept=".json,application/json" hidden onChange={openJSON} /><input ref={pythonInput} type="file" accept=".py,text/x-python,text/plain" hidden onChange={openPythonFile} />
    <header className="app-header"><a className="brand" href="#" aria-label="ML Forge 工作台" onClick={(event) => { event.preventDefault(); navigate(null); }}><span className="brand-symbol"><Cube size={23} weight="duotone" /></span><span>ML FORGE<small>模型工程工作台</small></span></a><div className="header-divider" /><div className="project-identity"><span>{project?.name || '模型工作区'}<Badge>本地项目</Badge></span><small>{project ? `r${project.revision} · ${saveState}` : '连接项目服务'}</small></div><div className="header-actions"><span className={`service-status ${offline ? 'offline' : ''}`} title={offline ? '本地服务离线' : `PyTorch ${health?.torch_version || ''}`}><i />{offline ? '服务离线' : health ? '本地服务已连接' : '连接中'}</span><Badge><Cpu size={12} /> CPU</Badge><Button icon={SlidersHorizontal} title="实验配置" aria-label="实验配置" onClick={() => setConfigOpen(true)} /><Button icon={CheckCircle} disabled={!project || validating} onClick={() => validate()} className="validate-button">{validating ? '校验中' : '校验'}</Button>{busy ? <Button icon={Stop} className="run-button cancel" onClick={cancelRun} disabled={starting || run?.status === 'cancel_requested'}>{starting ? '准备中' : run?.status === 'cancel_requested' ? '正在取消' : '停止实验'}</Button> : <Button icon={Play} className="run-button" onClick={startRun} disabled={!project}>运行实验</Button>}</div></header>
    <div className="workspace">
      <Library project={project} scopeId={scopeId} selectedInstance={selectedInstance} selectedId={selectedId} navigate={(id, callId) => navigate(id, callId, true)} select={(id) => { setScopeId(null); setTrail([]); selectOnly(id); }} addPrimitive={addPrimitive} addIO={addIO} newChip={() => openChip('template')} />
      <main className="canvas-panel"><div className="canvas-toolbar"><div className="breadcrumbs"><button onClick={() => navigate(null)}><GitBranch size={14} /> 模型</button><CaretRight size={12} />{trail.length ? trail.map((entry, index) => <React.Fragment key={`${index}-${entry.id}`}><button onClick={() => navigateCrumb(index)} title={entry.callId || '仅定义：未选择调用上下文'}><Cube size={13} />{entry.name}</button>{index < trail.length - 1 && <CaretRight size={12} />}</React.Fragment>) : <span>根计算图</span>}</div><div className="file-actions">{scopeId ? <Button icon={Cube} title="将当前内部图保存为一个新的芯片定义" onClick={() => openChip('copy')}>另存芯片</Button> : <Button icon={Cube} title="按住 Shift 多选调用，然后封装为新芯片" disabled={!selectedNodeIds.some((id) => id.startsWith('call:'))} onClick={() => openChip('selection')}>封装</Button>}<span className="toolbar-divider" /><Button icon={FolderOpen} title="打开项目 JSON" aria-label="打开项目 JSON" onClick={() => fileInput.current.click()} /><Button icon={FloppyDisk} title="保存项目 JSON" aria-label="保存项目 JSON" onClick={saveJSON} disabled={!project} /><span className="toolbar-divider" /><Button icon={Code} title="Python 双向编辑" disabled={!project} onClick={openPython}>Python</Button></div></div><div className="canvas-context"><div>{scopeId && <Button icon={ArrowLeft} title="返回模型" onClick={() => trail.length > 1 ? navigateCrumb(trail.length - 2) : navigate(null)} />}<span>{scopeId ? '芯片内部计算图' : `${project?.inputs.length || 0} 路输入 · 有向计算图`}</span>{scopeId && <Badge tone="blue">定义 · {definition?.version}</Badge>}{scopeId && <Badge tone={trail.at(-1)?.callId ? 'blue' : 'amber'}>{trail.at(-1)?.callId || '未选择调用上下文'}</Badge>}</div><div className="canvas-validation">{project && <span className="revision">设计 r{project.revision}</span>}{currentValidation && (currentValidation.valid ? <span className="valid-state"><CheckCircle size={13} /> 结构有效</span> : <button onClick={() => setTab('console')} className="error-state"><WarningCircle size={13} /> {currentValidation.errors?.length} 个问题</button>)}</div></div>
        <div className="graph-area">{project ? <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} onInit={handleFlowInit} onNodesChange={handleNodesChange} onEdgesChange={handleEdgesChange} onNodeClick={handleNodeClick} onPaneClick={handlePaneClick} selectionOnDrag multiSelectionKeyCode="Shift" onNodeDoubleClick={handleNodeDoubleClick} onNodeDragStop={handleNodeDragStop} onConnect={handleConnect} onNodesDelete={handleNodesDelete} onEdgesDelete={handleEdgesDelete} fitView fitViewOptions={FIT_VIEW_OPTIONS} minZoom={0.15} maxZoom={1.8} deleteKeyCode={DELETE_KEYS} colorMode="dark" proOptions={PRO_OPTIONS}><Background color="#334450" gap={22} size={1} /><Controls showInteractive={false} position="bottom-left" /></ReactFlow> : <div className="canvas-empty"><Cube size={38} weight="thin" /><h2>{loading ? '正在载入模型…' : '连接你的模型工作区'}</h2><p>{loading ? '读取项目结构与芯片定义' : '本地服务连接后，即可载入示例或打开项目 JSON。'}</p>{!loading && <div><Button icon={ArrowClockwise} onClick={() => { fetchHealth(); loadExample(); }}>重试连接</Button><Button icon={FolderOpen} onClick={() => fileInput.current.click()}>打开 JSON</Button></div>}</div>}<div className="canvas-legend"><span><i className="legend-chip" /> 可复用芯片</span><span><i className="legend-primitive" /> 基础算子</span><span><i className="legend-output" /> 张量接口</span></div>{scopeId && <div className="definition-banner"><Info size={14} /><span>正在编辑共享定义，修改会同步到此芯片的所有实例。</span></div>}{project && <div className="canvas-tip">{scopeId ? '拖动端口连接 · 修改内部算子以更新芯片定义' : '双击芯片进入内部 · 拖动端口连接'}<button title="适应画布" aria-label="适应画布" onClick={fit}><CornersOut size={15} /></button></div>}</div>
      </main>
      <Inspector project={project} scope={scope} scopeId={scopeId} selectedId={selectedId} setSelectedId={selectOnly} selectedCall={selectedCall} selectedInstance={selectedInstance} change={change} validation={currentValidation} navigate={navigate} loadExample={loadExample} addShared={addShared} deleteSelected={deleteSelected} notify={setNotice} renamePort={renamePort} updatePublic={(id, changes) => { try { setProject(updatePublicParameter(project, scopeId, id, changes)); return true; } catch (error) { setNotice(error.message); return false; } }} />
      <ResultPanel tab={tab} setTab={setTab} run={run} result={result} resultRevision={resultRevision} stale={stale} project={project} validation={currentValidation} scopeId={scopeId} probePrefix={probePrefixForTrail(trail)} selectedCall={selectedCall} selectedId={selectedId} scope={scope} busy={busy} config={config} />
    </div>
    {notice && <div className={`notice ${offline ? 'error' : ''}`} role="status"><Info size={16} /><span>{notice}</span>{offline && <button onClick={() => { fetchHealth(); validate(); }}>重试</button>}<button title="关闭消息" onClick={() => setNotice('')}><X size={14} /></button></div>}
    <footer className="statusbar"><div><span className={`status-dot ${offline ? 'offline' : ''}`} />{offline ? '本地服务不可用' : '本地工作区'}<span className="status-separator" />{scopeId ? definition?.name : '根计算图'}<span className="status-separator" />{scope?.calls.length || 0} 个调用</div><div><span>{project ? saveState : '未载入项目'}</span><span className="status-separator" /><span>CPU / float32</span><span className="status-separator" /><span>S1 v0.1</span></div></footer>
    {chipDraft && <Modal title={chipDraft.mode === 'selection' ? '封装所选调用' : chipDraft.mode === 'copy' ? '保存为新芯片' : '新建自定义芯片'} eyebrow="REUSABLE CHIP DEFINITION" onClose={() => setChipDraft(null)} className="config-modal"><div className="modal-body"><p className="config-description">{chipDraft.mode === 'selection' ? `将所选 ${chipDraft.callIds.length} 个调用保存为独立芯片。跨越边界的张量成为命名输入；为芯片指定一个输出节点。` : chipDraft.mode === 'copy' ? '保存当前芯片内部结构、公开参数和绑定映射，创建一个独立的新定义。' : '从 Linear → ReLU 模板创建。进入芯片后可增删算子、修改接口和连接，自定义内部结构。'}</p><Field label="芯片名称"><input aria-label="芯片名称" value={chipDraft.name} maxLength={64} onChange={(event) => setChipDraft((current) => ({ ...current, name: event.target.value }))} /></Field><Field label="定义 ID" hint="字母 / 数字 / 下划线"><input aria-label="芯片定义 ID" value={chipDraft.id} maxLength={64} onChange={(event) => setChipDraft((current) => ({ ...current, id: event.target.value }))} /></Field>{chipDraft.mode === 'selection' && <Field label="芯片的唯一输出" hint="tensor"><select value={chipDraft.outputCallId} onChange={(event) => setChipDraft((current) => ({ ...current, outputCallId: event.target.value }))}>{chipDraft.callIds.map((id) => <option value={id} key={id}>call:{id}</option>)}</select></Field>}{chipDraft.mode === 'template' && <NumericField label="初始输入维度" name="in_features" value={chipDraft.inputFeatures} min={1} onChange={(value) => setChipDraft((current) => ({ ...current, inputFeatures: Math.max(1, Math.round(value)) }))} />}<div className="config-note"><Cube size={18} /><div><strong>保存定义 · 新增根图实例</strong><span>现有计算图保留。新芯片拥有独立的模型实例；之后可添加共享调用。</span></div></div>{chipError && <div className="inline-error" role="alert"><WarningCircle size={14} />{chipError}</div>}</div><footer><Button onClick={() => setChipDraft(null)}>取消</Button><Button icon={Cube} className="primary-button" onClick={saveChip}>保存并新增实例</Button></footer></Modal>}
    {configOpen && <Modal title="实验配置" eyebrow="LOCAL CPU EXPERIMENT" onClose={() => setConfigOpen(false)} className="config-modal"><div className="modal-body"><p className="config-description">使用合成输入验证模型的前向、梯度与参数更新。固定随机种子以便复现。</p><div className="config-fields"><NumericField label="训练步数" name="steps · 1–200" value={config.steps} min={1} max={200} onChange={(value) => setConfig((current) => ({ ...current, steps: Math.max(1, Math.min(200, Math.round(value))) }))} /><NumericField label="学习率" name="learning rate" value={config.lr} min={0.001} max={1} step={0.001} onChange={(value) => setConfig((current) => ({ ...current, lr: Math.max(0.001, Math.min(1, value)) }))} /><NumericField label="随机种子" name="seed" value={config.seed} min={0} max={2147483647} onChange={(value) => setConfig((current) => ({ ...current, seed: Math.max(0, Math.min(2147483647, Math.round(value))) }))} /><NumericField label="批次大小" name="batch size · 2–64" value={config.batch_size} min={2} max={64} onChange={(value) => setConfig((current) => ({ ...current, batch_size: Math.max(2, Math.min(64, Math.round(value))) }))} /></div><div className="config-note"><Cpu size={18} /><div><strong>本地 CPU · synthetic_s1</strong><span>每个根图输入生成 float32 张量，使用合成二分类标签训练。</span></div></div></div><footer><Button onClick={() => setConfigOpen(false)}>完成配置</Button><Button icon={Play} className="run-button" disabled={busy || !project} onClick={() => { setConfigOpen(false); startRun(); }}>运行实验</Button></footer></Modal>}
    {pythonOpen && <Modal title="Python 与画布" eyebrow="S1 PYTORCH · 双向编辑" onClose={closePython} closeDisabled={pythonBusy} className="python-modal">
      <div className="python-toolbar"><span><Code size={14} /><code>model.py</code>{pythonDirty && <Badge tone="amber">有修改 · 尚未同步</Badge>}{!pythonDirty && pythonSession.sourceKey && pythonSession.sourceKey !== key && <Badge tone="amber">保留的旧画布草稿</Badge>}</span><div><Button icon={ArrowClockwise} onClick={requestPythonRegeneration} disabled={pythonBusy}>从画布重新生成</Button><Button icon={UploadSimple} onClick={() => pythonInput.current.click()} disabled={pythonBusy}>打开 .py</Button><Button icon={DownloadSimple} onClick={() => download(`${project?.id || 'model'}.py`, pythonCode, 'text/x-python')} disabled={!pythonCode || pythonBusy}>导出 .py</Button></div></div>
      {pythonSession.confirmRegenerate && <div className="python-confirm" role="alert"><span>重新生成将替换当前 Python 草稿。生成失败时会保留原文。</span><Button onClick={() => dispatchPython({ type: 'KEEP_DRAFT' })}>保留草稿</Button><Button className="primary-button" onClick={regeneratePython}>替换并重新生成</Button></div>}
      <div className="code-editor"><div className="line-numbers" aria-hidden="true">{pythonCode.split('\n').map((_, index) => <span key={index}>{index + 1}</span>)}</div><textarea aria-label="Python 模型代码" value={pythonCode} readOnly={pythonBusy} aria-busy={pythonBusy} onChange={(event) => dispatchPython({ type: 'EDIT', code: event.target.value })} spellCheck={false} placeholder={pythonBusy ? '正在处理，请稍候…' : '打开 .py 文件，或在此编辑导出的模型代码'} /></div>
      {pythonError && <div className="python-error" role="alert"><WarningCircle size={15} />{pythonError}</div>}<div className="python-limits"><Info size={15} /><span>S1 导入支持该配置的 PyTorch 构造器和显式 forward 计算。任意 Python、动态控制流与不支持的算子会被拒绝；导入仅解析代码，不执行代码。</span></div><footer><span>{pythonBusy ? '请求处理中，编辑与关闭暂时锁定。' : '关闭后保留代码草稿；重新解析才会更新画布。'}</span><Button icon={ArrowClockwise} className="primary-button" disabled={pythonBusy || !pythonCode} onClick={importPython}>{pythonBusy ? '处理中…' : '重新解析到画布'}</Button></footer>
    </Modal>}
  </div>;
}








