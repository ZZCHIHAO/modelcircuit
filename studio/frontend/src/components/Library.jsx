import React, { useState } from 'react';
import { TreeStructure, CaretDown, CaretRight, GitBranch, Cube, SquaresFour, BracketsCurly, ArrowRight, ArrowLeft, LinkSimple, Stack, MagnifyingGlass, Plus } from '@phosphor-icons/react';
import { PRIMITIVES } from '../graph.mjs';
import { hierarchyInstanceTargets } from '../flow-state.mjs';
import { Badge } from './common.jsx';
export function Library({ project, scopeId, selectedInstance, selectedId, navigate, select, addPrimitive, addIO, newChip }) {
  const [query, setQuery] = useState(''), [open, setOpen] = useState(true);
  const match = (text) => text.toLowerCase().includes(query.toLowerCase());
  return <aside className="left-panel">
    <div className="panel-title"><span><TreeStructure size={15} /> 项目层级</span><Badge>{project?.instances?.length || 0}</Badge></div>
    <div className="hierarchy">
      <button className={`tree-item tree-root ${!scopeId && !selectedId ? 'active' : ''}`} onClick={() => navigate(null)}><CaretDown size={12} /><GitBranch size={15} /><span>{project?.name || '模型'}</span><code>root</code></button>
      {project?.instances.map((instance) => {
        const chip = instance.kind === 'chip', calls = project.calls.filter((call) => call.instance_id === instance.id);
        const { inspectionNodeId, internalCallId } = hierarchyInstanceTargets(instance.id, project.calls);
        return <div key={instance.id}>
          <button className={`tree-item tree-child ${selectedInstance?.id === instance.id && !scopeId ? 'active' : ''}`} onClick={() => select(inspectionNodeId)} onDoubleClick={() => chip && navigate(instance.definition_id, internalCallId)} title={calls.length > 1 ? '单击检查共享实例；通过下方具体调用进入内部探针；双击仅编辑定义' : instance.name}>
            {chip ? <Cube size={15} /> : <SquaresFour size={15} />}<span>{instance.name || instance.kind}</span>{calls.length > 1 && <span className="tree-count">×{calls.length}</span>}{chip && <CaretRight size={11} />}
          </button>
          {chip && (calls.length ? calls.map((call) => <button key={call.id} className="tree-item tree-grandchild" title={`${instance.id} / ${call.id} · 进入此调用的内部结构`} onClick={() => navigate(instance.definition_id, call.id)}><span className="tree-branch" /><BracketsCurly size={13} /><span>{call.id} · 内部结构</span></button>) : <button className="tree-item tree-grandchild" onClick={() => navigate(instance.definition_id, null)}><span className="tree-branch" /><BracketsCurly size={13} /><span>内部结构 · 仅定义</span></button>)}
          {!chip && calls.length > 1 && calls.map((call) => <button key={call.id} className="tree-item tree-grandchild" onClick={() => select(`call:${call.id}`)}><span className="tree-branch" /><LinkSimple size={12} /><span>{call.id}</span></button>)}
        </div>;
      })}
      {project && <div className="tree-output"><ArrowRight size={13} /><span>{project.outputs.length} 个模型输出</span></div>}
    </div>
    <div className="hierarchy-note"><LinkSimple size={13} /><span>一次定义，多次调用<br /><small>共享同一组模型参数</small></span></div>
    <button className="panel-title library-heading" onClick={() => setOpen(!open)}><span><Stack size={15} /> 组件库</span>{open ? <CaretDown size={12} /> : <CaretRight size={12} />}</button>
    {open && <div className="library">
      <label className="search"><MagnifyingGlass size={14} /><input placeholder="搜索组件…" value={query} onChange={(event) => setQuery(event.target.value)} aria-label="搜索组件" /><kbd>/</kbd></label>
      <div className="library-section-label">可复用芯片<button className="new-chip" title="新建自定义芯片" aria-label="新建自定义芯片" disabled={!project} onClick={newChip}><Plus size={11} />新建</button></div>
      {project?.definitions?.filter((item) => match(`${item.name}${item.id}`)).map((item) => <button className="library-chip" key={item.id} disabled={!project || scopeId === item.id} onClick={() => addPrimitive('chip', item.id)}><Cube size={18} /><span>{item.name}<small>{item.version} · 参数可绑定</small></span><Plus size={13} /></button>)}
      <div className="library-section-label">基础算子 <span>PYTORCH</span></div><div className="primitive-list">{PRIMITIVES.filter((item) => match(`${item.kind}${item.label}${item.group}`)).map((item) => <button className="library-primitive" key={item.kind} disabled={!project} onClick={() => addPrimitive(item.kind)}><span className={`primitive-glyph kind-${item.kind}`}><SquaresFour size={14} /></span><span>{item.kind}<small>{item.label}</small></span><Plus size={12} /></button>)}</div>
      <div className="library-section-label">图接口</div><div className="io-actions"><button disabled={!project} onClick={() => addIO('input')}><ArrowLeft size={13} /> 输入<Plus size={11} /></button><button disabled={!project} onClick={() => addIO('output')}><ArrowRight size={13} /> 输出<Plus size={11} /></button></div>
    </div>}
    <div className="library-bottom"><span className="version-dot" /> S1 · PyTorch 实验配置</div>
  </aside>;
}
