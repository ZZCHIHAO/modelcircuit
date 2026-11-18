import React, { useEffect, useState } from 'react';
import { X } from '@phosphor-icons/react';
export function Button({ icon: Icon, children, className = '', ...props }) { return <button className={`button ${children ? '' : 'icon-button'} ${className}`} {...props}>{Icon && <Icon size={16} />}{children && <span>{children}</span>}</button>; }
export function Badge({ children, tone = '' }) { return <span className={`badge ${tone}`}>{children}</span>; }
export function Field({ label, hint, children }) { return <label className="field"><span className="field-label">{label}{hint && <code>{hint}</code>}</span>{children}</label>; }
export function NumericField({ label, name, value, onChange, min, max, step = 1 }) {
  const [draft, setDraft] = useState(String(value ?? ''));
  useEffect(() => setDraft(Array.isArray(value) ? JSON.stringify(value) : String(value ?? '')), [value]);
  function commit() { let parsed; try { parsed = draft.startsWith('[') ? JSON.parse(draft) : Number(draft); } catch { setDraft(String(value)); return; } if (draft.trim() && (Array.isArray(parsed) || Number.isFinite(parsed))) onChange(parsed); else setDraft(String(value ?? '')); }
  return <Field label={label} hint={name}><input type={Array.isArray(value) ? 'text' : 'number'} value={draft} min={min} max={max} step={step} onChange={(event) => setDraft(event.target.value)} onBlur={commit} onKeyDown={(event) => { if (event.key === 'Enter') event.currentTarget.blur(); }} /></Field>;
}
export function Modal({ title, eyebrow, children, onClose, closeDisabled = false, className = '' }) { useEffect(() => { const handler = (event) => event.key === 'Escape' && onClose(); window.addEventListener('keydown', handler); return () => window.removeEventListener('keydown', handler); }, [onClose]); return <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}><section className={`modal ${className}`} role="dialog" aria-modal="true" aria-label={title}><header><div><span className="eyebrow">{eyebrow}</span><h2>{title}</h2></div><Button icon={X} title="关闭" onClick={onClose} disabled={closeDisabled} /></header>{children}</section></div>; }
export function EmptyState({ icon: Icon, text, detail }) { return <div className="empty-state"><Icon size={27} weight="thin" /><div><strong>{text}</strong><p>{detail}</p></div></div>; }
export function fmt(value, precision = 4) { return typeof value === 'number' && Number.isFinite(value) ? Number(value.toFixed(precision)).toString() : '—'; }

