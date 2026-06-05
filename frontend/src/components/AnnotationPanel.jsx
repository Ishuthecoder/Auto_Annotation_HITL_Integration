import React, { useState } from 'react'
import styles from './AnnotationPanel.module.css'
import { postEdit, postDelete, postUndoDelete } from '../api/client'

const COLORS = ['#4f8ef7','#00e5a0','#f59e0b','#ef4444','#a855f7','#ec4899','#06b6d4','#84cc16','#f97316','#e11d48']

const MATERIALS = [
  "Unknown", "PET", "HDPE", "PVC", "LLDPE", "LDPE", "HM - LD",
  "PP_Blow_moulded", "PP_Injection moulded", "PP", "PS", "PC", "HIPS",
  "MLP-Aluminized", "MLP", "Raffia", "Newspaper", "Magazine Paper",
  "Paper", "Print", "Greyboard", "Cardboard", "Steel", "Aluminum",
  "Textile", "WEEE", "Moulded-pulp", "Mixed-plastic", "Glass", "Organic"
];

const COLORS_OPTS = [
  "Unapplicable", "Mixed", "Clear transparent", "White transparent",
  "Blue transparent", "Green transparent", "Brown transparent",
  "Red transparent", "Coloured transparent", "White opaque",
  "Blue opaque", "Green opaque", "Brown opaque", "Yellow opaque",
  "Black opaque", "Coloured opaque", "Grey opaque", "Red Opaque",
  "Silver opaque"
];

export default function AnnotationPanel({
  item, selectedAnnId, onSelectAnn, onDeleteAnn, categories, drawMode,
  onToggleDrawMode, onDrawCategoryChange, drawCategory,
  onRefresh, toast,
}) {
  const [busy, setBusy] = useState(false)

  const catMap = Object.fromEntries(categories.map(c => [c.id, c.name]))
  const ann = item?.annotations?.find(a => a.id === selectedAnnId)

  async function handleEdit(newCatId, newAttrs) {
    if (!ann) return
    setBusy(true)
    try {
      await postEdit(ann.id, parseInt(newCatId), newAttrs || ann.attributes || {})
      toast('Label updated', 'ok')
      onRefresh()
    } catch (e) {
      toast('Edit failed: ' + e.message, 'err')
    } finally { setBusy(false) }
  }

  async function handleDelete() {
    if (!ann) return
    setBusy(true)
    try {
      await onDeleteAnn(ann.id)
    } finally { setBusy(false) }
  }

  async function handleUndoDelete() {
    if (!ann) return
    setBusy(true)
    try {
      await postUndoDelete(ann.id)
      toast('Deletion undone', 'info')
      onRefresh()
    } catch (e) {
      toast('Undo failed: ' + e.message, 'err')
    } finally { setBusy(false) }
  }

  return (
    <aside className={styles.panel}>
      <div className={styles.section}>
        <div className={styles.sectionTitle}>TOOLS</div>
        <button
          className={`btn ${drawMode ? 'btn-warn' : 'btn-ghost'}`}
          style={{ width: '100%', marginBottom: 8 }}
          onClick={onToggleDrawMode}
        >
          {drawMode ? '✏️ Drawing… (click to cancel)' : '+ Draw Polygon'}
        </button>
        {drawMode && (
          <div style={{ marginBottom: 8 }}>
            <div className={styles.label}>New annotation category</div>
            <select value={drawCategory} onChange={e => onDrawCategoryChange(parseInt(e.target.value))}>
              {categories.map(c => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>
          </div>
        )}
      </div>

      <div className={styles.divider} />

      <div className={styles.section}>
        <div className={styles.sectionTitle}>SELECTED ANNOTATION</div>

        {!ann ? (
          <div className={styles.empty}>Click a polygon to select it</div>
        ) : (
          <>
            <div className={styles.annMeta}>
              <span className={styles.annId}>#{ann.id}</span>
              {ann.corrected && <span className={styles.tag} style={{ background: 'rgba(245,158,11,0.2)', color: '#f59e0b' }}>edited</span>}
              {ann.added    && <span className={styles.tag} style={{ background: 'rgba(79,142,247,0.2)', color: '#4f8ef7' }}>added</span>}
            </div>

            <div className={styles.label}>Category</div>
            <div className={styles.colorRow}>
              <span className={styles.dot} style={{ background: COLORS[ann.category_id % COLORS.length] }} />
              <select
                value={ann.category_id}
                disabled={busy}
                onChange={e => handleEdit(e.target.value, ann.attributes)}
              >
                {categories.map(c => (
                  <option key={c.id} value={c.id}>{c.name}</option>
                ))}
              </select>
            </div>

            <div className={styles.label} style={{ marginTop: '12px' }}>Material</div>
            <select
              value={ann.attributes?.Material || "Unknown"}
              disabled={busy}
              onChange={e => handleEdit(ann.category_id, { ...(ann.attributes || {}), Material: e.target.value })}
              style={{ width: '100%', padding: '4px' }}
            >
              {MATERIALS.map(m => <option key={m} value={m}>{m}</option>)}
            </select>

            <div className={styles.label} style={{ marginTop: '12px' }}>Color</div>
            <select
              value={ann.attributes?.Color || "Unapplicable"}
              disabled={busy}
              onChange={e => handleEdit(ann.category_id, { ...(ann.attributes || {}), Color: e.target.value })}
              style={{ width: '100%', padding: '4px' }}
            >
              {COLORS_OPTS.map(c => <option key={c} value={c}>{c}</option>)}
            </select>

            {ann.segmentation?.length > 0 && (
              <div className={styles.metaRow}>
                <span>{ann.segmentation[0].length / 2} points</span>
              </div>
            )}

            <div className={styles.actions}>
              <button
                className="btn btn-danger"
                style={{ flex: 1 }}
                disabled={busy}
                onClick={handleDelete}
              >
                🗑 Delete
              </button>
            </div>
          </>
        )}
      </div>

      <div className={styles.divider} />

      {/* Annotation list for current image */}
      <div className={styles.section} style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div className={styles.sectionTitle}>
          ANNOTATIONS ({(item?.annotations || []).length})
        </div>
        <div className={styles.annList}>
          {(item?.annotations || []).map(a => (
            <div
              key={a.id}
              className={`${styles.annItem} ${a.id === selectedAnnId ? styles.annItemActive : ''}`}
              onClick={() => onSelectAnn(a.id === selectedAnnId ? null : a.id)}
            >
              <span className={styles.dot} style={{ background: COLORS[a.category_id % COLORS.length] }} />
              <span className={styles.annItemName}>{catMap[a.category_id] ?? a.category_id}</span>
              {a.corrected && <span className={styles.dot} style={{ background: '#f59e0b', width: 6, height: 6 }} title="edited" />}
              {a.added     && <span className={styles.dot} style={{ background: '#4f8ef7', width: 6, height: 6 }} title="added" />}
              <button
                className={styles.annDeleteBtn}
                title="Delete polygon"
                onClick={(e) => { e.stopPropagation(); onDeleteAnn(a.id) }}
              >✕</button>
            </div>
          ))}
        </div>
      </div>
    </aside>
  )
}
