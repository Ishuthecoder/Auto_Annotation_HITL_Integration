import React, { useState, useCallback, useRef, useEffect } from 'react'
import AnnotationCanvas from './AnnotationCanvas'
import AnnotationPanel from './AnnotationPanel'
import { postAdd, postApprove, postUnapprove, postDelete } from '../api/client'
import styles from './ReviewPanel.module.css'

export default function ReviewPanel({ batchData, categories, onBatchUpdate, toast }) {
  const [imageIdx,       setImageIdx]     = useState(0)
  const [selectedAnn,    setSelectedAnn]  = useState(null)
  const [drawMode,       setDrawMode]     = useState(false)
  const [drawCatId,      setDrawCatId]    = useState(categories[0]?.id ?? 0)
  const [refreshKey,     setRefreshKey]   = useState(0)
  const [busy,           setBusy]         = useState(false)
  // IDs removed locally before server confirms — gives instant visual feedback
  const [localDeleted,   setLocalDeleted] = useState(new Set())

  const rawItem = batchData?.items?.[imageIdx] ?? null
  const total   = batchData?.items?.length ?? 0

  // Filter out locally-deleted annotations so polygon disappears instantly
  const item = rawItem ? {
    ...rawItem,
    annotations: (rawItem.annotations || []).filter(a => !localDeleted.has(a.id))
  } : null

  const refresh = useCallback(() => setRefreshKey(k => k + 1), [])

  // ── Delete annotation — polygon disappears instantly, server syncs in background ──
  const handleDeleteAnn = useCallback(async (annId) => {
    if (annId == null) return
    // Immediately hide the polygon — don't wait for server
    setLocalDeleted(prev => new Set([...prev, annId]))
    setSelectedAnn(null)
    try {
      await postDelete(annId)
      toast('Annotation deleted', 'ok')
      // Sync server counts (header stats) — do NOT clear localDeleted here
      // because React may not have flushed the new batchData prop yet,
      // which would cause the polygon to flash back momentarily.
      onBatchUpdate()
    } catch (e) {
      // Rollback the local hide only if the server actually rejected it
      setLocalDeleted(prev => { const s = new Set(prev); s.delete(annId); return s })
      toast('Delete failed: ' + e.message, 'err')
    }
  }, [toast, onBatchUpdate])

  useEffect(() => {
    const handler = (e) => {
      if (!selectedAnn) return
      // Don't fire when typing inside an input / select / textarea
      const tag = document.activeElement?.tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault()
        handleDeleteAnn(selectedAnn)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [selectedAnn, handleDeleteAnn])

  const handleDrawComplete = useCallback(async (points) => {
    if (!item) return
    const seg = [points]
    try {
      await postAdd(item.image_id, seg, drawCatId)
      toast('Annotation added', 'ok')
      setDrawMode(false)
      onBatchUpdate()
    } catch (e) {
      toast('Add failed: ' + e.message, 'err')
    }
  }, [item, drawCatId, toast, onBatchUpdate])

  const handleApprove = async () => {
    setBusy(true)
    try {
      await postApprove(batchData.batch_idx)
      toast(`Batch ${batchData.batch_idx + 1} approved ✓`, 'ok')
      onBatchUpdate()
    } catch (e) {
      toast('Approve failed: ' + e.message, 'err')
    } finally { setBusy(false) }
  }

  const handleUnapprove = async () => {
    setBusy(true)
    try {
      await postUnapprove(batchData.batch_idx)
      toast(`Batch ${batchData.batch_idx + 1} re-opened`, 'info')
      onBatchUpdate()
    } catch (e) {
      toast('Unapprove failed: ' + e.message, 'err')
    } finally { setBusy(false) }
  }

  const navigate = (delta) => {
    setImageIdx(i => Math.max(0, Math.min(total - 1, i + delta)))
    setSelectedAnn(null)
    setDrawMode(false)
    setLocalDeleted(new Set())  // clear per-image local state
  }

  if (!batchData) {
    return (
      <div className={styles.welcome}>
        <div className={styles.welcomeIcon}>👁</div>
        <div>Select a batch from the left to begin reviewing</div>
      </div>
    )
  }

  return (
    <div className={styles.wrapper}>
      {/* Toolbar */}
      <div className={styles.toolbar}>
        <span className={styles.toolbarInfo}>
          Batch <strong>{batchData.batch_idx + 1}</strong> / {batchData.total_batches}
          &nbsp;·&nbsp;
          Image <strong>{imageIdx + 1}</strong> / {total}
          {batchData.completed && <span className={styles.approved}> · ✓ APPROVED</span>}
        </span>
        <div className={styles.toolbarActions}>
          <button className="btn btn-ghost" disabled={imageIdx === 0} onClick={() => navigate(-1)}>◀ Prev</button>
          <button className="btn btn-ghost" disabled={imageIdx === total - 1} onClick={() => navigate(1)}>Next ▶</button>
          {batchData.completed
            ? <button className="btn btn-ghost" disabled={busy} onClick={handleUnapprove}>Re-open</button>
            : <button className="btn btn-success" disabled={busy} onClick={handleApprove}>✅ Approve Batch</button>
          }
        </div>
      </div>

      {/* Main area */}
      <div className={styles.main}>
        <div className={styles.canvasArea}>
          {item && (
            <div className={styles.imageMeta}>
              <span>{item.file_name?.split('/').pop()}</span>
              <span>{item.width} × {item.height}  ·  {(item.annotations||[]).length} annotations</span>
            </div>
          )}
          <AnnotationCanvas
            key={`${item?.image_id}-${refreshKey}`}
            item={item}
            categories={categories}
            selectedAnnId={selectedAnn}
            onSelectAnn={setSelectedAnn}
            drawMode={drawMode}
            onDrawComplete={handleDrawComplete}
          />
        </div>

        <AnnotationPanel
          item={item}
          selectedAnnId={selectedAnn}
          onSelectAnn={setSelectedAnn}
          onDeleteAnn={handleDeleteAnn}
          categories={categories}
          drawMode={drawMode}
          onToggleDrawMode={() => { setDrawMode(d => !d); setSelectedAnn(null) }}
          drawCategory={drawCatId}
          onDrawCategoryChange={setDrawCatId}
          onRefresh={() => { refresh(); onBatchUpdate() }}
          toast={toast}
        />
      </div>
    </div>
  )
}
