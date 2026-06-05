import React, { useState, useEffect, useCallback } from 'react'
import BatchList from './components/BatchList'
import ReviewPanel from './components/ReviewPanel'
import { getInfo, getBatch } from './api/client'
import styles from './App.module.css'

export default function App() {
  const [info,         setInfo]        = useState(null)
  const [currentBatch, setCurrentBatch] = useState(null)
  const [batchData,    setBatchData]   = useState(null)
  const [loading,      setLoading]     = useState(false)
  const [toasts,       setToasts]      = useState([])

  const toast = useCallback((msg, type = 'ok') => {
    const id = Date.now()
    setToasts(ts => [...ts, { id, msg, type }])
    setTimeout(() => setToasts(ts => ts.filter(t => t.id !== id)), 3500)
  }, [])

  const refreshInfo = useCallback(async () => {
    try { setInfo(await getInfo()) }
    catch (e) { toast('Cannot reach backend — is the server running?', 'err') }
  }, [toast])

  useEffect(() => { refreshInfo() }, [refreshInfo])

  const loadBatch = useCallback(async (idx) => {
    setLoading(true)
    setCurrentBatch(idx)
    setBatchData(null)
    try {
      const data = await getBatch(idx)
      setBatchData(data)
    } catch (e) {
      toast('Failed to load batch: ' + e.message, 'err')
    } finally { setLoading(false) }
  }, [toast])

  const handleBatchUpdate = useCallback(async () => {
    await refreshInfo()
    if (currentBatch !== null) {
      try { setBatchData(await getBatch(currentBatch)) }
      catch {}
    }
  }, [refreshInfo, currentBatch])

  const categories = info?.categories ?? []

  return (
    <div className={styles.app}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.logo}>HITL<span>/</span>REVIEW</div>
        <div className={styles.stats}>
          {info ? (
            <>
              <div className={styles.stat}><strong>{info.total_images.toLocaleString()}</strong> Images</div>
              <div className={styles.stat}><strong>{info.total_annotations.toLocaleString()}</strong> Anns</div>
              <div className={styles.stat}><strong>{info.total_batches}</strong> Batches</div>
              <div className={styles.stat}><strong>{(info.completed_batches?.length ?? 0)}</strong> Done</div>
              <div className={styles.stat}><strong>{info.total_edits ?? 0}</strong> Edits</div>
              <div className={styles.stat}><strong>{info.total_deletions ?? 0}</strong> Del</div>
              <div className={styles.stat}><strong>{info.total_additions ?? 0}</strong> Add</div>
            </>
          ) : (
            <span className={styles.statMuted}>Connecting to backend…</span>
          )}
        </div>
      </header>

      {/* Body */}
      <div className={styles.body}>
        <BatchList
          info={info}
          currentBatch={currentBatch}
          onSelect={loadBatch}
        />

        {loading ? (
          <div className={styles.loading}>
            <div className={styles.spinner} />
            <span>Loading batch…</span>
          </div>
        ) : (
          <ReviewPanel
            batchData={batchData}
            categories={categories}
            onBatchUpdate={handleBatchUpdate}
            toast={toast}
          />
        )}
      </div>

      {/* Toasts */}
      <div className="toasts">
        {toasts.map(t => (
          <div key={t.id} className={`toast toast-${t.type}`}>{t.msg}</div>
        ))}
      </div>
    </div>
  )
}
