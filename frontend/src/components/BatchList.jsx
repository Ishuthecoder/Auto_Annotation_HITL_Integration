import React from 'react'
import styles from './BatchList.module.css'

export default function BatchList({ info, currentBatch, onSelect }) {
  if (!info) return <div className={styles.sidebar}><div className={styles.empty}>Loading…</div></div>

  const completed = new Set(info.completed_batches)

  return (
    <aside className={styles.sidebar}>
      <div className={styles.header}>
        <span className={styles.title}>BATCHES</span>
        <span className={styles.count}>{info.total_batches} total</span>
      </div>
      <div className={styles.list}>
        {Array.from({ length: info.total_batches }, (_, i) => {
          const isActive    = i === currentBatch
          const isDone      = completed.has(i)
          const start       = i * info.batch_size + 1
          const end         = Math.min((i + 1) * info.batch_size, info.total_images)
          return (
            <div
              key={i}
              className={`${styles.item} ${isActive ? styles.active : ''} ${isDone ? styles.done : ''}`}
              onClick={() => onSelect(i)}
            >
              <div className={styles.itemLeft}>
                <span className={styles.batchName}>Batch {i + 1}</span>
                <span className={styles.batchSub}>{start.toLocaleString()} – {end.toLocaleString()}</span>
              </div>
              <span className={`${styles.badge} ${isActive ? styles.badgeActive : isDone ? styles.badgeDone : styles.badgePending}`}>
                {isActive ? 'OPEN' : isDone ? '✓' : '—'}
              </span>
            </div>
          )
        })}
      </div>
    </aside>
  )
}
