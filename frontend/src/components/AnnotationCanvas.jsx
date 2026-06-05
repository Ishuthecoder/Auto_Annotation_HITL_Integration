import React, { useRef, useEffect, useState, useCallback } from 'react'
import { Stage, Layer, Image as KonvaImage, Line, Circle, Group } from 'react-konva'
import useImage from 'use-image'
import { getImageUrl } from '../api/client'
import styles from './AnnotationCanvas.module.css'

const COLORS = [
  '#4f8ef7','#00e5a0','#f59e0b','#ef4444','#a855f7',
  '#ec4899','#06b6d4','#84cc16','#f97316','#e11d48',
]

function polyColor(catId, alpha = 0.3) {
  const hex = COLORS[catId % COLORS.length]
  const r = parseInt(hex.slice(1,3),16)
  const g = parseInt(hex.slice(3,5),16)
  const b = parseInt(hex.slice(5,7),16)
  return `rgba(${r},${g},${b},${alpha})`
}

function ImageLayer({ imageId, width, height }) {
  const [img] = useImage(getImageUrl(imageId), 'anonymous')
  if (!img) return null
  return <KonvaImage image={img} width={width} height={height} />
}

export default function AnnotationCanvas({
  item, categories, selectedAnnId, onSelectAnn,
  drawMode, onDrawComplete,
}) {
  const containerRef = useRef(null)
  const [dims, setDims] = useState({ w: 800, h: 500 })
  const [drawPoints, setDrawPoints] = useState([])
  const [mousePos, setMousePos]     = useState(null)

  // Resize observer
  useEffect(() => {
    if (!containerRef.current) return
    const ro = new ResizeObserver(([e]) => {
      const { width } = e.contentRect
      if (!item) return
      const ratio = item.height / item.width
      setDims({ w: Math.floor(width), h: Math.floor(width * ratio) })
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [item])

  useEffect(() => {
    if (!containerRef.current || !item) return
    const { width } = containerRef.current.getBoundingClientRect()
    const ratio = item.height / item.width
    setDims({ w: Math.floor(width), h: Math.floor(width * ratio) })
  }, [item])

  // Reset draw when leaving draw mode
  useEffect(() => { if (!drawMode) setDrawPoints([]) }, [drawMode])

  const scaleX = dims.w / (item?.width  || 1)
  const scaleY = dims.h / (item?.height || 1)

  // Keep a ref to always read fresh points in dblclick handler (avoids stale closure)
  const drawPointsRef = useRef([])
  useEffect(() => { drawPointsRef.current = drawPoints }, [drawPoints])

  // Debounce single clicks so double-click doesn't add 2 extra points
  const clickTimer = useRef(null)

  const handleStageClick = useCallback((e) => {
    if (!drawMode) return
    // If this click is part of a dblclick, the timer will be cleared before firing
    const pos = e.target.getStage().getPointerPosition()
    const x = pos.x / scaleX
    const y = pos.y / scaleY
    if (clickTimer.current) clearTimeout(clickTimer.current)
    clickTimer.current = setTimeout(() => {
      clickTimer.current = null
      setDrawPoints(prev => [...prev, x, y])
    }, 220) // 220 ms — shorter than typical dblclick interval
  }, [drawMode, scaleX, scaleY])

  const handleStageDblClick = useCallback((e) => {
    if (!drawMode) return
    // Cancel the pending single-click so no extra point is added
    if (clickTimer.current) {
      clearTimeout(clickTimer.current)
      clickTimer.current = null
    }
    e.evt.preventDefault()
    const pts = drawPointsRef.current
    if (pts.length >= 6) {
      onDrawComplete(pts)
    }
    setDrawPoints([])
  }, [drawMode, onDrawComplete])

  const handleMouseMove = useCallback((e) => {
    if (!drawMode) return
    const pos = e.target.getStage().getPointerPosition()
    setMousePos(pos)
  }, [drawMode])

  const catMap = Object.fromEntries(categories.map(c => [c.id, c.name]))

  if (!item) return <div className={styles.empty}>Select a batch to start reviewing</div>

  return (
    <div ref={containerRef} className={styles.container}>
      <Stage
        width={dims.w}
        height={dims.h}
        onClick={handleStageClick}
        onDblClick={handleStageDblClick}
        onMouseMove={handleMouseMove}
        style={{ cursor: drawMode ? 'crosshair' : 'default' }}
      >
        <Layer>
          <ImageLayer imageId={item.image_id} width={dims.w} height={dims.h} />
        </Layer>

        {/* Annotations layer */}
        <Layer>
          {(item.annotations || []).map(ann => {
            const isSelected = ann.id === selectedAnnId
            const color = polyColor(ann.category_id, isSelected ? 0.45 : 0.25)
            const stroke = COLORS[ann.category_id % COLORS.length]

            return (ann.segmentation || []).map((seg, si) => {
              if (!seg || seg.length < 6) return null
              const pts = []
              for (let i = 0; i < seg.length; i += 2) {
                pts.push(seg[i] * scaleX, seg[i + 1] * scaleY)
              }
              return (
                <Line
                  key={`${ann.id}-${si}`}
                  points={pts}
                  closed
                  fill={color}
                  stroke={stroke}
                  strokeWidth={isSelected ? 2.5 : 1.5}
                  dash={ann.added ? [6, 3] : undefined}
                  onClick={(e) => { e.cancelBubble = true; onSelectAnn(ann.id) }}
                  onMouseEnter={e => { e.target.getStage().container().style.cursor = 'pointer' }}
                  onMouseLeave={e => { if (!drawMode) e.target.getStage().container().style.cursor = 'default' }}
                />
              )
            })
          })}
        </Layer>

        {/* Draw mode layer */}
        {drawMode && drawPoints.length > 0 && (
          <Layer>
            <Line
              points={[
                ...drawPoints.map((p, i) => i % 2 === 0 ? p * scaleX : p * scaleY),
                ...(mousePos ? [mousePos.x, mousePos.y] : []),
              ]}
              stroke="#f59e0b"
              strokeWidth={2}
              dash={[5, 3]}
            />
            {Array.from({ length: drawPoints.length / 2 }, (_, i) => (
              <Circle
                key={i}
                x={drawPoints[i * 2] * scaleX}
                y={drawPoints[i * 2 + 1] * scaleY}
                radius={4}
                fill="#f59e0b"
                stroke="#fff"
                strokeWidth={1}
              />
            ))}
          </Layer>
        )}
      </Stage>

      {drawMode && (
        <div className={styles.drawHint}>
          ✏️ Click to place points &nbsp;·&nbsp; Double-click to finish
          {drawPoints.length >= 6 && <span className={styles.ready}> · Ready ({drawPoints.length / 2} pts)</span>}
        </div>
      )}
    </div>
  )
}
