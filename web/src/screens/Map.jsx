import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'

const CITIES = [
  { name: 'Bangkok', lat: 13.7563, lng: 100.5018 },
  { name: 'Chiang Mai', lat: 18.7883, lng: 98.9853 },
  { name: 'Phuket', lat: 7.8804, lng: 98.3923 },
  { name: 'Khon Kaen', lat: 16.4419, lng: 102.8356 },
  { name: 'Hat Yai', lat: 7.0086, lng: 100.4747 },
  { name: 'Nakhon Ratchasima', lat: 14.9799, lng: 102.0978 },
  { name: 'Udon Thani', lat: 17.4157, lng: 102.7859 },
  { name: 'Chiang Rai', lat: 19.9072, lng: 99.8329 }
]

// Thailand province boundaries (simplified polygon for visual outline)
const THAILAND_OUTLINE = [
  [97.3, 17.8], [97.5, 18.3], [98.0, 18.8], [98.5, 19.5], [99.0, 20.0],
  [99.5, 20.3], [100.0, 20.4], [100.5, 20.2], [101.0, 19.8], [101.5, 19.5],
  [102.0, 19.0], [102.5, 18.5], [103.0, 18.0], [103.5, 17.5], [104.0, 17.0],
  [104.5, 16.5], [105.0, 16.0], [105.5, 15.5], [105.8, 15.0], [105.9, 14.5],
  [105.8, 14.0], [105.5, 13.5], [105.3, 13.0], [105.0, 12.5], [104.5, 12.0],
  [104.0, 11.5], [103.5, 11.0], [103.0, 10.5], [102.5, 10.0], [102.0, 9.5],
  [101.5, 9.0], [101.0, 8.5], [100.5, 8.0], [100.0, 7.5], [99.5, 7.0],
  [99.0, 6.5], [98.5, 6.0], [98.2, 5.8], [98.0, 6.0], [97.8, 6.5],
  [97.5, 7.0], [97.3, 7.5], [97.2, 8.0], [97.3, 8.5], [97.5, 9.0],
  [97.8, 9.5], [98.0, 10.0], [98.2, 10.5], [98.3, 11.0], [98.2, 11.5],
  [98.0, 12.0], [97.8, 12.5], [97.6, 13.0], [97.4, 13.5], [97.3, 14.0],
  [97.2, 14.5], [97.2, 15.0], [97.2, 15.5], [97.2, 16.0], [97.2, 16.5],
  [97.2, 17.0], [97.3, 17.5], [97.3, 17.8]
]

function Map() {
  const [date, setDate] = useState(new Date().toISOString().split('T')[0])
  const [threshold, setThreshold] = useState(38.0)
  const [forecast, setForecast] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [hoveredCell, setHoveredCell] = useState(null)
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const canvasRef = useRef(null)
  const containerRef = useRef(null)

  const fetchForecast = async () => {
    setError(null)
    setLoading(true)
    try {
      const res = await axios.get('http://localhost:8000/api/map/forecast', {
        params: { date, threshold }
      })
      setForecast(res.data)
    } catch (err) {
      console.error('Failed to fetch forecast:', err)
      setError('Failed to load forecast data. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchForecast()
  }, [])

  const getTempColor = useCallback((temp) => {
    const minTemp = 20
    const maxTemp = 42
    const t = Math.max(0, Math.min(1, (temp - minTemp) / (maxTemp - minTemp)))
    const hue = (1 - t) * 240
    return `hsl(${hue}, 75%, 55%)`
  }, [])

  // Canvas rendering
  useEffect(() => {
    if (!forecast || !canvasRef.current) return
    
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)
    
    const width = rect.width
    const height = rect.height
    const bounds = forecast.bounds
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height)
    
    // Fill background
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-base').trim() || '#0c0e12'
    ctx.fillRect(0, 0, width, height)
    
    // Draw Thailand outline
    ctx.beginPath()
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border-strong').trim() || 'rgba(148,163,184,0.2)'
    ctx.lineWidth = 2
    THAILAND_OUTLINE.forEach((pt, i) => {
      const x = ((pt[0] - bounds.lon_min) / (bounds.lon_max - bounds.lon_min)) * width
      const y = ((bounds.lat_max - pt[1]) / (bounds.lat_max - bounds.lat_min)) * height
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.closePath()
    ctx.stroke()
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-elevated').trim() || 'rgba(26,30,42,0.3)'
    ctx.fill()
    
    // Draw heatmap cells
    const cols = new Set(forecast.grid.map(p => p.lon)).size
    const rows = new Set(forecast.grid.map(p => p.lat)).size
    const cellW = width / cols
    const cellH = height / rows
    
    forecast.grid.forEach(point => {
      const x = ((point.lon - bounds.lon_min) / (bounds.lon_max - bounds.lon_min)) * width
      const y = ((bounds.lat_max - point.lat) / (bounds.lat_max - bounds.lat_min)) * height
      
      ctx.fillStyle = getTempColor(point.temp)
      ctx.globalAlpha = 0.7
      ctx.fillRect(x - cellW/2, y - cellH/2, cellW + 1, cellH + 1)
      ctx.globalAlpha = 1
      
      // Heatwave border
      if (point.heatwave) {
        ctx.strokeStyle = '#f43f5e'
        ctx.lineWidth = 1.5
        ctx.strokeRect(x - cellW/2, y - cellH/2, cellW, cellH)
      }
    })
    
    // Draw city markers
    CITIES.forEach(city => {
      const x = ((city.lng - bounds.lon_min) / (bounds.lon_max - bounds.lon_min)) * width
      const y = ((bounds.lat_max - city.lat) / (bounds.lat_max - bounds.lat_min)) * height
      
      // Glow effect
      ctx.beginPath()
      ctx.arc(x, y, 8, 0, Math.PI * 2)
      ctx.fillStyle = 'rgba(255, 255, 255, 0.15)'
      ctx.fill()
      
      // Dot
      ctx.beginPath()
      ctx.arc(x, y, 4, 0, Math.PI * 2)
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim() || '#f1f5f9'
      ctx.fill()
      ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-base').trim() || '#0c0e12'
      ctx.lineWidth = 2
      ctx.stroke()
      
      // Label
      ctx.font = 'bold 11px var(--font-ui), sans-serif'
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-primary').trim() || '#f1f5f9'
      ctx.textAlign = 'center'
      ctx.fillText(city.name, x, y - 12)
    })
    
    // Draw grid lines
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--border-subtle').trim() || 'rgba(148,163,184,0.08)'
    ctx.lineWidth = 0.5
    for (let i = 1; i < cols; i++) {
      const x = (i / cols) * width
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()
    }
    for (let i = 1; i < rows; i++) {
      const y = (i / rows) * height
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }
    
  }, [forecast, getTempColor])

  const handleCanvasHover = (e) => {
    if (!forecast || !canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    const bounds = forecast.bounds
    const width = rect.width
    const height = rect.height
    
    // Find nearest grid point
    let nearest = null
    let minDist = Infinity
    
    forecast.grid.forEach(point => {
      const px = ((point.lon - bounds.lon_min) / (bounds.lon_max - bounds.lon_min)) * width
      const py = ((bounds.lat_max - point.lat) / (bounds.lat_max - bounds.lat_min)) * height
      const dist = Math.sqrt((x - px) ** 2 + (y - py) ** 2)
      if (dist < minDist && dist < 30) {
        minDist = dist
        nearest = point
      }
    })
    
    setHoveredCell(nearest)
    setMousePos({ x: e.clientX, y: e.clientY })
  }

  const handleZoomIn = () => setZoom(prev => Math.min(4, prev + 0.3))
  const handleZoomOut = () => setZoom(prev => Math.max(0.5, prev - 0.3))
  const handleReset = () => { setZoom(1); setPan({ x: 0, y: 0 }) }

  const handleMouseDown = (e) => {
    if (e.button === 0) {
      setIsDragging(true)
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y })
    }
  }

  const handleMouseMove = (e) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y })
    }
  }

  const handleMouseUp = () => setIsDragging(false)

  return (
    <div className="map-screen">
      <div className="map-toolbar">
        <div className="toolbar-group">
          <label className="toolbar-label">Forecast Date</label>
          <input
            type="date"
            className="toolbar-input"
            value={date}
            onChange={(e) => setDate(e.target.value)}
          />
        </div>
        <div className="toolbar-group threshold-group">
          <label className="toolbar-label">
            Heatwave Threshold
            <span className="threshold-value">{threshold}°C</span>
          </label>
          <input
            type="range"
            className="threshold-slider"
            min="35"
            max="42"
            step="0.5"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
          />
        </div>
        <button className="toolbar-btn" onClick={fetchForecast} disabled={loading}>
          {loading ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="spin">
              <path d="M21 12a9 9 0 1 1-6.219-8.56" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
            </svg>
          )}
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      {error && (
        <div className="map-error">
          <span className="error-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </span>
          {error}
          <button className="retry-btn" onClick={fetchForecast}>Retry</button>
        </div>
      )}

      <div className="map-wrapper">
        <div className="map-body">
          <div 
            ref={containerRef}
            className="map-container"
            style={{ cursor: isDragging ? 'grabbing' : 'crosshair' }}
            onMouseDown={handleMouseDown}
            onMouseMove={(e) => { handleMouseMove(e); handleCanvasHover(e); }}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => { handleMouseUp(); setHoveredCell(null); }}
          >
            {!forecast ? (
              <div className="map-placeholder">
                <div className="map-placeholder-icon">
                  <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" />
                    <path d="M2 17l10 5 10-5" />
                    <path d="M2 12l10 5 10-5" />
                  </svg>
                </div>
                <p className="map-placeholder-title">Thailand Heatwave Map</p>
                <p className="map-placeholder-hint">Click Refresh to load forecast data</p>
              </div>
            ) : (
              <canvas
                ref={canvasRef}
                className="map-canvas"
                style={{
                  transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`,
                  transformOrigin: 'center center'
                }}
              />
            )}
          </div>

          {/* Axes */}
          {forecast && (
            <div className="map-axes">
              <div className="map-axis-x">
                <span>{forecast.bounds?.lon_min?.toFixed(1)}°E</span>
                <span>{(((forecast.bounds?.lon_min || 0) + (forecast.bounds?.lon_max || 0)) / 2).toFixed(1)}°E</span>
                <span>{forecast.bounds?.lon_max?.toFixed(1)}°E</span>
              </div>
              <div className="map-axis-y">
                <span>{forecast.bounds?.lat_max?.toFixed(1)}°N</span>
                <span>{(((forecast.bounds?.lat_max || 0) + (forecast.bounds?.lat_min || 0)) / 2).toFixed(1)}°N</span>
                <span>{forecast.bounds?.lat_min?.toFixed(1)}°N</span>
              </div>
            </div>
          )}
        </div>

        {/* Controls sidebar */}
        {forecast && (
          <div className="map-sidebar">
            <div className="legend-section">
              <div className="legend-title">Temperature</div>
              <div className="legend-bar-container">
                <div className="legend-bar" />
                <div className="legend-labels">
                  <span>42°C</span>
                  <span>31°C</span>
                  <span>20°C</span>
                </div>
              </div>
            </div>
            
            <div className="legend-section">
              <div className="legend-title">Indicators</div>
              <div className="legend-item">
                <div className="legend-dot heatwave" />
                <span>≥ {threshold}°C Heatwave</span>
              </div>
              <div className="legend-item">
                <div className="legend-dot city" />
                <span>Major Cities</span>
              </div>
            </div>

            <div className="zoom-section">
              <div className="zoom-title">Zoom {Math.round(zoom * 100)}%</div>
              <div className="zoom-controls">
                <button className="zoom-btn" onClick={handleZoomIn} title="Zoom In">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="12" y1="5" x2="12" y2="19" />
                    <line x1="5" y1="12" x2="19" y2="12" />
                  </svg>
                </button>
                <button className="zoom-btn" onClick={handleZoomOut} title="Zoom Out">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="5" y1="12" x2="19" y2="12" />
                  </svg>
                </button>
                <button className="zoom-btn" onClick={handleReset} title="Reset View">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                    <path d="M3 3v5h5" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Floating tooltip */}
      {hoveredCell && (
        <div
          className="map-tooltip"
          style={{ left: mousePos.x + 16, top: mousePos.y - 50 }}
        >
          <div className="tooltip-temp" style={{ color: getTempColor(hoveredCell.temp) }}>
            {hoveredCell.temp}°C
          </div>
          <div className="tooltip-coords">
            {hoveredCell.lat.toFixed(3)}°N, {hoveredCell.lon.toFixed(3)}°E
          </div>
          {hoveredCell.heatwave && (
            <div className="tooltip-alert">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
              Heatwave Warning
            </div>
          )}
        </div>
      )}

      {/* Metrics bar */}
      {forecast && (
        <div className="map-metrics">
          <div className="map-metric">
            <span className="map-metric-label">Max Temp</span>
            <span className="map-metric-value" style={{ color: getTempColor(forecast.max_temp) }}>
              {forecast.max_temp}°C
            </span>
          </div>
          <div className="map-metric">
            <span className="map-metric-label">Avg Temp</span>
            <span className="map-metric-value">{forecast.avg_temp}°C</span>
          </div>
          <div className="map-metric">
            <span className="map-metric-label">Heatwave Area</span>
            <span className="map-metric-value" style={{ color: forecast.heatwave_area_pct > 30 ? 'var(--accent-rose)' : 'var(--accent-emerald)' }}>
              {forecast.heatwave_area_pct}%
            </span>
          </div>
          <div className="map-metric">
            <span className="map-metric-label">Grid Points</span>
            <span className="map-metric-value">{forecast.grid_points.toLocaleString()}</span>
          </div>
          <div className="map-metric">
            <span className="map-metric-label">Date</span>
            <span className="map-metric-value">{forecast.date}</span>
          </div>
        </div>
      )}

      <style>{`
        .map-screen {
          animation: fadeIn 0.4s ease-out;
          display: flex;
          flex-direction: column;
          gap: var(--space-lg);
        }

        .map-error {
          display: flex;
          align-items: center;
          gap: var(--space-md);
          padding: var(--space-md) var(--space-lg);
          background: var(--accent-rose-dim);
          border: 1px solid rgba(244, 63, 94, 0.2);
          border-radius: var(--radius-md);
          color: var(--accent-rose);
          font-size: 0.875rem;
        }

        .map-error .retry-btn {
          margin-left: auto;
          padding: var(--space-xs) var(--space-md);
          background: var(--accent-rose);
          color: var(--bg-base);
          border: none;
          border-radius: var(--radius-sm);
          font-family: var(--font-ui);
          font-size: 0.8125rem;
          font-weight: 600;
          cursor: pointer;
        }

        .map-toolbar {
          display: flex;
          align-items: flex-end;
          gap: var(--space-xl);
          padding: var(--space-lg) var(--space-xl);
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
          flex-wrap: wrap;
        }

        .toolbar-group {
          display: flex;
          flex-direction: column;
          gap: var(--space-sm);
          min-width: 180px;
          flex: 1;
        }

        .toolbar-label {
          font-size: 0.6875rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--text-muted);
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: var(--space-md);
        }

        .threshold-value {
          font-family: var(--font-mono);
          font-size: 0.75rem;
          color: var(--accent-amber);
          background: var(--accent-amber-dim);
          padding: 2px var(--space-sm);
          border-radius: 100px;
        }

        .toolbar-input {
          padding: var(--space-sm) var(--space-md);
          background: var(--bg-base);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-sm);
          color: var(--text-primary);
          font-family: var(--font-ui);
          font-size: 0.875rem;
          outline: none;
        }

        .toolbar-input:focus {
          border-color: var(--accent-cyan);
          box-shadow: 0 0 0 3px var(--accent-cyan-dim);
        }

        .threshold-slider {
          -webkit-appearance: none;
          appearance: none;
          height: 4px;
          background: var(--bg-overlay);
          border-radius: 2px;
          outline: none;
        }

        .threshold-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 14px;
          height: 14px;
          border-radius: 50%;
          background: var(--accent-amber);
          cursor: pointer;
          border: 2px solid var(--bg-base);
          box-shadow: 0 0 0 2px var(--accent-amber-dim);
        }

        .toolbar-btn {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
          padding: var(--space-sm) var(--space-lg);
          background: var(--accent-cyan);
          color: var(--bg-base);
          border: none;
          border-radius: var(--radius-sm);
          font-family: var(--font-ui);
          font-size: 0.875rem;
          font-weight: 600;
          cursor: pointer;
          transition: all var(--transition-fast);
          margin-left: auto;
        }

        .toolbar-btn:hover:not(:disabled) {
          filter: brightness(1.1);
        }

        .toolbar-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .map-wrapper {
          display: flex;
          gap: var(--space-sm);
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
          padding: var(--space-lg);
          min-height: 480px;
          align-items: stretch;
        }

        .map-body {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: var(--space-sm);
        }

        .map-container {
          flex: 1;
          border-radius: var(--radius-sm);
          overflow: hidden;
          background: var(--bg-base);
          border: 1px solid var(--border-subtle);
        }

        .map-placeholder {
          width: 100%;
          height: 100%;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          color: var(--text-dim);
          gap: var(--space-md);
        }

        .map-placeholder-icon {
          color: var(--text-muted);
          opacity: 0.5;
        }

        .map-placeholder-title {
          font-size: 1rem;
          font-weight: 600;
          color: var(--text-secondary);
        }

        .map-placeholder-hint {
          font-size: 0.8125rem;
          color: var(--text-dim);
        }

        .heatmap-grid {
          display: grid;
          width: 100%;
          height: 100%;
          gap: 1px;
          position: relative;
          background: var(--border-subtle);
        }

        .heatmap-cell {
          display: flex;
          align-items: center;
          justify-content: center;
          border: 1px solid transparent;
          cursor: ${isDragging ? 'grabbing' : 'crosshair'};
          transition: transform 0.15s, box-shadow 0.15s;
          position: relative;
          user-select: none;
        }

        .heatmap-cell:hover {
          transform: scale(1.15);
          z-index: 10;
          box-shadow: 0 0 0 2px var(--text-primary), 0 4px 12px rgba(0,0,0,0.3);
        }

        .heatmap-cell.heatwave {
          animation: pulseBorder 2s ease-in-out infinite;
        }

        @keyframes pulseBorder {
          0%, 100% { box-shadow: inset 0 0 0 1px rgba(244,63,94,0.4); }
          50% { box-shadow: inset 0 0 0 2px rgba(244,63,94,0.7); }
        }

        .city-marker {
          position: absolute;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 2px;
          transform: translate(-50%, -50%);
          pointer-events: none;
          z-index: 5;
        }

        .city-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--text-primary);
          border: 2px solid var(--bg-base);
          box-shadow: 0 0 0 1px var(--text-primary);
        }

        .city-label {
          font-size: 0.625rem;
          font-weight: 700;
          color: var(--text-primary);
          background: rgba(0, 0, 0, 0.6);
          backdrop-filter: blur(4px);
          padding: 1px 4px;
          border-radius: 4px;
          white-space: nowrap;
          text-shadow: 0 1px 2px rgba(0,0,0,0.7);
          letter-spacing: 0.02em;
        }

        .map-axis-y {
          display: flex;
          flex-direction: column;
          justify-content: space-between;
          padding: 0 var(--space-xs);
          min-width: 40px;
          text-align: right;
        }

        .map-axis-y span {
          font-family: var(--font-mono);
          font-size: 0.625rem;
          color: var(--text-dim);
        }

        .map-axis-x {
          display: flex;
          justify-content: space-between;
          padding: var(--space-xs) 0;
        }

        .map-axis-x span {
          font-family: var(--font-mono);
          font-size: 0.625rem;
          color: var(--text-dim);
        }

        .map-legend {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: var(--space-xs);
          min-width: 60px;
          padding: var(--space-sm) 0;
        }

        .legend-bar {
          flex: 1;
          width: 12px;
          border-radius: 6px;
          background: linear-gradient(to bottom, hsla(240,75%,52%,0.85), hsla(180,75%,52%,0.85), hsla(120,75%,52%,0.85), hsla(60,75%,52%,0.85), hsla(0,75%,52%,0.85));
          border: 1px solid var(--border-default);
        }

        .legend-label {
          font-family: var(--font-mono);
          font-size: 0.625rem;
          color: var(--text-dim);
        }

        .legend-heatwave {
          display: flex;
          align-items: center;
          gap: var(--space-xs);
          margin-top: var(--space-sm);
        }

        .legend-dot {
          width: 8px;
          height: 8px;
          border-radius: 2px;
          background: hsla(0, 75%, 52%, 0.85);
          border: 1px solid hsla(0, 75%, 52%, 0.4);
        }

        .legend-text {
          font-size: 0.625rem;
          color: var(--text-muted);
          white-space: nowrap;
        }

        .zoom-controls {
          display: flex;
          flex-direction: column;
          gap: var(--space-xs);
          margin-top: var(--space-sm);
        }

        .zoom-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 24px;
          height: 24px;
          background: var(--bg-elevated);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-sm);
          color: var(--text-secondary);
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .zoom-btn:hover {
          background: var(--bg-overlay);
          color: var(--text-primary);
          border-color: var(--border-strong);
        }

        .zoom-level {
          font-family: var(--font-mono);
          font-size: 0.625rem;
          color: var(--text-dim);
          text-align: center;
        }

        .map-tooltip {
          position: fixed;
          pointer-events: none;
          z-index: 1000;
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-sm);
          padding: var(--space-sm) var(--space-md);
          box-shadow: 0 4px 16px rgba(0,0,0,0.4);
          backdrop-filter: blur(8px);
          min-width: 140px;
        }

        .tooltip-temp {
          font-family: var(--font-mono);
          font-size: 1.125rem;
          font-weight: 700;
        }

        .tooltip-coords {
          font-size: 0.6875rem;
          color: var(--text-dim);
          margin-top: 2px;
        }

        .tooltip-alert {
          font-size: 0.625rem;
          font-weight: 700;
          color: var(--accent-rose);
          text-transform: uppercase;
          letter-spacing: 0.06em;
          margin-top: var(--space-xs);
          padding-top: var(--space-xs);
          border-top: 1px solid var(--border-subtle);
        }

        .map-metrics {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: var(--space-lg);
        }

        .map-metric {
          display: flex;
          flex-direction: column;
          gap: var(--space-xs);
          padding: var(--space-lg) var(--space-xl);
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
        }

        .map-metric-label {
          font-size: 0.6875rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: var(--text-muted);
        }

        .map-metric-value {
          font-family: var(--font-mono);
          font-size: 1.5rem;
          font-weight: 700;
          color: var(--text-primary);
          line-height: 1.2;
        }

        .map-metric-unit {
          font-size: 0.875rem;
          font-weight: 500;
          color: var(--text-muted);
          margin-left: 2px;
        }

        @media (max-width: 768px) {
          .map-toolbar {
            flex-direction: column;
            align-items: stretch;
          }
          .toolbar-btn {
            margin-left: 0;
          }
          .map-wrapper {
            flex-direction: column;
            min-height: auto;
          }
          .map-legend {
            flex-direction: row;
            min-width: auto;
          }
          .legend-bar {
            width: 100%;
            height: 12px;
            background: linear-gradient(to right, hsla(240,75%,52%,0.85), hsla(180,75%,52%,0.85), hsla(120,75%,52%,0.85), hsla(60,75%,52%,0.85), hsla(0,75%,52%,0.85));
          }
          .map-axis-y {
            display: none;
          }
          .map-metrics {
            grid-template-columns: repeat(2, 1fr);
          }
          .city-label {
            display: none;
          }
          .zoom-controls {
            flex-direction: row;
            margin-top: var(--space-xs);
          }
          .zoom-btn {
            width: 28px;
            height: 28px;
          }
          .zoom-level {
            min-width: 40px;
          }
        }
      `}</style>
    </div>
  )
}

export default Map
