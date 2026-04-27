import { useState, useEffect } from 'react'
import axios from 'axios'

const LOG_LEVELS = [
  { id: 'all', label: 'All' },
  { id: 'info', label: 'Info' },
  { id: 'warning', label: 'Warning' },
  { id: 'error', label: 'Error' }
]

function Logs() {
  const [logs, setLogs] = useState([])
  const [filter, setFilter] = useState('all')
  const [lastUpdate, setLastUpdate] = useState(new Date())
  const [error, setError] = useState(null)

  const fetchLogs = async () => {
    setError(null)
    try {
      const res = await axios.get('http://localhost:8000/api/system/logs')
      setLogs(res.data.logs || [])
      setLastUpdate(new Date())
    } catch (err) {
      console.error('Failed to fetch logs:', err)
      setError('Failed to fetch logs. Is the backend running?')
    }
  }

  useEffect(() => {
    fetchLogs()
    const interval = setInterval(fetchLogs, 5000)
    return () => clearInterval(interval)
  }, [])

  const filteredLogs = logs.filter(log => filter === 'all' || log.toLowerCase().includes(filter))

  const getLevelColor = (log) => {
    const lower = log.toLowerCase()
    if (lower.includes('error')) return 'var(--accent-rose)'
    if (lower.includes('warn')) return 'var(--accent-amber)'
    if (lower.includes('info')) return 'var(--accent-cyan)'
    return 'var(--text-muted)'
  }

  return (
    <div className="logs-screen">
      <div className="logs-toolbar">
        <div className="log-filters">
          {LOG_LEVELS.map(level => (
            <button
              key={level.id}
              className={`filter-pill ${filter === level.id ? 'active' : ''}`}
              onClick={() => setFilter(level.id)}
            >
              {level.label}
              <span className="filter-count">
                {level.id === 'all' ? logs.length : logs.filter(l => l.toLowerCase().includes(level.id)).length}
              </span>
            </button>
          ))}
        </div>
        <div className="logs-meta">
          <span className="refresh-indicator">
            <span className="refresh-dot" />
            Auto-refresh
          </span>
          <span className="last-update">
            Last: {lastUpdate.toLocaleTimeString()}
          </span>
          <button className="logs-refresh-btn" onClick={fetchLogs}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
            </svg>
          </button>
        </div>
      </div>

      {error && (
        <div className="logs-error">
          <span className="error-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </span>
          {error}
          <button className="retry-btn" onClick={fetchLogs}>Retry</button>
        </div>
      )}

      <div className="logs-panel">
        {filteredLogs.length === 0 ? (
          <div className="logs-empty">
            <div className="logs-empty-icon">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
                <polyline points="10 9 9 9 8 9" />
              </svg>
            </div>
            <p className="logs-empty-text">No logs to display</p>
            <p className="logs-empty-hint">Logs will appear here automatically</p>
          </div>
        ) : (
          <div className="logs-list">
            {filteredLogs.map((log, i) => (
              <div key={i} className="log-entry">
                <span className="log-level-dot" style={{ background: getLevelColor(log) }} />
                <span className="log-text">{log}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <style>{`
        .logs-screen {
          animation: fadeIn 0.4s ease-out;
          display: flex;
          flex-direction: column;
          gap: var(--space-lg);
          height: calc(100vh - 140px);
        }

        .logs-toolbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-shrink: 0;
        }

        .log-filters {
          display: flex;
          gap: var(--space-sm);
        }

        .filter-pill {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
          padding: var(--space-xs) var(--space-md);
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: 100px;
          color: var(--text-muted);
          font-family: var(--font-ui);
          font-size: 0.8125rem;
          font-weight: 500;
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .filter-pill:hover {
          background: var(--bg-elevated);
          border-color: var(--border-strong);
          color: var(--text-secondary);
        }

        .filter-pill.active {
          background: var(--accent-cyan-dim);
          border-color: bgb,(6, 182, 212, 0.25 0.25);
          color: var(--accent-cyan);
        }

        .filter-count {
          font-family: var(--font-mono);
          font-size: 0.6875rem;
          background: var(--bg-base);
          padding: 1px var(--space-sm);
          border-radius: 100px;
          min-width: 20px;
          text-align: center;
        }

        .filter-pill.active .filter-count {
          background: bgbaa6, 182, 212, 0.155);
        }

        .logs-meta {
          display: flex;
          align-items: center;
          gap: var(--space-lg);
        }

        .refresh-indicator {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
          font-size: 0.75rem;
          color: var(--text-muted);
        }

        .refresh-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: var(--status-live);
          box-shadow: 0 0 0 2px var(--accent-emerald-dim);
          animation: pulse 2s ease-in-out infinite;
        }

        .last-update {
          font-family: var(--font-mono);
          font-size: 0.6875rem;
          color: var(--text-dim);
        }

        .logs-refresh-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 32px;
          height: 32px;
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-sm);
          color: var(--text-muted);
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .logs-refresh-btn:hover {
          background: var(--bg-elevated);
          border-color: var(--border-strong);
          color: var(--text-secondary);
        }

        .logs-error {
          display: flex;
          align-items: center;
          gap: var(--space-md);
          padding: var(--space-md) var(--space-lg);
          background: var(--accent-rose-dim);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
          color: var(--accent-rose);
          font-size: 0.875rem;
          margin-bottom: var(--space-lg);
        }

        .logs-error .retry-btn {
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

        .logs-panel {
          flex: 1;
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }

        .logs-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          flex: 1;
          text-align: center;
          color: var(--text-dim);
          gap: var(--space-md);
        }

        .logs-empty-icon {
          color: var(--text-muted);
        }

        .logs-empty-text {
          font-size: 0.9375rem;
          font-weight: 600;
          color: var(--text-secondary);
        }

        .logs-empty-hint {
          font-size: 0.8125rem;
          color: var(--text-dim);
        }

        .logs-list {
          flex: 1;
          overflow-y: auto;
          padding: var(--space-md) 0;
          font-family: var(--font-mono);
          font-size: 0.8125rem;
          line-height: 1.6;
        }

        .log-entry {
          display: flex;
          align-items: flex-start;
          gap: var(--space-md);
          padding: var(--space-xs) var(--space-lg);
          color: var(--text-secondary);
          transition: background var(--transition-fast);
        }

        .log-entry:hover {
          background: var(--bg-elevated);
        }

        .log-level-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          margin-top: 7px;
          flex-shrink: 0;
        }

        .log-text {
          word-break: break-all;
        }

        @media (max-width: 768px) {
          .logs-toolbar {
            flex-direction: column;
            align-items: stretch;
            gap: var(--space-md);
          }
        }
      `}</style>
    </div>
  )
}

export default Logs
