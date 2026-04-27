import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

function Train() {
  const [isTraining, setIsTraining] = useState(false)
  const [logs, setLogs] = useState([])
  const [modelType, setModelType] = useState('xgboost')
  const [jobId, setJobId] = useState(null)
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState(null)
  const pollInterval = useRef(null)

  const startTraining = async () => {
    try {
      setIsTraining(true)
      setLogs([])
      setProgress(0)
      setStatus('starting')

      const res = await axios.post('http://localhost:8000/api/train/start', {
        model_type: modelType,
        config: {}
      })

      const newJobId = res.data.job_id
      setJobId(newJobId)
      console.log('Training started:', res.data)

      // Start polling
      pollInterval.current = setInterval(() => pollTrainingStatus(newJobId), 2000)

    } catch (err) {
      console.error('Training failed:', err)
      setIsTraining(false)
      setStatus('failed')
    }
  }

  const pollTrainingStatus = async (id) => {
    try {
      const [statusRes, logsRes] = await Promise.all([
        axios.get(`http://localhost:8000/api/train/${id}/status`),
        axios.get(`http://localhost:8000/api/train/${id}/logs`)
      ])

      const jobStatus = statusRes.data
      setStatus(jobStatus.status)
      setProgress(jobStatus.progress || 0)

      if (logsRes.data.logs) {
        setLogs(logsRes.data.logs)
      }

      // Stop polling if completed or failed
      if (jobStatus.status === 'completed' || jobStatus.status === 'failed') {
        clearInterval(pollInterval.current)
        setIsTraining(false)
      }
    } catch (err) {
      console.error('Poll failed:', err)
    }
  }

  useEffect(() => {
    return () => {
      if (pollInterval.current) {
        clearInterval(pollInterval.current)
      }
    }
  }, [])

  return (
    <div className="train-screen">
      <div className="train-layout">
        {/* Left: Controls */}
        <div className="train-controls">
          <div className="control-panel">
            <div className="panel-head">
              <span className="panel-badge">Configuration</span>
            </div>
            <div className="panel-body">
              <div className="field-group">
                <label className="field-label">Algorithm</label>
                <div className="algo-options">
                  <button
                    className={`algo-btn ${modelType === 'xgboost' ? 'active' : ''}`}
                    onClick={() => setModelType('xgboost')}
                  >
                    <span className="algo-dot" />
                    <div className="algo-info">
                      <div className="algo-name">XGBoost</div>
                      <div className="algo-desc">Daily classification, fast training</div>
                    </div>
                  </button>
                  <button
                    className={`algo-btn ${modelType === 'convlstm' ? 'active' : ''}`}
                    onClick={() => setModelType('convlstm')}
                  >
                    <span className="algo-dot" />
                    <div className="algo-info">
                      <div className="algo-name">ConvLSTM</div>
                      <div className="algo-desc">Sequence forecasting, spatial</div>
                    </div>
                  </button>
                </div>
              </div>

              <div className="field-group">
                <label className="field-label">Training Parameters</label>
                <div className="param-grid">
                  <div className="param-item">
                    <span className="param-key">Epochs</span>
                    <span className="param-val">100</span>
                  </div>
                  <div className="param-item">
                    <span className="param-key">Batch Size</span>
                    <span className="param-val">32</span>
                  </div>
                  <div className="param-item">
                    <span className="param-key">LR</span>
                    <span className="param-val">1e-4</span>
                  </div>
                  <div className="param-item">
                    <span className="param-key">Split</span>
                    <span className="param-val">75/10/15</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="action-bar">
            <button
              className={`train-btn ${isTraining ? 'running' : ''}`}
              onClick={startTraining}
              disabled={isTraining}
            >
              {isTraining ? (
                <>
                  <span className="btn-pulse" />
                  Training in Progress
                </>
              ) : (
                <>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  Start Training
                </>
              )}
            </button>
            <button className="stop-btn" disabled={!isTraining}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="3" width="18" height="18" rx="2" />
              </svg>
              Stop
            </button>
          </div>
        </div>

        {/* Right: Logs */}
        <div className="train-logs">
          <div className="logs-header">
            <span className="logs-title">Training Logs</span>
            <div className="logs-meta">
              {status && <span className={`status-badge ${status}`}>{status}</span>}
              <span className="logs-count">{logs.length} entries</span>
            </div>
          </div>

          {isTraining && (
            <div className="progress-bar-wrap">
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
              </div>
              <span className="progress-text">{(progress * 100).toFixed(0)}%</span>
            </div>
          )}

          <div className="logs-body">
            {logs.length === 0 ? (
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
                <p className="logs-empty-text">No training logs yet</p>
                <p className="logs-empty-hint">Start a training run to see logs here</p>
              </div>
            ) : (
              <div className="logs-list">
                {logs.map((log, i) => (
                  <div key={i} className="log-line">
                    <span className="log-time">{new Date().toLocaleTimeString()}</span>
                    <span className="log-msg">{log}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        .train-screen {
          animation: fadeIn 0.4s ease-out;
        }

        .train-layout {
          display: grid;
          grid-template-columns: 380px 1fr;
          gap: var(--space-xl);
          height: calc(100vh - 140px);
        }

        .train-controls {
          display: flex;
          flex-direction: column;
          gap: var(--space-lg);
        }

        .control-panel {
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
          overflow: hidden;
          flex: 1;
        }

        .panel-head {
          padding: var(--space-md) var(--space-lg);
          border-bottom: 1px solid var(--border-subtle);
          background: var(--bg-elevated);
        }

        .panel-badge {
          font-size: 0.6875rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--text-muted);
        }

        .panel-body {
          padding: var(--space-xl);
          display: flex;
          flex-direction: column;
          gap: var(--space-xl);
        }

        .field-label {
          font-size: 0.8125rem;
          font-weight: 600;
          color: var(--text-secondary);
          margin-bottom: var(--space-md);
          display: block;
        }

        .algo-options {
          display: flex;
          flex-direction: column;
          gap: var(--space-sm);
        }

        .algo-btn {
          display: flex;
          align-items: center;
          gap: var(--space-md);
          padding: var(--space-md) var(--space-lg);
          background: var(--bg-base);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-sm);
          color: inherit;
          cursor: pointer;
          text-align: left;
          transition: all var(--transition-fast);
        }

        .algo-btn:hover {
          border-color: var(--border-strong);
          background: var(--bg-elevated);
        }

        .algo-btn.active {
          border-color: var(--accent-cyan);
          background: var(--accent-cyan-dim);
        }

        .algo-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          border: 2px solid var(--border-strong);
          flex-shrink: 0;
          transition: all var(--transition-fast);
        }

        .algo-btn.active .algo-dot {
          border-color: var(--accent-cyan);
          background: var(--accent-cyan);
          box-shadow: 0 0 0 3px var(--accent-cyan-dim);
        }

        .algo-name {
          font-weight: 600;
          font-size: 0.875rem;
          color: var(--text-primary);
        }

        .algo-desc {
          font-size: 0.75rem;
          color: var(--text-dim);
          margin-top: 2px;
        }

        .param-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: var(--space-sm);
        }

        .param-item {
          display: flex;
          flex-direction: column;
          gap: 2px;
          padding: var(--space-md);
          background: var(--bg-base);
          border: 1px solid var(--border-subtle);
          border-radius: var(--radius-sm);
        }

        .param-key {
          font-size: 0.625rem;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: var(--text-dim);
        }

        .param-val {
          font-family: var(--font-mono);
          font-size: 0.875rem;
          font-weight: 600;
          color: var(--text-secondary);
        }

        .action-bar {
          display: flex;
          gap: var(--space-md);
        }

        .train-btn {
          flex: 1;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: var(--space-sm);
          padding: var(--space-md) var(--space-xl);
          background: var(--accent-emerald);
          color: var(--bg-base);
          border: none;
          border-radius: var(--radius-sm);
          font-family: var(--font-ui);
          font-size: 0.9375rem;
          font-weight: 600;
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .train-btn:hover {
          filter: brightness(1.1);
        }

        .train-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .train-btn.running {
          background: var(--accent-amber);
        }

        .btn-pulse {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: var(--bg-base);
          animation: pulse 1.5s ease-in-out infinite;
        }

        .stop-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: var(--space-sm);
          padding: var(--space-md) var(--space-lg);
          background: var(--bg-elevated);
          color: var(--accent-rose);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-sm);
          font-family: var(--font-ui);
          font-size: 0.875rem;
          font-weight: 600;
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .stop-btn:hover {
          background: var(--accent-rose-dim);
          border-color: var(--accent-rose);
        }

        .stop-btn:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .train-logs {
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .logs-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: var(--space-md) var(--space-lg);
          border-bottom: 1px solid var(--border-subtle);
          background: var(--bg-elevated);
        }

        .logs-title {
          font-size: 0.8125rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: var(--text-secondary);
        }

        .logs-count {
          font-family: var(--font-mono);
          font-size: 0.6875rem;
          color: var(--text-dim);
          background: var(--bg-base);
          padding: 2px var(--space-sm);
          border-radius: 100px;
        }

        .logs-meta {
          display: flex;
          align-items: center;
          gap: var(--space-md);
        }

        .status-badge {
          font-size: 0.6875rem;
          font-weight: 600;
          text-transform: uppercase;
          padding: 2px var(--space-sm);
          border-radius: 100px;
        }

        .status-badge.running {
          background: var(--accent-amber-dim);
          color: var(--accent-amber);
        }

        .status-badge.completed {
          background: var(--accent-emerald-dim);
          color: var(--accent-emerald);
        }

        .status-badge.failed {
          background: var(--accent-rose-dim);
          color: var(--accent-rose);
        }

        .progress-bar-wrap {
          display: flex;
          align-items: center;
          gap: var(--space-md);
          padding: var(--space-md) var(--space-lg);
          background: var(--bg-base);
          border-bottom: 1px solid var(--border-subtle);
        }

        .progress-bar {
          flex: 1;
          height: 6px;
          background: var(--bg-overlay);
          border-radius: 3px;
          overflow: hidden;
        }

        .progress-fill {
          height: 100%;
          background: var(--accent-emerald);
          border-radius: 3px;
          transition: width 0.3s ease;
        }

        .progress-text {
          font-family: var(--font-mono);
          font-size: 0.75rem;
          font-weight: 600;
          color: var(--accent-emerald);
          min-width: 40px;
          text-align: right;
        }

        .logs-body {
          flex: 1;
          overflow-y: auto;
          padding: var(--space-lg);
        }

        .logs-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          text-align: center;
          color: var(--text-dim);
        }

        .logs-empty-icon {
          margin-bottom: var(--space-md);
        }

        .logs-empty-text {
          font-size: 0.875rem;
          font-weight: 500;
          color: var(--text-muted);
          margin-bottom: var(--space-xs);
        }

        .logs-empty-hint {
          font-size: 0.75rem;
          color: var(--text-dim);
        }

        .logs-list {
          font-family: var(--font-mono);
          font-size: 0.8125rem;
          display: flex;
          flex-direction: column;
          gap: var(--space-xs);
        }

        .log-line {
          display: flex;
          gap: var(--space-md);
          padding: var(--space-xs) 0;
          color: var(--text-secondary);
        }

        .log-time {
          color: var(--text-dim);
          flex-shrink: 0;
        }

        .log-msg {
          word-break: break-all;
        }

        @media (max-width: 1024px) {
          .train-layout {
            grid-template-columns: 1fr;
            height: auto;
          }
        }
      `}</style>
    </div>
  )
}

export default Train
