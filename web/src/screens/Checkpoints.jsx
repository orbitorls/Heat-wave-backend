import { useState, useEffect } from 'react'
import axios from 'axios'

function Checkpoints() {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(null)
  const [message, setMessage] = useState(null)

  const fetchModels = async () => {
    try {
      const res = await axios.get('http://localhost:8000/api/models/')
      setModels(res.data.models || [])
    } catch (err) {
      console.error('Failed to fetch models:', err)
    }
  }

  const loadModel = async (modelName) => {
    setLoading(modelName)
    setMessage(null)
    try {
      const res = await axios.post('http://localhost:8000/api/models/load', null, {
        params: { model_name: modelName }
      })
      setMessage({ type: 'success', text: `Loaded ${modelName} successfully` })
    } catch (err) {
      console.error('Failed to load model:', err)
      setMessage({ type: 'error', text: `Failed to load ${modelName}` })
    } finally {
      setLoading(null)
    }
  }

  const deleteModel = async (modelName) => {
    if (!confirm(`Are you sure you want to delete ${modelName}?`)) {
      return
    }

    setLoading(modelName)
    setMessage(null)
    try {
      await axios.delete(`http://localhost:8000/api/models/${modelName}`)
      setMessage({ type: 'success', text: `Deleted ${modelName}` })
      fetchModels()  // Refresh list
    } catch (err) {
      console.error('Failed to delete model:', err)
      setMessage({ type: 'error', text: err.response?.data?.detail || `Failed to delete ${modelName}` })
    } finally {
      setLoading(null)
    }
  }

  useEffect(() => {
    fetchModels()
  }, [])

  return (
    <div className="checkpoints-screen">
      <div className="checkpoints-header">
        <div className="checkpoints-stat">
          <span className="checkpoints-count">{models.length}</span>
          <span className="checkpoints-label">Checkpoints</span>
        </div>
        <button className="checkpoints-refresh" onClick={fetchModels}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
          </svg>
          Refresh
        </button>
      </div>

      {message && (
        <div className={`load-message ${message.type}`}>
          {message.text}
        </div>
      )}

      <div className="checkpoints-grid">
        {models.length === 0 ? (
          <div className="checkpoints-empty">
            <div className="checkpoints-empty-icon">
              <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
            </div>
            <p className="checkpoints-empty-title">No checkpoints found</p>
            <p className="checkpoints-empty-hint">Train a model to create your first checkpoint</p>
          </div>
        ) : (
          models.map((model, i) => (
            <div key={i} className="checkpoint-card" style={{ animationDelay: `${i * 60}ms` }}>
              <div className="checkpoint-main">
                <div className="checkpoint-icon">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                  </svg>
                </div>
                <div className="checkpoint-info">
                  <div className="checkpoint-name">{model}</div>
                  <div className="checkpoint-meta">
                    <span className="meta-tag">.pth</span>
                    <span className="meta-tag">checkpoint</span>
                  </div>
                </div>
              </div>
              <div className="checkpoint-actions">
                <button
                  className={`cp-btn load ${loading === model ? 'loading' : ''}`}
                  onClick={() => loadModel(model)}
                  disabled={loading === model}
                >
                  {loading === model ? (
                    <>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="spin">
                        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                      </svg>
                      Loading
                    </>
                  ) : (
                    <>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
                      Load
                    </>
                  )}
                </button>
                <button
                  className="cp-btn delete"
                  onClick={() => deleteModel(model)}
                  disabled={loading === model}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  </svg>
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      <style>{`
        .checkpoints-screen {
          animation: fadeIn 0.4s ease-out;
        }

        .checkpoints-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: var(--space-xl);
        }

        .checkpoints-stat {
          display: flex;
          align-items: baseline;
          gap: var(--space-sm);
        }

        .checkpoints-count {
          font-family: var(--font-mono);
          font-size: 1.25rem;
          font-weight: 700;
          color: var(--text-primary);
        }

        .checkpoints-label {
          font-size: 0.8125rem;
          color: var(--text-muted);
        }

        .checkpoints-refresh {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
          padding: var(--space-sm) var(--space-lg);
          background: var(--bg-elevated);
          color: var(--text-secondary);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-sm);
          font-family: var(--font-ui);
          font-size: 0.875rem;
          font-weight: 600;
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .checkpoints-refresh:hover {
          background: var(--bg-overlay);
          border-color: var(--border-strong);
        }

        .checkpoints-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
          gap: var(--space-lg);
        }

        .checkpoints-empty {
          grid-column: 1 / -1;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: var(--space-3xl);
          text-align: center;
          color: var(--text-dim);
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
        }

        .checkpoints-empty-icon {
          margin-bottom: var(--space-md);
          color: var(--text-muted);
        }

        .checkpoints-empty-title {
          font-size: 0.9375rem;
          font-weight: 600;
          color: var(--text-secondary);
          margin-bottom: var(--space-xs);
        }

        .checkpoints-empty-hint {
          font-size: 0.8125rem;
          color: var(--text-dim);
        }

        .checkpoint-card {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: var(--space-lg);
          padding: var(--space-lg) var(--space-xl);
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
          transition: all var(--transition-fast);
          animation: fadeIn 0.4s ease-out backwards;
        }

        .checkpoint-card:hover {
          border-color: var(--border-strong);
          transform: translateY(-1px);
        }

        .checkpoint-main {
          display: flex;
          align-items: center;
          gap: var(--space-md);
          min-width: 0;
          flex: 1;
        }

        .checkpoint-icon {
          color: var(--text-dim);
          flex-shrink: 0;
        }

        .checkpoint-info {
          min-width: 0;
        }

        .checkpoint-name {
          font-family: var(--font-mono);
          font-size: 0.8125rem;
          font-weight: 600;
          color: var(--text-primary);
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .checkpoint-meta {
          display: flex;
          gap: var(--space-sm);
          margin-top: 4px;
        }

        .meta-tag {
          font-size: 0.625rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          color: var(--text-dim);
          background: var(--bg-base);
          padding: 2px var(--space-sm);
          border-radius: 100px;
        }

        .checkpoint-actions {
          display: flex;
          gap: var(--space-sm);
          flex-shrink: 0;
        }

        .cp-btn {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
          padding: var(--space-sm) var(--space-md);
          border-radius: var(--radius-sm);
          font-family: var(--font-ui);
          font-size: 0.8125rem;
          font-weight: 600;
          cursor: pointer;
          transition: all var(--transition-fast);
          pointer-events: auto;
          border: 1px solid transparent;
        }

        .cp-btn.load {
          background: var(--accent-emerald-dim);
          color: var(--accent-emerald);
          border-color: var(--accent-emerald);
          opacity: 0.85;
        }

        .cp-btn.load:hover {
          background: var(--accent-emerald);
          color: var(--bg-base);
          opacity: 1;
        }

        .cp-btn.delete {
          background: var(--accent-rose-dim);
          color: var(--accent-rose);
          border-color: var(--accent-rose);
          opacity: 0.85;
          padding: var(--space-xs);
        }

        .cp-btn.delete:hover {
          background: var(--accent-rose);
          color: var(--bg-base);
          opacity: 1;
        }

        .cp-btn:disabled {
          opacity: 0.4;
          cursor: not-allowed;
        }

        .cp-btn.loading {
          opacity: 0.7;
          cursor: wait;
        }

        .cp-btn.loading .spin {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .load-message {
          padding: var(--space-md) var(--space-lg);
          border-radius: var(--radius-sm);
          font-size: 0.8125rem;
          font-weight: 500;
          margin-bottom: var(--space-lg);
          animation: fadeIn 0.3s ease-out;
        }

        .load-message.success {
          background: var(--accent-emerald-dim);
          color: var(--accent-emerald);
          border: 1px solid var(--accent-emerald);
        }

        .load-message.error {
          background: var(--accent-rose-dim);
          color: var(--accent-rose);
          border: 1px solid var(--accent-rose);
        }

        @media (max-width: 768px) {
          .checkpoints-grid {
            grid-template-columns: 1fr;
          }
          .checkpoint-card {
            flex-direction: column;
            align-items: stretch;
            gap: var(--space-md);
          }
          .checkpoint-actions {
            justify-content: flex-end;
          }
        }
      `}</style>
    </div>
  )
}

export default Checkpoints
