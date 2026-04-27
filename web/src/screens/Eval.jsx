import { useState, useEffect } from 'react'
import axios from 'axios'

function Eval() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [currentModel, setCurrentModel] = useState(null)

  const fetchMetrics = async () => {
    setLoading(true)
    setError(null)
    try {
      const [metricsRes, currentRes] = await Promise.all([
        axios.get('http://localhost:8000/api/eval/metrics'),
        axios.get('http://localhost:8000/api/models/current')
      ])
      console.log('Metrics fetched:', metricsRes.data)
      console.log('Current model:', currentRes.data)
      setMetrics(metricsRes.data)
      setCurrentModel(currentRes.data)
    } catch (err) {
      console.error('Failed to fetch metrics:', err)
      setError('Failed to fetch metrics')
    } finally {
      setLoading(false)
    }
  }

  // Auto-fetch on mount
  useEffect(() => {
    fetchMetrics()
  }, [])

  const getStatusColor = (assessment) => {
    if (assessment === 'GOOD') return 'var(--status-live)'
    if (assessment === 'MODERATE') return 'var(--status-warning)'
    return 'var(--status-error)'
  }

  const getStatusBg = (assessment) => {
    if (assessment === 'GOOD') return 'var(--accent-emerald-dim)'
    if (assessment === 'MODERATE') return 'var(--accent-amber-dim)'
    return 'var(--accent-rose-dim)'
  }

  return (
    <div className="eval-screen">
      <div className="eval-header">
        <button className="eval-btn" onClick={fetchMetrics} disabled={loading}>
          {loading ? (
            <>
              <span className="btn-spinner" />
              Evaluating...
            </>
          ) : (
            <>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
              </svg>
              Run Evaluation
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="eval-error">
          {error}
        </div>
      )}

      {metrics && metrics.models && metrics.models.length > 0 && (
        <div className="models-list">
          <div className="models-count">{metrics.models.length} models found {currentModel?.loaded && `(Current: ${currentModel.model})`}</div>
          {metrics.models.map((model, i) => (
            <div
              key={i}
              className={`model-card ${currentModel?.loaded && currentModel.model === model.name ? 'loaded' : ''}`}
              style={{ animationDelay: `${i * 100}ms` }}
            >
              <div className="model-card-header">
                <div className="model-info">
                  <div className="model-name">{model.name}</div>
                  <div className="model-badges">
                    {currentModel?.loaded && currentModel.model === model.name && (
                      <div className="loaded-badge">LOADED</div>
                    )}
                    <div
                      className="model-status"
                      style={{ color: getStatusColor(model.assessment), background: getStatusBg(model.assessment) }}
                    >
                      {model.assessment}
                    </div>
                  </div>
                </div>
              </div>

              <div className="metrics-row">
                <div className="metric-box">
                  <span className="metric-name">F1 Score</span>
                  <span className="metric-number">{model.test_f1.toFixed(3)}</span>
                </div>
                <div className="metric-box">
                  <span className="metric-name">Precision</span>
                  <span className="metric-number">{model.test_precision.toFixed(3)}</span>
                </div>
                <div className="metric-box">
                  <span className="metric-name">Recall</span>
                  <span className="metric-number">{model.test_recall.toFixed(3)}</span>
                </div>
                <div className="metric-box">
                  <span className="metric-name">Accuracy</span>
                  <span className="metric-number">{model.test_accuracy.toFixed(3)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {(!metrics || !metrics.models || metrics.models.length === 0) && !loading && (
        <div className="eval-empty">
          <div className="eval-empty-icon">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 20v-6M6 20V10M18 20V4"/>
            </svg>
          </div>
          <p className="eval-empty-title">No evaluation data</p>
          <p className="eval-empty-hint">Click "Run Evaluation" to check model metrics</p>
        </div>
      )}

      <style>{`
        .eval-screen {
          animation: fadeIn 0.4s ease-out;
        }

        .eval-header {
          display: flex;
          justify-content: flex-end;
          margin-bottom: var(--space-xl);
        }

        .eval-btn {
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
        }

        .eval-btn:hover {
          filter: brightness(1.1);
        }

        .eval-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .btn-spinner {
          width: 12px;
          height: 12px;
          border: 2px solid var(--border-strong);
          border-top-color: var(--bg-base);
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .models-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-lg);
        }

        .model-card {
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
          overflow: hidden;
          animation: fadeIn 0.5s ease-out backwards;
        }

        .model-card-header {
          padding: var(--space-md) var(--space-lg);
          border-bottom: 1px solid var(--border-subtle);
          background: var(--bg-elevated);
        }

        .model-info {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: var(--space-md);
        }

        .model-name {
          font-family: var(--font-mono);
          font-size: 0.875rem;
          font-weight: 600;
          color: var(--text-primary);
        }

        .model-status {
          font-size: 0.6875rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          padding: 2px var(--space-sm);
          border-radius: 100px;
        }

        .metrics-row {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: var(--space-md);
          padding: var(--space-xl);
        }

        .metric-box {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          gap: var(--space-xs);
          padding: var(--space-md);
          background: var(--bg-base);
          border: 1px solid var(--border-subtle);
          border-radius: var(--radius-sm);
        }

        .metric-name {
          font-size: 0.625rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--text-muted);
        }

        .metric-number {
          font-family: var(--font-mono);
          font-size: 1.5rem;
          font-weight: 700;
          color: var(--text-primary);
          line-height: 1;
        }

        .eval-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: var(--space-3xl);
          text-align: center;
          color: var(--text-dim);
        }

        .eval-empty-icon {
          margin-bottom: var(--space-md);
          color: var(--text-muted);
        }

        .eval-empty-title {
          font-size: 0.9375rem;
          font-weight: 600;
          color: var(--text-secondary);
          margin-bottom: var(--space-xs);
        }

        .eval-empty-hint {
          font-size: 0.8125rem;
          color: var(--text-dim);
        }

        .eval-error {
          padding: var(--space-md) var(--space-lg);
          background: var(--accent-rose-dim);
          color: var(--accent-rose);
          border: 1px solid rgba(244, 63, 94, 0.2);
          border-radius: var(--radius-sm);
          font-size: 0.8125rem;
          margin-bottom: var(--space-lg);
        }

        .models-count {
          font-size: 0.75rem;
          color: var(--text-muted);
          margin-bottom: var(--space-md);
          font-family: var(--font-mono);
        }

        .model-badges {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
        }

        .loaded-badge {
          font-size: 0.625rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          padding: 2px var(--space-sm);
          background: var(--accent-cyan);
          color: var(--bg-base);
          border-radius: 100px;
        }

        .model-card.loaded {
          border-color: var(--accent-cyan);
          box-shadow: 0 0 0 1px var(--accent-cyan), 0 0 20px var(--accent-cyan-dim);
        }

        @media (max-width: 768px) {
          .metrics-row {
            grid-template-columns: repeat(2, 1fr);
          }
        }
      `}</style>
    </div>
  )
}

export default Eval
