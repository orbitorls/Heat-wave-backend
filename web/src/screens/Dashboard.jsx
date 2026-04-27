import { useState, useEffect } from 'react'
import axios from 'axios'

function Dashboard() {
  const [status, setStatus] = useState(null)
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [sysRes, modelRes] = await Promise.all([
        axios.get('http://localhost:8000/api/system/gpu-status').catch(() => null),
        axios.get('http://localhost:8000/api/models/').catch(() => null)
      ])
      if (sysRes) setStatus(sysRes.data)
      if (modelRes) setModels(modelRes.data.models || [])
      if (!sysRes && !modelRes) {
        setError('Backend connection failed. Is the API server running?')
      }
    } catch (err) {
      console.error('Dashboard fetch error:', err)
      setError('Failed to load dashboard data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  const metrics = [
    {
      label: 'Compute Device',
      value: status?.device || 'CPU',
      sub: status?.cuda_available ? 'CUDA Available' : 'CPU Mode',
      status: status?.cuda_available ? 'live' : 'warning',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="4" y="4" width="16" height="16" rx="2" />
          <rect x="9" y="9" width="6" height="6" />
          <path d="M15 2v2" /><path d="M15 20v2" /><path d="M2 15h2" /><path d="M2 9h2" /><path d="M20 15h2" /><path d="M20 9h2" /><path d="M9 2v2" /><path d="M9 20v2" />
        </svg>
      )
    },
    {
      label: 'Model Checkpoints',
      value: models.length.toString(),
      sub: models.length > 0 ? 'Available for load' : 'No checkpoints',
      status: models.length > 0 ? 'live' : 'offline',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
      )
    },
    {
      label: 'System Status',
      value: 'Online',
      sub: 'All services running',
      status: 'live',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2v4" /><path d="m16.2 7.8 2.9-2.9" /><path d="M18 12h4" /><path d="m16.2 16.2 2.9 2.9" /><path d="M12 18v4" /><path d="m7.8 16.2-2.9 2.9" /><path d="M6 12H2" /><path d="m7.8 7.8-2.9-2.9" />
        </svg>
      )
    }
  ]

  const quickActions = [
    { label: 'Train Model', path: '/train', desc: 'Start XGBoost or ConvLSTM training', color: 'amber' },
    { label: 'Run Prediction', path: '/predict', desc: 'Predict heatwave for a location', color: 'cyan' },
    { label: 'View Map', path: '/map', desc: 'Thailand heatwave visualization', color: 'rose' },
    { label: 'Evaluate', path: '/eval', desc: 'Check model accuracy metrics', color: 'emerald' }
  ]

  return (
    <div className="dashboard">
      {error && (
        <div className="dashboard-error">
          <span className="error-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </span>
          {error}
          <button className="retry-btn" onClick={fetchData}>Retry</button>
        </div>
      )}

      {/* Metrics Row */}
      <div className="metrics-grid">
        {metrics.map((m, i) => (
          <div key={i} className={`metric-card status-${m.status}`} style={{ animationDelay: `${i * 80}ms` }}>
            <div className="metric-icon">{m.icon}</div>
            <div className="metric-content">
              <div className="metric-label">{m.label}</div>
              <div className="metric-value">{m.value}</div>
              <div className="metric-sub">{m.sub}</div>
            </div>
            <div className="metric-glow" />
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="section-header">
        <h2 className="section-title">Quick Actions</h2>
        <span className="section-count">{quickActions.length} workflows</span>
      </div>

      <div className="actions-grid">
        {quickActions.map((action, i) => (
          <a key={i} href={action.path} className={`action-card color-${action.color}`} style={{ animationDelay: `${(i + 3) * 80}ms` }}>
            <div className="action-content">
              <div className="action-label">{action.label}</div>
              <div className="action-desc">{action.desc}</div>
            </div>
            <div className="action-arrow">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12h14" /><path d="m12 5 7 7-7 7" />
              </svg>
            </div>
          </a>
        ))}
      </div>

      {/* Recent Activity / Status */}
      <div className="section-header">
        <h2 className="section-title">System Overview</h2>
      </div>

      <div className="overview-grid">
        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">Recent Models</span>
            <span className="panel-badge">{models.length}</span>
          </div>
          <div className="panel-body">
            {models.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                </div>
                <p className="empty-text">No model checkpoints found</p>
                <p className="empty-hint">Train a model to create your first checkpoint</p>
              </div>
            ) : (
              <div className="model-list">
                {models.slice(0, 5).map((model, i) => (
                  <div key={i} className="model-item">
                    <div className="model-dot" />
                    <span className="model-name">{model}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">
            <span className="panel-title">Environment</span>
          </div>
          <div className="panel-body">
            <div className="env-grid">
              <div className="env-item">
                <span className="env-key">PyTorch</span>
                <span className="env-value">{status?.torch_version || '--'}</span>
              </div>
              <div className="env-item">
                <span className="env-key">CUDA</span>
                <span className="env-value">{status?.cuda_version || 'N/A'}</span>
              </div>
              <div className="env-item">
                <span className="env-key">Device</span>
                <span className="env-value">{status?.device || 'CPU'}</span>
              </div>
              <div className="env-item">
                <span className="env-key">Uptime</span>
                <span className="env-value">--</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .dashboard {
          animation: fadeIn 0.5s ease-out;
        }

        .dashboard-error {
          display: flex;
          align-items: center;
          gap: var(--space-md);
          padding: var(--space-md) var(--space-lg);
          background: var(--accent-rose-dim);
          border: 1px solid rgba(244, 63, 94, 0.2);
          border-radius: var(--radius-md);
          color: var(--accent-rose);
          font-size: 0.875rem;
          margin-bottom: var(--space-xl);
        }

        .error-icon {
          display: flex;
          flex-shrink: 0;
        }

        .retry-btn {
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
          transition: opacity var(--transition-fast);
        }

        .retry-btn:hover {
          opacity: 0.85;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: var(--space-lg);
          margin-bottom: var(--space-3xl);
        }

        .metric-card {
          position: relative;
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
          padding: var(--space-xl);
          display: flex;
          align-items: flex-start;
          gap: var(--space-md);
          overflow: hidden;
          animation: fadeIn 0.5s ease-out backwards;
          transition: all var(--transition-fast);
        }

        .metric-card:hover {
          border-color: var(--border-strong);
          transform: translateY(-2px);
        }

        .metric-card.status-live {
          border-color: rgba(16, 185, 129, 0.25);
        }

        .metric-card.status-warning {
          border-color: rgba(245, 158, 11, 0.25);
        }

        .metric-card.status-offline {
          border-color: rgba(100, 116, 139, 0.25);
        }

        .metric-icon {
          color: var(--text-muted);
          flex-shrink: 0;
          margin-top: 2px;
        }

        .metric-content {
          flex: 1;
          min-width: 0;
        }

        .metric-label {
          font-size: 0.75rem;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.06em;
          margin-bottom: var(--space-xs);
        }

        .metric-value {
          font-family: var(--font-mono);
          font-size: 1.5rem;
          font-weight: 600;
          color: var(--text-primary);
          line-height: 1.2;
        }

        .metric-sub {
          font-size: 0.8125rem;
          color: var(--text-dim);
          margin-top: var(--space-xs);
        }

        .metric-glow {
          position: absolute;
          top: -50%;
          right: -50%;
          width: 100%;
          height: 100%;
          border-radius: 50%;
          opacity: 0.03;
          pointer-events: none;
        }

        .metric-card.status-live .metric-glow {
          background: radial-gradient(circle, var(--status-live) 0%, transparent 70%);
        }

        .metric-card.status-warning .metric-glow {
          background: radial-gradient(circle, var(--status-warning) 0%, transparent 70%);
        }

        .section-header {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          margin-bottom: var(--space-lg);
          margin-top: var(--space-2xl);
        }

        .section-title {
          font-size: 0.875rem;
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.08em;
        }

        .section-count {
          font-size: 0.75rem;
          color: var(--text-dim);
          font-family: var(--font-mono);
        }

        .actions-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: var(--space-lg);
          margin-bottom: var(--space-3xl);
        }

        .action-card {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: var(--space-lg) var(--space-xl);
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
          text-decoration: none;
          color: inherit;
          transition: all var(--transition-fast);
          animation: fadeIn 0.5s ease-out backwards;
        }

        .action-card:hover {
          background: var(--bg-elevated);
          border-color: var(--border-strong);
          transform: translateY(-1px);
        }

        .action-card.color-amber:hover {
          border-color: rgba(245, 158, 11, 0.3);
        }

        .action-card.color-cyan:hover {
          border-color: rgba(6, 182, 212, 0.3);
        }

        .action-card.color-rose:hover {
          border-color: rgba(244, 63, 94, 0.3);
        }

        .action-card.color-emerald:hover {
          border-color: rgba(16, 185, 129, 0.3);
        }

        .action-label {
          font-weight: 600;
          font-size: 0.9375rem;
          color: var(--text-primary);
          margin-bottom: 2px;
        }

        .action-desc {
          font-size: 0.8125rem;
          color: var(--text-muted);
        }

        .action-arrow {
          color: var(--text-dim);
          transition: all var(--transition-fast);
        }

        .action-card:hover .action-arrow {
          color: var(--text-secondary);
          transform: translateX(3px);
        }

        .overview-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: var(--space-lg);
        }

        .panel {
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
          overflow: hidden;
        }

        .panel-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: var(--space-md) var(--space-lg);
          border-bottom: 1px solid var(--border-subtle);
        }

        .panel-title {
          font-size: 0.8125rem;
          font-weight: 600;
          color: var(--text-secondary);
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }

        .panel-badge {
          font-family: var(--font-mono);
          font-size: 0.6875rem;
          color: var(--text-dim);
          background: var(--bg-elevated);
          padding: 2px var(--space-sm);
          border-radius: 100px;
        }

        .panel-body {
          padding: var(--space-lg);
        }

        .empty-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          padding: var(--space-2xl) 0;
        }

        .empty-icon {
          color: var(--text-dim);
          margin-bottom: var(--space-md);
        }

        .empty-text {
          font-size: 0.875rem;
          color: var(--text-muted);
          font-weight: 500;
          margin-bottom: var(--space-xs);
        }

        .empty-hint {
          font-size: 0.75rem;
          color: var(--text-dim);
        }

        .model-list {
          display: flex;
          flex-direction: column;
          gap: var(--space-sm);
        }

        .model-item {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
          padding: var(--space-sm) 0;
          font-size: 0.875rem;
        }

        .model-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          background: var(--accent-cyan);
          flex-shrink: 0;
        }

        .model-name {
          color: var(--text-secondary);
          font-family: var(--font-mono);
          font-size: 0.8125rem;
        }

        .env-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: var(--space-md);
        }

        .env-item {
          display: flex;
          flex-direction: column;
          gap: 2px;
          padding: var(--space-md);
          background: var(--bg-base);
          border-radius: var(--radius-sm);
        }

        .env-key {
          font-size: 0.6875rem;
          color: var(--text-dim);
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }

        .env-value {
          font-family: var(--font-mono);
          font-size: 0.875rem;
          color: var(--text-secondary);
          font-weight: 500;
        }

        @media (max-width: 768px) {
          .metrics-grid,
          .actions-grid,
          .overview-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  )
}

export default Dashboard
