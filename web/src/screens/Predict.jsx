import { useState, useEffect } from 'react'
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

function Predict() {
  const [model, setModel] = useState('')
  const [availableModels, setAvailableModels] = useState([])
  const [city, setCity] = useState(CITIES[0])
  const [date, setDate] = useState(new Date().toISOString().split('T')[0])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [modelError, setModelError] = useState(null)

  const fetchModels = async () => {
    setModelError(null)
    try {
      const res = await axios.get('http://localhost:8000/api/models/')
      setAvailableModels(res.data.models || [])
      if (res.data.models && res.data.models.length > 0 && !model) {
        setModel(res.data.models[0])
      }
    } catch (err) {
      console.error('Failed to fetch models:', err)
      setModelError('Failed to load models. Is the backend running?')
    }
  }

  const validateThailandLocation = (lat, lon) => {
    // Thailand bounds: lat ~5.6 to 20.5, lon ~97.3 to 105.6
    if (lat < 5 || lat > 21) return 'Latitude out of Thailand bounds (5°N - 21°N)'
    if (lon < 97 || lon > 106) return 'Longitude out of Thailand bounds (97°E - 106°E)'
    return null
  }

  const runPrediction = async () => {
    if (!model) {
      alert('Please select a model first')
      return
    }
    const validation = validateThailandLocation(city.lat, city.lng)
    if (validation) {
      alert(validation)
      return
    }
    setLoading(true)
    setResult(null)
    try {
      const res = await axios.post('http://localhost:8000/api/predict/', {
        model_name: model,
        date,
        lat: city.lat,
        lon: city.lng
      })
      setResult(res.data)
    } catch (err) {
      console.error('Prediction failed:', err)
      alert('Prediction failed: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchModels()
  }, [])

  return (
    <div className="predict-screen">
      <div className="form-panel">
        <div className="form-header">
          <span className="form-badge">Input Parameters</span>
        </div>

        <div className="form-body">
          <div className="field-group">
            <label className="field-label">
              Model
              <span className="field-required">Required</span>
            </label>
            <select
              className="field-input"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={availableModels.length === 0}
              key="model-select"
            >
              <option value="">Select a model checkpoint...</option>
              {availableModels.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            {modelError && (
              <span className="field-error">{modelError}</span>
            )}
            <span className="field-hint">
              {modelError ? '' : availableModels.length > 0 ? `${availableModels.length} models available` : 'Loading models...'}
            </span>
          </div>

          <div className="field-group">
            <label className="field-label">
              Target Date
            </label>
            <input
              type="date"
              className="field-input"
              value={date}
              onChange={(e) => setDate(e.target.value)}
            />
          </div>

          <div className="field-group">
            <label className="field-label">
              Location
            </label>
            <select
              className="field-input"
              value={city.name}
              onChange={(e) => {
                const c = CITIES.find(c => c.name === e.target.value)
                if (c) setCity(c)
              }}
            >
              {CITIES.map(c => (
                <option key={c.name} value={c.name}>{c.name}</option>
              ))}
            </select>
            <div className="coords-display">
              <span className="coord-pill">{city.lat.toFixed(4)}°N</span>
              <span className="coord-pill">{city.lng.toFixed(4)}°E</span>
            </div>
          </div>

          <button
            className={`predict-btn ${loading ? 'loading' : ''}`}
            onClick={runPrediction}
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="btn-spinner" />
                Processing...
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
                </svg>
                Run Prediction
              </>
            )}
          </button>
        </div>
      </div>

      {result && (
        <div className="results-panel">
          <div className="results-header">
            <span className="results-badge">Prediction Results</span>
            <span className="results-time">{new Date().toLocaleTimeString()}</span>
          </div>

          <div className="results-grid">
            <div className={`result-card ${result.is_heatwave ? 'heatwave' : 'normal'}`}>
              <div className="result-icon">
                {result.is_heatwave ? (
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 9a4 4 0 0 0-2 7.5" /><path d="M12 3v2" /><path d="M6.6 18.4A9 9 0 0 0 20 9" /><path d="M20 9v4" /><path d="M16.2 7.8 19 5" />
                  </svg>
                ) : (
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 3v2" /><path d="M12 19v2" /><path d="M3 12h2" /><path d="M19 12h2" /><path d="m5.64 5.64 1.41 1.41" /><path d="m16.95 16.95 1.41 1.41" /><path d="m5.64 18.36 1.41-1.41" /><path d="m16.95 7.05 1.41-1.41" />
                  </svg>
                )}
              </div>
              <div className="result-status">{result.is_heatwave ? 'Heatwave Detected' : 'Normal Conditions'}</div>
            </div>

            <div className="result-card">
              <div className="result-label">Temperature</div>
              <div className="result-value">{result.temperature?.toFixed(1)}<span className="result-unit">°C</span></div>
            </div>

            <div className="result-card">
              <div className="result-label">Probability</div>
              <div className="result-value">{(result.heatwave_probability * 100).toFixed(0)}<span className="result-unit">%</span></div>
              <div className="result-bar">
                <div className="result-bar-fill" style={{ width: `${result.heatwave_probability * 100}%` }} />
              </div>
            </div>
          </div>

          <div className="result-meta">
            <div className="meta-item">
              <span className="meta-key">Location</span>
              <span className="meta-val">{city.name}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Coordinates</span>
              <span className="meta-val">{city.lat.toFixed(4)}, {city.lng.toFixed(4)}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Model</span>
              <span className="meta-val">{model || 'Default'}</span>
            </div>
          </div>
        </div>
      )}

      <style>{`
        .predict-screen {
          max-width: 680px;
          margin: 0 auto;
          animation: fadeIn 0.5s ease-out;
        }

        .form-panel {
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
          overflow: hidden;
          margin-bottom: var(--space-xl);
          box-shadow: 0 4px 24px rgba(0,0,0,0.08);
          transition: box-shadow 0.3s ease;
        }

        .form-panel:hover {
          box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        }

        .form-header {
          padding: var(--space-lg) var(--space-xl);
          border-bottom: 1px solid var(--border-subtle);
          background: linear-gradient(135deg, var(--bg-elevated), var(--bg-surface));
          display: flex;
          align-items: center;
          gap: var(--space-md);
        }

        .form-badge {
          font-size: 0.6875rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          color: var(--accent-cyan);
          background: var(--accent-cyan-dim);
          padding: var(--space-xs) var(--space-md);
          border-radius: 100px;
        }

        .form-body {
          padding: var(--space-2xl);
          display: flex;
          flex-direction: column;
          gap: var(--space-xl);
        }

        .field-group {
          display: flex;
          flex-direction: column;
          gap: var(--space-sm);
          position: relative;
        }

        .field-group:focus-within .field-label {
          color: var(--accent-cyan);
        }

        .field-label {
          display: flex;
          align-items: center;
          justify-content: space-between;
          font-size: 0.8125rem;
          font-weight: 600;
          color: var(--text-secondary);
          transition: color 0.2s ease;
        }

        .field-required {
          font-size: 0.625rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--accent-amber);
          background: var(--accent-amber-dim);
          padding: 2px var(--space-sm);
          border-radius: 4px;
        }

        .field-input {
          width: 100%;
          padding: var(--space-md) var(--space-lg);
          background: var(--bg-base);
          border: 2px solid var(--border-default);
          border-radius: var(--radius-sm);
          color: var(--text-primary);
          font-family: var(--font-ui);
          font-size: 0.875rem;
          outline: none;
          transition: all 0.2s ease;
        }

        .field-input:focus {
          border-color: var(--accent-cyan);
          box-shadow: 0 0 0 3px var(--accent-cyan-dim);
        }

        .field-input:hover:not(:focus) {
          border-color: var(--border-strong);
        }

        .field-hint {
          font-size: 0.75rem;
          color: var(--text-dim);
          margin-top: var(--space-xs);
          display: flex;
          align-items: center;
          gap: var(--space-xs);
        }

        .field-error {
          display: block;
          font-size: 0.75rem;
          color: var(--accent-rose);
          margin-top: var(--space-xs);
          padding: var(--space-xs) var(--space-sm);
          background: var(--accent-rose-dim);
          border-radius: var(--radius-sm);
          border-left: 3px solid var(--accent-rose);
        }

        .coords-display {
          display: flex;
          gap: var(--space-sm);
          margin-top: var(--space-xs);
        }

        .coord-pill {
          font-family: var(--font-mono);
          font-size: 0.75rem;
          color: var(--text-muted);
          background: var(--bg-elevated);
          padding: var(--space-xs) var(--space-md);
          border-radius: 100px;
          border: 1px solid var(--border-subtle);
          transition: all 0.2s ease;
        }

        .coord-pill:hover {
          background: var(--bg-overlay);
          border-color: var(--accent-cyan);
          color: var(--accent-cyan);
        }

        .predict-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: var(--space-sm);
          padding: var(--space-md) var(--space-xl);
          background: linear-gradient(135deg, var(--accent-cyan), #0891b2);
          color: var(--bg-base);
          border: none;
          border-radius: var(--radius-sm);
          font-family: var(--font-ui);
          font-size: 0.9375rem;
          font-weight: 700;
          cursor: pointer;
          transition: all 0.2s ease;
          margin-top: var(--space-sm);
          box-shadow: 0 4px 12px rgba(6, 182, 212, 0.25);
          position: relative;
          overflow: hidden;
        }

        .predict-btn::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
          transition: left 0.5s ease;
        }

        .predict-btn:hover::before {
          left: 100%;
        }

        .predict-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 24px rgba(6, 182, 212, 0.35);
        }

        .predict-btn:active {
          transform: translateY(0);
        }

        .predict-btn.loading {
          opacity: 0.8;
          cursor: not-allowed;
          transform: none;
        }

        .btn-spinner {
          width: 16px;
          height: 16px;
          border: 2px solid rgba(255,255,255,0.3);
          border-top-color: var(--bg-base);
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .results-panel {
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
          overflow: hidden;
          animation: slideUp 0.5s ease-out;
          box-shadow: 0 4px 24px rgba(0,0,0,0.08);
        }

        @keyframes slideUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }

        .results-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: var(--space-lg) var(--space-xl);
          border-bottom: 1px solid var(--border-subtle);
          background: linear-gradient(135deg, var(--bg-elevated), var(--bg-surface));
        }

        .results-badge {
          font-size: 0.6875rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          color: var(--accent-emerald);
          background: var(--accent-emerald-dim);
          padding: var(--space-xs) var(--space-md);
          border-radius: 100px;
        }

        .results-time {
          font-family: var(--font-mono);
          font-size: 0.6875rem;
          color: var(--text-dim);
        }

        .results-grid {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          gap: var(--space-lg);
          padding: var(--space-xl);
        }

        .result-card {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          padding: var(--space-xl);
          background: var(--bg-base);
          border: 2px solid var(--border-subtle);
          border-radius: var(--radius-md);
          gap: var(--space-sm);
          transition: all 0.3s ease;
          position: relative;
          overflow: hidden;
        }

        .result-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          height: 3px;
          background: var(--accent-cyan);
          opacity: 0;
          transition: opacity 0.3s ease;
        }

        .result-card:hover::before {
          opacity: 1;
        }

        .result-card:hover {
          transform: translateY(-4px);
          box-shadow: 0 12px 32px rgba(0,0,0,0.15);
          border-color: var(--border-strong);
        }

        .result-card.heatwave {
          border-color: var(--accent-rose);
          background: linear-gradient(135deg, var(--accent-rose-dim), var(--bg-base));
        }

        .result-card.heatwave::before {
          background: var(--accent-rose);
          opacity: 1;
        }

        .result-card.normal {
          border-color: var(--accent-emerald);
          background: linear-gradient(135deg, var(--accent-emerald-dim), var(--bg-base));
        }

        .result-card.normal::before {
          background: var(--accent-emerald);
          opacity: 1;
        }

        .result-icon {
          color: var(--text-secondary);
          transition: transform 0.3s ease;
        }

        .result-card:hover .result-icon {
          transform: scale(1.1);
        }

        .result-card.heatwave .result-icon {
          color: var(--accent-rose);
        }

        .result-card.normal .result-icon {
          color: var(--accent-emerald);
        }

        .result-status {
          font-size: 0.875rem;
          font-weight: 700;
          color: var(--text-primary);
        }

        .result-label {
          font-size: 0.625rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--text-dim);
          font-weight: 600;
        }

        .result-value {
          font-family: var(--font-mono);
          font-size: 2rem;
          font-weight: 800;
          color: var(--text-primary);
          line-height: 1;
          background: linear-gradient(135deg, var(--text-primary), var(--text-secondary));
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .result-card.heatwave .result-value {
          background: linear-gradient(135deg, var(--accent-rose), #fb7185);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .result-card.normal .result-value {
          background: linear-gradient(135deg, var(--accent-emerald), #34d399);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }

        .result-unit {
          font-size: 1rem;
          font-weight: 500;
          color: var(--text-muted);
          margin-left: 2px;
        }

        .result-bar {
          width: 100%;
          height: 6px;
          background: var(--bg-overlay);
          border-radius: 3px;
          overflow: hidden;
          margin-top: var(--space-xs);
          position: relative;
        }

        .result-bar-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--accent-cyan), #22d3ee);
          border-radius: 3px;
          transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
        }

        .result-bar-fill::after {
          content: '';
          position: absolute;
          top: 0;
          right: 0;
          bottom: 0;
          width: 20px;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3));
        }

        .result-meta {
          display: flex;
          gap: var(--space-xl);
          padding: var(--space-lg) var(--space-xl);
          border-top: 1px solid var(--border-subtle);
          background: var(--bg-elevated);
        }

        .meta-item {
          display: flex;
          flex-direction: column;
          gap: var(--space-xs);
          padding: var(--space-sm) var(--space-md);
          background: var(--bg-base);
          border-radius: var(--radius-sm);
          border: 1px solid var(--border-subtle);
          flex: 1;
          transition: all 0.2s ease;
        }

        .meta-item:hover {
          border-color: var(--border-strong);
          transform: translateY(-2px);
        }

        .meta-key {
          font-size: 0.625rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--text-dim);
          font-weight: 600;
        }

        .meta-val {
          font-family: var(--font-mono);
          font-size: 0.875rem;
          color: var(--text-secondary);
          font-weight: 600;
        }

        @media (max-width: 768px) {
          .predict-screen {
            max-width: 100%;
            padding: 0 var(--space-md);
          }
          .form-body {
            padding: var(--space-lg);
          }
          .results-grid {
            grid-template-columns: 1fr;
            gap: var(--space-md);
          }
          .result-meta {
            flex-direction: column;
            gap: var(--space-sm);
          }
          .meta-item {
            flex: none;
          }
        }
      `}</style>
    </div>
  )
}

export default Predict
