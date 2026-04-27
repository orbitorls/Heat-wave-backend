import { useState, useEffect } from 'react'
import axios from 'axios'

function Data() {
  const [files, setFiles] = useState([])
  const [downloadStatus, setDownloadStatus] = useState(null)
  const [isDownloading, setIsDownloading] = useState(false)
  const [error, setError] = useState(null)

  const fetchFiles = async () => {
    setError(null)
    try {
      const res = await axios.get('http://localhost:8000/api/data/files')
      setFiles(res.data.files || [])
    } catch (err) {
      console.error('Failed to fetch data files:', err)
      setError('Failed to load data files. Is the backend running?')
    }
  }

  const downloadERA5 = async () => {
    try {
      setIsDownloading(true)
      setDownloadStatus({ type: 'info', text: 'Starting download...' })

      const res = await axios.post('http://localhost:8000/api/data/download', {
        year: 2023,
        month: 1
      })

      const jobId = res.data.job_id
      setDownloadStatus({ type: 'info', text: `Download started (Job: ${jobId})` })

      // Poll for status
      const checkStatus = setInterval(async () => {
        try {
          const statusRes = await axios.get(`http://localhost:8000/api/data/download/${jobId}`)
          const job = statusRes.data

          if (job.status === 'completed') {
            clearInterval(checkStatus)
            setDownloadStatus({ type: 'success', text: `Download completed: ${job.file}` })
            setIsDownloading(false)
            fetchFiles()  // Refresh file list
          } else if (job.status === 'failed') {
            clearInterval(checkStatus)
            setDownloadStatus({ type: 'error', text: 'Download failed. Check logs.' })
            setIsDownloading(false)
          }
        } catch (err) {
          console.error('Status check failed:', err)
        }
      }, 3000)

    } catch (err) {
      console.error('Download failed:', err)
      setDownloadStatus({ type: 'error', text: 'Failed to start download' })
      setIsDownloading(false)
    }
  }

  useEffect(() => {
    fetchFiles()
  }, [])

  return (
    <div className="data-screen">
      <div className="data-toolbar">
        <div className="data-stats">
          <div className="data-stat">
            <span className="data-stat-value">{files.length}</span>
            <span className="data-stat-label">ERA5 Files</span>
          </div>
        </div>
        <div className="data-actions">
          <button className="data-btn secondary" onClick={fetchFiles}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21.5 2v6h-6M2.5 22v-6h6M2 11.5a10 10 0 0 1 18.8-4.3M22 12.5a10 10 0 0 1-18.8 4.2"/>
            </svg>
            Refresh
          </button>
          <button
            className="data-btn primary"
            onClick={downloadERA5}
            disabled={isDownloading}
          >
            {isDownloading ? (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="spin">
                  <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                </svg>
                Downloading...
              </>
            ) : (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Download ERA5
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="data-error">
          <span className="error-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </span>
          {error}
          <button className="retry-btn" onClick={fetchFiles}>Retry</button>
        </div>
      )}

      {downloadStatus && (
        <div className={`download-status ${downloadStatus.type}`}>
          {downloadStatus.text}
        </div>
      )}

      <div className="data-table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th className="col-actions">Actions</th>
            </tr>
          </thead>
          <tbody>
            {files.length === 0 ? (
              <tr>
                <td colSpan="2" className="empty-cell">
                  <div className="table-empty">
                    <div className="table-empty-icon">
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                      </svg>
                    </div>
                    <p className="table-empty-text">No ERA5 data files found</p>
                    <p className="table-empty-hint">Download ERA5 data or check your data directory</p>
                  </div>
                </td>
              </tr>
            ) : (
              files.map((file, i) => (
                <tr key={i} className="data-row">
                  <td>
                    <div className="file-cell">
                      <span className="file-icon">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                          <polyline points="14 2 14 8 20 8" />
                        </svg>
                      </span>
                      <span className="file-name">{file}</span>
                    </div>
                  </td>
                  <td className="col-actions">
                    <button className="action-btn audit" onClick={async () => {
                      try {
                        const res = await axios.post('http://localhost:8000/api/data/audit')
                        alert(`Audit result: ${res.data.status}\nFiles checked: ${res.data.files_checked}\nIssues: ${res.data.issues.length > 0 ? res.data.issues.join(', ') : 'None'}`)
                      } catch (err) {
                        alert('Audit failed: ' + (err.response?.data?.detail || err.message))
                      }
                    }}>Audit</button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <style>{`
        .data-screen {
          animation: fadeIn 0.4s ease-out;
        }

        .data-toolbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: var(--space-xl);
        }

        .data-stat {
          display: flex;
          align-items: baseline;
          gap: var(--space-sm);
        }

        .data-stat-value {
          font-family: var(--font-mono);
          font-size: 1.25rem;
          font-weight: 700;
          color: var(--text-primary);
        }

        .data-stat-label {
          font-size: 0.8125rem;
          color: var(--text-muted);
        }

        .data-actions {
          display: flex;
          gap: var(--space-md);
        }

        .data-btn {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
          padding: var(--space-sm) var(--space-lg);
          border-radius: var(--radius-sm);
          font-family: var(--font-ui);
          font-size: 0.875rem;
          font-weight: 600;
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .data-btn.secondary {
          background: var(--bg-elevated);
          color: var(--text-secondary);
          border: 1px solid var(--border-default);
        }

        .data-btn.secondary:hover {
          background: var(--bg-overlay);
          border-color: var(--border-strong);
        }

        .data-btn.primary {
          background: var(--accent-emerald);
          color: var(--bg-base);
          border: none;
        }

        .data-btn.primary:hover {
          filter: brightness(1.1);
        }

        .data-table-wrap {
          background: var(--bg-surface);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-md);
          overflow: hidden;
        }

        .data-table {
          width: 100%;
          border-collapse: collapse;
        }

        .data-table thead th {
          padding: var(--space-md) var(--space-lg);
          text-align: left;
          font-size: 0.6875rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--text-muted);
          background: var(--bg-elevated);
          border-bottom: 1px solid var(--border-subtle);
        }

        .data-table tbody td {
          padding: var(--space-md) var(--space-lg);
          border-bottom: 1px solid var(--border-subtle);
        }

        .data-table tbody tr:last-child td {
          border-bottom: none;
        }

        .data-table tbody tr:last-child td.empty-cell {
          padding: 0;
        }

        .file-cell {
          display: flex;
          align-items: center;
          gap: var(--space-md);
        }

        .file-icon {
          color: var(--text-dim);
          flex-shrink: 0;
        }

        .file-name {
          font-family: var(--font-mono);
          font-size: 0.8125rem;
          color: var(--text-secondary);
        }

        .col-actions {
          text-align: right;
          width: 120px;
        }

        .action-btn {
          padding: var(--space-xs) var(--space-md);
          background: var(--bg-base);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-sm);
          color: var(--text-muted);
          font-family: var(--font-ui);
          font-size: 0.75rem;
          font-weight: 600;
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .action-btn:hover {
          background: var(--bg-elevated);
          border-color: var(--border-strong);
          color: var(--text-secondary);
        }

        .table-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: var(--space-3xl);
          text-align: center;
          color: var(--text-dim);
        }

        .table-empty-icon {
          margin-bottom: var(--space-md);
          color: var(--text-muted);
        }

        .table-empty-text {
          font-size: 0.9375rem;
          font-weight: 600;
          color: var(--text-secondary);
          margin-bottom: var(--space-xs);
        }

        .table-empty-hint {
          font-size: 0.8125rem;
          color: var(--text-dim);
        }

        .data-error {
          display: flex;
          align-items: center;
          gap: var(--space-md);
          padding: var(--space-md) var(--space-lg);
          background: var(--accent-rose-dim);
          border: 1px solid rgba(244, 63, 94, 0.2);
          border-radius: var(--radius-md);
          color: var(--accent-rose);
          font-size: 0.875rem;
          margin-bottom: var(--space-lg);
        }

        .data-error .retry-btn {
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

        .download-status {
          padding: var(--space-md) var(--space-lg);
          border-radius: var(--radius-sm);
          font-size: 0.8125rem;
          font-weight: 500;
          margin-bottom: var(--space-lg);
          animation: fadeIn 0.3s ease-out;
        }

        .download-status.success {
          background: var(--accent-emerald-dim);
          color: var(--accent-emerald);
          border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .download-status.error {
          background: var(--accent-rose-dim);
          color: var(--accent-rose);
          border: 1px solid rgba(244, 63, 94, 0.2);
        }

        .download-status.info {
          background: var(--accent-cyan-dim);
          color: var(--accent-cyan);
          border: 1px solid rgba(6, 182, 212, 0.2);
        }

        .spin {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
          .data-toolbar {
            flex-direction: column;
            align-items: stretch;
            gap: var(--space-md);
          }
          .data-actions {
            justify-content: flex-end;
          }
          .data-table thead th {
            padding: var(--space-sm) var(--space-md);
          }
          .data-table tbody td {
            padding: var(--space-sm) var(--space-md);
          }
        }
      `}</style>
    </div>
  )
}

export default Data
