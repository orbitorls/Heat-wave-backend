import { Outlet, Link, useLocation } from 'react-router-dom'
import { useState, useEffect } from 'react'

const navItems = [
  { path: '/', label: 'Dashboard', id: 'dash' },
  { path: '/predict', label: 'Prediction', id: 'pred' },
  { path: '/train', label: 'Training', id: 'train' },
  { path: '/map', label: 'Map View', id: 'map' },
  { path: '/eval', label: 'Evaluation', id: 'eval' },
  { path: '/data', label: 'Data', id: 'data' },
  { path: '/checkpoints', label: 'Checkpoints', id: 'chk' },
  { path: '/logs', label: 'System Logs', id: 'logs' },
]

function Layout() {
  const location = useLocation()
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'dark')
  const currentPage = navItems.find(i => i.path === location.pathname)?.label || 'Dashboard'

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark')
  }

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
            </svg>
          </div>
          <div className="brand-text">
            <div className="brand-title">AGNI</div>
            <div className="brand-subtitle">Heatwave Forecast</div>
          </div>
        </div>

        <nav className="sidebar-nav">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path
            return (
              <Link
                key={item.id}
                to={item.path}
                className={`nav-item ${isActive ? 'active' : ''}`}
              >
                <span className="nav-indicator" />
                <span className="nav-label">{item.label}</span>
              </Link>
            )
          })}
        </nav>

        <div className="sidebar-footer">
          <button className="theme-toggle" onClick={toggleTheme}>
            {theme === 'dark' ? (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="5" />
                  <line x1="12" y1="1" x2="12" y2="3" />
                  <line x1="12" y1="21" x2="12" y2="23" />
                  <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                  <line x1="1" y1="12" x2="3" y2="12" />
                  <line x1="21" y1="12" x2="23" y2="12" />
                  <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                  <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                </svg>
                <span>Light Mode</span>
              </>
            ) : (
              <>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                </svg>
                <span>Dark Mode</span>
              </>
            )}
          </button>
          <div className="status-pill">
            <span className="status-dot live" />
            <span className="status-text">System Online</span>
          </div>
          <div className="version-text">v2.1.0</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="page-header">
          <h1 className="page-title">{currentPage}</h1>
          <div className="page-meta">
            <span className="meta-label">Thailand Heatwave Prediction System</span>
            <span className="meta-divider" />
            <span className="meta-time">{new Date().toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}</span>
          </div>
        </header>
        <div className="content-area">
          <Outlet />
        </div>
      </main>

      <style>{`
        .app-container {
          display: flex;
          height: 100vh;
          background: var(--bg-base);
          overflow: hidden;
        }

        .sidebar {
          width: 240px;
          background: var(--bg-surface);
          border-right: 1px solid var(--border-default);
          display: flex;
          flex-direction: column;
          padding: var(--space-xl) 0;
          flex-shrink: 0;
        }

        .sidebar-brand {
          display: flex;
          align-items: center;
          gap: var(--space-md);
          padding: 0 var(--space-xl);
          margin-bottom: var(--space-2xl);
        }

        .brand-icon {
          color: var(--accent-amber);
          display: flex;
          align-items: center;
        }

        .brand-text {
          display: flex;
          flex-direction: column;
        }

        .brand-title {
          font-family: var(--font-display);
          font-size: 1.25rem;
          font-weight: 700;
          letter-spacing: -0.02em;
          line-height: 1.2;
        }

        .brand-subtitle {
          font-size: 0.7rem;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 0.08em;
          margin-top: 2px;
        }

        .sidebar-nav {
          display: flex;
          flex-direction: column;
          gap: 2px;
          padding: 0 var(--space-sm);
          flex: 1;
        }

        .nav-item {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
          padding: var(--space-md) var(--space-lg);
          border-radius: var(--radius-sm);
          color: var(--text-muted);
          text-decoration: none;
          font-size: 0.875rem;
          font-weight: 500;
          transition: all var(--transition-fast);
          position: relative;
        }

        .nav-item:hover {
          color: var(--text-secondary);
          background: var(--bg-elevated);
        }

        .nav-item.active {
          color: var(--text-primary);
          background: var(--bg-elevated);
        }

        .nav-indicator {
          width: 3px;
          height: 0;
          border-radius: 2px;
          background: var(--accent-amber);
          transition: height var(--transition-fast);
        }

        .nav-item.active .nav-indicator {
          height: 20px;
        }

        .nav-label {
          transition: transform var(--transition-fast);
        }

        .nav-item:hover .nav-label {
          transform: translateX(2px);
        }

        .sidebar-footer {
          padding: 0 var(--space-xl);
          margin-top: auto;
          display: flex;
          flex-direction: column;
          gap: var(--space-sm);
        }

        .theme-toggle {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: var(--space-sm);
          padding: var(--space-sm) var(--space-md);
          background: var(--bg-elevated);
          border: 1px solid var(--border-default);
          border-radius: var(--radius-sm);
          color: var(--text-secondary);
          font-family: var(--font-ui);
          font-size: 0.75rem;
          font-weight: 600;
          cursor: pointer;
          transition: all var(--transition-fast);
        }

        .theme-toggle:hover {
          background: var(--bg-overlay);
          border-color: var(--border-strong);
          color: var(--text-primary);
        }

        .status-pill {
          display: flex;
          align-items: center;
          gap: var(--space-sm);
          padding: var(--space-sm) var(--space-md);
          background: var(--bg-elevated);
          border: 1px solid var(--border-subtle);
          border-radius: 100px;
          font-size: 0.75rem;
          color: var(--text-muted);
        }

        .status-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          flex-shrink: 0;
        }

        .status-dot.live {
          background: var(--status-live);
          box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
          animation: pulse 2s ease-in-out infinite;
        }

        .version-text {
          font-size: 0.6875rem;
          color: var(--text-dim);
          font-family: var(--font-mono);
        }

        .main-content {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .page-header {
          padding: var(--space-2xl) var(--space-2xl) var(--space-lg);
          border-bottom: 1px solid var(--border-subtle);
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          flex-shrink: 0;
        }

        .page-title {
          font-family: var(--font-display);
          font-size: 1.5rem;
          font-weight: 700;
          letter-spacing: -0.02em;
          color: var(--text-primary);
        }

        .page-meta {
          display: flex;
          align-items: center;
          gap: var(--space-md);
          font-size: 0.75rem;
          color: var(--text-muted);
        }

        .meta-divider {
          width: 3px;
          height: 3px;
          border-radius: 50%;
          background: var(--text-dim);
        }

        .meta-time {
          font-family: var(--font-mono);
        }

        .content-area {
          flex: 1;
          padding: var(--space-xl) var(--space-2xl);
          overflow-y: auto;
          animation: fadeIn 0.4s ease-out;
        }

        @media (max-width: 768px) {
          .sidebar {
            display: none;
          }
        }
      `}</style>
    </div>
  )
}

export default Layout
