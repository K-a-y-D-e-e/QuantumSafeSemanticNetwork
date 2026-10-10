import { lazy, Suspense } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { LabSettingsProvider } from './lib/labSettings'

const Overview = lazy(() => import('./pages/Overview'))
const Scheduling = lazy(() => import('./pages/Scheduling'))
const SemanticReplay = lazy(() => import('./pages/SemanticReplay'))
const Pqc = lazy(() => import('./pages/Pqc'))
const Walkthrough = lazy(() => import('./pages/Walkthrough'))
const EndToEndLab = lazy(() => import('./pages/EndToEndLab'))

const links = [
  { to: '/', label: 'Overview', end: true },
  { to: '/demo', label: 'Interactive Demo' },
  { to: '/results', label: 'Results' },
]

export default function App() {
  return (
    <LabSettingsProvider>
      <div className="app-shell">
        <aside className="sidebar">
          <div>
            <h1 className="brand-title">Illustrative Interactive Dashboard</h1>
          </div>
          <nav className="nav" aria-label="Dashboard">
            {links.map((link) => <NavLink key={link.to} to={link.to} end={link.end}>{link.label}</NavLink>)}
          </nav>
        </aside>
        <main className="content">
          <Suspense fallback={<p className="muted">Loading dashboard page...</p>}>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/demo" element={<EndToEndLab />} />
              <Route path="/results" element={<Scheduling />} />
              {/* Preserve the earlier URLs as working routes. */}
              <Route path="/lab" element={<EndToEndLab />} />
              <Route path="/scheduling" element={<Scheduling />} />
              <Route path="/semantic" element={<SemanticReplay />} />
              <Route path="/pqc" element={<Pqc />} />
              <Route path="/walkthrough" element={<Walkthrough />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </main>
      </div>
    </LabSettingsProvider>
  )
}
