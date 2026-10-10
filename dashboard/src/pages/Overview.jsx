import { Link } from 'react-router-dom'
import { useRecordedData } from '../lib/useRecordedData'

const stages = [
  { name: 'JIGSAWS input', detail: 'Source data', to: '/demo', stage: 'input' },
  { name: 'Semantic encoder', detail: 'Retention assumption', to: '/demo', stage: 'semantic' },
  { name: 'Orchestration + PPO', detail: 'Rule replay', to: '/demo', stage: 'orchestration' },
  { name: 'PQC security', detail: 'Overhead assumptions', to: '/demo', stage: 'security' },
  { name: 'Network scheduler', detail: 'Recorded policy results', to: '/results' },
  { name: 'End-to-end output', detail: 'Scenario estimates', to: '/demo', stage: 'output' },
]

export default function Overview() {
  const { error, comparison } = useRecordedData()
  const results = comparison?.results ?? {}
  const latency = results.DQN?.avg_latency_us
  const missRate = results.EDF?.deadline_miss_rate

  return <section>
    <header className="page-heading overview-heading">
      <div><h1>Project Interactive Dashboard</h1></div>
      <Link className="run-button" to="/demo">Open Interactive Demo</Link>
    </header>

    <div className="panel overview-pipeline">
      <div className="panel-head"><div><h2>Six-stage system path</h2><p className="faint">Choose a stage to open its controls or results.</p></div></div>
      <div className="overview-stage-grid">{stages.map((stage, index) => <Link to={stage.to} state={{ stage: stage.stage }} key={stage.name} className="overview-stage"><span className="pipeline-node-index">{String(index + 1).padStart(2, '0')}</span><strong>{stage.name}</strong><small>{stage.detail}</small></Link>)}</div>
    </div>

    <div className="panel overview-findings">
      <div className="panel-head"><div><h2>Key findings</h2><p className="faint">Recorded result · Python event-simulator evaluation.</p></div><Link className="badge recorded" to="/results">View recorded results</Link></div>
      {error ? <p className="muted">Recorded findings unavailable: {error}</p> : <div className="grid-2">
        <div className="finding"><span className="finding-mark cyan-mark">01</span><div><strong>DQN has the lowest average latency</strong><p>{latency == null ? 'Loading saved comparison…' : `${latency.toFixed(1)} us in the saved evaluation.`}</p></div></div>
        <div className="finding"><span className="finding-mark amber-mark">02</span><div><strong>EDF has the lowest deadline-miss rate</strong><p>{missRate == null ? 'Loading saved comparison…' : `${(missRate * 100).toFixed(1)}% in the saved evaluation.`}</p></div></div>
      </div>}
      <p className="overview-caption">Scenario outputs use editable assumptions; recorded results are shown separately.</p>
    </div>
  </section>
}
