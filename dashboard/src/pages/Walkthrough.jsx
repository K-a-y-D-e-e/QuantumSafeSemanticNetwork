import { Link } from 'react-router-dom'
import { useRecordedData } from '../lib/useRecordedData'

const stages = [
  { title: 'Inputs and environment', text: 'The simulator generates network traffic and state for a single-link event-driven scheduling experiment.' },
  { title: 'Policy training', text: 'DQN training records episode reward, latency, deadline misses and exploration epsilon.' },
  { title: 'Policy evaluation', text: 'Saved comparison metrics cover Random, Priority, EDF and DQN on a recorded evaluation seed.' },
  { title: 'Dashboard review', text: 'This UI reads the committed CSV/JSON copies and renders them without running research code.' },
  { title: 'Next validation', text: 'NS-3 comparison, end-to-end semantic coupling and measured PQC overhead.' },
]

export default function Walkthrough() {
  const { comparison, training, loading, error } = useRecordedData()
  return (
    <section>
      <header className="page-heading"><div><h1>From simulation to saved result</h1><p className="lede">A guide to the artifacts behind the saved experiment.</p></div></header>

      <div className="panel"><h2>Experiment path</h2><p className="faint">The saved network experiment is one component of the broader semantic and quantum-safe research architecture.</p><div className="timeline" style={{ marginTop: 20 }}>{stages.map((stage, i) => <div className="timeline-item" key={stage.title}><strong><span className="mono" style={{ color: 'var(--cyan)', marginRight: 7 }}>{String(i + 1).padStart(2, '0')}</span>{stage.title}</strong><p>{stage.text}</p></div>)}</div></div>

      <div className="grid-3">
        <div className="metric"><div className="label">Evaluation source</div><div className="value" style={{ fontSize: 17 }}>comparison_results.json</div><div className="hint">Scheduler aggregates and evaluation metadata</div></div>
        <div className="metric"><div className="label">Training source</div><div className="value" style={{ fontSize: 17 }}>training_log.csv</div><div className="hint">Episode-level DQN history</div></div>
        <div className="metric"><div className="label">Current run data</div><div className="value" style={{ fontSize: 17 }}>{loading ? 'Loading…' : error ? 'Unavailable' : `${comparison?.num_episodes ?? '—'} eval · ${training.length} train`}</div><div className="hint">Loaded from dashboard/public/data</div></div>
      </div>

      <div className="panel"><h2>Review the evidence</h2><p className="muted">The recorded evaluation shows a latency and deadline-reliability tradeoff: DQN leads on average latency while EDF has fewer deadline misses. The training log contains episode-level reward, latency, deadline-miss and exploration history.</p><div className="subnav" style={{ marginTop: 15 }}><Link to="/scheduling">Open scheduling results →</Link><Link to="/semantic">Inspect rule replay →</Link><Link to="/pqc">Review PQC implementation scope →</Link><Link to="/">Back to system overview →</Link></div></div>

      <div className="panel"><h2>Next evaluation steps</h2><div className="grid-2" style={{ marginTop: 12 }}><div><strong>Network</strong><p className="muted">Compare the simulator against NS-3 and report repeated-seed variation.</p></div><div><strong>Semantic policy</strong><p className="muted">Connect trained encoder/PPO artifacts and reconstruction metrics to the network evaluation.</p></div><div><strong>Cryptography</strong><p className="muted">Record handshake, encryption and payload-size measurements using the Python implementation.</p></div><div><strong>Reproducibility</strong><p className="muted">Store configuration, model revision and per-run results alongside aggregate comparisons.</p></div></div></div>
    </section>
  )
}
