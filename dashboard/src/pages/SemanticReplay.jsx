import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import {
  ACTION_LABELS,
  LATENT_DIM,
  STATE_FIELDS,
  replayController,
} from '../lib/orchestration'
import { useLabSettings } from '../lib/labSettingsContext'
import { derivePipeline } from '../lib/pipeline'

export default function SemanticReplay() {
  const { settings, updateSetting, updateSemanticState } = useLabSettings()
  const { semanticState: state, assumedAction, applyBias } = settings

  const result = useMemo(
    () => replayController(state, assumedAction, applyBias),
    [state, assumedAction, applyBias],
  )
  const pipeline = useMemo(() => derivePipeline(settings), [settings])

  function setField(key, value) {
    updateSemanticState(key, value)
  }

  return (
    <section>
      <header className="page-heading">
        <div><h1>Semantic compression and orchestration</h1>
      <p className="lede">
        Rule replay of the published objective table in <span className="mono">orchestration/agent.py</span> and
        analytic reward in <span className="mono">SemanticCompressionEnv._step_analytic</span>.
        </p></div>
        <div className="heading-actions"><span className="badge replay">Rule-based replay</span></div>
      </header>

      <div className="grid-2">
        <div className="panel">
          <h2>Observation (7-D)</h2>
          <p className="faint">
            Values are normalized proxies in [0, 1], matching the environment interface. Changing a slider updates the
            rule evaluation immediately.
          </p>
          {STATE_FIELDS.map((field) => (
            <label className="slider-block" key={field.key}>
              <span>{field.label}</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={state[field.key]}
                onChange={(e) => setField(field.key, Number(e.target.value))}
              />
              <span className="mono">{state[field.key].toFixed(2)}</span>
            </label>
          ))}
          <p className="faint" style={{ marginTop: 8 }}>
            <span className="mono">decide_objective()</span> uses only task criticality, network load, and latency.
            The communicator passes <span className="mono">state[5]</span> as <span className="mono">security_requirement</span>
            , but that argument is unused in the Python function.
          </p>
        </div>

        <div className="stack">
          <div className="panel">
            <h2>Hypothetical policy action</h2>
            <p className="faint">Select an action (0–3); the objective rule can bias the selected action.</p>
            <div className="action-pills">
              {ACTION_LABELS.map((label, index) => (
                <button
                  key={label}
                  type="button"
                  className={`pill ${assumedAction === index ? 'active' : ''}`}
                  onClick={() => updateSetting('assumedAction', index)}
                >
                  {index}: keep {label}
                </button>
              ))}
            </div>
            <label className="toggle" style={{ marginTop: 12 }}>
              <input
                type="checkbox"
                checked={applyBias}
                onChange={(e) => updateSetting('applyBias', e.target.checked)}
              />
              Apply <span className="mono">bias_action_for_objective</span>
            </label>
          </div>

          <div className="panel">
            <div className="row">
              <span className="badge replay">Replay output</span>
            </div>
            <h2 style={{ marginTop: 10 }}>{result.objective.replaceAll('_', ' ')}</h2>
            <p className="muted">
              Fired rule: <span className="mono">{result.rule}</span>
            </p>
            <p className="muted">
              Assumed action {result.rawAction} ({ACTION_LABELS[result.rawAction]}) → biased action {result.action} (
              {ACTION_LABELS[result.action]}) → retained fraction{' '}
              <span className="mono">{result.compression.toFixed(2)}</span> → keep {result.keepDim} / {LATENT_DIM}{' '}
              latent dims (<span className="mono">max(1, int(16 * level))</span>).
            </p>
            <div className="latent" aria-hidden="true">
              {Array.from({ length: LATENT_DIM }, (_, i) => (
                <i key={i} className={i < result.keepDim ? 'on' : ''} />
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="grid-2" style={{ marginTop: 14 }}>
        <div className="panel">
          <h2>Objective reward weights</h2>
          <p className="faint">
            From <span className="mono">AIOrchestrationAgent.REWARD_WEIGHTS</span>. Used in reconstruction-reward
            training mode in Python; the analytic replay below uses its reward formula directly.
          </p>
          <table>
            <thead>
              <tr>
                <th>Term</th>
                <th>Multiplier</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.weights).map(([key, value]) => (
                <tr key={key}>
                  <td>{key}</td>
                  <td className="num">{value.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h2>Analytic reward proxy</h2>
          <p className="faint">
            Exact formula from <span className="mono">_step_analytic</span>: quality = compression; bandwidth_cost =
            compression; estimated_latency = latency + 0.3·bandwidth_cost + 0.2·network_load; reward = quality −
            0.5·estimated_latency − 0.3·bandwidth_cost; then −0.5 if criticality &gt; 0.7 and quality &lt; 0.75.
          </p>
          <div className="grid-2">
            <div className="metric">
              <div className="label">Reward</div>
              <div className="value">{result.analytic.reward.toFixed(4)}</div>
            </div>
            <div className="metric">
              <div className="label">Est. latency (clipped)</div>
              <div className="value">{result.analytic.estimated_latency.toFixed(4)}</div>
              <div className="hint">raw {result.analytic.estimated_latency_raw.toFixed(4)}</div>
            </div>
          </div>
          <p className="muted" style={{ marginTop: 10 }}>
            Criticality penalty applied: {result.analytic.criticalityPenalty ? 'yes' : 'no'}. Semantic quality proxy
            equals retained fraction ({result.analytic.semantic_quality.toFixed(2)}).
          </p>
        </div>
      </div>

      <div className="panel layer-experiment" style={{ marginTop: 14 }}>
        <div className="panel-head"><div><h2>Layer experiment: retention to packet footprint</h2><p className="faint">These shared settings continue into the End-to-End Lab. Byte counts are user assumptions.</p></div><Link className="badge replay" to="/lab">Open End-to-End Lab</Link></div>
        <div className="pipeline-strip"><div><span>Semantic retention</span><strong>{pipeline.replay.keepDim} / {LATENT_DIM} dims</strong></div><b>→</b><div><span>Orchestration objective</span><strong>{pipeline.replay.objective.replaceAll('_', ' ')}</strong></div><b>→</b><div><span>Modeled packet</span><strong>{pipeline.packetBytes} bytes</strong></div></div>
        <p className="faint" style={{ marginTop: 12 }}>Packet estimate = ceil(source bytes × retention) + header assumption + signature assumption + amortized KEM-session assumption.</p>
      </div>

      <div className="panel" style={{ marginTop: 14 }}>
        <h2>Integration work</h2>
        <ul className="muted">
          <li>Load <span className="mono">rl/checkpoints/ppo_semantic_agent.zip</span> for true policy actions.</li>
          <li>
            Reconstruction-reward path needs encoder/decoder weights and real JIGSAWS sequences.
          </li>
          <li>Connect network simulator outcomes to this reward.</li>
          <li>Add fixed-fraction baselines to the PPO comparison.</li>
        </ul>
      </div>
    </section>
  )
}
