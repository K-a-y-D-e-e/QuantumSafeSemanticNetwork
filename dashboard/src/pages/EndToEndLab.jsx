import { useMemo, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { ACTION_LABELS, LATENT_DIM, STATE_FIELDS } from '../lib/orchestration'
import { derivePipeline } from '../lib/pipeline'
import { simulateScenario } from '../lib/scenario'
import { useLabSettings } from '../lib/labSettingsContext'

const stages = [
  { id: 'input', name: 'JIGSAWS input', short: 'Input' },
  { id: 'semantic', name: 'Semantic encoder / compressor', short: 'Semantic' },
  { id: 'orchestration', name: 'Rule orchestration + PPO', short: 'Orchestration' },
  { id: 'security', name: 'PQC packet assumptions', short: 'PQC' },
  { id: 'network', name: 'DQN network scheduler', short: 'Network' },
  { id: 'output', name: 'Output metrics', short: 'Output' },
]

function NumberSetting({ label, value, min, max, step = 1, suffix, onChange, hint }) {
  return <label className="lab-number"><span>{label}</span><span className="lab-number-field"><input type="number" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Math.min(max, Math.max(min, Number(event.target.value) || min)))} /><span>{suffix}</span></span>{hint && <small>{hint}</small>}</label>
}

function Selection({ label, value, values, onChange, hint }) {
  return <label className="lab-number"><span>{label}</span><select className="select-control" value={value} onChange={(event) => onChange(event.target.value)}>{values.map((item) => typeof item === 'string' ? <option key={item}>{item}</option> : <option key={item.value} value={item.value}>{item.label}</option>)}</select>{hint && <small>{hint}</small>}</label>
}

const schedulerOption = { value: 'DQN scheduler', label: 'DQN scheduler' }
const schedulerLabel = (value) => value

export default function EndToEndLab() {
  const location = useLocation()
  const { settings, updateSetting, updateSemanticState, scenarios, saveScenario, clearScenarios } = useLabSettings()
  const [activeStage, setActiveStage] = useState(location.state?.stage || 'input')
  const scenario = scenarios[scenarios.length - 1] ?? null
  const pipeline = useMemo(() => derivePipeline(settings), [settings])

  function runScenario() {
    const result = {
      ...simulateScenario({ scheduler: settings.scheduler, trafficLoad: settings.trafficLoad, deadlineMs: settings.deadlineMs, packetBytes: pipeline.packetBytes }),
      inputSource: settings.inputSource,
      objective: pipeline.replay.objective,
      retention: pipeline.replay.compression,
    }
    saveScenario(result)
    setActiveStage('output')
  }

  function stageSummary(stageId) {
    if (stageId === 'input') return settings.inputSource
    if (stageId === 'semantic') return `${ACTION_LABELS[settings.assumedAction]} proposed`
    if (stageId === 'orchestration') return pipeline.replay.objective.replaceAll('_', ' ')
    if (stageId === 'security') return `${pipeline.packetBytes} B modeled`
    if (stageId === 'network') return `${schedulerLabel(settings.scheduler)} / ${Math.round(settings.trafficLoad * 100)}% load`
    return scenario ? `${scenario.deadlineMisses} misses / modeled` : 'Run scenario'
  }

  return (
    <section>
      <header className="page-heading demo-heading"><div><h1>Interactive Demo</h1><p className="lede">Adjust a stage, then run the shared scenario to update the packet path and estimates.</p></div><div className="heading-actions"><button type="button" className="run-button" onClick={runScenario}>Run Scenario</button></div></header>

      <nav className="pipeline-nav" aria-label="Experiment stages">
        {stages.map((stage, index) => <button key={stage.id} className={`pipeline-node ${activeStage === stage.id ? 'active' : ''}`} type="button" aria-current={activeStage === stage.id ? 'step' : undefined} onClick={() => setActiveStage(stage.id)}><span className="pipeline-node-index">{String(index + 1).padStart(2, '0')}</span><strong>{stage.name}</strong><small>{stageSummary(stage.id)}</small></button>)}
      </nav>

      <div className="panel packet-flow-panel"><div className="panel-head"><div><h2>Packet flow</h2><p className="packet-flow-note">Current packet path from the selected assumptions.</p></div></div><div className="packet-flow"><div className="packet-flow-step"><span>Input</span><strong>{settings.inputSource.startsWith('JIGSAWS') ? 'JIGSAWS · assumed' : 'Synthetic payload'}</strong></div><div className="packet-flow-step"><span>Semantic</span><strong>{pipeline.retainedPayloadBytes} B retained</strong></div><div className="packet-flow-step"><span>Orchestration</span><strong>{pipeline.replay.objective.replaceAll('_', ' ')}</strong></div><div className="packet-flow-step"><span>PQC</span><strong>+{pipeline.overheadBytes} B assumed</strong></div><div className="packet-flow-step"><span>Network</span><strong>{schedulerLabel(settings.scheduler)} / {Math.round(settings.trafficLoad * 100)}%</strong></div><div className="packet-flow-step"><span>Output</span><strong>{scenario ? `${scenario.deadlineMisses} / 10 misses` : 'Run scenario'}</strong></div></div></div>

      <div className="panel stage-editor">
        {activeStage === 'input' && <>
          <div className="panel-head"><div><h2>JIGSAWS input</h2><p className="faint">Choose a source profile and payload-size assumption.</p></div></div>
          <div className="grid-2"><Selection label="Input profile" value={settings.inputSource} values={['JIGSAWS kinematics · assumed', 'Synthetic network payload']} onChange={(value) => updateSetting('inputSource', value)} hint="Profile label used in the scenario." /><NumberSetting label="Source packet payload assumption" value={settings.sourcePayloadBytes} min={64} max={10000} suffix="B" onChange={(value) => updateSetting('sourcePayloadBytes', value)} hint="Default 1,200 B is a scenario assumption within the configured network traffic size range." /></div>
        </>}

        {activeStage === 'semantic' && <>
          <div className="panel-head"><div><h2>Semantic encoder / compressor</h2><p className="faint">Choose the retention level used in packet-size calculations.</p></div></div>
          <div className="action-pills">{ACTION_LABELS.map((label, index) => <button type="button" key={label} className={`pill ${settings.assumedAction === index ? 'active' : ''}`} onClick={() => updateSetting('assumedAction', index)}>{label} proposed retention</button>)}</div>
          <div className="lab-result"><span>Proposed retention action</span><strong>{ACTION_LABELS[settings.assumedAction]}</strong><small>Orchestration may bias this action in the next stage.</small></div>
        </>}

        {activeStage === 'orchestration' && <>
          <div className="panel-head"><div><h2>Rule-based orchestration + PPO</h2><p className="faint">Rule replay evaluates normalized inputs and can bias the selected action.</p></div></div>
          <div className="grid-2"><div>{STATE_FIELDS.map((field) => <label className="slider-block" key={field.key}><span>{field.label}</span><input type="range" min="0" max="1" step="0.01" value={settings.semanticState[field.key]} onChange={(event) => updateSemanticState(field.key, Number(event.target.value))} /><span className="mono">{settings.semanticState[field.key].toFixed(2)}</span></label>)}<label className="toggle"><input type="checkbox" checked={settings.applyBias} onChange={(event) => updateSetting('applyBias', event.target.checked)} />Apply objective action bias</label></div><div className="stack"><div className="lab-result"><span>Selected objective</span><strong className="lab-objective">{pipeline.replay.objective.replaceAll('_', ' ')}</strong><small>Rule: {pipeline.replay.rule}</small></div><div className="lab-result"><span>Effective retention</span><strong>{ACTION_LABELS[pipeline.replay.action]}</strong><small>{pipeline.replay.keepDim} / {LATENT_DIM} latent dimensions kept</small></div><div className="lab-result"><span>Analytic reward proxy</span><strong>{pipeline.replay.analytic.reward.toFixed(3)}</strong><small>Analytic rule formula output.</small></div></div></div>
        </>}

        {activeStage === 'security' && <>
          <div className="panel-head"><div><h2>PQC packet assumptions</h2><p className="faint">Choose algorithm profiles and byte overhead assumptions.</p></div></div>
          <div className="grid-2"><div className="stack"><Selection label="KEM algorithm label" value={settings.kemProfile} values={['No KEM overhead', 'ML-KEM-512 · assumed', 'ML-KEM-768 · assumed', 'ML-KEM-1024 · assumed', 'Custom KEM overhead']} onChange={(value) => updateSetting('kemProfile', value)} /><NumberSetting label="KEM ciphertext bytes / session" value={settings.kemSessionBytes} min={0} max={20000} suffix="B" onChange={(value) => updateSetting('kemSessionBytes', value)} hint="Amortized across 10 modeled packets." /></div><div className="stack"><Selection label="Signature algorithm label" value={settings.signatureProfile} values={['No signature overhead', 'ML-DSA-44 · assumed', 'ML-DSA-65 · assumed', 'ML-DSA-87 · assumed', 'Custom signature overhead']} onChange={(value) => updateSetting('signatureProfile', value)} /><NumberSetting label="Signature bytes / packet" value={settings.signatureBytes} min={0} max={20000} suffix="B" onChange={(value) => updateSetting('signatureBytes', value)} hint="Applied to each modeled packet." /></div></div>
          <div className="grid-2"><NumberSetting label="Fixed packet header assumption" value={settings.headerBytes} min={0} max={2048} suffix="B" onChange={(value) => updateSetting('headerBytes', value)} /><div className="lab-result"><span>Modeled packet size</span><strong>{pipeline.packetBytes} B</strong><small>{pipeline.retainedPayloadBytes} B retained payload + {pipeline.overheadBytes} B assumed overhead</small></div></div>
        </>}

        {activeStage === 'network' && <>
          <div className="panel-head"><div><h2>DQN network scheduler</h2><p className="faint">The browser DQN option uses a rule-based score; recorded DQN results are shown on Results.</p></div></div>
          <div className="grid-3"><Selection label="Scenario scheduler" value={settings.scheduler} values={['EDF', 'Priority', 'Random', schedulerOption]} onChange={(value) => updateSetting('scheduler', value)} hint={settings.scheduler === 'DQN scheduler' ? 'Score: priority rank × 0.5 + remaining-deadline fraction.' : 'Recorded DQN metrics are available on Results.'} /><label className="slider-block"><span>Traffic load</span><input type="range" min="0.1" max="1" step="0.05" value={settings.trafficLoad} onChange={(event) => updateSetting('trafficLoad', Number(event.target.value))} /><span className="mono">{Math.round(settings.trafficLoad * 100)}%</span></label><label className="slider-block"><span>Deadline target</span><input type="range" min="0.1" max="15" step="0.1" value={settings.deadlineMs} onChange={(event) => updateSetting('deadlineMs', Number(event.target.value))} /><span className="mono">{settings.deadlineMs.toFixed(1)} ms</span></label></div>
          <p className="faint">Scenario assumptions: traffic load changes the synthetic arrival window from 1,000 to 100 us. Per-flow deadlines are deterministic values between half the target and the target. Link values mirror the checked-in config: 10 Mbps bandwidth and 100 us propagation.</p>
        </>}

        {activeStage === 'output' && <>
          <div className="panel-head"><div><h2>Output metrics</h2></div></div>
          {scenario ? <><div className="row" style={{ marginBottom: 12 }}><span className="badge replay">Scenario estimate · 10 synthetic packets</span></div><div className="grid-4"><div className="metric"><div className="label">Average latency</div><div className="value">{scenario.avgLatencyUs.toFixed(1)}</div><div className="hint">us estimate</div></div><div className="metric"><div className="label">Average queue</div><div className="value">{scenario.avgQueueingUs.toFixed(1)}</div><div className="hint">us estimate</div></div><div className="metric"><div className="label">Deadline misses</div><div className="value">{scenario.deadlineMisses} / {scenario.flowCount}</div><div className="hint">generated flows</div></div><div className="metric"><div className="label">Packet size</div><div className="value">{scenario.packetBytes} B</div><div className="hint">modeled assumption</div></div></div><div className="grid-2 before-after"><div className="metric"><div className="label">Before · source payload</div><div className="value">{settings.sourcePayloadBytes} B</div><div className="hint">input assumption</div></div><div className="metric"><div className="label">After · transmitted packet</div><div className="value">{scenario.packetBytes} B</div><div className="hint">retention plus header and PQC assumptions</div></div></div><details className="compact-details"><summary>Inspect {scenario.flowCount} modeled packet results</summary><div className="table-wrap"><table><thead><tr><th>Flow</th><th>Priority</th><th>Arrival (us)</th><th>Deadline (ms)</th><th>Latency estimate (us)</th><th>Deadline status</th></tr></thead><tbody>{scenario.results.map((flow) => <tr key={flow.flowId}><td>{flow.flowId}</td><td>{flow.priority}</td><td className="num">{flow.arrivalUs}</td><td className="num">{flow.deadlineMs.toFixed(2)}</td><td className="num">{flow.latencyUs.toFixed(1)}</td><td><span className={`badge ${flow.deadlineMet ? 'replay' : 'warn'}`}>{flow.deadlineMet ? 'Within target' : 'Miss'} · estimate</span></td></tr>)}</tbody></table></div></details></> : <div className="callout">Run the scenario to see modeled output.</div>}
          {scenarios.length > 0 && <div className="scenario-history"><div className="panel-head"><div><h3>Before / after scenario comparison</h3><p className="faint">Each run preserves its settings and modeled output.</p></div><button type="button" className="pill" onClick={clearScenarios}>Clear</button></div><details className="compact-details"><summary>Compare {scenarios.length} runs</summary><div className="table-wrap"><table><thead><tr><th>Run</th><th>Input</th><th>Objective</th><th>Retention</th><th>Scheduler</th><th>Load</th><th>Deadline</th><th>Packet</th><th>Latency est.</th><th>Misses</th></tr></thead><tbody>{scenarios.map((run) => <tr key={run.runId}><td>Scenario {run.runId}</td><td>{run.inputSource.startsWith('JIGSAWS') ? 'JIGSAWS assumption' : 'Synthetic'}</td><td>{run.objective.replaceAll('_', ' ')}</td><td className="num">{(run.retention * 100).toFixed(0)}%</td><td>{schedulerLabel(run.scheduler)}</td><td>{Math.round(run.trafficLoad * 100)}%</td><td className="num">{run.deadlineTargetMs.toFixed(1)} ms</td><td className="num">{run.packetBytes} B</td><td className="num">{run.avgLatencyUs.toFixed(1)} us</td><td className="num">{run.deadlineMisses} / {run.flowCount}</td></tr>)}</tbody></table></div></details></div>}
        </>}
      </div>

    </section>
  )
}
