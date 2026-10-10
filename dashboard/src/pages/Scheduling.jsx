import { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useRecordedData } from '../lib/useRecordedData'

const tooltipStyle = { background: '#FAF8F2', border: '1px solid #D8D0C5', borderRadius: 12, color: '#292536', fontSize: 11, boxShadow: '0 2px 8px rgba(41,37,54,.05)' }
const schedulerColors = { Random: '#817A88', Priority: '#7866A6', EDF: '#9A812A', DQN: '#64785C' }
const chartMetrics = {
  avg_latency_us: { label: 'Average latency', format: (v) => `${Number(v).toFixed(1)} us` },
  deadline_miss_rate: { label: 'Deadline miss rate', format: (v) => `${(Number(v) * 100).toFixed(1)}%` },
  avg_queueing_us: { label: 'Average queueing', format: (v) => `${Number(v).toFixed(1)} us` },
  jitter_us: { label: 'Jitter', format: (v) => `${Number(v).toFixed(1)} us` },
}
const trainingMetrics = {
  reward: { label: 'Episode reward', format: (v) => Number(v).toFixed(2), color: '#7866A6' },
  avg_latency_us: { label: 'Average latency', format: (v) => `${Number(v).toFixed(0)} us`, color: '#9A812A' },
  deadline_miss_rate: { label: 'Deadline miss rate', format: (v) => `${(Number(v) * 100).toFixed(0)}%`, color: '#9A5E70' },
  epsilon: { label: 'Exploration epsilon', format: (v) => Number(v).toFixed(3), color: '#64785C' },
}

export default function Scheduling() {
  const { loading, error, comparison, training } = useRecordedData()
  const [metric, setMetric] = useState('avg_latency_us')
  const [historyMetric, setHistoryMetric] = useState('reward')
  const [visible, setVisible] = useState(['Random', 'Priority', 'EDF', 'DQN'])
  const rows = useMemo(() => comparison ? Object.entries(comparison.results).map(([scheduler, values]) => ({ scheduler, ...values })) : [], [comparison])

  if (loading) return <p className="muted">Loading recorded experiment files…</p>
  if (error || !comparison || !training.length) return <div className="callout">Could not load recorded scheduling data. {error}</div>

  const last = training[training.length - 1]
  const bestReward = training.reduce((best, row) => row.reward > best.reward ? row : best, training[0])
  const dqn = comparison.results.DQN
  const edf = comparison.results.EDF
  const selectedRows = rows.filter((row) => visible.includes(row.scheduler))
  const history = trainingMetrics[historyMetric]
  const chart = chartMetrics[metric]
  const toggleScheduler = (name) => setVisible((current) => current.includes(name) ? (current.length === 1 ? current : current.filter((value) => value !== name)) : [...current, name])

  return <section>
    <header className="page-heading"><div><h1>Results</h1><p className="lede">Policy comparison and DQN training history from the saved evaluation files.</p></div><div className="heading-actions"><span className="badge recorded">Recorded result · {comparison.num_episodes} episodes · seed {comparison.eval_seed}</span></div></header>

    <div className="grid-4 results-kpis">
      <div className="metric"><div className="label">Lowest latency</div><div className="value">{dqn.avg_latency_us.toFixed(1)}</div><div className="hint">us · DQN</div></div>
      <div className="metric"><div className="label">Lowest miss rate</div><div className="value">{(edf.deadline_miss_rate * 100).toFixed(1)}%</div><div className="hint">EDF</div></div>
      <div className="metric"><div className="label">Training episodes</div><div className="value">{training.length}</div><div className="hint">log ends at episode {last.episode}</div></div>
      <div className="metric"><div className="label">Peak episode reward</div><div className="value">{Number(bestReward.reward).toFixed(2)}</div><div className="hint">episode {bestReward.episode}</div></div>
    </div>

    <div className="panel">
      <div className="panel-head"><div><h2>Recorded scheduler comparison</h2><p className="faint">CSV values cross-checked against the saved JSON file.</p></div><select className="select-control" value={metric} onChange={(event) => setMetric(event.target.value)} aria-label="Recorded comparison metric">{Object.entries(chartMetrics).map(([key, item]) => <option value={key} key={key}>{item.label}</option>)}</select></div>
      <div className="row results-filters">{rows.map((row) => <button className={`pill ${visible.includes(row.scheduler) ? 'active' : ''}`} aria-pressed={visible.includes(row.scheduler)} type="button" key={row.scheduler} onClick={() => toggleScheduler(row.scheduler)}><span style={{ color: schedulerColors[row.scheduler] }}>●</span> {row.scheduler}</button>)}</div>
      <div className="chart-box"><ResponsiveContainer><BarChart data={selectedRows} margin={{ top: 10, right: 18, left: 4, bottom: 2 }}><CartesianGrid stroke="#D8D0C5" vertical={false} /><XAxis dataKey="scheduler" stroke="#847E88" tick={{ fontSize: 11, fill: '#706A78' }} tickLine={false} axisLine={false} /><YAxis stroke="#847E88" tick={{ fontSize: 10, fill: '#706A78' }} tickLine={false} axisLine={false} tickFormatter={metric === 'deadline_miss_rate' ? (value) => `${(value * 100).toFixed(0)}%` : undefined} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => chart.format(value)} /><Bar dataKey={metric} name={chart.label} radius={[5, 5, 0, 0]}>{selectedRows.map((row) => <Cell key={row.scheduler} fill={schedulerColors[row.scheduler]} />)}</Bar></BarChart></ResponsiveContainer></div>
    </div>

    <div className="panel">
      <div className="panel-head"><div><h2>DQN training history</h2><p className="faint">{training.length} saved rows from <span className="mono">training_log.csv</span>.</p></div><select className="select-control" value={historyMetric} onChange={(event) => setHistoryMetric(event.target.value)} aria-label="Training history metric">{Object.entries(trainingMetrics).map(([key, item]) => <option value={key} key={key}>{item.label}</option>)}</select></div>
      <div className="chart-box tall"><ResponsiveContainer><LineChart data={training} margin={{ top: 12, right: 18, left: 5, bottom: 2 }}><CartesianGrid stroke="#D8D0C5" /><XAxis dataKey="episode" stroke="#847E88" tick={{ fontSize: 10, fill: '#706A78' }} tickLine={false} axisLine={false} /><YAxis stroke="#847E88" tick={{ fontSize: 10, fill: '#706A78' }} tickLine={false} axisLine={false} /><Tooltip contentStyle={tooltipStyle} formatter={(value, name) => [history.format(value), name]} /><Legend wrapperStyle={{ fontSize: 11, color: '#706A78' }} />{historyMetric === 'reward' && <Line type="monotone" dataKey="reward_roll20" name="20-episode mean" stroke="#9A812A" dot={false} strokeWidth={2} />}<Line type="monotone" dataKey={historyMetric} name={history.label} stroke={history.color} dot={false} strokeWidth={1.8} activeDot={{ r: 4 }} /></LineChart></ResponsiveContainer></div>
      <p className="overview-caption">Latest reward: {Number(last.reward).toFixed(2)}. The saved log may end before the configured training run.</p>
    </div>

    <div className="panel results-findings"><div className="panel-head"><div><h2>Key findings</h2><p className="faint">Recorded evaluation only; results apply to this workload and seed.</p></div></div><div className="grid-2"><div className="finding"><span className="finding-mark cyan-mark">01</span><div><strong>DQN led on average latency</strong><p>{dqn.avg_latency_us.toFixed(1)} us in the saved comparison.</p></div></div><div className="finding"><span className="finding-mark amber-mark">02</span><div><strong>EDF led on deadline misses</strong><p>{(edf.deadline_miss_rate * 100).toFixed(1)}% miss rate in the saved comparison.</p></div></div></div></div>
  </section>
}
