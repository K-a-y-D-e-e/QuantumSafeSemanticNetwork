import { useEffect, useState } from 'react'
import { parseCsv, rollingMean } from './csv'

const comparisonFields = [
  'avg_latency_us', 'min_latency_us', 'max_latency_us', 'avg_queueing_us',
  'jitter_us', 'deadline_misses', 'deadline_miss_rate', 'completed_flows',
]

function requireFields(rows, fields, source) {
  if (!rows.length) throw new Error(`${source} contains no data rows`)
  for (const [index, row] of rows.entries()) {
    for (const field of fields) {
      if (!Number.isFinite(row[field])) throw new Error(`${source} row ${index + 2} has an invalid ${field}`)
    }
  }
}

export function useRecordedData() {
  const [state, setState] = useState({
    loading: true,
    error: null,
    comparison: null,
    training: [],
  })

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const [jsonRes, comparisonRes, logRes] = await Promise.all([
          fetch('/data/comparison_results.json'),
          fetch('/data/comparison_results.csv'),
          fetch('/data/training_log.csv'),
        ])
        if (!jsonRes.ok) throw new Error(`comparison_results.json HTTP ${jsonRes.status}`)
        if (!comparisonRes.ok) throw new Error(`comparison_results.csv HTTP ${comparisonRes.status}`)
        if (!logRes.ok) throw new Error(`training_log.csv HTTP ${logRes.status}`)
        const comparison = await jsonRes.json()
        if (!comparison?.results || !Number.isFinite(comparison.eval_seed) || !Number.isFinite(comparison.num_episodes)) {
          throw new Error('comparison_results.json has an unexpected schema')
        }
        const comparisonRows = parseCsv(await comparisonRes.text())
        requireFields(comparisonRows, comparisonFields, 'comparison_results.csv')
        const csvNames = new Set(comparisonRows.map((row) => row.scheduler))
        const jsonNames = Object.keys(comparison.results)
        if (csvNames.size !== jsonNames.length || jsonNames.some((name) => !csvNames.has(name))) {
          throw new Error('comparison_results.csv and JSON contain different schedulers')
        }
        if (['Random', 'Priority', 'EDF', 'DQN'].some((name) => !csvNames.has(name))) {
          throw new Error('Recorded comparison is missing a supported scheduler row')
        }
        for (const row of comparisonRows) {
          const jsonRow = comparison.results[row.scheduler]
          if (!jsonRow || comparisonFields.some((field) => !Number.isFinite(jsonRow[field]) || Math.abs(jsonRow[field] - row[field]) > 0.011)) {
            throw new Error(`comparison_results.csv and JSON disagree for ${row.scheduler}`)
          }
        }
        comparison.results = Object.fromEntries(comparisonRows.map(({ scheduler, ...metrics }) => [scheduler, metrics]))
        const rawTraining = parseCsv(await logRes.text())
        requireFields(rawTraining, ['episode', 'reward', 'avg_latency_us', 'deadline_misses', 'deadline_miss_rate', 'epsilon'], 'training_log.csv')
        if (rawTraining.some((row, index) => index > 0 && row.episode <= rawTraining[index - 1].episode)) {
          throw new Error('training_log.csv episodes must be strictly increasing')
        }
        const training = rollingMean(rawTraining, 'reward', 20)
        if (!cancelled) {
          setState({ loading: false, error: null, comparison, training })
        }
      } catch (error) {
        if (!cancelled) {
          setState({
            loading: false,
            error: error instanceof Error ? error.message : String(error),
            comparison: null,
            training: [],
          })
        }
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [])

  return state
}
