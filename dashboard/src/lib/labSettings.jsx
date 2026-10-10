import { useRef, useState } from 'react'
import { defaults, LabSettingsContext } from './labSettingsContext'

export function LabSettingsProvider({ children }) {
  const [settings, setSettings] = useState(defaults)
  const [scenarios, setScenarios] = useState([])
  const nextRunId = useRef(0)

  function updateSetting(key, value) {
    setSettings((current) => ({ ...current, [key]: value }))
  }

  function updateSemanticState(key, value) {
    setSettings((current) => ({ ...current, semanticState: { ...current.semanticState, [key]: value } }))
  }

  function saveScenario(result) {
    nextRunId.current += 1
    const runId = nextRunId.current
    setScenarios((current) => [...current, { ...result, runId }].slice(-6))
  }

  function clearScenarios() {
    setScenarios([])
  }

  return <LabSettingsContext.Provider value={{ settings, updateSetting, updateSemanticState, scenarios, saveScenario, clearScenarios }}>{children}</LabSettingsContext.Provider>
}
