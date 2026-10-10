import { createContext, useContext } from 'react'

export const defaults = {
  inputSource: 'JIGSAWS kinematics · assumed',
  semanticState: {
    task_criticality: 0.85,
    bandwidth: 0.4,
    network_load: 0.55,
    latency: 0.75,
    packet_loss: 0.1,
    deadline: 0.6,
    semantic_quality: 0.5,
  },
  assumedAction: 2,
  applyBias: true,
  sourcePayloadBytes: 1200,
  headerBytes: 64,
  kemProfile: 'No KEM overhead',
  kemSessionBytes: 0,
  signatureProfile: 'No signature overhead',
  signatureBytes: 0,
  scheduler: 'EDF',
  trafficLoad: 0.5,
  deadlineMs: 2,
}

export const LabSettingsContext = createContext(null)

export function useLabSettings() {
  const context = useContext(LabSettingsContext)
  if (!context) throw new Error('useLabSettings must be used inside LabSettingsProvider')
  return context
}
