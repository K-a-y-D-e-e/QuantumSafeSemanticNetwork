/**
 * Faithful JS port of orchestration/agent.py and the analytic reward in
 * rl/environment.py::_step_analytic. Used for labelled replay only.
 */

export const OBJECTIVES = [
  'LATENCY_CRITICAL',
  'QUALITY_CRITICAL',
  'BANDWIDTH_EFFICIENT',
  'BALANCED',
]

export const REWARD_WEIGHTS = {
  LATENCY_CRITICAL: {
    quality: 0.6,
    latency: 1.8,
    bandwidth: 1.4,
    deadline: 1.2,
    criticality: 0.8,
  },
  QUALITY_CRITICAL: {
    quality: 2.0,
    latency: 0.5,
    bandwidth: 0.4,
    deadline: 1.0,
    criticality: 1.5,
  },
  BANDWIDTH_EFFICIENT: {
    quality: 0.5,
    latency: 0.8,
    bandwidth: 2.0,
    deadline: 0.8,
    criticality: 0.6,
  },
  BALANCED: {
    quality: 1.0,
    latency: 1.0,
    bandwidth: 1.0,
    deadline: 1.0,
    criticality: 1.0,
  },
}

export const COMPRESSION_LEVELS = [0.25, 0.5, 0.75, 1.0]
export const ACTION_LABELS = ['25%', '50%', '75%', '100%']
export const LATENT_DIM = 16

export const STATE_FIELDS = [
  { key: 'task_criticality', index: 0, label: 'Task criticality' },
  { key: 'bandwidth', index: 1, label: 'Available bandwidth' },
  { key: 'network_load', index: 2, label: 'Network load' },
  { key: 'latency', index: 3, label: 'Latency' },
  { key: 'packet_loss', index: 4, label: 'Packet loss' },
  { key: 'deadline', index: 5, label: 'Deadline requirement' },
  { key: 'semantic_quality', index: 6, label: 'Semantic quality (prior)' },
]

export function decideObjective({ task_criticality, network_load, latency }) {
  if (task_criticality > 0.8 && latency > 0.7) {
    return {
      objective: 'LATENCY_CRITICAL',
      rule: 'task_criticality > 0.8 and latency > 0.7',
    }
  }
  if (task_criticality > 0.8) {
    return {
      objective: 'QUALITY_CRITICAL',
      rule: 'task_criticality > 0.8',
    }
  }
  if (network_load > 0.7) {
    return {
      objective: 'BANDWIDTH_EFFICIENT',
      rule: 'network_load > 0.7',
    }
  }
  return { objective: 'BALANCED', rule: 'default (no high-criticality / congestion rule fired)' }
}

export function biasActionForObjective(action, objective) {
  let next = Math.max(0, Math.min(3, Math.trunc(action)))
  if (objective === 'BANDWIDTH_EFFICIENT') next = Math.min(next, 1)
  if (objective === 'QUALITY_CRITICAL') next = Math.max(next, 2)
  if (objective === 'LATENCY_CRITICAL') next = Math.min(next, 2)
  return next
}

export function keptLatentDims(compression) {
  return Math.max(1, Math.trunc(LATENT_DIM * compression))
}

export function analyticStep({ compression, task_criticality, network_load, latency }) {
  const semantic_quality = compression
  const bandwidth_cost = compression
  const estimated_latency = latency + 0.3 * bandwidth_cost + 0.2 * network_load
  let reward = semantic_quality - 0.5 * estimated_latency - 0.3 * bandwidth_cost
  const criticalityPenalty = task_criticality > 0.7 && semantic_quality < 0.75
  if (criticalityPenalty) reward -= 0.5
  return {
    semantic_quality,
    bandwidth_cost,
    estimated_latency: Math.min(1.0, estimated_latency),
    estimated_latency_raw: estimated_latency,
    reward,
    criticalityPenalty,
  }
}

export function replayController(state, assumedPpoAction, applyBias) {
  const { objective, rule } = decideObjective(state)
  const raw = Math.max(0, Math.min(3, Math.trunc(assumedPpoAction)))
  const action = applyBias ? biasActionForObjective(raw, objective) : raw
  const compression = COMPRESSION_LEVELS[action]
  const analytic = analyticStep({
    compression,
    task_criticality: state.task_criticality,
    network_load: state.network_load,
    latency: state.latency,
  })
  return {
    objective,
    rule,
    rawAction: raw,
    action,
    compression,
    keepDim: keptLatentDims(compression),
    weights: REWARD_WEIGHTS[objective],
    analytic,
    unusedInputs: {
      bandwidth: state.bandwidth,
      packet_loss: state.packet_loss,
      deadline: state.deadline,
      semantic_quality: state.semantic_quality,
      security_requirement: state.deadline,
    },
  }
}
