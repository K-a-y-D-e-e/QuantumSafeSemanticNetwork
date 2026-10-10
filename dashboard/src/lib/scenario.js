/**
 * Deterministic browser-side queue model based on network/simulation/
 * event_simulator.py and link.py. Workload generation is synthetic and is
 * deliberately reported as a modeled scenario, separate from recorded evals.
 */
const priorityRank = { HIGH: 0, MEDIUM: 1, LOW: 2 }

function createWorkload({ packetBytes, trafficLoad, deadlineMs, flowCount = 10 }) {
  let seed = 999
  function random() {
    seed = (seed * 1664525 + 1013904223) >>> 0
    return seed / 0x100000000
  }
  const arrivalSpreadUs = Math.round(1000 - (trafficLoad - 0.1) * (900 / 0.9))
  return Array.from({ length: flowCount }, (_, index) => ({
    flowId: index + 1,
    priority: ['HIGH', 'MEDIUM', 'LOW'][Math.floor(random() * 3)],
    deadlineMs: deadlineMs * (0.5 + random() * 0.5),
    arrivalUs: Math.floor(random() * (arrivalSpreadUs + 1)),
    packetBytes,
    tie: index,
  })).sort((a, b) => a.arrivalUs - b.arrivalUs || a.tie - b.tie)
}

export function simulateScenario({ scheduler, trafficLoad, deadlineMs, packetBytes }) {
  const packets = createWorkload({ packetBytes, trafficLoad, deadlineMs })
  let seed = 999
  function randomIndex(max) {
    seed = (seed * 1664525 + 1013904223) >>> 0
    return seed % max
  }
  const waiting = []
  const results = []
  let currentTimeUs = 0
  let cursor = 0

  while (cursor < packets.length || waiting.length) {
    while (cursor < packets.length && packets[cursor].arrivalUs <= currentTimeUs) {
      waiting.push(packets[cursor])
      cursor += 1
    }
    if (!waiting.length) {
      currentTimeUs = packets[cursor].arrivalUs
      continue
    }

    let chosenIndex
    if (scheduler === 'Random') {
      chosenIndex = randomIndex(waiting.length)
    } else if (scheduler === 'Priority') {
      chosenIndex = waiting.reduce((best, packet, index) =>
        priorityRank[packet.priority] < priorityRank[waiting[best].priority] ? index : best, 0)
    } else if (scheduler === 'DQN scheduler') {
      chosenIndex = waiting.reduce((best, packet, index) => {
        const score = priorityRank[packet.priority] * 0.5 + (packet.deadlineMs * 1000 - currentTimeUs) / (deadlineMs * 1000)
        const bestScore = priorityRank[waiting[best].priority] * 0.5 + (waiting[best].deadlineMs * 1000 - currentTimeUs) / (deadlineMs * 1000)
        return score < bestScore ? index : best
      }, 0)
    } else {
      chosenIndex = waiting.reduce((best, packet, index) => {
        const remaining = packet.deadlineMs * 1000 - currentTimeUs
        const bestRemaining = waiting[best].deadlineMs * 1000 - currentTimeUs
        return remaining < bestRemaining || (remaining === bestRemaining && priorityRank[packet.priority] < priorityRank[waiting[best].priority]) ? index : best
      }, 0)
    }

    const [packet] = waiting.splice(chosenIndex, 1)
    const txDelayUs = packet.packetBytes * 8 / 10
    const startUs = currentTimeUs
    const completionUs = startUs + txDelayUs + 100
    const latencyUs = completionUs - packet.arrivalUs
    results.push({
      ...packet,
      startUs,
      completionUs,
      latencyUs,
      queueingUs: startUs - packet.arrivalUs,
      deadlineMet: latencyUs <= packet.deadlineMs * 1000,
    })
    currentTimeUs = completionUs
  }

  const mean = (key) => results.reduce((sum, item) => sum + item[key], 0) / results.length
  const latencies = results.map((item) => item.latencyUs)
  const misses = results.filter((item) => !item.deadlineMet).length
  return {
    scheduler,
    trafficLoad,
    deadlineTargetMs: deadlineMs,
    flowCount: results.length,
    packetBytes,
    arrivalSpreadUs: Math.round(1000 - (trafficLoad - 0.1) * 1000),
    avgLatencyUs: mean('latencyUs'),
    avgQueueingUs: mean('queueingUs'),
    minLatencyUs: Math.min(...latencies),
    maxLatencyUs: Math.max(...latencies),
    deadlineMisses: misses,
    deadlineMissRate: misses / results.length,
    results,
  }
}
