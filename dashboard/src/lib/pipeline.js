import { replayController } from './orchestration'

export function derivePipeline(settings) {
  const replay = replayController(settings.semanticState, settings.assumedAction, settings.applyBias)
  const retainedPayloadBytes = Math.ceil(settings.sourcePayloadBytes * replay.compression)
  const kemAmortizedBytes = Math.ceil(settings.kemSessionBytes / 10)
  const packetBytes = retainedPayloadBytes + settings.headerBytes + settings.signatureBytes + kemAmortizedBytes
  return {
    replay,
    retainedPayloadBytes,
    kemAmortizedBytes,
    packetBytes,
    totalWireBytes: packetBytes * 10,
    overheadBytes: packetBytes - retainedPayloadBytes,
  }
}
