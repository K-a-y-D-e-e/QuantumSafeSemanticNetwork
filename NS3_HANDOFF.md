# NS-3 Handoff Guide — Linux Run Instructions

**Project:** Quantum-Safe Semantic Communication for Deterministic Tactile Networks  
**Prepared by:** RL Engineer  
**For:** Linux teammate  
**Date:** 2026-10-11  

> [!IMPORTANT]
> All metrics and traces in `rl/results/` are **Python simulation results**, not NS-3 measurements.
> NS-3 is not installed and has not been run. The workload traces are exported for your NS-3 integration.

---

## What Has Been Done (Windows/Python Side)

All RL training, evaluation, and export is complete. The following artifacts are ready for NS-3 ingestion:

### Available Workload Traces for NS-3

| Trace File | Episodes / Rows | Description |
|---|---|---|
| `rl/results/ai_vs_baseline/ns3_workload_A.csv` | 100 episodes | Conventional baseline: Fixed 50% retention + EDF scheduler |
| `rl/results/ai_vs_baseline/ns3_workload_B.csv` | 100 episodes | Full AI system: PPO adaptive compression + DQN scheduler + Orchestration |
| `rl/results/rl_evaluation_run/ns3_workload.csv` | 1000 packets | Single-packet resolution trace with deadline / priority tags |
| `rl/results/rl_evaluation_run/ns3_workload.jsonl` | 1000 packets | Same trace in JSON Lines format |

### ns3_workload.csv Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `packet_seq_id` | int | Global packet index (0–999) |
| `episode` | int | Simulation episode (1–100) |
| `flow_id` | int | Flow identifier within episode |
| `timestamp_sec` | float | Packet creation time (seconds) |
| `arrival_time_us` | int | Packet arrival time (microseconds) |
| `deadline_sec` | float | Deadline in seconds |
| `deadline_us` | float | Deadline in microseconds |
| `packet_size_bytes` | int | Payload size (encoded latent dims × 4 bytes + header) |
| `retained_fraction` | float | Fraction of latent dimensions retained (0.25/0.50/0.75) |
| `task_criticality` | float | Task criticality [0,1] |
| `priority_class` | str | `TACTILE_CRITICAL` or `BEST_EFFORT` |
| `semantic_quality` | float | Semantic reconstruction quality = 1/(1+MSE) |
| `reconstruction_mse` | float | Encoder-decoder reconstruction MSE |

---

## Minimal Linux NS-3 Setup

### 1. Clone and Install NS-3 (Ubuntu 22.04 / Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y g++ python3 cmake ninja-build git \
    libxml2-dev libboost-all-dev python3-pip

# Download NS-3 (tested with 3.41)
wget https://www.nsnam.org/releases/ns-allinone-3.41.tar.bz2
tar xjf ns-allinone-3.41.tar.bz2
cd ns-allinone-3.41
./build.py --enable-examples --enable-tests

# Verify build
cd ns-3.41
./ns3 run hello-simulator
```

### 2. Copy Workload Traces

```bash
# Copy from the QuantumSafeSemanticNetwork project
cp /path/to/QuantumSafeSemanticNetwork-main/rl/results/rl_evaluation_run/ns3_workload.csv \
   /path/to/ns-allinone-3.41/ns-3.41/scratch/

cp /path/to/QuantumSafeSemanticNetwork-main/rl/results/rl_evaluation_run/ns3_workload.jsonl \
   /path/to/ns-allinone-3.41/ns-3.41/scratch/
```

### 3. Suggested NS-3 Simulation Approach

Write a C++ NS-3 script (`scratch/semantic_tactile.cc`) that:

1. **Reads the workload CSV** row by row (use NS-3's `CsvReader` or standard C++ file I/O).
2. **Creates point-to-point links** matching the Python simulation parameters:
   - Bandwidth: 10 Mbps (`DataRate("10Mbps")`)
   - Propagation delay: 100 µs (`Delay("0.1ms")`)
3. **Schedules packet transmissions** at `arrival_time_us` with `packet_size_bytes` payload.
4. **Sets per-packet deadlines** from `deadline_us`.
5. **Computes latency** = reception time − arrival time.
6. **Reports**: mean latency, deadline miss rate, throughput — per `priority_class`.

```cpp
// Example skeleton (adapt as needed)
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include <fstream>
#include <sstream>

using namespace ns3;

int main(int argc, char *argv[]) {
    // Load workload
    std::ifstream file("scratch/ns3_workload.csv");
    std::string line;
    std::getline(file, line); // skip header

    // Setup point-to-point link
    NodeContainer nodes;
    nodes.Create(2);
    PointToPointHelper p2p;
    p2p.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
    p2p.SetChannelAttribute("Delay", StringValue("0.1ms"));
    NetDeviceContainer devices = p2p.Install(nodes);

    // ... schedule packets from CSV, measure latency vs deadlines ...
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}
```

### 4. Key Parameters to Match Python Simulation

| Parameter | Python Value | NS-3 Equivalent |
|-----------|-------------|-----------------|
| Link bandwidth | 10 Mbps | `DataRate("10Mbps")` |
| Propagation delay | 100 µs | `Delay("0.1ms")` |
| Tactile deadline (HIGH) | 2–5 ms | Per `deadline_ms` column |
| Packet size range | 548–2096 B | Per `packet_size_bytes` column |
| Flow types | HIGH/MED/LOW | Per `priority_class` column |

---

## What to Measure and Report

To validate the Python simulation predictions, measure in NS-3:

1. **Mean latency** per priority class — compare to Python results:
   - PPO Raw: 2.92 ms mean (Python simulation)
   - PPO + Orchestrated: 3.41 ms mean (Python simulation)
2. **Deadline miss rate** — Python predicts 10.9% (PPO Raw), 13.1% (PPO + Orchestrated).
3. **Throughput** (Mbps) — verify link is not saturated.
4. **Queue depth** at the bottleneck link.

---

## What NOT to Claim

- Do **not** label Python simulation results as NS-3 measurements.
- Do **not** add NS-3 results to `rl/results/` — keep them in a separate `ns3/results/` directory.
- The Python simulation uses a simplified queueing model; NS-3 will give more realistic numbers.
- Differences of ±20–30% between Python sim and NS-3 are expected and should be documented.

---

## Files Modified / Not Modified

**Modified:** None — this is a read-only handoff document.

**Artifacts ready for NS-3 use:**
- `rl/results/rl_evaluation_run/ns3_workload.csv` (1000 packets, 13 columns)
- `rl/results/rl_evaluation_run/ns3_workload.jsonl` (1000 packets, JSON)
- `rl/results/rl_evaluation_run/config.json` (experiment configuration)

**Not needed for NS-3:** Python checkpoints (`.pt`, `.zip`) and decision logs.
