# QuantumSafeSemanticNetwork

A student research project integrating post-quantum cryptography,
network simulation, and reinforcement learning for adaptive scheduling.

---

## Project Structure

```
QuantumSafeSemanticNetwork-main/
│
├── crypto/                   Phase 1 — Cryptographic foundation
│   ├── ml_kem.py             ML-KEM-768 post-quantum key encapsulation
│   ├── ml_dsa.py             ML-DSA-65 post-quantum digital signatures
│   ├── aes_gcm.py            AES-256-GCM authenticated encryption
│   ├── classical_crypto.py   X25519 / Ed25519 baselines
│   ├── secure_channel.py     Secure channel integration
│   └── benchmark*.py         Classical vs PQC performance benchmarks
│
├── network/                  Phase 2 — Network simulation
│   ├── flows/flow.py         NetworkFlow model
│   ├── nodes/node.py         NetworkNode model
│   └── simulation/
│       ├── link.py           NetworkLink (bandwidth + propagation delay)
│       ├── scheduler.py      PriorityScheduler (HIGH > MEDIUM > LOW)
│       ├── deadline_scheduler.py  DeadlineAwareScheduler (EDF)
│       ├── event_simulator.py     EventDrivenSimulator
│       ├── analyzer.py       NetworkAnalyzer (latency / jitter / deadline)
│       └── traffic.py        TrafficSimulator
│
├── rl/                       Phase 3 — Reinforcement learning scheduler
│   ├── environment.py        SemanticCompressionEnv + NetworkSchedulingEnv
│   ├── dqn_agent.py          DQN with action masking (QNetwork, ReplayBuffer)
│   ├── traffic_generator.py  Reproducible workload generator
│   ├── random_scheduler.py   Random scheduling baseline
│   ├── train.py              DQN training script (--smoke / full)
│   ├── evaluate.py           Comparison evaluation (Random/Priority/EDF/DQN)
│   └── test_rl_environment.py  RL unit tests (10 tests)
│
├── configs/
│   └── rl_config.yaml        All RL hyperparameters
│
└── rl/
    ├── checkpoints/          Saved DQN model weights
    ├── logs/                 training_log.csv (training curves)
    └── results/              comparison_results.csv / .json
```

---

## Phase 1 — Cryptographic Foundation

Implemented and tested post-quantum and classical cryptographic primitives:

| Algorithm    | Type            | Key / Output sizes                        |
|--------------|-----------------|-------------------------------------------|
| ML-KEM-768   | PQC KEM         | PK 1184 B · CT 1088 B · SS 32 B          |
| ML-DSA-65    | PQC Signature   | PK 1952 B · SK 4032 B · Sig 3309 B       |
| X25519       | Classical KEM   | PK 32 B · SS 32 B                         |
| Ed25519      | Classical Sig   | PK 32 B · Sig 64 B                        |
| AES-256-GCM  | Symmetric AEAD  | 1 KB benchmark included                   |

The benchmarks demonstrate that classical algorithms have lower
computational and communication overhead, while PQC provides
protection against future quantum attacks.

### Run Phase 1 tests

```bash
cd crypto
python test_ml_kem.py
python test_ml_dsa.py
python test_classical_crypto.py
python test_secure_channel.py
python test_secure_channel_integration.py
python benchmark.py
python benchmark_comparison.py
```

---

## Phase 2 — Network Simulation

A deterministic, event-driven single-link network simulator.

**Key concepts:** nodes, flows, packets, link bandwidth, propagation delay,
transmission delay, queueing contention, dynamic arrivals, event-driven
scheduling, latency, jitter, deadline satisfaction.

**Existing schedulers:**

| Scheduler        | Policy                      |
|------------------|-----------------------------|
| PriorityScheduler | Strict priority (HIGH > MEDIUM > LOW) |
| DeadlineAwareScheduler | EDF — earliest deadline first |

**Existing 10-flow stress-test baseline results (unchanged):**

| Metric            | Strict Priority | EDF     |
|-------------------|-----------------|---------|
| Avg latency (µs)  | 5464            | 5432    |
| Max latency (µs)  | 9800            | 9800    |
| Avg queueing (µs) | 4444            | 4412    |
| Avg jitter (µs)   | 988.89          | 988.89  |
| Deadline misses   | 6/10            | 8/10    |

### Run Phase 2 baseline

```bash
cd network
python test_network_stress.py
python test_deadline_scheduler.py
```

---

## Phase 3 — Reinforcement Learning Adaptive Scheduler

### Motivation

Fixed scheduling policies (Priority, EDF) apply the same rule regardless of
traffic conditions.  An RL agent can *learn* to make scheduling decisions
that better balance latency, queueing delay, and deadline satisfaction across
varying workloads.

### What the RL agent does

At each scheduling step the agent observes the current waiting queue and
selects **which flow to transmit next**.  This replaces the fixed ordering
imposed by Priority or EDF with a learned, adaptive policy.

### State / Observation (42 features)

The observation is a fixed-size float32 array of shape `(42,)`:

```
Slots 0–9 (4 features each = 40 features):
  priority_norm           HIGH=1.0, MEDIUM=0.5, LOW=0.0
  remaining_deadline_norm (deadline_us − current_time) / 20 000, clipped [0,1]
  size_norm               size_bytes / 2000, clipped [0,1]
  wait_time_norm          (current_time − arrival_time) / 10 000, clipped [0,1]

Empty slots are padded with zeros.

Global features (2):
  queue_fill_ratio        len(waiting_queue) / 10
  time_norm               current_time_us / 50 000
```

### Action Space

`Discrete(10)` — select index `i` from the current waiting queue.

**Action masking:** Q-values of empty slots are set to −10⁹ before argmax,
so the agent never selects a non-existent slot.  If an invalid action
bypasses masking (e.g. from a random policy), it is clamped to the last
valid index and a configurable `mask_violation_penalty` is applied.

### Reward Function

```
R = w_deadline × deadline_bonus
  − w_latency  × latency_norm
  − w_queue    × queue_norm
  + w_miss     × deadline_penalty
  [+ mask_violation_penalty if invalid action clamped]
```

Where:
- `deadline_bonus`   = +1.0 if deadline met, else 0.0
- `deadline_penalty` = −1.0 if deadline missed, else 0.0
- `latency_norm`     = latency_µs / 15 000, clipped [0,1]
- `queue_norm`       = queueing_µs / 15 000, clipped [0,1]

Default weights: `w_deadline=2.0, w_latency=1.0, w_queue=0.5, w_miss=3.0`.
These weights are **configurable** in `configs/rl_config.yaml` and are not
claimed to be mathematically optimal.

### Episode Design

One episode = one complete workload of N flows (default 10).
The episode terminates when all flows have been scheduled and transmitted.
A fresh random workload is generated on each `reset()`.

### DQN Architecture

```
Input (42)
  → Linear(128) → ReLU
  → Linear(128) → ReLU
  → Linear(10)         ← Q-values for each queue slot
```

Training details:
- Epsilon-greedy exploration with per-episode multiplicative decay
- Experience replay (circular buffer, capacity 10 000)
- Hard target-network update every 50 gradient steps
- Huber loss (smooth L1)
- Gradient clipping (max norm 10)

### Limitations

- Single-link simulation (no multi-hop routing)
- Fixed workload size per episode (10 flows by default)
- DQN does not guarantee optimal scheduling; it learns an adaptive policy
- Reward weights are heuristic starting points, not analytically derived
- Generalisation to very different traffic distributions not yet validated

---

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies already in `requirements.txt`:
- `gymnasium==1.0.0`
- `torch==2.5.1`
- `stable-baselines3==2.4.0`
- `numpy==1.26.4`
- `PyYAML==6.0.2`
- `pytest==8.3.4`

---

## Commands

### Run RL unit tests (10 tests)

```bash
# From project root
pytest rl/test_rl_environment.py -v
```

### Smoke test (10 episodes — pipeline validation only)

```bash
python rl/train.py --smoke
```

> ⚠️ Smoke test results do NOT represent a trained model.

### Full training run (500 episodes)

```bash
python rl/train.py
```

Or with a custom config:

```bash
python rl/train.py --config configs/rl_config.yaml
```

### Evaluate all schedulers (after training)

```bash
python rl/evaluate.py --model rl/checkpoints/dqn_best.pt
```

### Reproduce Phase 2 stress-test baseline (unchanged)

```bash
cd network
python test_network_stress.py
```

---

## Comparison Experiment

`rl/evaluate.py` runs Random, Priority, EDF, and DQN on **identical**
workloads (same `eval_seed`, separate from the training seed).
Results are saved to:

- `rl/results/comparison_results.csv`
- `rl/results/comparison_results.json`

The experiment is intended to answer:
> "Can a learned scheduling policy adaptively select flows under varying
> traffic conditions, and how does its observed performance compare with
> fixed Priority, EDF, and Random scheduling?"

The results do not pre-suppose RL superiority — the comparison is
empirical and the outcome depends on training quality and workload characteristics.

---

## Training Curves

The training log `rl/logs/training_log.csv` contains:

| Column             | Description                         |
|--------------------|-------------------------------------|
| episode            | Episode number                      |
| reward             | Total episode reward                |
| avg_latency_us     | Average end-to-end latency (µs)     |
| deadline_misses    | Number of deadline misses           |
| deadline_miss_rate | Fraction of flows missing deadline  |
| epsilon            | Exploration rate at end of episode  |

These columns are sufficient to generate reward vs episode,
latency vs episode, and deadline-miss vs episode plots.
