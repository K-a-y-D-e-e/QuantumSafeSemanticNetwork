# AI-Enabled Semantic Network vs Conventional Baseline

**Branch:** `experiment/rl-evaluation`  
**Date:** 2026-10-11  
**Seeds:** 3 × 100 episodes = 300 scheduling evaluations per config  
**Semantic eval samples:** 256 held-out JIGSAWS kinematics  

> **Measurement disclaimer:** All latency and deadline figures are from the Python
> `EventDrivenSimulator` / `NetworkSchedulingEnv`. These are simulation proxies, NOT
> NS-3 measurements. Semantic quality figures (MSE, quality) are from the trained
> encoder/decoder checkpoint evaluated on JIGSAWS kinematics.

---

## Configurations

| ID | Label | Compression | Scheduler | Orchestration |
|----|-------|------------|-----------|---------------|
| A  | Conv-50%+EDF      | Fixed 50% | EDF | No |
| B  | PPO+DQN+Orch      | PPO       | DQN | Yes |
| C  | PPO+DQN (no orch) | PPO       | DQN | No (ablation) |
| D  | Fixed-25%+EDF     | Fixed 25% | EDF | No |
| E  | Fixed-75%+EDF     | Fixed 75% | EDF | No |
| F  | PPO+EDF           | PPO       | EDF | No |
| G  | PPO+Orch+EDF      | PPO+Orch  | EDF | Yes |

**Integration design:** PPO decides compression (packet size). DQN decides scheduling
order. Orchestration biases PPO's action — it does NOT modify DQN's behavior.
This is the minimal honest joint harness: packet sizes from PPO feed directly into
the DQN scheduling environment.

---

## Results Summary

### Semantic Layer

| Config | MSE ↓ | Quality ↑ | MAE ↓ | RMSE ↓ | Retention | Payload (B) |
|--------|-------|-----------|-------|--------|-----------|-------------|
| **A** Conv-50%+EDF | 0.4181 | 0.7279 | 0.4287 | 0.6164 | 0.50 | 1064 |
| **B** PPO+DQN+Orch | 0.5707 | 0.6844 | 0.4935 | 0.6937 | 0.46 | 975 |
| **C** PPO+DQN (no orch) | 0.6026 | 0.6675 | 0.5139 | 0.7214 | 0.43 | 911 |
| **D** Fixed-25%+EDF | 0.8269 | 0.5711 | 0.6419 | 0.8845 | 0.25 | 548 |
| **E** Fixed-75%+EDF | 0.2033 | 0.8432 | 0.2869 | 0.4232 | 0.75 | 1580 |
| **F** PPO+EDF | 0.6026 | 0.6675 | 0.5139 | 0.7214 | 0.43 | 911 |
| **G** PPO+Orch+EDF | 0.5707 | 0.6844 | 0.4935 | 0.6937 | 0.46 | 975 |

### Network / Scheduling Layer [Python sim — NOT NS-3]

| Config | Avg Lat (µs) | P95 Lat (µs) | Deadline Miss ↓ | Deadline Sat ↑ |
|--------|-------------|-------------|-----------------|----------------|
| **A** Conv-50%+EDF | 4824.4 | 8627.6 | 0.051 | 0.949 |
| **B** PPO+DQN+Orch | 4133.7 | 7632.8 | 0.157 | 0.843 |
| **C** PPO+DQN (no orch) | 3882.1 | 7125.5 | 0.147 | 0.853 |
| **D** Fixed-25%+EDF | 2554.0 | 4689.3 | 0.000 | 1.000 |
| **E** Fixed-75%+EDF | 7094.8 | 12569.8 | 0.329 | 0.671 |
| **F** PPO+EDF | 4120.7 | 7381.4 | 0.042 | 0.958 |
| **G** PPO+Orch+EDF | 4390.8 | 7854.0 | 0.055 | 0.945 |

---

## Key Findings

### Does AI (B) improve over Conventional (A)?

**Semantic quality:** B=0.6844 vs A=0.7279 → -6.0%

**Reconstruction MSE:** B=0.5707 vs A=0.4181 → +36.5% (negative = better)

**Network latency:** B=4133.7 µs vs A=4824.4 µs → -14.3%

**Deadline miss rate:** B=0.157 vs A=0.051 → +207.2%

### Does Orchestration add value (B vs C)?

**Semantic quality:** B=0.6844 vs C=0.6675 → +2.5%
**Network latency:** B=4133.7 µs vs C=3882.1 µs → +6.5%
**Deadline miss:** B=0.157 vs C=0.147 → +6.8%

### Does DQN add value over EDF (B vs G)?

**Network latency:** B(DQN)=4133.7 µs vs G(EDF)=4390.8 µs → -5.9%
**Deadline miss:** B(DQN)=0.157 vs G(EDF)=0.055 → +183.1%

---

## What Remains Unverified

- **NS-3 results:** All scheduling metrics are Python simulation proxies.
  See `NS3_HANDOFF.md` for Linux NS-3 run instructions.
- **DQN deadline miss vs EDF:** DQN achieves lower mean latency but higher
  deadline miss rate than EDF in some configurations (latency-vs-deadline trade-off).
- **Statistical significance of scheduling metrics:** CI reported but not corrected
  for multiple comparisons across 7 configurations.

---

## Reproduction

```powershell
# Full evaluation (3 seeds × 100 episodes)
python ai_vs_baseline_evaluation.py

# Quick smoke test (1 seed × 5 episodes)
python ai_vs_baseline_evaluation.py --num-episodes 5 --num-seeds 1

# Run tests
python -m pytest tests/test_ai_vs_baseline.py -v
```

All outputs are in `rl/results/ai_vs_baseline/`.