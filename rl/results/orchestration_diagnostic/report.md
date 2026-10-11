# AI-Enabled Semantic Network vs Conventional Baseline

**Branch:** `experiment/rl-evaluation`  
**Date:** 2026-10-11  
**Seeds:** 1 × 10 episodes = 10 scheduling evaluations per config  
**Semantic eval samples:** 100 held-out JIGSAWS kinematics  

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
| **A** Conv-50%+EDF | 0.4254 | 0.7194 | 0.4390 | 0.6284 | 0.50 | 1064 |
| **B** PPO+DQN+Orch | 0.5769 | 0.6771 | 0.5014 | 0.7046 | 0.45 | 950 |
| **C** PPO+DQN (no orch) | 0.6311 | 0.6497 | 0.5347 | 0.7498 | 0.40 | 852 |
| **D** Fixed-25%+EDF | 0.8460 | 0.5606 | 0.6524 | 0.9001 | 0.25 | 548 |
| **E** Fixed-75%+EDF | 0.2066 | 0.8393 | 0.2920 | 0.4305 | 0.75 | 1580 |
| **F** PPO+EDF | 0.6311 | 0.6497 | 0.5347 | 0.7498 | 0.40 | 852 |
| **G** PPO+Orch+EDF | 0.5769 | 0.6771 | 0.5014 | 0.7046 | 0.45 | 950 |

### Network / Scheduling Layer [Python sim — NOT NS-3]

| Config | Avg Lat (µs) | P95 Lat (µs) | Deadline Miss ↓ | Deadline Sat ↑ |
|--------|-------------|-------------|-----------------|----------------|
| **A** Conv-50%+EDF | 4820.4 | 8625.9 | 0.080 | 0.920 |
| **B** PPO+DQN+Orch | 4052.6 | 7248.7 | 0.150 | 0.850 |
| **C** PPO+DQN (no orch) | 3507.7 | 6296.7 | 0.140 | 0.860 |
| **D** Fixed-25%+EDF | 2550.0 | 4700.4 | 0.000 | 1.000 |
| **E** Fixed-75%+EDF | 7090.8 | 12568.2 | 0.390 | 0.610 |
| **F** PPO+EDF | 3648.0 | 6571.4 | 0.030 | 0.970 |
| **G** PPO+Orch+EDF | 4151.6 | 7441.6 | 0.070 | 0.930 |

---

## Key Findings

### Does AI (B) improve over Conventional (A)?

**Semantic quality:** B=0.6771 vs A=0.7194 → -5.9%

**Reconstruction MSE:** B=0.5769 vs A=0.4254 → +35.6% (negative = better)

**Network latency:** B=4052.6 µs vs A=4820.4 µs → -15.9%

**Deadline miss rate:** B=0.150 vs A=0.080 → +87.5%

### Does Orchestration add value (B vs C)?

**Semantic quality:** B=0.6771 vs C=0.6497 → +4.2%
**Network latency:** B=4052.6 µs vs C=3507.7 µs → +15.5%
**Deadline miss:** B=0.150 vs C=0.140 → +7.1%

### Does DQN add value over EDF (B vs G)?

**Network latency:** B(DQN)=4052.6 µs vs G(EDF)=4151.6 µs → -2.4%
**Deadline miss:** B(DQN)=0.150 vs G(EDF)=0.070 → +114.3%

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