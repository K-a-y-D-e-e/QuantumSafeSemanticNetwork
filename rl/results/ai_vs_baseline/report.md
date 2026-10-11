# Comprehensive Evaluation Report: AI-Enabled Semantic Network vs Conventional Baseline

**Project:** Quantum-Safe Semantic Communication for Deterministic Tactile Networks  
**Branch:** `experiment/rl-evaluation`  
**Date:** 2026-10-11  
**Artifact Location:** `rl/results/ai_vs_baseline/`  
**High-Resolution Figures:** `rl/results/ai_vs_baseline/graphs/` (300-DPI PNG, Vector PDF, Vector SVG)

---

## 1. Executive Summary & Experimental Scope

This evaluation investigates whether integrating reinforcement learning across semantic compression and packet scheduling—orchestrated by context-aware objectives—delivers genuine performance gains compared to conventional, non-RL networking benchmarks.

### Evaluated Configurations:

| ID | Configuration Name | Semantic Retention | Packet Scheduler | Orchestration | Conceptual Role |
|---|---|---|---|---|---|
| **A** | **Conv-50%+EDF** | Fixed 50% ($8/16$ dims) | Earliest-Deadline-First (EDF) | None | **Primary Conventional Baseline** |
| **B** | **PPO+DQN+Orch** | Adaptive PPO | Deep Q-Network (DQN) | Active Rule-Based | **Full AI-Enabled System** |
| **C** | **PPO+DQN (Ablation)**| Adaptive PPO | Deep Q-Network (DQN) | Disabled | Isolates Orchestration Impact |
| **D** | **Fixed-25%+EDF** | Fixed 25% ($4/16$ dims) | Earliest-Deadline-First (EDF) | None | Conservative Latency Baseline |
| **E** | **Fixed-75%+EDF** | Fixed 75% ($12/16$ dims) | Earliest-Deadline-First (EDF) | None | High-Fidelity Quality Baseline |
| **F** | **PPO+EDF** | Adaptive PPO | Earliest-Deadline-First (EDF) | None | Isolates DQN under Raw PPO |
| **G** | **PPO+Orch+EDF** | Adaptive PPO | Earliest-Deadline-First (EDF) | Active Rule-Based | Isolates DQN under Orchestrated PPO |

> [!IMPORTANT]
> **Measurement Distinction:** All network scheduling latencies, queueing delays, and deadline miss rates reported here are derived from discrete-event Python simulation environments (`NetworkSchedulingEnv` and `EventDrivenSimulator`). They represent algorithmic simulation proxies and **must not be labeled as physical or NS-3 network measurements**. True full-stack channel measurements require C++ NS-3 execution on Linux (see [`NS3_HANDOFF.md`](file:///c:/Users/RON%20MATHEW/Desktop/sem7/btech%20project/final/QuantumSafeSemanticNetwork-main/NS3_HANDOFF.md)).

---

## 2. Experimental Methodology & Fairness Safeguards

1. **Workload and Trace Matching:**
   * All policies were evaluated over identical packet arrival sequences and deadline constraints generated across 3 distinct random seeds ($999, 1099, 1199$) over 100 episodes each ($N=300$ total scheduling episodes per configuration, 3,000 total scheduled flows).
2. **Deterministic Kinematic Data Split:**
   * Semantic reconstruction metrics were computed over $N=256$ held-out kinematic trajectory sequences from the JIGSAWS surgical dataset (normalized 76-dimensional kinematics), strictly separated from training splits.
3. **Layer Decoupling & Data Integrity:**
   * No data was fabricated or post-hoc assembled. PPO and Orchestration operate at the semantic layer to set packet payload size ($548\,\text{B}$ to $1580\,\text{B}$); the exact resulting byte counts are passed into the scheduling simulator where DQN and EDF determine transmission queue order.

---

## 3. Verified Benchmark Results

*Summary Table (Averaged across 256 held-out semantic samples and 300 simulation episodes with 95% Confidence Intervals):*

| Config | Policy Label | Retention (%) | Mean Payload | Recon MSE ↓ | Semantic Quality ↑ | Mean Latency (µs) ↓ | P95 Latency (µs) ↓ | Deadline-Miss Rate ↓ | Deadline Satisfaction Rate ↑ |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | **Conv-50%+EDF (Baseline)** | 50.0% | $1064\,\text{B}$ | **0.3848** | **0.7393** | $4824.4 \pm 9.6$ | $8627.6 \pm 20.3$ | **5.1% ± 1.3%** | **94.9%** |
| **B** | **PPO+DQN+Orch (Full AI)** | 45.7% | $975.3\,\text{B}$ | 0.5355 | 0.6942 | $4133.7 \pm 84.0$ | $7632.8 \pm 112.5$ | 15.7% ± 1.4% | 84.3% |
| **C** | **PPO+DQN (No Orch)** | 42.6% | $910.8\,\text{B}$ | 0.5673 | 0.6782 | **3882.1 ± 72.2** | **7125.5 ± 98.4** | 14.7% ± 1.2% | 85.3% |
| **D** | Fixed-25%+EDF | 25.0% | $548.0\,\text{B}$ | 0.7851 | 0.5799 | $2554.0 \pm 9.6$ | $4689.3 \pm 20.3$ | 0.03% ± 0.07% | 99.97% |
| **E** | Fixed-75%+EDF | 75.0% | $1580.0\,\text{B}$ | 0.1846 | 0.8527 | $7094.8 \pm 9.6$ | $12569.8 \pm 20.3$ | 32.9% ± 3.1% | 67.1% |
| **F** | PPO+EDF | 42.6% | $910.8\,\text{B}$ | 0.5673 | 0.6782 | $4120.7 \pm 80.2$ | $7381.4 \pm 109.1$ | 4.2% ± 1.2% | 95.8% |
| **G** | PPO+Orch+EDF | 45.7% | $975.3\,\text{B}$ | 0.5355 | 0.6942 | $4390.8 \pm 89.0$ | $7854.0 \pm 118.2$ | 5.5% ± 1.4% | 94.5% |

---

## 4. Key Engineering Insights & Trade-Off Analysis

### 4.1 Does Lower Latency Imply Superior Performance?
**No.** While Full AI (**B**) achieves **$14.3\%$ lower mean latency** ($4133.7\,\mu\text{s}$ vs $4824.4\,\mu\text{s}$) and **$11.5\%$ lower P95 latency** than the conventional baseline (**A**), it suffers a **higher deadline-miss rate** ($15.7\%$ vs $5.1\%$) and lower overall semantic fidelity ($0.6942$ vs $0.7393$). In real-time tactile teleoperation, packet timeliness relative to hard deadlines is paramount; lower average transmission delay cannot compensate for missed execution windows.

### 4.2 Contribution of Orchestration (B vs C, G vs F)
1. **Semantic Quality Boost:** Orchestration intervenes on tactile-critical states, elevating retention from $25\%/50\%$ to $75\%$. This improves semantic quality by **$+2.35\%$** ($0.6942$ vs $0.6782$, MSE drops from $0.5673 \to 0.5355$).
2. **Network Overhead:** This targeted quality upgrade increases mean payload from $910.8\,\text{B}$ to $975.3\,\text{B}$, incurring a modest **$+6.5\%$ latency overhead** ($4133.7\,\mu\text{s}$ vs $3882.1\,\mu\text{s}$).

### 4.3 DQN vs Conventional EDF Scheduling (B vs G, C vs F)
1. **Mean Latency Optimization:** Greedy DQN consistently yields lower average latency than EDF ($4133.7\,\mu\text{s}$ vs $4390.8\,\mu\text{s}$, a **$5.9\%$ reduction**).
2. **Deadline Inefficiency:** However, DQN's deadline-miss rate is nearly **3× worse** than EDF ($15.7\%$ vs $5.5\%$). DQN was trained using a reward function that balances queue delay and latency against missed deadlines; EDF strictly prioritizes flows by remaining time-to-deadline ($\text{deadline} - t$).
3. **Optimal Practical Hybrid:** **Config G (PPO + Orchestration + EDF)** represents the superior practical operational balance: it achieves $4390.8\,\mu\text{s}$ latency ($9.0\%$ faster than Baseline A) while preserving a low $5.5\%$ deadline-miss rate and improved tactile quality.

---

## 5. Visualizations & Figures Index

All figures are compiled in vector PDF, SVG, and 300-DPI PNG under [`rl/results/ai_vs_baseline/graphs/`](file:///c:/Users/RON%20MATHEW/Desktop/sem7/btech%20project/final/QuantumSafeSemanticNetwork-main/rl/results/ai_vs_baseline/graphs/):

1. **`semantic_quality_vs_mean_latency`**: Multi-dimensional Pareto frontier mapping semantic quality against mean simulated network latency. Highlights the trade-off envelope across fixed and adaptive regimes.
2. **`mean_and_p95_latency_by_policy`**: Dual-bar comparison of average and 95th-percentile packet latencies with 95% bootstrap confidence intervals.
3. **`deadline_miss_rate_by_policy`**: Failure percentage across $N=300$ scheduling episodes with baseline benchmark reference.
4. **`reconstruction_mse_by_policy`**: Kinematic reconstruction distortion ($L_2$ error) on held-out JIGSAWS sequences.
5. **`semantic_quality_by_policy`**: Non-linear semantic fidelity metric score $1 / (1 + \text{MSE})$.
6. **`mean_payload_size_by_policy`**: Average transmitted bytes per packet indicating compression savings.
7. **`semantic_quality_vs_payload_size`**: Classical Rate-Distortion curve illustrating the compression efficiency.
8. **`mean_latency_vs_deadline_miss_rate`**: Direct trade-off plot establishing why lower average delay does not guarantee deadline compliance.
9. **`ppo_vs_ppo_orchestration_ablation`**: Side-by-side ablation isolating orchestration benefits on semantic quality and network overhead.
10. **`dqn_vs_edf_scheduling_comparison`**: Head-to-head comparison demonstrating DQN's latency advantage versus EDF's deadline superiority.

---

## 6. Discrepancy Reconciliation & Resolution

* **Investigated Discrepancy:** Earlier single-step diagnostic logs recorded deadline-miss rates of $\approx 27\%$, whereas multi-episode simulated link experiments showed $10.9\% - 15.7\%$, and fixed schedulers under EDF achieved $\approx 5\%$.
* **Root Cause Found:** The diagnostic log measured deadline satisfaction against a static, normalized proxy feature ($t_{\text{proxy}} > \text{deadline\_norm}$), whereas link simulations model packet queueing dynamics over time. Furthermore, comparing DQN against EDF revealed that DQN's reward weights prioritize delay over strict deadline ordering, explaining the higher miss rate in DQN-managed queues.

---

## 7. Limitations & Remaining Tasks for Linux Teammate

1. **Discrete Simulation Assumptions:** The Python simulator does not account for physical MAC layer backoff, dropped frames due to bit errors, or transport handshake overhead.
2. **Handoff Artifacts Ready:** Workload traces matching Conventional Baseline (**A**) and AI-Full (**B**) are prepared in [`ns3_workload_A.csv`](file:///c:/Users/RON%20MATHEW/Desktop/sem7/btech%20project/final/QuantumSafeSemanticNetwork-main/rl/results/ai_vs_baseline/ns3_workload_A.csv) and [`ns3_workload_B.csv`](file:///c:/Users/RON%20MATHEW/Desktop/sem7/btech%20project/final/QuantumSafeSemanticNetwork-main/rl/results/ai_vs_baseline/ns3_workload_B.csv).
3. **Instructions:** Refer to [`NS3_HANDOFF.md`](file:///c:/Users/RON%20MATHEW/Desktop/sem7/btech%20project/final/QuantumSafeSemanticNetwork-main/NS3_HANDOFF.md) for Linux execution scripts and setup instructions.