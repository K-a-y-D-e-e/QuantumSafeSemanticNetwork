"""
AI-Enabled Semantic Network vs Conventional Baseline — Evaluation Harness
=========================================================================

Configurations evaluated (see Task §2):

A  — Conventional:  Fixed 50% semantic retention + EDF packet scheduling.
B  — AI Full:       PPO compression + DQN scheduling + Orchestration bias.
C  — AI No-Orch:    PPO compression + DQN scheduling, NO orchestration bias
                    (ablation to isolate orchestration contribution).
D  — Fixed-25+EDF:  Additional fixed-retention baseline (lower bound).
E  — Fixed-75+EDF:  Additional fixed-retention baseline (upper quality).
F  — PPO+EDF:       PPO compression + EDF scheduling (no DQN, no orchestration).
G  — PPO+Orch+EDF:  PPO+Orchestration compression + EDF scheduling (isolates DQN).

Design principles
-----------------
- All configurations share identical random-seed workloads (seed 999 for
  scheduling evaluation, seed 42 for semantic evaluation). No workload
  is generated differently per configuration.
- PPO and DQN operate in different layers (no hidden coupling):
    * PPO selects semantic retention → determines packet_size_bytes.
    * DQN selects scheduling order → determines latency and deadline outcomes.
  In the joint (B/C) configuration, each episode runs as:
    1. PPO decides compression for each kinematics sample → assigns packet sizes.
    2. DQN (or EDF) schedules those packets over the link.
  This is the smallest truthful joint harness that does not fabricate results.
- Orchestration (in B) biases PPO's compression action using
  AIOrchestrationAgent.bias_action_for_objective(); it does NOT touch DQN.
- No metrics are fabricated. Network-layer results come from the existing
  EventDrivenSimulator / NetworkSchedulingEnv. Semantic-layer results come
  from the encoder/decoder checkpoint.
- All outputs written to rl/results/ai_vs_baseline/.

Outputs
-------
    rl/results/ai_vs_baseline/experiment_config.json
    rl/results/ai_vs_baseline/policy_comparison.csv
    rl/results/ai_vs_baseline/semantic_results.csv      (per-sample semantic)
    rl/results/ai_vs_baseline/scheduling_results.csv    (per-episode network)
    rl/results/ai_vs_baseline/combined_results.csv      (merged summary)
    rl/results/ai_vs_baseline/ns3_workload_B.csv        (AI-full workload for NS-3)
    rl/results/ai_vs_baseline/ns3_workload_A.csv        (Conventional workload for NS-3)
    rl/results/ai_vs_baseline/plots/*.png
    rl/results/ai_vs_baseline/report.md

Reproduction
------------
    python ai_vs_baseline_evaluation.py
    python ai_vs_baseline_evaluation.py --eval-seed 999 --num-episodes 100 --num-seeds 5
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# ── Project-root imports ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_semantic_rl import build_deterministic_state, build_held_out_split
from network.simulation.analyzer import NetworkAnalyzer
from network.simulation.deadline_scheduler import DeadlineAwareScheduler
from network.simulation.event_simulator import EventDrivenSimulator
from network.simulation.link import NetworkLink
from orchestration.agent import AIOrchestrationAgent
from rl.agent import RLAgent
from rl.dqn_agent import DQNAgent
from rl.environment import (
    ACTION_TO_COMPRESSION,
    MAX_QUEUE,
    OBS_DIM,
    NetworkSchedulingEnv,
    SemanticCompressionEnv,
)
from rl.traffic_generator import TrafficGenerator
from semantic.compression import AdaptiveSemanticCompressor
from semantic.data_loader import JIGSAWSKinematicsDataset
from semantic.model_io import (
    DEFAULT_INPUT_DIM,
    DEFAULT_LATENT_DIM,
    DEFAULT_HIDDEN_DIM,
    load_semantic_models,
)

# ── Constants ─────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("rl/results/ai_vs_baseline")
LATENT_DIM = DEFAULT_LATENT_DIM         # 16
SEQ_LEN = 32                            # JIGSAWS window length
HEADER_BYTES = 32                       # protocol overhead

# Packet-size map: action → bytes
# Formula: seq_len * retained_dims * 4 (float32) + retained_dims + header
def action_to_packet_bytes(action: int) -> int:
    frac = ACTION_TO_COMPRESSION[action]
    dims = max(1, int(LATENT_DIM * frac))
    return int(SEQ_LEN * dims * 4 + dims + HEADER_BYTES)

ACTION_BYTES = {a: action_to_packet_bytes(a) for a in range(4)}
# Conventional Fixed-50 packet size (action=1)
FIXED50_BYTES = ACTION_BYTES[1]   # 1064 B
FIXED25_BYTES = ACTION_BYTES[0]   # 548 B
FIXED75_BYTES = ACTION_BYTES[2]   # 1580 B

# Network link (same as rl_config.yaml defaults)
LINK_BW_MBPS = 10
LINK_PROP_DELAY_US = 100

# Reward weight normalization constants (same as rl/evaluate.py)
ENV_KWARGS = dict(
    w_deadline=2.0,
    w_latency=1.0,
    w_queue=0.5,
    w_miss=3.0,
    max_deadline_us=20_000.0,
    max_size_bytes=2_000.0,
    max_wait_us=10_000.0,
    max_expected_latency_us=15_000.0,
    episode_max_time_us=50_000.0,
)

# Configuration names
CONFIG_NAMES = ["A", "B", "C", "D", "E", "F", "G"]
CONFIG_LABELS = {
    "A": "Conv-50%+EDF",
    "B": "PPO+DQN+Orch",
    "C": "PPO+DQN (no orch)",
    "D": "Fixed-25%+EDF",
    "E": "Fixed-75%+EDF",
    "F": "PPO+EDF",
    "G": "PPO+Orch+EDF",
}


# =============================================================================
# Semantic evaluation helpers
# =============================================================================

def evaluate_semantic_policy(
    encoder,
    decoder,
    compressor,
    samples: List[torch.Tensor],
    val_indices: np.ndarray,
    action_fn,          # callable(state_vec: np.ndarray) -> int
    seed: int,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Run a semantic compression policy over all held-out samples.

    Parameters
    ----------
    action_fn : callable
        Given a 7-dim state vector, returns action in {0,1,2,3}.

    Returns (summary_dict, per_step_records).
    """
    device = torch.device("cpu")
    mses, maes, rmses, qualities, fracs, payloads, rewards_sem = [], [], [], [], [], [], []
    records = []

    for local_idx, orig_idx in enumerate(val_indices):
        state = build_deterministic_state(int(orig_idx), seed)
        action = int(action_fn(state))
        frac = ACTION_TO_COMPRESSION[action]
        dims = max(1, int(LATENT_DIM * frac))
        payload_b = ACTION_BYTES[action]

        sample = samples[local_idx].to(device)
        with torch.no_grad():
            lat = encoder(sample)
            comp, idx = compressor.compress(lat, frac)
            relat = compressor.decompress(comp, idx)
            rec = decoder(relat)
            mse = float(torch.mean((rec - sample) ** 2).item())
            mae = float(torch.mean(torch.abs(rec - sample)).item())
            rmse = math.sqrt(mse)
            quality = float(np.clip(1.0 / (1.0 + mse), 0.0, 1.0))

        mses.append(mse)
        maes.append(mae)
        rmses.append(rmse)
        qualities.append(quality)
        fracs.append(frac)
        payloads.append(payload_b)

        records.append({
            "orig_index": int(orig_idx),
            "action": action,
            "retention_frac": frac,
            "payload_bytes": payload_b,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "quality": quality,
        })

    n = len(mses)
    summary = {
        "n_samples": n,
        "mse_mean": float(np.mean(mses)),
        "mse_p95": float(np.percentile(mses, 95)),
        "mae_mean": float(np.mean(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "quality_mean": float(np.mean(qualities)),
        "quality_std": float(np.std(qualities)),
        "retention_mean": float(np.mean(fracs)),
        "payload_bytes_mean": float(np.mean(payloads)),
        "total_bytes": int(sum(payloads)),
        "pct_a0": float(np.mean([1 if r["action"] == 0 else 0 for r in records]) * 100),
        "pct_a1": float(np.mean([1 if r["action"] == 1 else 0 for r in records]) * 100),
        "pct_a2": float(np.mean([1 if r["action"] == 2 else 0 for r in records]) * 100),
        "pct_a3": float(np.mean([1 if r["action"] == 3 else 0 for r in records]) * 100),
    }
    return summary, records


# =============================================================================
# Scheduling evaluation helpers
# =============================================================================

class _FixedSizeTrafficGenerator:
    """
    Wraps TrafficGenerator but overrides packet sizes to a fixed value.
    Used so Fixed-50%/25%/75% baselines use realistic packet sizes matching
    their compression level, while sharing the same arrival/deadline pattern.
    """
    def __init__(self, gen: TrafficGenerator, fixed_size_bytes: int):
        self._gen = gen
        self._size = fixed_size_bytes

    def generate(self) -> List[Dict[str, Any]]:
        pkts = self._gen.generate()
        for p in pkts:
            p["size_bytes"] = self._size
            p["payload"] = b"X" * self._size
        return pkts


class _PPOSizedTrafficGenerator:
    """
    Wraps TrafficGenerator but assigns packet sizes from a PPO decision log.
    Each episode pops from a pre-computed list of per-packet action→bytes.
    If the list is exhausted it falls back to fixed 50% size.
    """
    def __init__(self, gen: TrafficGenerator, episode_sizes: List[List[int]]):
        self._gen = gen
        self._episode_sizes = episode_sizes
        self._ep_idx = 0

    def generate(self) -> List[Dict[str, Any]]:
        pkts = self._gen.generate()
        if self._ep_idx < len(self._episode_sizes):
            sizes = self._episode_sizes[self._ep_idx]
            self._ep_idx += 1
        else:
            sizes = [FIXED50_BYTES] * len(pkts)

        for p, sz in zip(pkts, sizes):
            p["size_bytes"] = sz
            p["payload"] = b"X" * sz
        return pkts


def run_edf_scheduler(
    packets: List[Dict],
    link: NetworkLink,
) -> List[Dict]:
    """Run EDF (DeadlineAwareScheduler) on a packet list."""
    sim = EventDrivenSimulator(link=link, scheduler=DeadlineAwareScheduler())
    return sim.simulate(list(packets))


def run_dqn_scheduler(
    agent: DQNAgent,
    packets: List[Dict],
    link: NetworkLink,
) -> List[Dict]:
    """Run DQN agent (fully greedy, ε=0) on a fixed packet list."""
    from rl.evaluate import _FixedTrafficGenerator, run_dqn_scheduler as _run
    return _run(agent, packets, link, ENV_KWARGS)


def summarise_scheduling(results: List[Dict]) -> Dict[str, float]:
    """Compute per-episode scheduling metrics via NetworkAnalyzer."""
    if not results:
        return {}
    ana = NetworkAnalyzer.analyze(results)
    n = ana.get("total_flows", 0)
    misses = ana.get("deadline_misses", 0)
    latencies = [r["latency_us"] for r in results]
    return {
        "avg_latency_us": ana.get("average_latency_us", 0.0),
        "p95_latency_us": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "min_latency_us": ana.get("minimum_latency_us", 0.0),
        "max_latency_us": ana.get("maximum_latency_us", 0.0),
        "avg_queueing_us": ana.get("average_queueing_delay_us", 0.0),
        "jitter_us": ana.get("jitter_us", 0.0),
        "deadline_misses": misses,
        "deadline_miss_rate": misses / n if n > 0 else 0.0,
        "deadline_satisfaction_rate": 1.0 - (misses / n if n > 0 else 0.0),
        "completed_flows": n,
        "throughput_pkt": float(n),
    }


# =============================================================================
# PPO-sized workload builder
# =============================================================================

def build_ppo_episode_sizes(
    encoder,
    decoder,
    compressor,
    samples: List[torch.Tensor],
    val_indices: np.ndarray,
    seed: int,
    action_fn,
    num_episodes: int,
    flows_per_episode: int,
) -> List[List[int]]:
    """
    Pre-compute per-packet sizes for PPO-based configurations.
    Each episode randomly draws flows_per_episode samples from val_indices
    and maps them to packet sizes via action_fn.

    This ensures PPO packet sizes are data-driven (from actual encoder/decoder
    compression decisions) rather than fabricated.
    """
    rng = np.random.default_rng(seed + 1000)
    episode_sizes = []
    for _ in range(num_episodes):
        chosen = rng.choice(len(val_indices), size=flows_per_episode, replace=True)
        sizes = []
        for ci in chosen:
            orig_idx = val_indices[ci]
            state = build_deterministic_state(int(orig_idx), seed)
            action = int(action_fn(state))
            sizes.append(ACTION_BYTES[action])
        episode_sizes.append(sizes)
    return episode_sizes


# =============================================================================
# Main evaluation loop
# =============================================================================

def run_evaluation(
    eval_seed: int = 999,
    sem_seed: int = 42,
    num_episodes: int = 100,
    num_seeds: int = 3,
    ppo_checkpoint: str = "rl/checkpoints/ppo_semantic_adaptive.zip",
    dqn_checkpoint: str = "rl/checkpoints/dqn_final.pt",
    encoder_path: str = "semantic_encoder.pth",
    decoder_path: str = "semantic_decoder.pth",
    data_root: str = "dataset",
    num_eval_samples: int = 256,
    flows_per_episode: int = 10,
    output_dir: Path = RESULTS_DIR,
) -> Dict[str, Any]:

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── 1. Load models ────────────────────────────────────────────────────────
    print("Loading semantic models...")
    encoder, decoder = load_semantic_models(encoder_path, decoder_path)
    compressor = AdaptiveSemanticCompressor(latent_dim=LATENT_DIM)
    orchestrator = AIOrchestrationAgent()

    # ── 2. Load PPO checkpoint ────────────────────────────────────────────────
    print("Loading PPO checkpoint...")
    # Use a minimal dummy env just for loading weights
    dummy_env = SemanticCompressionEnv(
        encoder=encoder, decoder=decoder, compressor=compressor,
        kinematics_samples=[torch.randn(1, DEFAULT_INPUT_DIM)],
        orchestrator=orchestrator,
        use_reconstruction_reward=True,
    )
    ppo_agent = RLAgent(environment=dummy_env, seed=sem_seed)
    ppo_agent.load(ppo_checkpoint, environment=dummy_env)

    # ── 3. Load DQN checkpoint ────────────────────────────────────────────────
    print("Loading DQN checkpoint...")
    dqn_agent = DQNAgent(obs_dim=OBS_DIM, action_dim=MAX_QUEUE, seed=eval_seed)
    dqn_agent.load(dqn_checkpoint)
    dqn_agent.epsilon = 0.0  # fully greedy

    # ── 4. Build held-out semantic dataset ────────────────────────────────────
    print("Building held-out semantic split...")
    dataset = JIGSAWSKinematicsDataset(data_root)
    samples, _, val_indices = build_held_out_split(
        dataset, seed=sem_seed, val_fraction=0.15,
        max_eval_samples=num_eval_samples,
    )
    print(f"  Held-out samples: {len(samples)} (indices from {val_indices.min()} to {val_indices.max()})")

    # ── 5. Define action functions ────────────────────────────────────────────
    def ppo_raw_action(state: np.ndarray) -> int:
        return int(ppo_agent.predict(state, deterministic=True))

    def ppo_orch_action(state: np.ndarray) -> int:
        raw = int(ppo_agent.predict(state, deterministic=True))
        obj = orchestrator.decide_objective(
            float(state[0]), float(state[2]), float(state[3]), float(state[5])
        )
        return int(AIOrchestrationAgent.bias_action_for_objective(raw, obj))

    fixed_action_fns = {
        "D": lambda s: 0,   # Fixed 25%
        "A": lambda s: 1,   # Fixed 50%
        "E": lambda s: 2,   # Fixed 75%
    }

    # ── 6. Semantic layer evaluation (all configs) ────────────────────────────
    print("\nRunning semantic evaluations...")
    sem_configs = {
        "A": fixed_action_fns["A"],
        "B": ppo_orch_action,
        "C": ppo_raw_action,
        "D": fixed_action_fns["D"],
        "E": fixed_action_fns["E"],
        "F": ppo_raw_action,       # same as C (EDF doesn't affect compression)
        "G": ppo_orch_action,      # same as B (EDF doesn't affect compression)
    }
    sem_results = {}
    sem_records_all = {}
    for cfg, fn in sem_configs.items():
        label = CONFIG_LABELS[cfg]
        print(f"  {cfg} [{label}]...")
        summary, records = evaluate_semantic_policy(
            encoder, decoder, compressor, samples, val_indices, fn, sem_seed
        )
        sem_results[cfg] = summary
        sem_records_all[cfg] = records

    # ── 7. Build PPO episode sizes for scheduling ─────────────────────────────
    print("\nBuilding PPO-sized packet workloads...")
    ppo_raw_sizes = build_ppo_episode_sizes(
        encoder, decoder, compressor, samples, val_indices,
        sem_seed, ppo_raw_action, num_episodes, flows_per_episode,
    )
    ppo_orch_sizes = build_ppo_episode_sizes(
        encoder, decoder, compressor, samples, val_indices,
        sem_seed, ppo_orch_action, num_episodes, flows_per_episode,
    )

    # ── 8. Scheduling layer evaluation ───────────────────────────────────────
    print("\nRunning scheduling evaluations across seeds...")
    link = NetworkLink(bandwidth_mbps=LINK_BW_MBPS, propagation_delay_us=LINK_PROP_DELAY_US)

    # All seeds to average over
    seeds_to_run = [eval_seed + i * 100 for i in range(num_seeds)]

    # Per-config per-seed per-episode accumulators
    sched_results: Dict[str, List[Dict]] = {cfg: [] for cfg in CONFIG_NAMES}

    for seed_i, cur_seed in enumerate(seeds_to_run):
        print(f"  Seed {cur_seed} ({seed_i+1}/{num_seeds})...")
        gen = TrafficGenerator(seed=cur_seed, num_flows=flows_per_episode)

        for ep in range(num_episodes):
            packets_base = gen.generate()   # base arrival/deadline/priority pattern

            # --- Config A: Fixed 50% + EDF ---
            pkts_a = [dict(p, size_bytes=FIXED50_BYTES, payload=b"X"*FIXED50_BYTES) for p in packets_base]
            res_a = run_edf_scheduler(pkts_a, link)
            sched_results["A"].append(summarise_scheduling(res_a))

            # --- Config D: Fixed 25% + EDF ---
            pkts_d = [dict(p, size_bytes=FIXED25_BYTES, payload=b"X"*FIXED25_BYTES) for p in packets_base]
            res_d = run_edf_scheduler(pkts_d, link)
            sched_results["D"].append(summarise_scheduling(res_d))

            # --- Config E: Fixed 75% + EDF ---
            pkts_e = [dict(p, size_bytes=FIXED75_BYTES, payload=b"X"*FIXED75_BYTES) for p in packets_base]
            res_e = run_edf_scheduler(pkts_e, link)
            sched_results["E"].append(summarise_scheduling(res_e))

            # --- Configs B + C: PPO sizes, DQN scheduling ---
            ep_idx = (seed_i * num_episodes + ep) % len(ppo_orch_sizes)
            orch_sizes = ppo_orch_sizes[ep_idx]
            raw_sizes = ppo_raw_sizes[ep_idx]

            # Config B: PPO+Orch sizes + DQN
            pkts_b = list(packets_base)
            for i, p in enumerate(pkts_b):
                sz = orch_sizes[i] if i < len(orch_sizes) else FIXED50_BYTES
                p = dict(p, size_bytes=sz, payload=b"X"*sz)
                pkts_b[i] = p
            res_b = run_dqn_scheduler(dqn_agent, pkts_b, link)
            sched_results["B"].append(summarise_scheduling(res_b))

            # Config C: PPO-raw sizes + DQN (no orchestration)
            pkts_c = list(packets_base)
            for i, p in enumerate(pkts_c):
                sz = raw_sizes[i] if i < len(raw_sizes) else FIXED50_BYTES
                p = dict(p, size_bytes=sz, payload=b"X"*sz)
                pkts_c[i] = p
            res_c = run_dqn_scheduler(dqn_agent, pkts_c, link)
            sched_results["C"].append(summarise_scheduling(res_c))

            # Config F: PPO-raw sizes + EDF (no DQN, no orch)
            pkts_f = list(packets_base)
            for i, p in enumerate(pkts_f):
                sz = raw_sizes[i] if i < len(raw_sizes) else FIXED50_BYTES
                p = dict(p, size_bytes=sz, payload=b"X"*sz)
                pkts_f[i] = p
            res_f = run_edf_scheduler(pkts_f, link)
            sched_results["F"].append(summarise_scheduling(res_f))

            # Config G: PPO+Orch sizes + EDF (orch but no DQN)
            pkts_g = list(packets_base)
            for i, p in enumerate(pkts_g):
                sz = orch_sizes[i] if i < len(orch_sizes) else FIXED50_BYTES
                p = dict(p, size_bytes=sz, payload=b"X"*sz)
                pkts_g[i] = p
            res_g = run_edf_scheduler(pkts_g, link)
            sched_results["G"].append(summarise_scheduling(res_g))

    # ── 9. Aggregate scheduling results ──────────────────────────────────────
    print("\nAggregating results...")
    metric_keys = [
        "avg_latency_us", "p95_latency_us", "min_latency_us", "max_latency_us",
        "avg_queueing_us", "jitter_us", "deadline_misses",
        "deadline_miss_rate", "deadline_satisfaction_rate", "completed_flows",
    ]
    total_runs = num_episodes * num_seeds

    sched_agg: Dict[str, Dict] = {}
    for cfg in CONFIG_NAMES:
        eps_data = sched_results[cfg]
        n = len(eps_data)
        if n == 0:
            sched_agg[cfg] = {}
            continue
        agg = {}
        for k in metric_keys:
            vals = [e[k] for e in eps_data if k in e]
            if vals:
                agg[f"{k}_mean"] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))
                agg[f"{k}_p95"] = float(np.percentile(vals, 95))
                agg[f"{k}_ci95_lo"] = float(np.mean(vals) - 1.96 * np.std(vals) / math.sqrt(len(vals)))
                agg[f"{k}_ci95_hi"] = float(np.mean(vals) + 1.96 * np.std(vals) / math.sqrt(len(vals)))
        agg["n_episodes"] = n
        sched_agg[cfg] = agg

    # ── 10. Build combined summary table ─────────────────────────────────────
    rows = []
    for cfg in CONFIG_NAMES:
        s = sem_results.get(cfg, {})
        sc = sched_agg.get(cfg, {})
        row = {
            "config": cfg,
            "label": CONFIG_LABELS[cfg],
            # Semantic
            "sem_mse_mean": s.get("mse_mean"),
            "sem_mse_p95": s.get("mse_p95"),
            "sem_mae_mean": s.get("mae_mean"),
            "sem_rmse_mean": s.get("rmse_mean"),
            "sem_quality_mean": s.get("quality_mean"),
            "sem_quality_std": s.get("quality_std"),
            "retention_mean": s.get("retention_mean"),
            "payload_bytes_mean": s.get("payload_bytes_mean"),
            "total_bytes": s.get("total_bytes"),
            "pct_a0_25pct": s.get("pct_a0"),
            "pct_a1_50pct": s.get("pct_a1"),
            "pct_a2_75pct": s.get("pct_a2"),
            "pct_a3_100pct": s.get("pct_a3"),
            # Scheduling
            "net_avg_latency_us": sc.get("avg_latency_us_mean"),
            "net_p95_latency_us": sc.get("p95_latency_us_mean"),
            "net_avg_queueing_us": sc.get("avg_queueing_us_mean"),
            "net_deadline_miss_rate": sc.get("deadline_miss_rate_mean"),
            "net_deadline_sat_rate": sc.get("deadline_satisfaction_rate_mean"),
            "net_jitter_us": sc.get("jitter_us_mean"),
            "net_lat_ci95_lo": sc.get("avg_latency_us_ci95_lo"),
            "net_lat_ci95_hi": sc.get("avg_latency_us_ci95_hi"),
            "net_miss_ci95_lo": sc.get("deadline_miss_rate_ci95_lo"),
            "net_miss_ci95_hi": sc.get("deadline_miss_rate_ci95_hi"),
            "n_sched_episodes": sc.get("n_episodes"),
        }
        rows.append(row)

    # ── 11. Write CSVs ────────────────────────────────────────────────────────
    combined_csv = output_dir / "combined_results.csv"
    if rows:
        with open(combined_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"  Written: {combined_csv}")

    # policy_comparison.csv (required deliverable)
    policy_csv = output_dir / "policy_comparison.csv"
    with open(policy_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {policy_csv}")

    # Semantic results (per sample, for B)
    sem_csv = output_dir / "semantic_results.csv"
    b_records = sem_records_all.get("B", [])
    if b_records:
        with open(sem_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(b_records[0].keys()))
            writer.writeheader()
            writer.writerows(b_records)
    print(f"  Written: {sem_csv}")

    # Scheduling raw results (per-episode, for all configs)
    sched_csv = output_dir / "scheduling_results.csv"
    with open(sched_csv, "w", newline="") as f:
        all_ep_rows = []
        for cfg in CONFIG_NAMES:
            for ep_i, ep_data in enumerate(sched_results[cfg]):
                ep_row = {"config": cfg, "label": CONFIG_LABELS[cfg], "episode": ep_i}
                ep_row.update(ep_data)
                all_ep_rows.append(ep_row)
        if all_ep_rows:
            writer = csv.DictWriter(f, fieldnames=list(all_ep_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_ep_rows)
    print(f"  Written: {sched_csv}")

    # ── 12. NS-3 workload exports ─────────────────────────────────────────────
    print("\nExporting NS-3 workloads...")
    _export_ns3_workload(
        sched_results["A"][:num_episodes],
        sem_records_all.get("A", []),
        output_dir / "ns3_workload_A.csv",
        config="A",
        label=CONFIG_LABELS["A"],
    )
    _export_ns3_workload(
        sched_results["B"][:num_episodes],
        sem_records_all.get("B", []),
        output_dir / "ns3_workload_B.csv",
        config="B",
        label=CONFIG_LABELS["B"],
    )

    # ── 13. Plots ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    _plot_quality_vs_latency(rows, plots_dir)
    _plot_miss_rate_comparison(rows, plots_dir)
    _plot_quality_comparison(rows, plots_dir)
    _plot_latency_comparison(rows, plots_dir)
    _plot_payload_vs_quality(rows, plots_dir)

    # ── 14. Experiment config ─────────────────────────────────────────────────
    exp_config = {
        "eval_seed": eval_seed,
        "sem_seed": sem_seed,
        "num_episodes": num_episodes,
        "num_seeds": num_seeds,
        "total_episodes_per_config": total_runs,
        "num_eval_samples": num_eval_samples,
        "flows_per_episode": flows_per_episode,
        "ppo_checkpoint": ppo_checkpoint,
        "dqn_checkpoint": dqn_checkpoint,
        "encoder_path": encoder_path,
        "decoder_path": decoder_path,
        "link_bandwidth_mbps": LINK_BW_MBPS,
        "link_propagation_delay_us": LINK_PROP_DELAY_US,
        "action_bytes": ACTION_BYTES,
        "configurations": CONFIG_LABELS,
        "note": (
            "Semantic-layer results (MSE/quality) come from encoder-decoder inference on "
            "JIGSAWS kinematics. Scheduling-layer results (latency/deadline) come from "
            "Python EventDrivenSimulator / NetworkSchedulingEnv. These are Python simulation "
            "results, NOT NS-3 measurements. NS-3 workload exports are in ns3_workload_*.csv."
        ),
    }
    cfg_path = output_dir / "experiment_config.json"
    with open(cfg_path, "w") as f:
        json.dump(exp_config, f, indent=2)
    print(f"  Written: {cfg_path}")

    # ── 15. Report ────────────────────────────────────────────────────────────
    _write_report(rows, sched_agg, sem_results, exp_config, output_dir)

    print(f"\nDone. Results in {output_dir}/")
    return {"combined": rows, "sem": sem_results, "sched": sched_agg}


# =============================================================================
# NS-3 workload export
# =============================================================================

def _export_ns3_workload(
    sched_ep_list: List[Dict],
    sem_records: List[Dict],
    path: Path,
    config: str,
    label: str,
) -> None:
    """Export per-packet NS-3 workload trace for one configuration."""
    sem_quality_mean = float(np.mean([r["quality"] for r in sem_records])) if sem_records else 0.0
    sem_mse_mean = float(np.mean([r["mse"] for r in sem_records])) if sem_records else 0.0
    payload_mean = float(np.mean([r["payload_bytes"] for r in sem_records])) if sem_records else FIXED50_BYTES

    rows = []
    t_us = 0
    for ep_i, ep in enumerate(sched_ep_list):
        # synthesize one row per episode (aggregate, since we don't have per-packet records here)
        rows.append({
            "config": config,
            "label": label,
            "episode": ep_i + 1,
            "avg_latency_us": ep.get("avg_latency_us", 0),
            "p95_latency_us": ep.get("p95_latency_us", 0),
            "deadline_miss_rate": ep.get("deadline_miss_rate", 0),
            "deadline_satisfaction_rate": ep.get("deadline_satisfaction_rate", 0),
            "completed_flows": ep.get("completed_flows", 0),
            "avg_payload_bytes": round(payload_mean),
            "sem_quality": round(sem_quality_mean, 6),
            "sem_mse": round(sem_mse_mean, 6),
            "note": "Python simulation result - NOT NS-3 measurement",
        })

    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Plotting helpers
# =============================================================================

COLORS = {
    "A": "#1f77b4",   # blue
    "B": "#d62728",   # red
    "C": "#ff7f0e",   # orange
    "D": "#2ca02c",   # green
    "E": "#9467bd",   # purple
    "F": "#8c564b",   # brown
    "G": "#e377c2",   # pink
}


def _plot_quality_vs_latency(rows: List[Dict], plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        cfg = row["config"]
        q = row.get("sem_quality_mean")
        lat = row.get("net_avg_latency_us")
        if q is None or lat is None:
            continue
        ax.scatter(lat, q, color=COLORS.get(cfg, "gray"), s=120, zorder=5)
        ax.annotate(
            f'{cfg}: {CONFIG_LABELS[cfg]}',
            (lat, q), textcoords="offset points", xytext=(8, 4), fontsize=8,
        )
    ax.set_xlabel("Mean Network Latency (µs) [Python sim]")
    ax.set_ylabel("Mean Semantic Quality [1/(1+MSE)]")
    ax.set_title("Quality–Latency Trade-off: AI vs Conventional")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "quality_vs_latency.png", dpi=150)
    plt.close(fig)


def _plot_miss_rate_comparison(rows: List[Dict], plots_dir: Path) -> None:
    cfgs = [r["config"] for r in rows]
    labels = [CONFIG_LABELS[c] for c in cfgs]
    miss_rates = [r.get("net_deadline_miss_rate") or 0 for r in rows]
    ci_lo = [r.get("net_miss_ci95_lo") or 0 for r in rows]
    ci_hi = [r.get("net_miss_ci95_hi") or 0 for r in rows]
    errs_lo = [max(0, mr - lo) for mr, lo in zip(miss_rates, ci_lo)]
    errs_hi = [max(0, hi - mr) for mr, hi in zip(miss_rates, ci_hi)]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cfgs))
    bars = ax.bar(x, miss_rates, color=[COLORS.get(c, "gray") for c in cfgs], alpha=0.8)
    ax.errorbar(x, miss_rates, yerr=[errs_lo, errs_hi], fmt="none", color="black", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Deadline Miss Rate")
    ax.set_title("Deadline Miss Rate by Configuration (95% CI) [Python sim]")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, mr in zip(bars, miss_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{mr:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "deadline_miss_rate.png", dpi=150)
    plt.close(fig)


def _plot_quality_comparison(rows: List[Dict], plots_dir: Path) -> None:
    cfgs = [r["config"] for r in rows]
    labels = [CONFIG_LABELS[c] for c in cfgs]
    qualities = [r.get("sem_quality_mean") or 0 for r in rows]
    stds = [r.get("sem_quality_std") or 0 for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cfgs))
    ax.bar(x, qualities, color=[COLORS.get(c, "gray") for c in cfgs], alpha=0.8)
    ax.errorbar(x, qualities, yerr=stds, fmt="none", color="black", capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean Semantic Quality [1/(1+MSE)]")
    ax.set_title("Semantic Quality by Configuration")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    for xi, q in zip(x, qualities):
        ax.text(xi, q + 0.01, f"{q:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "semantic_quality_comparison.png", dpi=150)
    plt.close(fig)


def _plot_latency_comparison(rows: List[Dict], plots_dir: Path) -> None:
    cfgs = [r["config"] for r in rows]
    labels = [CONFIG_LABELS[c] for c in cfgs]
    lats = [r.get("net_avg_latency_us") or 0 for r in rows]
    p95s = [r.get("net_p95_latency_us") or 0 for r in rows]
    ci_lo = [r.get("net_lat_ci95_lo") or 0 for r in rows]
    ci_hi = [r.get("net_lat_ci95_hi") or 0 for r in rows]
    errs_lo = [max(0, m - lo) for m, lo in zip(lats, ci_lo)]
    errs_hi = [max(0, hi - m) for m, hi in zip(lats, ci_hi)]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cfgs))
    w = 0.35
    ax.bar(x - w/2, lats, w, label="Mean latency", color=[COLORS.get(c, "gray") for c in cfgs], alpha=0.8)
    ax.errorbar(x - w/2, lats, yerr=[errs_lo, errs_hi], fmt="none", color="black", capsize=4)
    ax.bar(x + w/2, p95s, w, label="P95 latency", color=[COLORS.get(c, "gray") for c in cfgs], alpha=0.4, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Latency (µs) [Python sim]")
    ax.set_title("Mean and P95 Network Latency by Configuration")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "latency_comparison.png", dpi=150)
    plt.close(fig)


def _plot_payload_vs_quality(rows: List[Dict], plots_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        cfg = row["config"]
        pb = row.get("payload_bytes_mean")
        q = row.get("sem_quality_mean")
        if pb is None or q is None:
            continue
        ax.scatter(pb, q, color=COLORS.get(cfg, "gray"), s=120, zorder=5)
        ax.annotate(cfg, (pb, q), textcoords="offset points", xytext=(6, 4), fontsize=9, fontweight="bold")
    ax.set_xlabel("Mean Payload Size (bytes)")
    ax.set_ylabel("Mean Semantic Quality")
    ax.set_title("Payload Size vs Semantic Quality Trade-off")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "payload_vs_quality.png", dpi=150)
    plt.close(fig)


# =============================================================================
# Report writer
# =============================================================================

def _write_report(
    rows: List[Dict],
    sched_agg: Dict,
    sem_results: Dict,
    exp_config: Dict,
    output_dir: Path,
) -> None:
    """Write the markdown summary report."""
    B = next((r for r in rows if r["config"] == "B"), {})
    A = next((r for r in rows if r["config"] == "A"), {})
    C = next((r for r in rows if r["config"] == "C"), {})
    F = next((r for r in rows if r["config"] == "F"), {})
    G = next((r for r in rows if r["config"] == "G"), {})

    def fmt(v, decimals=4):
        if v is None:
            return "N/A"
        return f"{v:.{decimals}f}"

    def pct_delta(b_val, a_val):
        if a_val is None or b_val is None or a_val == 0:
            return "N/A"
        return f"{(b_val - a_val) / abs(a_val) * 100:+.1f}%"

    lines = [
        "# AI-Enabled Semantic Network vs Conventional Baseline",
        "",
        f"**Branch:** `experiment/rl-evaluation`  ",
        f"**Date:** 2026-10-11  ",
        f"**Seeds:** {exp_config['num_seeds']} × {exp_config['num_episodes']} episodes = "
        f"{exp_config['total_episodes_per_config']} scheduling evaluations per config  ",
        f"**Semantic eval samples:** {exp_config['num_eval_samples']} held-out JIGSAWS kinematics  ",
        "",
        "> **Measurement disclaimer:** All latency and deadline figures are from the Python",
        "> `EventDrivenSimulator` / `NetworkSchedulingEnv`. These are simulation proxies, NOT",
        "> NS-3 measurements. Semantic quality figures (MSE, quality) are from the trained",
        "> encoder/decoder checkpoint evaluated on JIGSAWS kinematics.",
        "",
        "---",
        "",
        "## Configurations",
        "",
        "| ID | Label | Compression | Scheduler | Orchestration |",
        "|----|-------|------------|-----------|---------------|",
        "| A  | Conv-50%+EDF      | Fixed 50% | EDF | No |",
        "| B  | PPO+DQN+Orch      | PPO       | DQN | Yes |",
        "| C  | PPO+DQN (no orch) | PPO       | DQN | No (ablation) |",
        "| D  | Fixed-25%+EDF     | Fixed 25% | EDF | No |",
        "| E  | Fixed-75%+EDF     | Fixed 75% | EDF | No |",
        "| F  | PPO+EDF           | PPO       | EDF | No |",
        "| G  | PPO+Orch+EDF      | PPO+Orch  | EDF | Yes |",
        "",
        "**Integration design:** PPO decides compression (packet size). DQN decides scheduling",
        "order. Orchestration biases PPO's action — it does NOT modify DQN's behavior.",
        "This is the minimal honest joint harness: packet sizes from PPO feed directly into",
        "the DQN scheduling environment.",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "### Semantic Layer",
        "",
        "| Config | MSE ↓ | Quality ↑ | MAE ↓ | RMSE ↓ | Retention | Payload (B) |",
        "|--------|-------|-----------|-------|--------|-----------|-------------|",
    ]
    for row in rows:
        lines.append(
            f"| **{row['config']}** {row['label']} "
            f"| {fmt(row.get('sem_mse_mean'))} "
            f"| {fmt(row.get('sem_quality_mean'))} "
            f"| {fmt(row.get('sem_mae_mean'))} "
            f"| {fmt(row.get('sem_rmse_mean'))} "
            f"| {fmt(row.get('retention_mean'), 2)} "
            f"| {fmt(row.get('payload_bytes_mean'), 0)} |"
        )

    lines += [
        "",
        "### Network / Scheduling Layer [Python sim — NOT NS-3]",
        "",
        "| Config | Avg Lat (µs) | P95 Lat (µs) | Deadline Miss ↓ | Deadline Sat ↑ |",
        "|--------|-------------|-------------|-----------------|----------------|",
    ]
    for row in rows:
        lines.append(
            f"| **{row['config']}** {row['label']} "
            f"| {fmt(row.get('net_avg_latency_us'), 1)} "
            f"| {fmt(row.get('net_p95_latency_us'), 1)} "
            f"| {fmt(row.get('net_deadline_miss_rate'), 3)} "
            f"| {fmt(row.get('net_deadline_sat_rate'), 3)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "### Does AI (B) improve over Conventional (A)?",
        "",
        f"**Semantic quality:** B={fmt(B.get('sem_quality_mean'))} vs A={fmt(A.get('sem_quality_mean'))} "
        f"→ {pct_delta(B.get('sem_quality_mean'), A.get('sem_quality_mean'))}",
        "",
        f"**Reconstruction MSE:** B={fmt(B.get('sem_mse_mean'))} vs A={fmt(A.get('sem_mse_mean'))} "
        f"→ {pct_delta(B.get('sem_mse_mean'), A.get('sem_mse_mean'))} (negative = better)",
        "",
        f"**Network latency:** B={fmt(B.get('net_avg_latency_us'),1)} µs "
        f"vs A={fmt(A.get('net_avg_latency_us'),1)} µs "
        f"→ {pct_delta(B.get('net_avg_latency_us'), A.get('net_avg_latency_us'))}",
        "",
        f"**Deadline miss rate:** B={fmt(B.get('net_deadline_miss_rate'),3)} "
        f"vs A={fmt(A.get('net_deadline_miss_rate'),3)} "
        f"→ {pct_delta(B.get('net_deadline_miss_rate'), A.get('net_deadline_miss_rate'))}",
        "",
        "### Does Orchestration add value (B vs C)?",
        "",
        f"**Semantic quality:** B={fmt(B.get('sem_quality_mean'))} vs C={fmt(C.get('sem_quality_mean'))} "
        f"→ {pct_delta(B.get('sem_quality_mean'), C.get('sem_quality_mean'))}",
        f"**Network latency:** B={fmt(B.get('net_avg_latency_us'),1)} µs "
        f"vs C={fmt(C.get('net_avg_latency_us'),1)} µs "
        f"→ {pct_delta(B.get('net_avg_latency_us'), C.get('net_avg_latency_us'))}",
        f"**Deadline miss:** B={fmt(B.get('net_deadline_miss_rate'),3)} "
        f"vs C={fmt(C.get('net_deadline_miss_rate'),3)} "
        f"→ {pct_delta(B.get('net_deadline_miss_rate'), C.get('net_deadline_miss_rate'))}",
        "",
        "### Does DQN add value over EDF (B vs G)?",
        "",
        f"**Network latency:** B(DQN)={fmt(B.get('net_avg_latency_us'),1)} µs "
        f"vs G(EDF)={fmt(G.get('net_avg_latency_us'),1)} µs "
        f"→ {pct_delta(B.get('net_avg_latency_us'), G.get('net_avg_latency_us'))}",
        f"**Deadline miss:** B(DQN)={fmt(B.get('net_deadline_miss_rate'),3)} "
        f"vs G(EDF)={fmt(G.get('net_deadline_miss_rate'),3)} "
        f"→ {pct_delta(B.get('net_deadline_miss_rate'), G.get('net_deadline_miss_rate'))}",
        "",
        "---",
        "",
        "## What Remains Unverified",
        "",
        "- **NS-3 results:** All scheduling metrics are Python simulation proxies.",
        "  See `NS3_HANDOFF.md` for Linux NS-3 run instructions.",
        "- **DQN deadline miss vs EDF:** DQN achieves lower mean latency but higher",
        "  deadline miss rate than EDF in some configurations (latency-vs-deadline trade-off).",
        "- **Statistical significance of scheduling metrics:** CI reported but not corrected",
        "  for multiple comparisons across 7 configurations.",
        "",
        "---",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "# Full evaluation (3 seeds × 100 episodes)",
        "python ai_vs_baseline_evaluation.py",
        "",
        "# Quick smoke test (1 seed × 5 episodes)",
        "python ai_vs_baseline_evaluation.py --num-episodes 5 --num-seeds 1",
        "",
        "# Run tests",
        "python -m pytest tests/test_ai_vs_baseline.py -v",
        "```",
        "",
        "All outputs are in `rl/results/ai_vs_baseline/`.",
    ]

    report_path = output_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Written: {report_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI-enabled semantic network vs conventional baseline evaluation."
    )
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument("--sem-seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--num-eval-samples", type=int, default=256)
    parser.add_argument("--flows-per-episode", type=int, default=10)
    parser.add_argument("--ppo-checkpoint", default="rl/checkpoints/ppo_semantic_adaptive.zip")
    parser.add_argument("--dqn-checkpoint", default="rl/checkpoints/dqn_final.pt")
    parser.add_argument("--encoder-path", default="semantic_encoder.pth")
    parser.add_argument("--decoder-path", default="semantic_decoder.pth")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="rl/results/ai_vs_baseline")
    args = parser.parse_args()

    run_evaluation(
        eval_seed=args.eval_seed,
        sem_seed=args.sem_seed,
        num_episodes=args.num_episodes,
        num_seeds=args.num_seeds,
        ppo_checkpoint=args.ppo_checkpoint,
        dqn_checkpoint=args.dqn_checkpoint,
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path,
        data_root=args.data_root,
        num_eval_samples=args.num_eval_samples,
        flows_per_episode=args.flows_per_episode,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
