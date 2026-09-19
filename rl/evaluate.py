"""
Scheduler comparison evaluation script.

Compares four scheduling policies on IDENTICAL workloads:
    1. Random   — uniformly random packet selection (lower-bound baseline)
    2. Priority — strict priority (HIGH > MEDIUM > LOW)
    3. EDF      — earliest-deadline-first
    4. DQN      — trained deep Q-network (fully greedy, ε=0)

All four schedulers receive the **same** packet workloads generated with
``eval_seed`` (separate from the training seed).  This ensures the
evaluation is fair and that the DQN is tested on workloads it has not
seen during training.

Output
------
    rl/results/comparison_results.csv
    rl/results/comparison_results.json

Usage
-----
    # After full training:
    python rl/evaluate.py --model rl/checkpoints/dqn_best.pt

    # Specify number of evaluation episodes:
    python rl/evaluate.py --model rl/checkpoints/dqn_best.pt --episodes 100

    # Custom config:
    python rl/evaluate.py --model rl/checkpoints/dqn_best.pt \\
                          --config configs/rl_config.yaml

Run from the project root.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Imports ───────────────────────────────────────────────────────────────────
from network.simulation.link import NetworkLink                      # noqa: E402
from network.simulation.event_simulator import EventDrivenSimulator  # noqa: E402
from network.simulation.scheduler import PriorityScheduler           # noqa: E402
from network.simulation.deadline_scheduler import DeadlineAwareScheduler  # noqa: E402
from network.simulation.analyzer import NetworkAnalyzer              # noqa: E402
from rl.traffic_generator import TrafficGenerator                    # noqa: E402
from rl.random_scheduler import RandomScheduler                      # noqa: E402
from rl.environment import NetworkSchedulingEnv, OBS_DIM, MAX_QUEUE  # noqa: E402
from rl.dqn_agent import DQNAgent                                    # noqa: E402

DEFAULT_CONFIG = "configs/rl_config.yaml"
DEFAULT_MODEL  = "rl/checkpoints/dqn_best.pt"


# =============================================================================
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
def run_fixed_scheduler(
    scheduler,
    packets: List[Dict[str, Any]],
    link: NetworkLink,
) -> List[Dict[str, Any]]:
    """
    Run a fixed scheduler (Random / Priority / EDF) on a packet list
    via the existing EventDrivenSimulator.

    Parameters
    ----------
    scheduler : object with .schedule() method
    packets   : list of packet dicts (will not be mutated)
    link      : NetworkLink

    Returns
    -------
    list of result dicts (same format as EventDrivenSimulator output)
    """
    sim = EventDrivenSimulator(link=link, scheduler=scheduler)
    return sim.simulate(list(packets))


# =============================================================================
class _FixedTrafficGenerator:
    """
    A traffic generator that returns the same fixed packet list every time.
    Used so the DQN environment uses the same workload as the fixed schedulers.
    """

    def __init__(self, packets: List[Dict[str, Any]]):
        self._packets = packets

    def generate(self) -> List[Dict[str, Any]]:
        return list(self._packets)


# =============================================================================
def run_dqn_scheduler(
    agent: DQNAgent,
    packets: List[Dict[str, Any]],
    link: NetworkLink,
    env_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Run the DQN agent (fully greedy) on a fixed packet workload.

    Parameters
    ----------
    agent      : DQNAgent  with ε=0
    packets    : list of packet dicts
    link       : NetworkLink
    env_kwargs : dict of extra kwargs for NetworkSchedulingEnv

    Returns
    -------
    list of result dicts compatible with NetworkAnalyzer.analyze()
    """
    gen = _FixedTrafficGenerator(packets)
    env = NetworkSchedulingEnv(
        traffic_generator=gen,
        link=link,
        **env_kwargs,
    )

    obs, _ = env.reset()
    done = False

    while not done:
        mask   = env.get_action_mask()
        action = agent.select_action(obs, mask, training=False)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return env.get_episode_results()


# =============================================================================
def _summarise(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute per-episode summary metrics using NetworkAnalyzer."""
    analysis = NetworkAnalyzer.analyze(results)
    n      = analysis.get("total_flows", 0)
    misses = analysis.get("deadline_misses", 0)
    return {
        "avg_latency_us":     round(analysis.get("average_latency_us", 0.0), 2),
        "min_latency_us":     round(analysis.get("minimum_latency_us", 0.0), 2),
        "max_latency_us":     round(analysis.get("maximum_latency_us", 0.0), 2),
        "avg_queueing_us":    round(analysis.get("average_queueing_delay_us", 0.0), 2),
        "jitter_us":          round(analysis.get("jitter_us", 0.0), 2),
        "deadline_misses":    misses,
        "deadline_miss_rate": round(misses / n if n > 0 else 0.0, 4),
        "completed_flows":    n,
    }


# =============================================================================
def evaluate(
    model_path: str,
    config: dict,
    num_episodes: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Run the full comparison evaluation.

    Parameters
    ----------
    model_path   : str   — path to the trained DQN checkpoint
    config       : dict  — parsed rl_config.yaml
    num_episodes : int   — number of evaluation episodes

    Returns
    -------
    dict mapping scheduler name → averaged metric dict
    """
    eval_seed   = config.get("eval_seed", config.get("eval", {}).get("random_seed", 999))
    traffic_cfg = config.get("traffic", {})
    net_cfg     = config.get("network", {})
    reward_cfg  = config.get("reward", {})
    norm_cfg    = config.get("normalisation", {})

    # ── Link ──────────────────────────────────────────────────────────
    link = NetworkLink(
        bandwidth_mbps=net_cfg.get("bandwidth_mbps", 10),
        propagation_delay_us=net_cfg.get("propagation_delay_us", 100),
    )

    # ── Evaluation traffic generator (separate eval_seed) ─────────────
    eval_gen = TrafficGenerator(
        seed=eval_seed,
        num_flows=traffic_cfg.get("num_flows", 10),
        min_size_bytes=traffic_cfg.get("min_size_bytes", 500),
        max_size_bytes=traffic_cfg.get("max_size_bytes", 2000),
        min_deadline_ms=traffic_cfg.get("min_deadline_ms", 2),
        max_deadline_ms=traffic_cfg.get("max_deadline_ms", 15),
        max_arrival_spread_us=traffic_cfg.get("max_arrival_spread_us", 1000),
    )

    # ── DQN Agent (fully greedy) ───────────────────────────────────────
    agent = DQNAgent(
        obs_dim=OBS_DIM,
        action_dim=MAX_QUEUE,
        seed=eval_seed,
    )
    agent.load(model_path)
    agent.epsilon = 0.0   # fully greedy during evaluation

    # ── Env kwargs for DQN evaluation ─────────────────────────────────
    env_kwargs = dict(
        w_deadline=reward_cfg.get("w_deadline", 2.0),
        w_latency=reward_cfg.get("w_latency", 1.0),
        w_queue=reward_cfg.get("w_queue", 0.5),
        w_miss=reward_cfg.get("w_miss", 3.0),
        mask_violation_penalty=reward_cfg.get("mask_violation_penalty", -5.0),
        max_deadline_us=norm_cfg.get("max_deadline_us", 20_000.0),
        max_size_bytes=norm_cfg.get("max_size_bytes", 2_000.0),
        max_wait_us=norm_cfg.get("max_wait_us", 10_000.0),
        max_expected_latency_us=norm_cfg.get("max_expected_latency_us", 15_000.0),
        episode_max_time_us=norm_cfg.get("episode_max_time_us", 50_000.0),
    )

    # -- Fixed schedulers ----------------------------------------------
    # Random uses the eval_seed for reproducibility
    schedulers = {
        "Random":   RandomScheduler(seed=eval_seed),
        "Priority": PriorityScheduler(),
        "EDF":      DeadlineAwareScheduler(),
    }

    # Accumulators: {scheduler_name: [per-episode summary dicts]}
    agg: Dict[str, list] = {
        name: [] for name in list(schedulers.keys()) + ["DQN"]
    }

    # -- Evaluation header ----------------------------------------------
    print("=" * 70)
    print(
        f"SCHEDULER COMPARISON EVALUATION\n"
        f"  eval_seed={eval_seed}  episodes={num_episodes}\n"
        f"  model={model_path}"
    )
    print("=" * 70)
    print(
        "  All four schedulers receive identical workloads per episode.\n"
        "  The DQN is evaluated with eps=0 (fully greedy).\n"
    )

    for ep in range(num_episodes):
        # Generate ONE workload for this episode (all schedulers use it)
        packets = eval_gen.generate()

        # Fixed schedulers
        for name, sched in schedulers.items():
            results = run_fixed_scheduler(sched, list(packets), link)
            agg[name].append(_summarise(results))

        # DQN on the same packets
        results_dqn = run_dqn_scheduler(agent, list(packets), link, env_kwargs)
        agg["DQN"].append(_summarise(results_dqn))

        if (ep + 1) % max(1, num_episodes // 10) == 0:
            print(f"  Episode {ep + 1}/{num_episodes} complete")

    # -- Aggregate averages --------------------------------------------
    metric_keys = [
        "avg_latency_us",
        "min_latency_us",
        "max_latency_us",
        "avg_queueing_us",
        "jitter_us",
        "deadline_misses",
        "deadline_miss_rate",
        "completed_flows",
    ]

    final: Dict[str, Dict[str, float]] = {}
    for name in ["Random", "Priority", "EDF", "DQN"]:
        eps = agg[name]
        n   = len(eps)
        final[name] = {
            k: round(sum(e[k] for e in eps) / n, 4)
            for k in metric_keys
        }

    # -- Print comparison table ----------------------------------------
    print(f"\n{'Metric':32s} {'Random':>10s} {'Priority':>10s} {'EDF':>10s} {'DQN':>10s}")
    print("-" * 74)
    labels = {
        "avg_latency_us":     "Avg latency (us)",
        "min_latency_us":     "Min latency (us)",
        "max_latency_us":     "Max latency (us)",
        "avg_queueing_us":    "Avg queueing (us)",
        "jitter_us":          "Jitter (us)",
        "deadline_misses":    "Deadline misses",
        "deadline_miss_rate": "Deadline miss rate",
        "completed_flows":    "Completed flows",
    }
    for k in metric_keys:
        row = f"{labels[k]:32s}"
        for sched_name in ["Random", "Priority", "EDF", "DQN"]:
            row += f" {final[sched_name][k]:>10.4f}"
        print(row)

    # -- Save results --------------------------------------------------
    results_dir = Path(config.get("results_dir", "rl/results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = results_dir / "comparison_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scheduler"] + metric_keys)
        for sched_name in ["Random", "Priority", "EDF", "DQN"]:
            writer.writerow(
                [sched_name] + [final[sched_name][k] for k in metric_keys]
            )
    print(f"\nCSV  -> {csv_path}")

    # JSON
    json_path = results_dir / "comparison_results.json"
    payload = {
        "eval_seed":    eval_seed,
        "num_episodes": num_episodes,
        "model_path":   model_path,
        "results":      final,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON -> {json_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return final


# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Random / Priority / EDF / DQN on identical workloads."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to trained DQN checkpoint (default: rl/checkpoints/dqn_best.pt)",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to RL config YAML (default: configs/rl_config.yaml)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes (overrides config)",
    )
    args = parser.parse_args()

    config       = load_config(args.config)
    num_episodes = args.episodes or config.get("eval", {}).get("num_episodes", 100)

    evaluate(args.model, config, num_episodes=num_episodes)


# =============================================================================
if __name__ == "__main__":
    main()
