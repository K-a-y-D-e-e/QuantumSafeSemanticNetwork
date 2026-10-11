"""
Scheduler comparison evaluation script.

Compares Random, Priority, EDF, DQN, and DQN-hybrid scheduling policies
on identical workloads per episode.

Outputs:
    rl/results/comparison_results.csv
    rl/results/comparison_results.json

Run from the project root:
    python rl/evaluate.py --model rl/checkpoints/dqn_best.pt --episodes 100
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Imports
from network.simulation.link import NetworkLink  # noqa: E402
from network.simulation.event_simulator import EventDrivenSimulator  # noqa: E402
from network.simulation.scheduler import PriorityScheduler  # noqa: E402
from network.simulation.deadline_scheduler import DeadlineAwareScheduler  # noqa: E402
from network.simulation.analyzer import NetworkAnalyzer  # noqa: E402
from rl.traffic_generator import TrafficGenerator  # noqa: E402
from rl.random_scheduler import RandomScheduler  # noqa: E402
from rl.environment import NetworkSchedulingEnv, OBS_DIM, MAX_QUEUE  # noqa: E402
from rl.dqn_agent import DQNAgent  # noqa: E402

DEFAULT_CONFIG = "configs/rl_config.yaml"
DEFAULT_MODEL = "rl/checkpoints/dqn_best.pt"


def load_config(path: str) -> dict:
    """Load the YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_fixed_scheduler(
    scheduler,
    packets: List[Dict[str, Any]],
    link: NetworkLink,
) -> List[Dict[str, Any]]:
    """Run a fixed scheduler through the event-driven simulator."""
    sim = EventDrivenSimulator(link=link, scheduler=scheduler)
    return sim.simulate(list(packets))


class _FixedTrafficGenerator:
    """Return the same packet workload each time the environment requests it."""

    def __init__(self, packets: List[Dict[str, Any]]):
        self._packets = packets

    def generate(self) -> List[Dict[str, Any]]:
        return list(self._packets)


def run_dqn_scheduler(
    agent: DQNAgent,
    packets: List[Dict[str, Any]],
    link: NetworkLink,
    env_kwargs: Dict[str, Any],
    hybrid: bool = False,
    safety_margin_us: float = 1_000.0,
) -> List[Dict[str, Any]]:
    """
    Run greedy DQN, optionally overriding its choice when a packet is near
    its deadline. The hybrid uses least remaining slack as its fallback rule.
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
        mask = env.get_action_mask()
        action = agent.select_action(obs, mask, training=False)

        if hybrid and env._waiting_queue:
            now_us = env._current_time_us
            queue = env._waiting_queue[:MAX_QUEUE]
            urgency = []

            for idx, packet in enumerate(queue):
                tx_us = link.transmission_delay_us(packet["size_bytes"])
                prop_us = link.propagation_delay_us

                elapsed_us = now_us - packet["arrival_time_us"]
                remaining_deadline_us = (
                    packet["deadline_ms"] * 1_000
                    - elapsed_us
                    - tx_us
                    - prop_us
                )
                urgency.append((remaining_deadline_us, idx))

            if urgency:
                tightest_slack_us, least_slack_action = min(urgency)
                if tightest_slack_us <= safety_margin_us:
                    action = least_slack_action

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    return env.get_episode_results()


def _summarise(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute per-episode summary metrics."""
    analysis = NetworkAnalyzer.analyze(results)
    n = analysis.get("total_flows", 0)
    misses = analysis.get("deadline_misses", 0)

    return {
        "avg_latency_us": round(analysis.get("average_latency_us", 0.0), 2),
        "min_latency_us": round(analysis.get("minimum_latency_us", 0.0), 2),
        "max_latency_us": round(analysis.get("maximum_latency_us", 0.0), 2),
        "avg_queueing_us": round(
            analysis.get("average_queueing_delay_us", 0.0), 2
        ),
        "jitter_us": round(analysis.get("jitter_us", 0.0), 2),
        "deadline_misses": misses,
        "deadline_miss_rate": round(misses / n if n > 0 else 0.0, 4),
        "completed_flows": n,
    }


def evaluate(
    model_path: str,
    config: dict,
    num_episodes: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Compare baselines, DQN, and four hybrid safety margins."""

    eval_seed = config.get(
        "eval_seed", config.get("eval", {}).get("random_seed", 999)
    )
    traffic_cfg = config.get("traffic", {})
    net_cfg = config.get("network", {})
    reward_cfg = config.get("reward", {})
    norm_cfg = config.get("normalisation", {})

    link = NetworkLink(
        bandwidth_mbps=net_cfg.get("bandwidth_mbps", 10),
        propagation_delay_us=net_cfg.get("propagation_delay_us", 100),
    )

    eval_gen = TrafficGenerator(
        seed=eval_seed,
        num_flows=traffic_cfg.get("num_flows", 10),
        min_size_bytes=traffic_cfg.get("min_size_bytes", 500),
        max_size_bytes=traffic_cfg.get("max_size_bytes", 2000),
        min_deadline_ms=traffic_cfg.get("min_deadline_ms", 2),
        max_deadline_ms=traffic_cfg.get("max_deadline_ms", 15),
        max_arrival_spread_us=traffic_cfg.get(
            "max_arrival_spread_us", 1000
        ),
    )

    agent = DQNAgent(
        obs_dim=OBS_DIM,
        action_dim=MAX_QUEUE,
        seed=eval_seed,
    )
    agent.load(model_path)
    agent.epsilon = 0.0

    env_kwargs = dict(
        w_deadline=reward_cfg.get("w_deadline", 2.0),
        w_latency=reward_cfg.get("w_latency", 1.0),
        w_queue=reward_cfg.get("w_queue", 0.5),
        w_miss=reward_cfg.get("w_miss", 3.0),
        mask_violation_penalty=reward_cfg.get(
            "mask_violation_penalty", -5.0
        ),
        max_deadline_us=norm_cfg.get("max_deadline_us", 20_000.0),
        max_size_bytes=norm_cfg.get("max_size_bytes", 2_000.0),
        max_wait_us=norm_cfg.get("max_wait_us", 10_000.0),
        max_expected_latency_us=norm_cfg.get(
            "max_expected_latency_us", 15_000.0
        ),
        episode_max_time_us=norm_cfg.get(
            "episode_max_time_us", 50_000.0
        ),
    )

    schedulers = {
        "Random": RandomScheduler(seed=eval_seed),
        "Priority": PriorityScheduler(),
        "EDF": DeadlineAwareScheduler(),
    }

    hybrid_margins = [500, 1000, 1500, 2000]
    scheduler_names = (
        list(schedulers.keys())
        + ["DQN"]
        + [f"HYBRID_{margin}" for margin in hybrid_margins]
    )
    agg: Dict[str, list] = {name: [] for name in scheduler_names}

    print("=" * 90)
    print(
        "SCHEDULER COMPARISON EVALUATION\n"
        f"  eval_seed={eval_seed}  episodes={num_episodes}\n"
        f"  model={model_path}\n"
        f"  hybrid safety margins (us): {hybrid_margins}"
    )
    print("=" * 90)
    print(
        "All schedulers receive identical workloads per episode.\n"
        "DQN evaluation uses epsilon=0 (fully greedy).\n"
    )

    for ep in range(num_episodes):
        packets = eval_gen.generate()

        for name, scheduler in schedulers.items():
            results = run_fixed_scheduler(scheduler, list(packets), link)
            agg[name].append(_summarise(results))

        results_dqn = run_dqn_scheduler(
            agent, list(packets), link, env_kwargs
        )
        agg["DQN"].append(_summarise(results_dqn))

        for margin_us in hybrid_margins:
            results_hybrid = run_dqn_scheduler(
                agent,
                list(packets),
                link,
                env_kwargs,
                hybrid=True,
                safety_margin_us=float(margin_us),
            )
            agg[f"HYBRID_{margin_us}"].append(
                _summarise(results_hybrid)
            )

        if (ep + 1) % max(1, num_episodes // 10) == 0:
            print(f"  Episode {ep + 1}/{num_episodes} complete")

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
    for name in scheduler_names:
        episodes = agg[name]
        if not episodes:
            raise RuntimeError(f"No evaluation results collected for {name}")
        final[name] = {
            key: round(
                sum(episode[key] for episode in episodes) / len(episodes), 4
            )
            for key in metric_keys
        }

    labels = {
        "avg_latency_us": "Avg latency (us)",
        "min_latency_us": "Min latency (us)",
        "max_latency_us": "Max latency (us)",
        "avg_queueing_us": "Avg queueing (us)",
        "jitter_us": "Jitter (us)",
        "deadline_misses": "Deadline misses",
        "deadline_miss_rate": "Deadline miss rate",
        "completed_flows": "Completed flows",
    }

    name_width = max(12, max(len(name) for name in scheduler_names) + 2)
    print(
        "\n"
        + f"{'Metric':26s}"
        + "".join(f"{name:>{name_width}s}" for name in scheduler_names)
    )
    print("-" * (26 + name_width * len(scheduler_names)))

    for key in metric_keys:
        row = f"{labels[key]:26s}"
        for name in scheduler_names:
            row += f"{final[name][key]:{name_width}.4f}"
        print(row)

    results_dir = Path(config.get("results_dir", "rl/results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "comparison_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scheduler"] + metric_keys)
        for name in scheduler_names:
            writer.writerow(
                [name] + [final[name][key] for key in metric_keys]
            )
    print(f"\nCSV  -> {csv_path}")

    json_path = results_dir / "comparison_results.json"
    payload = {
        "eval_seed": eval_seed,
        "num_episodes": num_episodes,
        "model_path": model_path,
        "hybrid_safety_margins_us": hybrid_margins,
        "results": final,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON -> {json_path}")

    print("\n" + "=" * 90)
    print("EVALUATION COMPLETE")
    print("=" * 90)

    return final


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Random / Priority / EDF / DQN and hybrid policies "
            "on identical workloads."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Path to trained DQN checkpoint (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to RL config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes (overrides config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    num_episodes = args.episodes or config.get("eval", {}).get(
        "num_episodes", 100
    )
    evaluate(args.model, config, num_episodes=num_episodes)


if __name__ == "__main__":
    main()
