"""
DQN training script for the NetworkSchedulingEnv.

This script has two modes:

1. Smoke test (--smoke flag)
   ─────────────────────────
   Runs exactly 10 episodes to verify the training loop executes without
   error.  No model is saved.  This is NOT a meaningful trained-model
   experiment — it only validates pipeline correctness.

   Usage:
       python rl/train.py --smoke

2. Full training run
   ──────────────────
   Runs the number of episodes defined in configs/rl_config.yaml (default 500).
   Saves:
       rl/logs/training_log.csv          — per-episode metrics
       rl/checkpoints/dqn_ep<N>.pt      — periodic checkpoints
       rl/checkpoints/dqn_best.pt        — best episode reward
       rl/checkpoints/dqn_final.pt       — final model

   Usage:
       python rl/train.py
       python rl/train.py --config configs/rl_config.yaml

Training Log Columns
--------------------
   episode, reward, avg_latency_us, deadline_misses,
   deadline_miss_rate, epsilon

The training log can be used to generate training curves:
   - reward vs episode
   - avg_latency_us vs episode
   - deadline_misses vs episode
   - deadline_miss_rate vs episode

Run from the project root:
    python rl/train.py --smoke
    python rl/train.py
"""

import argparse
import csv
import sys
from pathlib import Path

import yaml

# ── Path setup so this script runs from project root ─────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Imports after path setup ──────────────────────────────────────────────────
from network.simulation.link import NetworkLink          # noqa: E402
from rl.traffic_generator import TrafficGenerator        # noqa: E402
from rl.environment import NetworkSchedulingEnv, OBS_DIM, MAX_QUEUE  # noqa: E402
from rl.dqn_agent import DQNAgent                       # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
SMOKE_EPISODES: int = 10
DEFAULT_CONFIG: str = "configs/rl_config.yaml"


# =============================================================================
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
def run_training(config: dict, smoke: bool = False) -> tuple:
    """
    Execute the DQN training loop.

    Parameters
    ----------
    config : dict   — parsed rl_config.yaml
    smoke  : bool   — if True, runs SMOKE_EPISODES only (no saving)

    Returns
    -------
    agent    : DQNAgent
    log_rows : list of dict  — per-episode metrics
    """
    seed          = config.get("seed", 42)
    num_episodes  = SMOKE_EPISODES if smoke else config.get("num_episodes", 500)
    net_cfg       = config.get("network", {})
    traffic_cfg   = config.get("traffic", {})
    reward_cfg    = config.get("reward", {})
    norm_cfg      = config.get("normalisation", {})

    # ── Network link ──────────────────────────────────────────────────
    link = NetworkLink(
        bandwidth_mbps=net_cfg.get("bandwidth_mbps", 10),
        propagation_delay_us=net_cfg.get("propagation_delay_us", 100),
    )

    # ── Traffic generator (training seed) ─────────────────────────────
    gen = TrafficGenerator(
        seed=seed,
        num_flows=traffic_cfg.get("num_flows", 10),
        min_size_bytes=traffic_cfg.get("min_size_bytes", 500),
        max_size_bytes=traffic_cfg.get("max_size_bytes", 2000),
        min_deadline_ms=traffic_cfg.get("min_deadline_ms", 2),
        max_deadline_ms=traffic_cfg.get("max_deadline_ms", 15),
        max_arrival_spread_us=traffic_cfg.get("max_arrival_spread_us", 1000),
    )

    # ── Environment ───────────────────────────────────────────────────
    env = NetworkSchedulingEnv(
        traffic_generator=gen,
        link=link,
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

    # ── DQN Agent ─────────────────────────────────────────────────────
    agent = DQNAgent(
        obs_dim=OBS_DIM,
        action_dim=MAX_QUEUE,
        lr=config.get("learning_rate", 1e-3),
        gamma=config.get("gamma", 0.99),
        epsilon_start=config.get("epsilon_start", 1.0),
        epsilon_min=config.get("epsilon_min", 0.05),
        epsilon_decay=config.get("epsilon_decay", 0.995),
        buffer_size=config.get("buffer_size", 10_000),
        batch_size=config.get("batch_size", 64),
        target_update_freq=config.get("target_update_freq", 50),
        hidden=config.get("hidden_size", 128),
        seed=seed,
    )

    # ── Output directories ────────────────────────────────────────────
    checkpoint_dir    = Path(config.get("checkpoint_dir", "rl/checkpoints"))
    log_dir           = Path(config.get("log_dir", "rl/logs"))
    checkpoint_interval = config.get("checkpoint_interval", 100)

    if not smoke:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

    # -- Header -------------------------------------------------------
    print("=" * 70)
    if smoke:
        print(
            "DQN SMOKE TEST  (10 episodes -- pipeline validation only)\n"
            "NOTE: Smoke test results do NOT represent a trained model."
        )
    else:
        print(f"DQN FULL TRAINING  ({num_episodes} episodes)")
        print(f"  seed={seed}  lr={config.get('learning_rate',1e-3)}"
              f"  gamma={config.get('gamma',0.99)}"
              f"  eps_decay={config.get('epsilon_decay',0.995)}")
    print("=" * 70)

    log_rows: list = []
    best_reward: float = float("-inf")
    print_interval = max(1, num_episodes // 10)

    # -- Training loop -------------------------------------------------
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            mask   = env.get_action_mask()
            action = agent.select_action(obs, mask, training=True)

            next_obs, reward, terminated, truncated, _ = env.step(action)

            next_mask = env.get_action_mask()
            done = terminated or truncated

            agent.store(obs, action, reward, next_obs, done, next_mask)
            agent.learn()

            obs = next_obs
            episode_reward += reward

        agent.decay_epsilon()

        # -- Per-episode metrics ------------------------------------
        results = env.get_episode_results()
        n = len(results)
        avg_latency    = sum(r["latency_us"] for r in results) / n if n else 0.0
        miss_count     = sum(1 for r in results if not r["deadline_met"])
        miss_rate      = miss_count / n if n else 0.0

        row = {
            "episode":            episode,
            "reward":             round(episode_reward, 4),
            "avg_latency_us":     round(avg_latency, 2),
            "deadline_misses":    miss_count,
            "deadline_miss_rate": round(miss_rate, 4),
            "epsilon":            round(agent.epsilon, 4),
        }
        log_rows.append(row)

        # -- Console output -----------------------------------------
        if episode % print_interval == 0 or episode == 1:
            print(
                f"  Ep {episode:4d}/{num_episodes} | "
                f"Reward: {episode_reward:7.3f} | "
                f"Avg Lat: {avg_latency:8.1f} us | "
                f"Misses: {miss_count}/{n} ({miss_rate:.0%}) | "
                f"eps: {agent.epsilon:.3f}"
            )

        if not smoke:
            # Periodic checkpoint
            if episode % checkpoint_interval == 0:
                ckpt_path = checkpoint_dir / f"dqn_ep{episode}.pt"
                agent.save(str(ckpt_path))

            # Best-model checkpoint
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(str(checkpoint_dir / "dqn_best.pt"))

    # ── Save training log CSV ─────────────────────────────────────────
    if not smoke and log_rows:
        log_path = log_dir / "training_log.csv"
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"\nTraining log -> {log_path}")

    # -- Save final model ----------------------------------------------
    if not smoke:
        final_path = checkpoint_dir / "dqn_final.pt"
        agent.save(str(final_path))
        print(f"Final model  -> {final_path}")
        print(f"Best model   -> {checkpoint_dir / 'dqn_best.pt'}")

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE" if smoke else "TRAINING COMPLETE")
    print("=" * 70)

    return agent, log_rows


# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train DQN network scheduler.\n"
            "  --smoke     : 10-episode pipeline smoke test (no saved model)\n"
            "  --episodes  : override number of training episodes\n"
            "  (default)   : full training run per configs/rl_config.yaml"
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to RL config YAML (default: configs/rl_config.yaml)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run 10-episode smoke test only. "
            "Results do NOT represent a trained model."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes (overrides config num_episodes)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.episodes is not None and not args.smoke:
        config["num_episodes"] = args.episodes
    run_training(config, smoke=args.smoke)


# =============================================================================
if __name__ == "__main__":
    main()
