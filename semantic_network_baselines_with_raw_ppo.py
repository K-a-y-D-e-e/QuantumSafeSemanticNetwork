"""
Controlled semantic-retention + DQN network baseline evaluation.

Compares fixed semantic retention (25%, 50%, 75%, 100%), raw PPO action
selection, and PPO + rule-based orchestration. Each condition uses identical
traffic workloads and the same DQN scheduler within each episode.

The packet-size model is an explicit assumption:
  bytes = sequence_length * retained_dims * 4 + retained_dims + header_bytes
No PQC overhead is included. Uses the project's Python network simulator,
not NS-3.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from orchestration.agent import AIOrchestrationAgent
from rl.agent import RLAgent
from rl.dqn_agent import DQNAgent
from rl.environment import MAX_QUEUE, OBS_DIM, NetworkSchedulingEnv, SemanticCompressionEnv
from rl.traffic_generator import TrafficGenerator
from semantic.compression import AdaptiveSemanticCompressor
from semantic.data_loader import JIGSAWSKinematicsDataset
from semantic.model_io import DEFAULT_LATENT_DIM, load_semantic_models
from network.simulation.link import NetworkLink
from evaluate_semantic_rl import build_held_out_split, build_deterministic_state


class FixedTrafficGenerator:
    def __init__(self, packets):
        self.packets = packets

    def generate(self):
        return [dict(p) for p in self.packets]


def packet_size_bytes(sequence_length, retained_dims, header_bytes):
    return int(sequence_length * retained_dims * 4 + retained_dims + header_bytes)


def env_kwargs(config):
    reward = config.get("reward", {})
    norm = config.get("normalisation", {})
    return {
        "w_deadline": reward.get("w_deadline", 2.0),
        "w_latency": reward.get("w_latency", 1.0),
        "w_queue": reward.get("w_queue", 0.5),
        "w_miss": reward.get("w_miss", 3.0),
        "mask_violation_penalty": reward.get("mask_violation_penalty", -5.0),
        "max_deadline_us": norm.get("max_deadline_us", 20000.0),
        "max_size_bytes": norm.get("max_size_bytes", 2000.0),
        "max_wait_us": norm.get("max_wait_us", 10000.0),
        "max_expected_latency_us": norm.get("max_expected_latency_us", 15000.0),
        "episode_max_time_us": norm.get("episode_max_time_us", 50000.0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="configs/rl_config.yaml")
    parser.add_argument("--ppo-checkpoint", default="rl/checkpoints/ppo_semantic_adaptive.zip")
    parser.add_argument("--dqn-checkpoint", default="rl/checkpoints/dqn_best.pt")
    parser.add_argument("--encoder-path", default="semantic_encoder.pth")
    parser.add_argument("--decoder-path", default="semantic_decoder.pth")
    parser.add_argument("--seed", type=int, default=42, help="Semantic sample/state seed")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--header-bytes", type=int, default=32)
    parser.add_argument("--output-prefix", default="rl/results/semantic_network_baselines")
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if args.header_bytes < 0:
        raise ValueError("--header-bytes cannot be negative")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    traffic_cfg = config.get("traffic", {})
    network_cfg = config.get("network", {})
    eval_seed = int(config.get("eval_seed", 999))
    n_flows = int(traffic_cfg.get("num_flows", 10))
    sequence_length = 32
    device = torch.device("cpu")

    dataset = JIGSAWSKinematicsDataset(
        root_dir=args.data_root, sequence_length=sequence_length,
        stride=16, normalize=True
    )
    samples, _, _ = build_held_out_split(
        dataset, seed=args.seed, val_fraction=0.2,
        max_eval_samples=max(args.episodes * n_flows, 256)
    )
    if len(samples) == 0:
        raise RuntimeError("No held-out kinematic samples were loaded.")

    encoder, decoder = load_semantic_models(
        encoder_path=args.encoder_path, decoder_path=args.decoder_path, device=device
    )
    compressor = AdaptiveSemanticCompressor(latent_dim=DEFAULT_LATENT_DIM)
    orchestrator = AIOrchestrationAgent()
    sem_env = SemanticCompressionEnv(
        encoder=encoder, decoder=decoder, compressor=compressor,
        kinematics_samples=samples, orchestrator=orchestrator,
        device=device, use_reconstruction_reward=True
    )

    ppo = RLAgent(sem_env, seed=args.seed, verbose=0)
    ppo_path = Path(args.ppo_checkpoint)
    if ppo_path.suffix == ".zip":
        ppo_path = ppo_path.with_suffix("")
    if not Path(str(ppo_path) + ".zip").exists() and not ppo_path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {args.ppo_checkpoint}")
    ppo.load(str(ppo_path), environment=sem_env)

    dqn = DQNAgent(obs_dim=OBS_DIM, action_dim=MAX_QUEUE, seed=eval_seed)
    dqn.load(args.dqn_checkpoint)
    dqn.epsilon = 0.0

    link = NetworkLink(
        bandwidth_mbps=network_cfg.get("bandwidth_mbps", 10),
        propagation_delay_us=network_cfg.get("propagation_delay_us", 100),
    )
    traffic_gen = TrafficGenerator(
        seed=eval_seed,
        num_flows=n_flows,
        min_size_bytes=traffic_cfg.get("min_size_bytes", 500),
        max_size_bytes=traffic_cfg.get("max_size_bytes", 2000),
        min_deadline_ms=traffic_cfg.get("min_deadline_ms", 2),
        max_deadline_ms=traffic_cfg.get("max_deadline_ms", 15),
        max_arrival_spread_us=traffic_cfg.get("max_arrival_spread_us", 1000),
    )
    fixed_conditions = [
        ("Fixed_25pct", 0), ("Fixed_50pct", 1),
        ("Fixed_75pct", 2), ("Fixed_100pct", 3),
    ]
    conditions = fixed_conditions + [
        ("PPO_Raw", "ppo_raw"),
        ("PPO_Orchestrated", "ppo_orchestrated"),
    ]
    all_rows = []

    print("=" * 78)
    print("SEMANTIC RETENTION BASELINE EVALUATION")
    print(f"Episodes={args.episodes}; flows/episode={n_flows}; eval_seed={eval_seed}")
    print("Every condition reuses the same generated traffic workload per episode.")
    print("=" * 78)

    for ep in range(args.episodes):
        base_packets = traffic_gen.generate()
        # Freeze one semantic context and sample assignment per packet so all
        # retention conditions see the same kinematic sequence and context.
        packet_contexts = []
        for i, packet in enumerate(base_packets):
            sample_index = (ep * n_flows + i) % len(samples)
            state = build_deterministic_state(sample_index, args.seed + ep)
            objective = orchestrator.decide_objective(
                float(state[0]), float(state[2]), float(state[3]), float(state[5])
            )
            packet_contexts.append((sample_index, state, objective))

        for condition, fixed_action in conditions:
            packets = [dict(p) for p in base_packets]
            semantic_records = []
            for i, packet in enumerate(packets):
                sample_index, state, objective = packet_contexts[i]
                obs, _ = sem_env.reset(
                    seed=args.seed + ep * n_flows + i,
                    options={
                        "state_vector": state,
                        "sample_index": sample_index,
                        "objective": objective,
                    },
                )
                if fixed_action in ("ppo_raw", "ppo_orchestrated"):
                    raw = ppo.predict(obs)
                    action = int(np.asarray(raw).reshape(-1)[0])
                    if fixed_action == "ppo_orchestrated":
                        action = orchestrator.bias_action_for_objective(action, objective)
                else:
                    action = fixed_action
                _, _, _, _, info = sem_env.step(action)
                retained_dims = int(info["retained_latent_dims"])
                size_bytes = packet_size_bytes(
                    sequence_length, retained_dims, args.header_bytes
                )
                packet["size_bytes"] = size_bytes
                packet["payload"] = b"X" * size_bytes
                semantic_records.append({
                    "episode": ep + 1,
                    "condition": condition,
                    "flow_id": packet["flow_id"],
                    "objective": str(objective),
                    "ppo_action_or_fixed_action": action,
                    "retained_latent_dims": retained_dims,
                    "retained_latent_fraction": float(info["retained_latent_fraction"]),
                    "reconstruction_mse": float(info["reconstruction_mse"]),
                    "semantic_quality": float(info["semantic_quality"]),
                    "modeled_packet_bytes": size_bytes,
                })

            net_env = NetworkSchedulingEnv(
                traffic_generator=FixedTrafficGenerator(packets),
                link=link,
                **env_kwargs(config)
            )
            obs, _ = net_env.reset(seed=eval_seed + ep)
            done = False
            while not done:
                mask = net_env.get_action_mask()
                action = dqn.select_action(obs, mask, training=False)
                obs, _, terminated, truncated, _ = net_env.step(action)
                done = terminated or truncated

            net_by_id = {r["flow_id"]: r for r in net_env.get_episode_results()}
            for sem in semantic_records:
                net = net_by_id.get(sem["flow_id"], {})
                arrival = float(net.get("arrival_time_us", 0.0))
                start = float(net.get("start_time_us", arrival))
                sem.update({
                    "arrival_time_us": arrival,
                    "start_time_us": start,
                    "completion_time_us": net.get("completion_time_us"),
                    "network_latency_us": net.get("latency_us"),
                    "queueing_delay_us": max(0.0, start - arrival),
                    "deadline_us": net.get("deadline_us"),
                    "deadline_met": net.get("deadline_met"),
                })
                all_rows.append(sem)

        if (ep + 1) % max(1, args.episodes // 10) == 0:
            print(f"Episode {ep + 1}/{args.episodes} complete")

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    packet_path = Path(str(out_prefix) + "_packets.csv")
    summary_path = Path(str(out_prefix) + "_summary.csv")
    fields = list(all_rows[0].keys())
    with packet_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    summaries = []
    for condition, _ in conditions:
        rows = [r for r in all_rows if r["condition"] == condition]
        n = len(rows)
        summaries.append({
            "condition": condition,
            "packets": n,
            "mean_retained_fraction": round(float(np.mean([r["retained_latent_fraction"] for r in rows])), 4),
            "mean_reconstruction_mse": round(float(np.mean([r["reconstruction_mse"] for r in rows])), 6),
            "mean_semantic_quality": round(float(np.mean([r["semantic_quality"] for r in rows])), 4),
            "mean_packet_bytes": round(float(np.mean([r["modeled_packet_bytes"] for r in rows])), 2),
            "mean_latency_us": round(float(np.mean([r["network_latency_us"] for r in rows])), 2),
            "mean_queueing_us": round(float(np.mean([r["queueing_delay_us"] for r in rows])), 2),
            "deadline_miss_rate": round(float(np.mean([not bool(r["deadline_met"]) for r in rows])), 4),
            "deadline_misses": int(sum(not bool(r["deadline_met"]) for r in rows)),
        })

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("\nSummary (all packets pooled across episodes)")
    print(f"{'Condition':20s} {'MSE':>10s} {'Quality':>10s} {'Bytes':>10s} {'Latency ms':>12s} {'Queue ms':>10s} {'Miss rate':>10s}")
    for r in summaries:
        print(f"{r['condition']:20s} {r['mean_reconstruction_mse']:10.4f} {r['mean_semantic_quality']:10.4f} {r['mean_packet_bytes']:10.1f} {r['mean_latency_us']/1000:12.3f} {r['mean_queueing_us']/1000:10.3f} {r['deadline_miss_rate']:10.3f}")
    print(f"\nPacket-level CSV: {packet_path}")
    print(f"Summary CSV:      {summary_path}")
    print("Note: modeled bytes exclude cryptographic overhead; simulator is Python, not NS-3.")


if __name__ == "__main__":
    main()
