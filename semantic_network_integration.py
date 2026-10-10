"""
End-to-end semantic-retention + DQN scheduling integration experiment.

This is a separate evaluator. It does not modify the existing environments,
traffic generator, checkpoints, or baseline evaluator.

Packet-size model (explicit simulation assumption):
  bytes = sequence_length * retained_dimensions * 4
        + retained_dimensions * 1
        + header_bytes
The latent values are modeled as float32 (4 bytes each); each retained feature
index is modeled as 1 byte (latent_dim=16); header_bytes defaults to 32.
No cryptographic overhead is included. This is a Python simulator, not NS-3.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml

from orchestration.agent import AIOrchestrationAgent
from rl.agent import RLAgent
from rl.dqn_agent import DQNAgent
from rl.environment import (
    MAX_QUEUE,
    OBS_DIM,
    NetworkSchedulingEnv,
    SemanticCompressionEnv,
)
from rl.traffic_generator import TrafficGenerator
from semantic.compression import AdaptiveSemanticCompressor
from semantic.data_loader import JIGSAWSKinematicsDataset
from semantic.model_io import (
    DEFAULT_LATENT_DIM,
    load_semantic_models,
)
from network.simulation.link import NetworkLink
from evaluate_semantic_rl import build_held_out_split, build_deterministic_state


class FixedTrafficGenerator:
    """Return one prebuilt packet workload without changing the baseline generator."""
    def __init__(self, packets):
        self.packets = packets

    def generate(self):
        return [dict(packet) for packet in self.packets]


def packet_size_bytes(sequence_length, retained_dims, header_bytes=32):
    """Estimate encoded payload size under the documented float32/index model."""
    latent_value_bytes = sequence_length * retained_dims * 4
    index_bytes = retained_dims  # latent_dim=16 indices fit in one byte each
    return int(latent_value_bytes + index_bytes + header_bytes)


def make_env_kwargs(config):
    reward_cfg = config.get("reward", {})
    norm_cfg = config.get("normalisation", {})
    return dict(
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                        help="Path to the JIGSAWS dataset used by evaluate_semantic_rl.py")
    parser.add_argument("--config", default="configs/rl_config.yaml")
    parser.add_argument("--ppo-checkpoint", default="rl/checkpoints/ppo_semantic_adaptive.zip")
    parser.add_argument("--dqn-checkpoint", default="rl/checkpoints/dqn_best.pt")
    parser.add_argument("--encoder-path", default="semantic_encoder.pth")
    parser.add_argument("--decoder-path", default="semantic_decoder.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--packets", type=int, default=10)
    parser.add_argument("--header-bytes", type=int, default=32)
    parser.add_argument("--output", default="rl/results/semantic_network_integration.csv")
    args = parser.parse_args()

    if args.packets < 1:
        raise ValueError("--packets must be at least 1")
    if args.header_bytes < 0:
        raise ValueError("--header-bytes cannot be negative")

    device = torch.device("cpu")
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    dataset = JIGSAWSKinematicsDataset(
        root_dir=args.data_root,
        sequence_length=32,
        stride=16,
        normalize=True,
    )
    val_samples, _train_indices, val_indices = build_held_out_split(
        dataset,
        seed=args.seed,
        val_fraction=0.2,
        max_eval_samples=max(args.packets, 256),
    )
    if len(val_samples) < args.packets:
        raise RuntimeError(
            f"Only {len(val_samples)} held-out samples available for "
            f"{args.packets} packets."
        )

    encoder, decoder = load_semantic_models(
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path,
        device=device,
    )
    compressor = AdaptiveSemanticCompressor(latent_dim=DEFAULT_LATENT_DIM)
    orchestrator = AIOrchestrationAgent()
    semantic_env = SemanticCompressionEnv(
        encoder=encoder,
        decoder=decoder,
        compressor=compressor,
        kinematics_samples=val_samples,
        orchestrator=orchestrator,
        device=device,
        use_reconstruction_reward=True,
    )

    # RLAgent.load wraps PPO.load and attaches the correct environment.
    ppo = RLAgent(semantic_env, seed=args.seed, verbose=0)
    ppo_path = Path(args.ppo_checkpoint)
    if ppo_path.suffix == ".zip":
        ppo_path = ppo_path.with_suffix("")
    if not Path(str(ppo_path) + ".zip").exists() and not ppo_path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {args.ppo_checkpoint}")
    ppo.load(str(ppo_path), environment=semantic_env)

    traffic_cfg = config.get("traffic", {})
    net_cfg = config.get("network", {})
    eval_seed = int(config.get("eval_seed", config.get("eval", {}).get("random_seed", 999)))
    generator = TrafficGenerator(
        seed=args.seed,
        num_flows=args.packets,
        min_size_bytes=traffic_cfg.get("min_size_bytes", 500),
        max_size_bytes=traffic_cfg.get("max_size_bytes", 2000),
        min_deadline_ms=traffic_cfg.get("min_deadline_ms", 2),
        max_deadline_ms=traffic_cfg.get("max_deadline_ms", 15),
        max_arrival_spread_us=traffic_cfg.get("max_arrival_spread_us", 1000),
    )
    packets = generator.generate()

    semantic_rows = []
    sequence_length = 32
    for i, packet in enumerate(packets):
        sample_index = i % len(val_samples)
        state = build_deterministic_state(sample_index, args.seed)
        objective = orchestrator.decide_objective(
            float(state[0]), float(state[2]), float(state[3]), float(state[5])
        )
        obs, _ = semantic_env.reset(
            seed=args.seed + i,
            options={
                "state_vector": state,
                "sample_index": sample_index,
                "objective": objective,
            },
        )
        raw_action = ppo.predict(obs)
        action = int(np.asarray(raw_action).reshape(-1)[0])
        action = orchestrator.bias_action_for_objective(action, objective)
        _next_obs, _reward, _terminated, _truncated, info = semantic_env.step(action)

        retained_dims = int(info["retained_latent_dims"])
        encoded_size = packet_size_bytes(
            sequence_length, retained_dims, args.header_bytes
        )
        packet["size_bytes"] = encoded_size
        packet["payload"] = b"X" * encoded_size
        packet["semantic_action"] = action
        packet["retained_latent_dims"] = retained_dims
        packet["retained_latent_fraction"] = float(info["retained_latent_fraction"])
        packet["reconstruction_mse"] = float(info["reconstruction_mse"])
        packet["semantic_quality"] = float(info["semantic_quality"])
        packet["objective"] = str(objective)

        semantic_rows.append({
            "flow_id": packet["flow_id"],
            "objective": str(objective),
            "ppo_action": action,
            "retained_latent_dims": retained_dims,
            "retained_latent_fraction": float(info["retained_latent_fraction"]),
            "reconstruction_mse": float(info["reconstruction_mse"]),
            "semantic_quality": float(info["semantic_quality"]),
            "estimated_packet_bytes": encoded_size,
        })

    link = NetworkLink(
        bandwidth_mbps=net_cfg.get("bandwidth_mbps", 10),
        propagation_delay_us=net_cfg.get("propagation_delay_us", 100),
    )
    dqn = DQNAgent(obs_dim=OBS_DIM, action_dim=MAX_QUEUE, seed=eval_seed)
    dqn.load(args.dqn_checkpoint)
    dqn.epsilon = 0.0

    net_env = NetworkSchedulingEnv(
        traffic_generator=FixedTrafficGenerator(packets),
        link=link,
        **make_env_kwargs(config),
    )
    obs, _ = net_env.reset(seed=args.seed)
    done = False
    while not done:
        mask = net_env.get_action_mask()
        action = dqn.select_action(obs, mask, training=False)
        obs, _reward, terminated, truncated, _info = net_env.step(action)
        done = terminated or truncated

    net_results = net_env.get_episode_results()
    result_by_flow = {row["flow_id"]: row for row in net_results}
    semantic_by_flow = {row["flow_id"]: row for row in semantic_rows}

    combined = []
    for flow_id, sem in semantic_by_flow.items():
        net = result_by_flow.get(flow_id, {})
        combined.append({
            **sem,
            "arrival_time_us": net.get("arrival_time_us"),
            "start_time_us": net.get("start_time_us"),
            "completion_time_us": net.get("completion_time_us"),
            "network_latency_us": net.get("latency_us"),
            "queueing_delay_us": max(
                0.0, net.get("start_time_us", 0.0) - net.get("arrival_time_us", 0.0)
            ) if net else None,
            "deadline_us": net.get("deadline_us"),
            "deadline_met": net.get("deadline_met"),
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0].keys()))
        writer.writeheader()
        writer.writerows(combined)

    latencies = [r["network_latency_us"] for r in combined if r["network_latency_us"] is not None]
    queues = [r["queueing_delay_us"] for r in combined if r["queueing_delay_us"] is not None]
    misses = sum(r["deadline_met"] is False for r in combined)
    print("\nIntegrated semantic + DQN network evaluation")
    print(f"Packets completed: {len(net_results)}/{args.packets}")
    print(f"Mean retained latent fraction: {np.mean([r['retained_latent_fraction'] for r in combined]):.3f}")
    print(f"Mean reconstruction MSE: {np.mean([r['reconstruction_mse'] for r in combined]):.6f}")
    print(f"Mean semantic quality: {np.mean([r['semantic_quality'] for r in combined]):.4f}")
    print(f"Mean modeled packet size: {np.mean([r['estimated_packet_bytes'] for r in combined]):.1f} bytes")
    print(f"Mean network latency: {np.mean(latencies):.2f} us" if latencies else "Mean network latency: n/a")
    print(f"Mean queueing delay: {np.mean(queues):.2f} us" if queues else "Mean queueing delay: n/a")
    print(f"Deadline misses: {misses}/{len(combined)}")
    print(f"CSV: {output_path}")


if __name__ == "__main__":
    main()
