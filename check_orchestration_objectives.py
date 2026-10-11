"""
Compact diagnostic for the four rule-based orchestration objectives.

Tests objective selection with controlled contexts and compares raw PPO action
selection with the objective-biased action on the same held-out sample/state.
This is a diagnostic, not a replacement for full network evaluation.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from orchestration.agent import AIOrchestrationAgent
from rl.agent import RLAgent
from rl.environment import SemanticCompressionEnv
from semantic.compression import AdaptiveSemanticCompressor
from semantic.data_loader import JIGSAWSKinematicsDataset
from semantic.model_io import DEFAULT_LATENT_DIM, load_semantic_models
from evaluate_semantic_rl import build_held_out_split, build_deterministic_state


CONTEXTS = {
    "LATENCY_CRITICAL": {"task_criticality": 0.95, "network_load": 0.40, "latency": 0.95},
    "QUALITY_CRITICAL": {"task_criticality": 0.95, "network_load": 0.40, "latency": 0.20},
    "BANDWIDTH_EFFICIENT": {"task_criticality": 0.40, "network_load": 0.95, "latency": 0.30},
    "BALANCED": {"task_criticality": 0.40, "network_load": 0.30, "latency": 0.30},
}


def controlled_state(sample_index, seed, context):
    state = np.asarray(build_deterministic_state(sample_index, seed), dtype=np.float32).copy()
    # SemanticCompressionEnv state layout:
    # [criticality, bandwidth, network_load, latency, packet_loss, deadline, semantic_quality]
    state[0] = context["task_criticality"]
    state[2] = context["network_load"]
    state[3] = context["latency"]
    return state


def evaluate_action(env, state, sample_index, objective, seed, action):
    env.reset(
        seed=seed,
        options={
            "state_vector": state,
            "sample_index": sample_index,
            "objective": objective,
        },
    )
    _, _, _, _, info = env.step(action)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--ppo-checkpoint", default="rl/checkpoints/ppo_semantic_adaptive.zip")
    parser.add_argument("--encoder-path", default="semantic_encoder.pth")
    parser.add_argument("--decoder-path", default="semantic_decoder.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-objective", type=int, default=25)
    parser.add_argument("--output-prefix", default="rl/results/orchestration_objective_check")
    args = parser.parse_args()

    if args.samples_per_objective < 1:
        raise ValueError("--samples-per-objective must be at least 1")

    device = torch.device("cpu")
    dataset = JIGSAWSKinematicsDataset(
        root_dir=args.data_root, sequence_length=32, stride=16, normalize=True
    )
    samples, _, _ = build_held_out_split(
        dataset, seed=args.seed, val_fraction=0.2,
        max_eval_samples=max(args.samples_per_objective * len(CONTEXTS), 256)
    )
    if len(samples) == 0:
        raise RuntimeError("No held-out kinematic samples loaded.")

    encoder, decoder = load_semantic_models(
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path,
        device=device,
    )
    orchestrator = AIOrchestrationAgent()
    compressor = AdaptiveSemanticCompressor(latent_dim=DEFAULT_LATENT_DIM)
    env = SemanticCompressionEnv(
        encoder=encoder,
        decoder=decoder,
        compressor=compressor,
        kinematics_samples=samples,
        orchestrator=orchestrator,
        device=device,
        use_reconstruction_reward=True,
    )

    ppo = RLAgent(env, seed=args.seed, verbose=0)
    checkpoint = Path(args.ppo_checkpoint)
    if checkpoint.suffix.lower() == ".zip":
        checkpoint = checkpoint.with_suffix("")
    if not Path(str(checkpoint) + ".zip").exists() and not checkpoint.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {args.ppo_checkpoint}")
    ppo.load(str(checkpoint), environment=env)

    rows = []
    print("=" * 76)
    print("ORCHESTRATION OBJECTIVE DIAGNOSTIC")
    print(f"{args.samples_per_objective} held-out samples per objective")
    print("For each sample, raw PPO and biased PPO use the same state and sample.")
    print("=" * 76)

    for objective, context in CONTEXTS.items():
        # Check that the objective selector actually produces the intended label.
        selected = orchestrator.decide_objective(
            context["task_criticality"],
            context["network_load"],
            context["latency"],
            security_requirement=0.5,
        )
        print(f"{objective}: selector returns {selected}")

        for j in range(args.samples_per_objective):
            sample_index = (j + list(CONTEXTS).index(objective) * args.samples_per_objective) % len(samples)
            state = controlled_state(sample_index, args.seed + j, context)
            obs, _ = env.reset(
                seed=args.seed + j,
                options={
                    "state_vector": state,
                    "sample_index": sample_index,
                    "objective": objective,
                },
            )
            raw_prediction = ppo.predict(obs)
            raw_action = int(np.asarray(raw_prediction).reshape(-1)[0])
            biased_action = orchestrator.bias_action_for_objective(raw_action, objective)

            raw_info = evaluate_action(
                env, state, sample_index, objective, args.seed + j, raw_action
            )
            biased_info = evaluate_action(
                env, state, sample_index, objective, args.seed + j, biased_action
            )

            rows.append({
                "objective": objective,
                "selector_output": selected,
                "sample_index": sample_index,
                "raw_action": raw_action,
                "biased_action": biased_action,
                "action_changed": raw_action != biased_action,
                "raw_retained_fraction": float(raw_info["retained_latent_fraction"]),
                "biased_retained_fraction": float(biased_info["retained_latent_fraction"]),
                "raw_retained_dims": int(raw_info["retained_latent_dims"]),
                "biased_retained_dims": int(biased_info["retained_latent_dims"]),
                "raw_mse": float(raw_info["reconstruction_mse"]),
                "biased_mse": float(biased_info["reconstruction_mse"]),
                "raw_quality": float(raw_info["semantic_quality"]),
                "biased_quality": float(biased_info["semantic_quality"]),
            })

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    packet_path = Path(str(prefix) + "_details.csv")
    summary_path = Path(str(prefix) + "_summary.csv")

    with packet_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for objective in CONTEXTS:
        group = [r for r in rows if r["objective"] == objective]
        n = len(group)
        summary.append({
            "objective": objective,
            "samples": n,
            "selector_matches_expected": all(r["selector_output"] == objective for r in group),
            "action_changed_count": sum(r["action_changed"] for r in group),
            "action_changed_rate": round(sum(r["action_changed"] for r in group) / n, 4),
            "mean_raw_action": round(float(np.mean([r["raw_action"] for r in group])), 4),
            "mean_biased_action": round(float(np.mean([r["biased_action"] for r in group])), 4),
            "mean_raw_retention": round(float(np.mean([r["raw_retained_fraction"] for r in group])), 4),
            "mean_biased_retention": round(float(np.mean([r["biased_retained_fraction"] for r in group])), 4),
            "mean_raw_mse": round(float(np.mean([r["raw_mse"] for r in group])), 6),
            "mean_biased_mse": round(float(np.mean([r["biased_mse"] for r in group])), 6),
            "mean_raw_quality": round(float(np.mean([r["raw_quality"] for r in group])), 4),
            "mean_biased_quality": round(float(np.mean([r["biased_quality"] for r in group])), 4),
        })

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\nSummary")
    print(f"{'Objective':22s} {'Changed':>8s} {'Retention raw→bias':>22s} {'MSE raw→bias':>22s} {'Quality raw→bias':>22s}")
    for r in summary:
        print(
            f"{r['objective']:22s} "
            f"{r['action_changed_rate']:8.1%} "
            f"{r['mean_raw_retention']:.3f} → {r['mean_biased_retention']:.3f}".rjust(22 + 8) + " "
            f"{r['mean_raw_mse']:.4f} → {r['mean_biased_mse']:.4f}".rjust(22) + " "
            f"{r['mean_raw_quality']:.3f} → {r['mean_biased_quality']:.3f}".rjust(22)
        )
    print(f"\nDetails: {packet_path}")
    print(f"Summary: {summary_path}")
    print("Diagnostic only: no network simulation, retraining, NS-3, or PQC measurements.")


if __name__ == "__main__":
    main()
