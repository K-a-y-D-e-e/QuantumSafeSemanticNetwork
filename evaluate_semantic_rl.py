"""
Evaluate semantic compression policies: fixed baselines vs trained PPO.

All policies are evaluated on the exact same held-out JIGSAWS split using
deterministic per-sample network states derived from ``seed``.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from orchestration.agent import AIOrchestrationAgent
from rl.agent import RLAgent, DEFAULT_PPO_CHECKPOINT
from rl.environment import SemanticCompressionEnv, ACTION_TO_COMPRESSION
from semantic.compression import AdaptiveSemanticCompressor
from semantic.data_loader import JIGSAWSKinematicsDataset
from semantic.model_io import (
    DEFAULT_INPUT_DIM,
    DEFAULT_LATENT_DIM,
    load_semantic_models,
)


RESULTS_DIR = Path("rl/results/semantic_rl")
BASELINE_NAMES = {
    0: "baseline_25pct",
    1: "baseline_50pct",
    2: "baseline_75pct",
    3: "baseline_100pct",
}


def split_dataset_indices(
    num_samples: int,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/validation index arrays."""
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    val_size = max(1, int(num_samples * val_fraction))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def build_deterministic_state(sample_index: int, seed: int) -> np.ndarray:
    """
    Build a reproducible network-state vector for a held-out sample index.

    Each sample index receives a unique but deterministic state so all
    policies observe identical conditions during evaluation.
    """
    rng = np.random.default_rng(seed + sample_index)
    return rng.uniform(0.0, 1.0, size=(7,)).astype(np.float32)


def build_held_out_split(
    dataset: JIGSAWSKinematicsDataset,
    seed: int,
    val_fraction: float,
    max_eval_samples: Optional[int] = None,
) -> Tuple[List[torch.Tensor], np.ndarray, np.ndarray]:
    """
    Create deterministic train/validation splits and materialize held-out samples.
    """
    train_indices, val_indices = split_dataset_indices(
        len(dataset),
        val_fraction=val_fraction,
        seed=seed,
    )

    if max_eval_samples is not None:
        val_indices = val_indices[:max_eval_samples]

    val_samples = [dataset[int(i)] for i in val_indices]
    return val_samples, train_indices, val_indices


def evaluate_policy_on_held_out(
    policy_name: str,
    val_samples: Sequence[torch.Tensor],
    val_indices: np.ndarray,
    env: SemanticCompressionEnv,
    seed: int,
    action_selector: Callable[[np.ndarray, int], int],
    fixed_action: Optional[int] = None,
) -> Dict:
    """
    Evaluate one policy across every held-out sample exactly once.
    """
    mse_values: List[float] = []
    fractions: List[float] = []
    costs: List[float] = []
    latencies: List[float] = []
    qualities: List[float] = []
    rewards: List[float] = []

    for local_idx, dataset_index in enumerate(val_indices):
        state = build_deterministic_state(int(local_idx), seed)
        action = (
            int(fixed_action)
            if fixed_action is not None
            else int(action_selector(state, int(local_idx)))
        )

        env.reset(
            seed=seed,
            options={
                "sample_index": local_idx,
                "state_vector": state,
            },
        )
        _, reward, _, _, info = env.step(action)

        mse_values.append(float(info["reconstruction_mse"]))
        fractions.append(float(info["compression"]))
        costs.append(float(info["communication_cost"]))
        latencies.append(float(info["estimated_latency"]))
        qualities.append(float(info["semantic_quality"]))
        rewards.append(float(reward))

    num_samples = len(val_samples)
    avg_fraction = float(np.mean(fractions))

    return {
        "policy": policy_name,
        "action": fixed_action,
        "compression_fraction": avg_fraction,
        "retained_latent_dims": avg_fraction * DEFAULT_LATENT_DIM,
        "communication_cost": float(np.mean(costs)),
        "reconstruction_mse": float(np.mean(mse_values)),
        "estimated_latency": float(np.mean(latencies)),
        "semantic_quality": float(np.mean(qualities)),
        "mean_reward": float(np.mean(rewards)),
        "num_eval_samples": num_samples,
        "evaluation_scope": "held_out",
        "held_out_indices": [int(i) for i in val_indices],
        "seed": seed,
    }


def save_results(results: List[Dict], output_prefix: Path):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    fieldnames = [
        "policy",
        "action",
        "compression_fraction",
        "retained_latent_dims",
        "communication_cost",
        "reconstruction_mse",
        "estimated_latency",
        "semantic_quality",
        "mean_reward",
        "num_eval_samples",
        "evaluation_scope",
        "seed",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})

    return json_path, csv_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic compression baselines and PPO."
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--encoder-path", default="semantic_encoder.pth")
    parser.add_argument("--decoder-path", default="semantic_decoder.pth")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_PPO_CHECKPOINT,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=256,
        help="Maximum held-out sequences evaluated for every policy.",
    )
    parser.add_argument(
        "--output-prefix",
        default=str(RESULTS_DIR / "semantic_rl_comparison"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")

    dataset = JIGSAWSKinematicsDataset(
        root_dir=args.data_root,
        sequence_length=32,
        stride=16,
        normalize=True,
    )

    val_samples, train_indices, val_indices = build_held_out_split(
        dataset,
        seed=args.seed,
        val_fraction=args.val_fraction,
        max_eval_samples=args.max_eval_samples,
    )

    encoder, decoder = load_semantic_models(
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path,
        device=device,
    )
    compressor = AdaptiveSemanticCompressor(latent_dim=DEFAULT_LATENT_DIM)
    orchestrator = AIOrchestrationAgent()

    env = SemanticCompressionEnv(
        encoder=encoder,
        decoder=decoder,
        compressor=compressor,
        kinematics_samples=val_samples,
        orchestrator=orchestrator,
        device=device,
        use_reconstruction_reward=True,
    )

    results: List[Dict] = []

    print(
        f"Evaluating all policies on the same held-out split "
        f"(seed={args.seed}, n={len(val_samples)})..."
    )

    for action in range(4):
        row = evaluate_policy_on_held_out(
            policy_name=BASELINE_NAMES[action],
            val_samples=val_samples,
            val_indices=val_indices,
            env=env,
            seed=args.seed,
            action_selector=lambda _state, _idx, action=action: action,
            fixed_action=action,
        )
        results.append(row)
        print(
            f"{row['policy']:16s} | "
            f"MSE={row['reconstruction_mse']:.6f} | "
            f"latency={row['estimated_latency']:.4f} | "
            f"quality={row['semantic_quality']:.4f} | "
            f"n={row['num_eval_samples']}"
        )

    checkpoint_path = Path(args.checkpoint).with_suffix(".zip")
    if checkpoint_path.exists():
        agent = RLAgent(env, seed=args.seed, verbose=0)
        agent.load(str(checkpoint_path.with_suffix("")), environment=env)

        def ppo_raw_action(state, _idx):
            return agent.predict(state)

        def ppo_orchestrated_action(state, _idx):
            objective = orchestrator.decide_objective(
                float(state[0]),
                float(state[2]),
                float(state[3]),
                float(state[5]),
            )
            raw_action = agent.predict(state)
            return orchestrator.bias_action_for_objective(
                raw_action,
                objective,
            )

        for policy_name, selector in (
            ("ppo_raw", ppo_raw_action),
            ("ppo_orchestrated", ppo_orchestrated_action),
        ):
            row = evaluate_policy_on_held_out(
                policy_name=policy_name,
                val_samples=val_samples,
                val_indices=val_indices,
                env=env,
                seed=args.seed,
                action_selector=selector,
            )
            results.append(row)
            print(
                f"{row['policy']:16s} | "
                f"MSE={row['reconstruction_mse']:.6f} | "
                f"latency={row['estimated_latency']:.4f} | "
                f"quality={row['semantic_quality']:.4f} | "
                f"fraction={row['compression_fraction']:.2f} | "
                f"n={row['num_eval_samples']}"
            )
    else:
        print(
            f"\nSkipping PPO evaluation: checkpoint not found at {checkpoint_path}"
        )

    json_path, csv_path = save_results(results, Path(args.output_prefix))
    print("\nEVALUATION COMPLETE")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
