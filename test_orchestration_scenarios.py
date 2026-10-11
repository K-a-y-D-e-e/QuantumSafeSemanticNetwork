
import csv
from pathlib import Path

import numpy as np
import torch

from orchestration.agent import AIOrchestrationAgent
from rl.agent import RLAgent
from rl.environment import SemanticCompressionEnv
from evaluate_semantic_rl import build_held_out_split
from semantic.compression import AdaptiveSemanticCompressor
from semantic.data_loader import JIGSAWSKinematicsDataset
from semantic.model_io import load_semantic_models, DEFAULT_LATENT_DIM


SEED = 42
CHECKPOINT = "rl/checkpoints/ppo_semantic_adaptive"
OUTPUT = Path(
    "rl/results/semantic_rl/orchestration_controlled_scenarios.csv"
)

# Each scenario changes the network/task conditions while keeping
# all other state features explicitly defined.
SCENARIOS = {
    "BALANCED": np.array(
        [0.50, 0.80, 0.30, 0.25, 0.01, 0.60, 0.70],
        dtype=np.float32,
    ),
    "QUALITY_CRITICAL": np.array(
        [0.95, 0.80, 0.40, 0.30, 0.02, 0.80, 0.75],
        dtype=np.float32,
    ),
    "LATENCY_CRITICAL": np.array(
        [0.95, 0.55, 0.60, 0.75, 0.05, 0.65, 0.70],
        dtype=np.float32,
    ),
    "BANDWIDTH_EFFICIENT": np.array(
        [0.50, 0.45, 0.85, 0.55, 0.08, 0.70, 0.60],
        dtype=np.float32,
    ),
}

ACTION_RETENTION = {
    0: 0.25,
    1: 0.50,
    2: 0.75,
    3: 1.00,
}


def evaluate_action(env, state, objective, action):
    """Evaluate one action on the same sample and scenario."""
    env.reset(
        seed=SEED,
        options={
            "sample_index": 0,
            "state_vector": state.copy(),
            "objective": objective,
        },
    )

    _, reward, _, _, info = env.step(int(action))

    return {
        "reward": float(reward),
        "mse": float(info["reconstruction_mse"]),
        "quality": float(info["semantic_quality"]),
        "estimated_latency": float(info["estimated_latency"]),
        "retention": float(info["compression"]),
    }


def main():
    device = torch.device("cpu")

    print("Loading JIGSAWS dataset...")
    dataset = JIGSAWSKinematicsDataset(
        root_dir="data",
        sequence_length=32,
        stride=16,
        normalize=True,
    )

    samples, _, _ = build_held_out_split(
        dataset,
        seed=SEED,
        val_fraction=0.2,
        max_eval_samples=256,
    )

    if len(samples) == 0:
        raise RuntimeError("No held-out samples were loaded.")

    encoder, decoder = load_semantic_models(
        encoder_path="semantic_encoder.pth",
        decoder_path="semantic_decoder.pth",
        device=device,
    )

    orchestrator = AIOrchestrationAgent()

    env = SemanticCompressionEnv(
        encoder=encoder,
        decoder=decoder,
        compressor=AdaptiveSemanticCompressor(
            latent_dim=DEFAULT_LATENT_DIM
        ),
        kinematics_samples=samples,
        orchestrator=orchestrator,
        device=device,
        use_reconstruction_reward=True,
    )

    agent = RLAgent(env, seed=SEED, verbose=0)
    agent.load(CHECKPOINT, environment=env)

    records = []

    try:
        for expected_objective, state in SCENARIOS.items():
            objective = orchestrator.decide_objective(
                float(state[0]),
                float(state[2]),
                float(state[3]),
                float(state[5]),
            )

            if objective != expected_objective:
                raise ValueError(
                    f"Expected {expected_objective}, "
                    f"but orchestration selected {objective}."
                )

            # The trained policy's actual choice for this state.
            ppo_action = int(agent.predict(state.copy()))

            print(
                f"\nScenario: {objective} | "
                f"PPO action: {ACTION_RETENTION[ppo_action]:.0%}"
            )

            # Sweep all four possible raw actions to test each rule,
            # not just the action selected by the trained PPO policy.
            for raw_action in range(4):
                orchestrated_action = int(
                    orchestrator.bias_action_for_objective(
                        raw_action, objective
                    )
                )

                raw_result = evaluate_action(
                    env, state, objective, raw_action
                )

                if orchestrated_action == raw_action:
                    # Identical action and inputs: reuse the result.
                    orchestrated_result = raw_result.copy()
                else:
                    orchestrated_result = evaluate_action(
                        env, state, objective, orchestrated_action
                    )

                row = {
                    "objective": objective,
                    "ppo_action": ppo_action,
                    "ppo_retention": ACTION_RETENTION[ppo_action],
                    "tested_raw_action": raw_action,
                    "raw_retention": ACTION_RETENTION[raw_action],
                    "orchestrated_action": orchestrated_action,
                    "orchestrated_retention": (
                        ACTION_RETENTION[orchestrated_action]
                    ),
                    "action_changed": (
                        raw_action != orchestrated_action
                    ),
                    "is_ppo_action": raw_action == ppo_action,
                }

                for metric in (
                    "reward",
                    "mse",
                    "quality",
                    "estimated_latency",
                    "retention",
                ):
                    row[f"raw_{metric}"] = raw_result[metric]
                    row[f"orchestrated_{metric}"] = (
                        orchestrated_result[metric]
                    )
                    row[f"delta_{metric}"] = (
                        orchestrated_result[metric]
                        - raw_result[metric]
                    )

                row["mse_reduction_pct"] = (
                    100.0
                    * (
                        raw_result["mse"]
                        - orchestrated_result["mse"]
                    )
                    / raw_result["mse"]
                    if raw_result["mse"] != 0
                    else float("nan")
                )

                records.append(row)

                print(
                    f"  Tested action {ACTION_RETENTION[raw_action]:.0%}"
                    f" -> {ACTION_RETENTION[orchestrated_action]:.0%}"
                    f" | MSE {raw_result['mse']:.4f}"
                    f" -> {orchestrated_result['mse']:.4f}"
                    f" | Latency proxy "
                    f"{raw_result['estimated_latency']:.4f}"
                    f" -> {orchestrated_result['estimated_latency']:.4f}"
                )

    finally:
        env.close()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved controlled results to: {OUTPUT}")

    print("\nACTION-BIAS RULE VERIFICATION")
    print("Objective | Tested actions changed | Total tested")

    for objective in SCENARIOS:
        group = [
            row for row in records
            if row["objective"] == objective
        ]
        changed = sum(row["action_changed"] for row in group)

        print(f"{objective} | {changed} | {len(group)}")

    print(
        "\nNote: The action sweep tests every possible raw action. "
        "Only rows marked is_ppo_action=True represent the trained "
        "PPO policy's actual choice for that scenario."
    )
    print(
        "Estimated latency is an environment proxy, not measured "
        "end-to-end network latency."
    )


if __name__ == "__main__":
    main()
