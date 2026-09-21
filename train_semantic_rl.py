"""
Train a PPO agent for adaptive semantic compression.

Pipeline:
    JIGSAWS features -> semantic encoder -> orchestration objective ->
    PPO compression action -> adaptive latent compression -> decoder metrics
"""

import argparse
from pathlib import Path

import torch

from orchestration.agent import AIOrchestrationAgent
from rl.agent import RLAgent, DEFAULT_PPO_CHECKPOINT
from rl.environment import SemanticCompressionEnv
from semantic.compression import AdaptiveSemanticCompressor
from semantic.data_loader import JIGSAWSKinematicsDataset
from semantic.model_io import (
    DEFAULT_INPUT_DIM,
    DEFAULT_LATENT_DIM,
    load_semantic_models,
)


def build_environment(
    data_root: str,
    encoder_path: str,
    decoder_path: str,
    max_train_samples: int,
    device: torch.device,
):
    dataset = JIGSAWSKinematicsDataset(
        root_dir=data_root,
        sequence_length=32,
        stride=16,
        normalize=True,
    )

    sample_count = min(max_train_samples, len(dataset))
    kinematics_samples = [dataset[i] for i in range(sample_count)]

    encoder, decoder = load_semantic_models(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        input_dim=DEFAULT_INPUT_DIM,
        latent_dim=DEFAULT_LATENT_DIM,
        device=device,
    )

    compressor = AdaptiveSemanticCompressor(latent_dim=DEFAULT_LATENT_DIM)
    orchestrator = AIOrchestrationAgent()

    env = SemanticCompressionEnv(
        encoder=encoder,
        decoder=decoder,
        compressor=compressor,
        kinematics_samples=kinematics_samples,
        orchestrator=orchestrator,
        device=device,
        use_reconstruction_reward=True,
    )
    return env, sample_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO semantic compression agent."
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--encoder-path",
        default="semantic_encoder.pth",
    )
    parser.add_argument(
        "--decoder-path",
        default="semantic_decoder.pth",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_PPO_CHECKPOINT,
        help="Output path for the trained PPO model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10000,
        help="Total PPO training timesteps.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=512,
        help="Number of JIGSAWS sequences used inside the RL environment.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short training pass for quick validation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        args.timesteps = min(args.timesteps, 256)
        args.max_train_samples = min(args.max_train_samples, 32)

    device = torch.device("cpu")

    print("=" * 60)
    print("Semantic RL Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Encoder: {args.encoder_path}")
    print(f"Decoder: {args.decoder_path}")

    env, sample_count = build_environment(
        data_root=args.data_root,
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path,
        max_train_samples=args.max_train_samples,
        device=device,
    )

    print(f"Environment samples: {sample_count}")
    print("Objective source: rule-based AIOrchestrationAgent")

    agent = RLAgent(env, seed=args.seed, verbose=1)
    agent.train(timesteps=args.timesteps)
    agent.save(args.checkpoint)

    print("\nTRAINING COMPLETE")
    print(f"Saved PPO model to: {Path(args.checkpoint).with_suffix('')}")


if __name__ == "__main__":
    main()
