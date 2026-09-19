"""
Focused tests for semantic RL, orchestration integration, and evaluation I/O.

Run from project root:
    pytest test_semantic_rl.py -q
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.agent import AIOrchestrationAgent
from orchestration.communicator import CommunicationController, COMPRESSION_LEVELS
from rl.agent import RLAgent, DEFAULT_PPO_CHECKPOINT
from rl.environment import (
    SemanticCompressionEnv,
    ACTION_TO_COMPRESSION,
    SEMANTIC_COMPRESSION_LEVELS,
)
from semantic.compression import AdaptiveSemanticCompressor
from semantic.model_io import load_semantic_models, DEFAULT_LATENT_DIM


@pytest.fixture(scope="module")
def semantic_models():
    encoder_path = PROJECT_ROOT / "semantic_encoder.pth"
    decoder_path = PROJECT_ROOT / "semantic_decoder.pth"
    if not encoder_path.exists() or not decoder_path.exists():
        pytest.skip("Semantic checkpoints not available")

    device = torch.device("cpu")
    encoder, decoder = load_semantic_models(
        encoder_path=str(encoder_path),
        decoder_path=str(decoder_path),
        device=device,
    )
    return encoder, decoder, device


@pytest.fixture
def kinematics_sample():
    sample = torch.randn(32, 76)
    return [sample]


def test_semantic_env_default_reset_step_unchanged():
    env = SemanticCompressionEnv()
    obs, info = env.reset()

    assert obs.shape == (7,)
    assert obs.dtype == np.float32
    assert np.allclose(obs, [
        0.8, 0.7, 0.3, 0.2, 0.05, 0.9, 0.8,
    ])

    next_obs, reward, terminated, truncated, info = env.step(2)
    assert next_obs.shape == (7,)
    assert math.isfinite(reward)
    assert terminated is False
    assert truncated is False
    assert info["compression"] == 0.75


def test_action_mapping_constants():
    assert SEMANTIC_COMPRESSION_LEVELS == [0.25, 0.50, 0.75, 1.00]
    assert ACTION_TO_COMPRESSION[3] == 1.00


def test_semantic_env_invalid_action_clamped():
    env = SemanticCompressionEnv()
    env.reset()
    _, reward, _, _, info = env.step(99)

    assert info["action"] == 3
    assert info["invalid_action"] is True
    assert math.isfinite(reward)


def test_semantic_env_reconstruction_mode(semantic_models, kinematics_sample):
    encoder, decoder, device = semantic_models
    compressor = AdaptiveSemanticCompressor(latent_dim=DEFAULT_LATENT_DIM)
    orchestrator = AIOrchestrationAgent()

    env = SemanticCompressionEnv(
        encoder=encoder,
        decoder=decoder,
        compressor=compressor,
        kinematics_samples=kinematics_sample,
        orchestrator=orchestrator,
        device=device,
        use_reconstruction_reward=True,
    )

    obs, info = env.reset(seed=7)
    assert obs.shape == (7,)
    assert info["objective"] in orchestrator.objectives

    _, reward, _, _, info = env.step(3)
    assert info["reconstruction_mse"] is not None
    assert info["reconstruction_mse"] >= 0.0
    assert info["retained_latent_dims"] == DEFAULT_LATENT_DIM
    assert math.isfinite(reward)


def test_orchestration_reward_weights_change_by_objective():
    agent = AIOrchestrationAgent()

    latency = agent.get_reward_weights("LATENCY_CRITICAL")
    quality = agent.get_reward_weights("QUALITY_CRITICAL")

    assert latency["latency"] > quality["latency"]
    assert quality["quality"] > latency["quality"]


def test_orchestration_action_bias():
    agent = AIOrchestrationAgent()

    assert agent.bias_action_for_objective(3, "BANDWIDTH_EFFICIENT") == 1
    assert agent.bias_action_for_objective(0, "QUALITY_CRITICAL") == 2
    assert agent.bias_action_for_objective(3, "BALANCED") == 3


def test_communication_controller_integration(semantic_models, kinematics_sample):
    encoder, decoder, device = semantic_models
    compressor = AdaptiveSemanticCompressor(latent_dim=DEFAULT_LATENT_DIM)

    env = SemanticCompressionEnv(
        encoder=encoder,
        decoder=decoder,
        compressor=compressor,
        kinematics_samples=kinematics_sample,
        orchestrator=AIOrchestrationAgent(),
        device=device,
        use_reconstruction_reward=True,
    )

    agent = RLAgent(env, seed=7, verbose=0)
    controller = CommunicationController(agent)

    state, _ = env.reset(seed=1)
    objective, compression, action = controller.select_compression(state)

    assert objective in AIOrchestrationAgent.OBJECTIVES
    assert compression == COMPRESSION_LEVELS[action]
    assert 0 <= action <= 3


def test_model_loading_tensor_shapes(semantic_models):
    encoder, decoder, device = semantic_models
    x = torch.randn(4, 32, 76)

    with torch.no_grad():
        latent = encoder(x)
        output = decoder(latent)

    assert latent.shape == (4, 32, 16)
    assert output.shape == (4, 32, 76)


def test_ppo_training_smoke(semantic_models, kinematics_sample, tmp_path):
    encoder, decoder, device = semantic_models
    compressor = AdaptiveSemanticCompressor(latent_dim=DEFAULT_LATENT_DIM)

    env = SemanticCompressionEnv(
        encoder=encoder,
        decoder=decoder,
        compressor=compressor,
        kinematics_samples=kinematics_sample,
        orchestrator=AIOrchestrationAgent(),
        device=device,
        use_reconstruction_reward=True,
    )

    checkpoint = tmp_path / "ppo_smoke"
    agent = RLAgent(env, seed=11, verbose=0)
    agent.train(timesteps=64)
    agent.save(str(checkpoint))

    saved_zip = checkpoint.with_suffix(".zip")
    assert saved_zip.exists()

    agent.load(str(checkpoint), environment=env)
    state, _ = env.reset()
    action = agent.predict(state)
    assert action in ACTION_TO_COMPRESSION


def test_evaluation_output_files(tmp_path, semantic_models, kinematics_sample):
    from evaluate_semantic_rl import save_results

    row = {
        "policy": "baseline_25pct",
        "action": 0,
        "compression_fraction": 0.25,
        "retained_latent_dims": 4,
        "communication_cost": 0.25,
        "reconstruction_mse": 0.12,
        "estimated_latency": 0.31,
        "semantic_quality": 0.89,
        "mean_reward": -0.1,
        "num_eval_samples": 8,
        "evaluation_scope": "held_out",
        "seed": 42,
    }

    prefix = tmp_path / "semantic_rl_comparison"
    json_path, csv_path = save_results([row], prefix)

    assert json_path.exists()
    assert csv_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload[0]["policy"] == "baseline_25pct"
    assert payload[0]["num_eval_samples"] == 8


def test_split_is_deterministic_and_disjoint():
    from evaluate_semantic_rl import split_dataset_indices

    train_a, val_a = split_dataset_indices(100, val_fraction=0.2, seed=42)
    train_b, val_b = split_dataset_indices(100, val_fraction=0.2, seed=42)
    train_other, val_other = split_dataset_indices(
        100,
        val_fraction=0.2,
        seed=99,
    )

    np.testing.assert_array_equal(val_a, val_b)
    np.testing.assert_array_equal(train_a, train_b)
    assert not np.array_equal(val_a, val_other)

    overlap = set(train_a).intersection(set(val_a))
    assert len(overlap) == 0
    assert len(val_a) == 20
    assert len(train_a) == 80


def test_deterministic_state_builder():
    from evaluate_semantic_rl import build_deterministic_state

    state_a = build_deterministic_state(sample_index=3, seed=42)
    state_b = build_deterministic_state(sample_index=3, seed=42)
    state_other = build_deterministic_state(sample_index=4, seed=42)

    np.testing.assert_array_equal(state_a, state_b)
    assert not np.allclose(state_a, state_other)
    assert state_a.shape == (7,)


def test_all_policies_use_same_held_out_indices(semantic_models):
    from evaluate_semantic_rl import (
        build_held_out_split,
        evaluate_policy_on_held_out,
    )
    from semantic.data_loader import JIGSAWSKinematicsDataset

    dataset = JIGSAWSKinematicsDataset(
        root_dir=str(PROJECT_ROOT / "data"),
        sequence_length=32,
        stride=16,
        normalize=True,
    )
    val_samples, _, val_indices = build_held_out_split(
        dataset,
        seed=42,
        val_fraction=0.2,
        max_eval_samples=16,
    )

    encoder, decoder, device = semantic_models
    compressor = AdaptiveSemanticCompressor(latent_dim=DEFAULT_LATENT_DIM)
    env = SemanticCompressionEnv(
        encoder=encoder,
        decoder=decoder,
        compressor=compressor,
        kinematics_samples=val_samples,
        orchestrator=AIOrchestrationAgent(),
        device=device,
        use_reconstruction_reward=True,
    )
    agent = RLAgent(env, seed=42, verbose=0)

    baseline_row = evaluate_policy_on_held_out(
        policy_name="baseline_50pct",
        val_samples=val_samples,
        val_indices=val_indices,
        env=env,
        seed=42,
        action_selector=lambda _state, _idx: 1,
        fixed_action=1,
    )
    ppo_row = evaluate_policy_on_held_out(
        policy_name="ppo_raw",
        val_samples=val_samples,
        val_indices=val_indices,
        env=env,
        seed=42,
        action_selector=lambda state, _idx: agent.predict(state),
    )

    assert baseline_row["held_out_indices"] == ppo_row["held_out_indices"]
    assert baseline_row["num_eval_samples"] == len(val_samples)
    assert ppo_row["num_eval_samples"] == len(val_samples)
    assert baseline_row["evaluation_scope"] == "held_out"
    assert ppo_row["evaluation_scope"] == "held_out"
