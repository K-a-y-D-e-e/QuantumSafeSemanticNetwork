# Semantic RL Components

This note documents the semantic compression reinforcement-learning prototype added for the assigned pipeline:

```
JIGSAWS features
  -> semantic encoder
  -> orchestration objective (rule-based)
  -> PPO compression action
  -> adaptive latent compression
  -> decoder
  -> reconstruction / communication metrics
```

## Architecture

| Component | File | Role |
|-----------|------|------|
| Semantic encoder/decoder | `semantic/encoder.py`, `semantic/decoder.py` | Map kinematic sequences to/from latent space |
| Adaptive compressor | `semantic/compression.py` | Keep top-|latent| dimensions by magnitude |
| RL environment | `rl/environment.py` (`SemanticCompressionEnv`) | PPO training interface |
| PPO wrapper | `rl/agent.py` (`RLAgent`) | Stable-Baselines3 training/inference |
| Orchestration | `orchestration/agent.py` | Rule-based objective selection |
| Controller | `orchestration/communicator.py` | Combines objective + PPO action |

Orchestration is **not learned**. Only the compression policy is trained with PPO.

## State / Action Definitions

**Observation** (`float32[7]`):

1. task criticality
2. available bandwidth
3. network load
4. latency
5. packet loss
6. deadline requirement
7. semantic quality (updated after each step)

**Actions** (discrete):

| Action | Retained latent fraction |
|--------|--------------------------|
| 0 | 25% |
| 1 | 50% |
| 2 | 75% |
| 3 | 100% |

Invalid actions are clamped to `[0, 3]` with a small penalty.

## Reward (extended reconstruction mode)

When `use_reconstruction_reward=True`:

```
reward =
    w_quality      * (1 / (1 + reconstruction_mse))
  - w_latency      * estimated_latency
  - w_bandwidth    * retained_latent_fraction
  - w_deadline     * deadline_violation
  - w_criticality  * criticality_penalty
  - 0.15           * packet_loss
```

Objective-specific weights (`w_*`) come from `AIOrchestrationAgent.get_reward_weights()`.

Assumptions:

- One JIGSAWS sequence is evaluated per environment step.
- Latency/bandwidth terms are normalized proxies, not ns-3 measurements.
- Objective is **not** included in the observation to avoid reward/action leakage.

Default mode (`use_reconstruction_reward=False`) keeps the original analytic reward for backward compatibility.

## Commands

Train semantic autoencoder (existing):

```bash
python train_semantic.py
```

Train PPO semantic compression policy:

```bash
python train_semantic_rl.py --seed 42 --timesteps 10000
python train_semantic_rl.py --smoke-test
```

Evaluate baselines + PPO (all policies on the same held-out split):

```bash
python evaluate_semantic_rl.py --seed 42
```

Every policy evaluates each held-out sample exactly once using deterministic
per-sample network states derived from `seed`.

Run tests:

```bash
python -m pytest semantic orchestration rl network test_semantic_rl.py -q
```

## Model / Result Paths

| Artifact | Path |
|----------|------|
| Semantic encoder | `semantic_encoder.pth` |
| Semantic decoder | `semantic_decoder.pth` |
| PPO checkpoint | `rl/checkpoints/ppo_semantic_agent.zip` |
| Evaluation JSON/CSV | `rl/results/semantic_rl/semantic_rl_comparison.{json,csv}` |

DQN network-scheduling checkpoints under `rl/` are untouched.

## Limitations / Not Yet Integrated

- ns-3 network simulation is not connected to semantic RL rewards.
- Cryptographic channel overhead is not modeled in the RL reward.
- Orchestration remains rule-based; no meta-learning across objectives.
- Held-out evaluation uses a deterministic index split from `seed`; training still uses separate train indices in `train_semantic_rl.py`.
- Training uses a capped subset of JIGSAWS sequences for CPU-friendly prototyping.
