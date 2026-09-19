"""
Unit tests for the Phase 3 RL components.

Tests
-----
 1. Environment reset works (returns obs and info dict)
 2. Observation has correct shape and dtype
 3. Action space matches MAX_QUEUE
 4. Valid action executes correctly (no exception, correct types)
 5. Invalid action is handled safely (clamped, no crash)
 6. Reward is finite on every step
 7. Episode terminates within a bounded number of steps
 8. Existing Phase 1/2 imports and SemanticCompressionEnv still work
 9. Same seed produces reproducible observations
10. Smoke-training run (1 episode, 5 flows) executes without error

Run from the project root:
    pytest rl/test_rl_environment.py -v

Or from within the rl/ directory:
    cd rl
    pytest test_rl_environment.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── RL imports ────────────────────────────────────────────────────────────────
from rl.environment import NetworkSchedulingEnv, OBS_DIM, MAX_QUEUE   # noqa: E402
from rl.traffic_generator import TrafficGenerator                      # noqa: E402
from rl.dqn_agent import DQNAgent                                      # noqa: E402

# ── Network imports ───────────────────────────────────────────────────────────
from network.simulation.link import NetworkLink                        # noqa: E402


# =============================================================================
# Shared helpers
# =============================================================================

def _make_link() -> NetworkLink:
    """Standard 10 Mbps link used across tests."""
    return NetworkLink(bandwidth_mbps=10, propagation_delay_us=100)


def _make_env(seed: int = 42, num_flows: int = 5) -> NetworkSchedulingEnv:
    """Create a minimal environment for testing."""
    link = _make_link()
    gen  = TrafficGenerator(seed=seed, num_flows=num_flows)
    return NetworkSchedulingEnv(traffic_generator=gen, link=link)


def _first_valid(env: NetworkSchedulingEnv) -> int:
    """Return the first valid action index for the current queue state."""
    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    assert len(valid) > 0, "No valid actions available"
    return int(valid[0])


# =============================================================================
# Test 1 — Environment reset
# =============================================================================

def test_01_reset_works():
    """reset() should return an observation and a non-None info dict."""
    env = _make_env()
    obs, info = env.reset()

    assert obs is not None, "reset() returned None observation"
    assert isinstance(info, dict), "reset() info must be a dict"


# =============================================================================
# Test 2 — Observation shape and dtype
# =============================================================================

def test_02_observation_shape_and_dtype():
    """Observation must have shape (OBS_DIM,) and dtype float32."""
    env = _make_env()
    obs, _ = env.reset()

    assert obs.shape == (OBS_DIM,), (
        f"Expected shape ({OBS_DIM},), got {obs.shape}"
    )
    assert obs.dtype == np.float32, (
        f"Expected float32, got {obs.dtype}"
    )
    # Values must be in [0, 1] (observation is normalised)
    assert obs.min() >= 0.0, "Observation contains values below 0"
    assert obs.max() <= 1.0, "Observation contains values above 1"


# =============================================================================
# Test 3 — Action space
# =============================================================================

def test_03_action_space_valid():
    """Action space must be Discrete(MAX_QUEUE) and sample within range."""
    env = _make_env()
    env.reset()

    assert env.action_space.n == MAX_QUEUE, (
        f"Expected action_space.n={MAX_QUEUE}, got {env.action_space.n}"
    )
    for _ in range(50):
        a = env.action_space.sample()
        assert 0 <= a < MAX_QUEUE, f"Sampled action {a} out of range"


# =============================================================================
# Test 4 — Valid action executes correctly
# =============================================================================

def test_04_valid_action_executes():
    """A valid action must not raise and must return correct types."""
    env = _make_env()
    obs, _ = env.reset()
    action  = _first_valid(env)

    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == (OBS_DIM,), "next_obs has wrong shape"
    assert isinstance(reward, float), "reward must be float"
    assert isinstance(terminated, bool), "terminated must be bool"
    assert isinstance(truncated, bool), "truncated must be bool"
    assert isinstance(info, dict), "info must be dict"


# =============================================================================
# Test 5 — Invalid action handled safely
# =============================================================================

def test_05_invalid_action_safe():
    """
    An out-of-range action (>= len(waiting_queue)) must not raise an
    exception.  It should be clamped and apply a mask-violation penalty.
    """
    env = _make_env(num_flows=2)
    obs, _ = env.reset()

    # With only 2 flows, action MAX_QUEUE-1 will almost certainly be invalid
    invalid_action = MAX_QUEUE - 1

    try:
        next_obs, reward, terminated, truncated, info = env.step(
            invalid_action
        )
    except Exception as exc:
        pytest.fail(
            f"step() raised an exception on invalid action: {exc}"
        )

    # The step must still return a valid next observation
    assert next_obs.shape == (OBS_DIM,), "next_obs shape wrong after invalid action"
    # Reward should be finite (penalty is a finite float)
    assert math.isfinite(reward), f"Reward after invalid action is not finite: {reward}"


# =============================================================================
# Test 6 — Reward is finite
# =============================================================================

def test_06_reward_is_finite():
    """Reward must be a finite float on every step of an episode."""
    env = _make_env(num_flows=5)
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < 100:
        action = _first_valid(env)
        obs, reward, terminated, truncated, info = env.step(action)
        assert math.isfinite(reward), (
            f"Non-finite reward at step {steps}: {reward}"
        )
        done = terminated or truncated
        steps += 1


# =============================================================================
# Test 7 — Episode terminates
# =============================================================================

def test_07_episode_terminates():
    """
    An episode driven by valid greedy actions must terminate (terminated=True)
    within a bounded number of steps.
    """
    env = _make_env(num_flows=10)
    obs, _ = env.reset()
    done  = False
    steps = 0
    limit = 200   # generous upper bound

    while not done and steps < limit:
        action = _first_valid(env)
        obs, _, terminated, truncated, _ = env.step(action)
        done  = terminated or truncated
        steps += 1

    assert done, (
        f"Episode did not terminate within {limit} steps "
        f"(completed {steps} steps)"
    )


# =============================================================================
# Test 8 — Phase 1/2 and SemanticCompressionEnv still importable
# =============================================================================

def test_08_existing_imports_not_broken():
    """
    All Phase 1/2 network-simulator imports must still work.
    SemanticCompressionEnv must remain importable from rl.environment.
    """
    # Phase 2 network simulator
    from network.simulation.scheduler import PriorityScheduler
    from network.simulation.deadline_scheduler import DeadlineAwareScheduler
    from network.simulation.event_simulator import EventDrivenSimulator
    from network.simulation.link import NetworkLink
    from network.simulation.analyzer import NetworkAnalyzer
    from network.flows.flow import NetworkFlow
    from network.nodes.node import NetworkNode
    from network.simulation.queue import NetworkQueue

    # Original semantic compression environment (preserved)
    from rl.environment import SemanticCompressionEnv
    assert SemanticCompressionEnv is not None

    # Verify SemanticCompressionEnv still operates correctly
    env = SemanticCompressionEnv()
    obs, _ = env.reset()
    assert obs.shape == (7,), (
        f"SemanticCompressionEnv obs shape changed: {obs.shape}"
    )
    next_obs, reward, terminated, truncated, info = env.step(2)  # action=75%
    assert math.isfinite(reward), "SemanticCompressionEnv reward not finite"


# =============================================================================
# Test 9 — Seed reproducibility
# =============================================================================

def test_09_seed_reproducibility():
    """
    Two environments created with the same seed must produce identical
    first observations on reset().
    """
    env1 = _make_env(seed=77, num_flows=8)
    env2 = _make_env(seed=77, num_flows=8)

    obs1, _ = env1.reset()
    obs2, _ = env2.reset()

    np.testing.assert_array_equal(
        obs1, obs2,
        err_msg="Observations differ between same-seed environments"
    )


# =============================================================================
# Test 10 — Smoke-training run
# =============================================================================

def test_10_smoke_training():
    """
    A single DQN training episode (5 flows) must complete without error.
    Verifies the full training pipeline: reset → action → store → learn.
    This is a smoke test — it does NOT validate trained-model quality.
    """
    env   = _make_env(num_flows=5)
    agent = DQNAgent(
        obs_dim=OBS_DIM,
        action_dim=MAX_QUEUE,
        seed=42,
        batch_size=4,   # small batch so learning triggers quickly
        buffer_size=50,
    )

    obs, _ = env.reset()
    done   = False
    steps  = 0

    while not done:
        mask   = env.get_action_mask()
        action = agent.select_action(obs, mask, training=True)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_mask = env.get_action_mask()
        done = terminated or truncated

        agent.store(obs, action, reward, next_obs, done, next_mask)
        agent.learn()

        obs    = next_obs
        steps += 1

    agent.decay_epsilon()

    results = env.get_episode_results()
    assert len(results) == 5, (
        f"Expected 5 flow results, got {len(results)}"
    )
    assert steps == 5, (
        f"Expected exactly 5 scheduling steps for 5 flows, got {steps}"
    )
