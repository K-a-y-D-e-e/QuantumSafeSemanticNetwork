"""
Reinforcement learning environments for the QuantumSafeSemanticNetwork project.

This module contains two independent environments:

SemanticCompressionEnv  (original, Phase 2)
--------------------------------------------
Models adaptive semantic compression level selection.
Action:  choose compression ratio — 25 / 50 / 75 / 100 %.
Purpose: research into bandwidth–quality trade-off under network load.
Status:  preserved exactly as originally implemented.

NetworkSchedulingEnv  (Phase 3)
--------------------------------
Models adaptive network flow scheduling over the existing single-link,
event-driven network simulator.  The RL agent selects *which waiting
flow to transmit next* at each scheduling decision, replacing the fixed
Priority and EDF schedulers.

Both environments are importable from this module and are completely
independent of each other.  The semantic-compression environment is
NOT modified or affected by the addition of the scheduling environment.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Dict, List, Optional


# =============================================================================
# ─── SemanticCompressionEnv (original — preserved unchanged) ─────────────────
# =============================================================================


class SemanticCompressionEnv(gym.Env):
    """
    Original semantic compression environment.

    State
    -----
    [task_criticality, bandwidth, network_load, latency,
     packet_loss, deadline, semantic_quality]

    Actions
    -------
    0 = 25 %  compression
    1 = 50 %  compression
    2 = 75 %  compression
    3 = 100 % compression
    """

    def __init__(self):
        super().__init__()

        # State:
        # [task_criticality,
        #  bandwidth,
        #  network_load,
        #  latency,
        #  packet_loss,
        #  deadline,
        #  semantic_quality]

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        # Actions:
        # 0 = 25%
        # 1 = 50%
        # 2 = 75%
        # 3 = 100%

        self.action_space = spaces.Discrete(4)

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.array([
            0.8,  # task criticality
            0.7,  # bandwidth
            0.3,  # network load
            0.2,  # latency
            0.05, # packet loss
            0.9,  # deadline requirement
            0.8   # semantic quality
        ], dtype=np.float32)

        return self.state, {}

    def step(self, action):

        compression_levels = [
            0.25,
            0.50,
            0.75,
            1.00
        ]

        compression = compression_levels[action]

        task_criticality = self.state[0]
        network_load = self.state[2]
        latency = self.state[3]

        # Higher compression level → better semantic quality
        semantic_quality = compression

        # Higher compression → larger packet
        bandwidth_cost = compression

        # Simple latency model
        estimated_latency = (
            latency
            + 0.3 * bandwidth_cost
            + 0.2 * network_load
        )

        # Reward
        reward = (
            semantic_quality
            - 0.5 * estimated_latency
            - 0.3 * bandwidth_cost
        )

        # Penalize poor quality when task is critical
        if task_criticality > 0.7 and semantic_quality < 0.75:
            reward -= 0.5

        self.state[3] = min(1.0, estimated_latency)
        self.state[6] = semantic_quality

        terminated = False
        truncated = False

        return self.state, reward, terminated, truncated, {}


# =============================================================================
# ─── NetworkSchedulingEnv (Phase 3 — new) ────────────────────────────────────
# =============================================================================

# Maximum number of waiting flows the observation encodes.
# Flows beyond this cap are not included in the observation but are
# still in the internal waiting queue and can be selected via action.
MAX_QUEUE: int = 10

# Normalised priority mapping used in the observation vector
PRIORITY_NORM: Dict[str, float] = {
    "HIGH": 1.0,
    "MEDIUM": 0.5,
    "LOW": 0.0,
}

# Number of features encoded per queue slot in the observation
FEATURES_PER_FLOW: int = 4  # [priority_norm, remaining_deadline_norm, size_norm, wait_time_norm]

# Number of global features appended after the per-slot block
GLOBAL_FEATURES: int = 2    # [queue_fill_ratio, time_norm]

# Total observation dimension
OBS_DIM: int = MAX_QUEUE * FEATURES_PER_FLOW + GLOBAL_FEATURES   # 42


class NetworkSchedulingEnv(gym.Env):
    """
    Gymnasium environment for adaptive network flow scheduling.

    The agent interacts with the existing single-link, event-driven
    network simulator step by step.  At each step the agent selects
    which packet in the current waiting queue to transmit next.

    This environment does **not** replace or modify the existing
    ``EventDrivenSimulator``.  Instead it reproduces the same
    simulation loop internally, giving the RL agent control at
    each scheduling decision point.

    Observation Space
    -----------------
    ``Box(low=0, high=1, shape=(OBS_DIM,), dtype=float32)``  where
    ``OBS_DIM = MAX_QUEUE * FEATURES_PER_FLOW + GLOBAL_FEATURES = 42``.

    The observation vector is structured as::

        [ slot_0_features (4) | slot_1_features (4) | ... | slot_9_features (4) |
          queue_fill_ratio (1) | time_norm (1) ]

    Per-slot features (padded with zeros if the slot is empty):
        priority_norm           HIGH=1.0, MEDIUM=0.5, LOW=0.0
        remaining_deadline_norm (deadline_us - current_time) / max_deadline_us,
                                clipped to [0, 1]
        size_norm               size_bytes / max_size_bytes, clipped to [0, 1]
        wait_time_norm          (current_time - arrival_time) / max_wait_us,
                                clipped to [0, 1]

    Global features:
        queue_fill_ratio        len(waiting_queue) / MAX_QUEUE
        time_norm               current_time_us / episode_max_time_us

    Action Space
    ------------
    ``Discrete(MAX_QUEUE)``  — select index ``i`` from the current
    waiting queue.

    Action Masking
    --------------
    The environment exposes ``get_action_mask()`` which returns a
    boolean array of shape ``(MAX_QUEUE,)`` where ``True`` means the
    slot has a waiting packet.  The ``DQNAgent`` applies this mask by
    setting Q-values of ``False`` slots to ``-1e9`` before argmax, so
    an invalid slot is never selected during inference or exploration.

    If an invalid action is nonetheless received (e.g. from a policy
    that ignores the mask), the action is **clamped** to the last valid
    index and a configurable ``mask_violation_penalty`` is added to the
    reward.  This is documented explicitly rather than silently ignored.

    Reward
    ------
    ::

        R = w_deadline * deadline_bonus
          - w_latency  * latency_norm
          - w_queue    * queue_norm
          + w_miss     * deadline_penalty   (w_miss acts as negative weight)
          + mask_violation_penalty           (only if invalid action clamped)

    where:
        deadline_bonus   = +1.0 if deadline met, else 0.0
        deadline_penalty = −1.0 if deadline missed, else 0.0
        latency_norm     = latency_us / max_expected_latency_us  (clipped [0,1])
        queue_norm       = queueing_delay_us / max_expected_latency_us (clipped [0,1])

    All weights are configurable via constructor arguments so they can
    be tuned experimentally.  The chosen defaults are not claimed to be
    mathematically optimal.

    Episode
    -------
    One episode corresponds to one complete scheduling workload.
    The episode terminates (``terminated=True``) when all packets in
    the workload have been transmitted.

    Parameters
    ----------
    traffic_generator : TrafficGenerator
        Used to generate a fresh packet workload on each ``reset()``.
    link : NetworkLink
        Defines link bandwidth and propagation delay (shared with the
        existing ``EventDrivenSimulator`` experiments).
    w_deadline : float
        Weight for the deadline-met bonus.
    w_latency : float
        Weight for the normalised latency penalty.
    w_queue : float
        Weight for the normalised queueing-delay penalty.
    w_miss : float
        Additional penalty multiplier on deadline miss.
    max_deadline_us : float
        Normalisation constant for remaining-deadline feature.
    max_size_bytes : float
        Normalisation constant for packet-size feature.
    max_wait_us : float
        Normalisation constant for wait-time feature.
    max_expected_latency_us : float
        Normalisation constant for reward latency/queue terms.
    episode_max_time_us : float
        Normalisation constant for time_norm global feature.
    mask_violation_penalty : float
        One-off penalty applied when an out-of-range action is clamped.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        traffic_generator,
        link,
        # Reward weights (configurable)
        w_deadline: float = 2.0,
        w_latency: float = 1.0,
        w_queue: float = 0.5,
        w_miss: float = 3.0,
        # Observation normalisation constants
        max_deadline_us: float = 20_000.0,
        max_size_bytes: float = 2_000.0,
        max_wait_us: float = 10_000.0,
        max_expected_latency_us: float = 15_000.0,
        episode_max_time_us: float = 50_000.0,
        # Penalty for bypassing action masking
        mask_violation_penalty: float = -5.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.traffic_generator = traffic_generator
        self.link = link

        self.w_deadline = w_deadline
        self.w_latency = w_latency
        self.w_queue = w_queue
        self.w_miss = w_miss

        self.max_deadline_us = max_deadline_us
        self.max_size_bytes = max_size_bytes
        self.max_wait_us = max_wait_us
        self.max_expected_latency_us = max_expected_latency_us
        self.episode_max_time_us = episode_max_time_us
        self.mask_violation_penalty = mask_violation_penalty

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(MAX_QUEUE)

        # Episode state (initialised by reset())
        self._pending_events: List[Dict[str, Any]] = []
        self._event_index: int = 0
        self._waiting_queue: List[Dict[str, Any]] = []
        self._current_time_us: float = 0.0
        self._results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Reset the environment and generate a fresh workload.

        Returns
        -------
        obs : np.ndarray  shape (OBS_DIM,)
        info : dict
        """
        super().reset(seed=seed)

        # Generate a fresh workload from the traffic generator
        packets = self.traffic_generator.generate()

        # Sort by arrival time (mirrors EventDrivenSimulator behaviour)
        self._pending_events = sorted(
            packets, key=lambda p: p["arrival_time_us"]
        )
        self._event_index = 0
        self._current_time_us = 0.0
        self._waiting_queue = []
        self._results = []

        # Admit packets that arrive at or before t=0
        self._admit_arrived_packets()

        obs = self._build_observation()
        info = {"num_waiting": len(self._waiting_queue)}
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Execute one scheduling decision.

        Parameters
        ----------
        action : int
            Index into the current waiting queue.
            Valid range: [0, len(waiting_queue) - 1].

            If ``action >= len(waiting_queue)`` (invalid action, mask
            bypassed), the action is clamped to ``len(waiting_queue) - 1``
            and ``mask_violation_penalty`` is added to the reward.

        Returns
        -------
        obs          : np.ndarray shape (OBS_DIM,)
        reward       : float
        terminated   : bool   — True when all packets transmitted
        truncated    : bool   — always False (no step limit enforced here)
        info         : dict
        """
        assert len(self._waiting_queue) > 0, (
            "step() called with an empty waiting queue. "
            "Check that 'terminated' is True before calling step() again."
        )

        # ── Action masking enforcement ─────────────────────────────────
        extra_penalty = 0.0
        n_waiting = len(self._waiting_queue)

        if action >= n_waiting:
            # Invalid action — clamp and apply documented penalty
            action = n_waiting - 1
            extra_penalty = self.mask_violation_penalty

        # ── Select and remove packet ───────────────────────────────────
        packet = self._waiting_queue.pop(action)

        # ── Simulate transmission (same logic as EventDrivenSimulator) ─
        transmission_delay_us = self.link.transmission_delay_us(
            packet["size_bytes"]
        )
        propagation_delay_us = self.link.propagation_delay_us

        start_time_us = self._current_time_us
        completion_time_us = (
            start_time_us + transmission_delay_us + propagation_delay_us
        )

        latency_us = completion_time_us - packet["arrival_time_us"]
        queueing_delay_us = max(
            start_time_us - packet["arrival_time_us"], 0.0
        )
        deadline_us = packet["deadline_ms"] * 1_000
        deadline_met = latency_us <= deadline_us

        result: Dict[str, Any] = {
            "flow_id": packet["flow_id"],
            "priority": packet["priority"],
            "arrival_time_us": packet["arrival_time_us"],
            "start_time_us": start_time_us,
            "completion_time_us": completion_time_us,
            "latency_us": latency_us,
            "deadline_us": deadline_us,
            "deadline_met": deadline_met,
        }
        self._results.append(result)

        # ── Advance simulation clock ───────────────────────────────────
        self._current_time_us = completion_time_us

        # ── Admit newly arrived packets ────────────────────────────────
        self._admit_arrived_packets()

        # ── Reward ────────────────────────────────────────────────────
        reward = (
            self._compute_reward(latency_us, queueing_delay_us, deadline_met)
            + extra_penalty
        )

        # ── Termination ───────────────────────────────────────────────
        terminated = (
            len(self._waiting_queue) == 0
            and self._event_index >= len(self._pending_events)
        )
        truncated = False

        obs = self._build_observation()
        info = {
            "flow_id": packet["flow_id"],
            "latency_us": latency_us,
            "deadline_met": deadline_met,
            "num_waiting": len(self._waiting_queue),
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def get_action_mask(self) -> np.ndarray:
        """
        Return a boolean mask of shape ``(MAX_QUEUE,)``.

        ``True``  → slot has a waiting packet (valid action).
        ``False`` → slot is empty / beyond queue length (invalid action).

        The ``DQNAgent`` uses this mask to set invalid action Q-values
        to ``-1e9`` before computing ``argmax``, ensuring the agent
        never selects a non-existent queue slot.

        Returns
        -------
        np.ndarray  dtype=bool, shape=(MAX_QUEUE,)
        """
        mask = np.zeros(MAX_QUEUE, dtype=bool)
        n = min(len(self._waiting_queue), MAX_QUEUE)
        mask[:n] = True
        return mask

    # ------------------------------------------------------------------
    def get_episode_results(self) -> List[Dict[str, Any]]:
        """
        Return a copy of all per-packet results for the current episode.

        Each entry matches the result-dict format of ``EventDrivenSimulator``
        so it can be passed directly to ``NetworkAnalyzer.analyze()``.

        Returns
        -------
        list of dict
        """
        return list(self._results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _admit_arrived_packets(self) -> None:
        """
        Move all packets whose ``arrival_time_us <= current_time_us``
        from ``_pending_events`` to ``_waiting_queue``.

        If the queue is empty after admitting arrivals and there are
        still pending events, advance the clock to the next arrival and
        admit it (mirrors ``EventDrivenSimulator``'s idle-link jump).
        """
        while (
            self._event_index < len(self._pending_events)
            and self._pending_events[self._event_index]["arrival_time_us"]
            <= self._current_time_us
        ):
            self._waiting_queue.append(
                self._pending_events[self._event_index]
            )
            self._event_index += 1

        # If nothing is waiting and more packets are pending, jump clock
        if (
            len(self._waiting_queue) == 0
            and self._event_index < len(self._pending_events)
        ):
            next_arrival = self._pending_events[self._event_index][
                "arrival_time_us"
            ]
            self._current_time_us = next_arrival
            self._waiting_queue.append(
                self._pending_events[self._event_index]
            )
            self._event_index += 1

    # ------------------------------------------------------------------
    def _compute_reward(
        self,
        latency_us: float,
        queueing_delay_us: float,
        deadline_met: bool,
    ) -> float:
        """
        Compute the per-step reward.

        Formula
        -------
        R = w_deadline * deadline_bonus
          - w_latency  * latency_norm
          - w_queue    * queue_norm
          + w_miss     * deadline_penalty

        The weights are not claimed to be optimal; they are configurable
        and should be tuned experimentally.
        """
        deadline_bonus = 1.0 if deadline_met else 0.0
        deadline_penalty = -1.0 if not deadline_met else 0.0

        latency_norm = float(
            np.clip(latency_us / self.max_expected_latency_us, 0.0, 1.0)
        )
        queue_norm = float(
            np.clip(
                queueing_delay_us / self.max_expected_latency_us, 0.0, 1.0
            )
        )

        reward = (
            self.w_deadline * deadline_bonus
            - self.w_latency * latency_norm
            - self.w_queue * queue_norm
            + self.w_miss * deadline_penalty
        )
        return float(reward)

    # ------------------------------------------------------------------
    def _build_observation(self) -> np.ndarray:
        """
        Build the fixed-size observation vector from the current queue state.
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        for i, packet in enumerate(self._waiting_queue[:MAX_QUEUE]):
            base = i * FEATURES_PER_FLOW

            # priority_norm
            obs[base + 0] = PRIORITY_NORM.get(packet["priority"], 0.0)

            # remaining_deadline_norm — clipped to [0, 1]
            deadline_us = packet["deadline_ms"] * 1_000
            remaining = deadline_us - self._current_time_us
            obs[base + 1] = float(
                np.clip(remaining / self.max_deadline_us, 0.0, 1.0)
            )

            # size_norm
            obs[base + 2] = float(
                np.clip(
                    packet["size_bytes"] / self.max_size_bytes, 0.0, 1.0
                )
            )

            # wait_time_norm
            wait = self._current_time_us - packet["arrival_time_us"]
            obs[base + 3] = float(
                np.clip(wait / self.max_wait_us, 0.0, 1.0)
            )

        # Global features
        global_base = MAX_QUEUE * FEATURES_PER_FLOW
        obs[global_base + 0] = float(
            min(len(self._waiting_queue) / MAX_QUEUE, 1.0)
        )
        obs[global_base + 1] = float(
            np.clip(
                self._current_time_us / self.episode_max_time_us, 0.0, 1.0
            )
        )

        return obs
