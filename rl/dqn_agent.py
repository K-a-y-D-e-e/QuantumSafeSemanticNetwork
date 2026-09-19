"""
DQN agent for network flow scheduling.

Components
----------
QNetwork      — 3-layer MLP mapping observations to Q-values
ReplayBuffer  — Circular experience replay buffer (stores next-state mask)
DQNAgent      — Epsilon-greedy DQN with action masking and target network

Action Masking Design
---------------------
At each scheduling step there are at most MAX_QUEUE=10 candidate actions,
but only ``len(waiting_queue)`` of them are valid (i.e. a packet exists at
that index).  Invalid actions are handled via **action masking**:

1. The environment exposes ``get_action_mask()`` returning a bool array
   of shape ``(MAX_QUEUE,)`` where ``True`` = valid slot.

2. ``DQNAgent.select_action()`` sets Q-values of invalid slots to
   ``-1e9`` (NEGATIVE_INF) before computing ``argmax``.  This
   guarantees the agent never selects a non-existent queue slot.

3. During training, the next-state mask is stored alongside each
   transition so that the target network's max-Q computation also
   respects future invalid actions.

4. If an invalid action somehow bypasses masking (e.g. from a random
   policy that ignores masks), ``NetworkSchedulingEnv.step()`` clamps
   the action and applies ``mask_violation_penalty``.  This fallback is
   documented explicitly rather than silently redirecting to a valid action.

Architecture
------------
Q-Network:
    obs_dim  → Linear(hidden) → ReLU
             → Linear(hidden) → ReLU
             → Linear(action_dim)

Default: hidden=128, obs_dim=42, action_dim=10.

Training
--------
- Experience replay with circular buffer (default capacity 10 000)
- Hard target-network update every ``target_update_freq`` gradient steps
- Huber loss (smooth L1) for robustness to outlier Q-value targets
- Gradient clipping (max norm 10.0) for training stability
- Epsilon-greedy exploration with per-episode multiplicative decay
"""

import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =============================================================================
# Q-Network
# =============================================================================


class QNetwork(nn.Module):
    """
    Three-layer fully-connected Q-network.

    Parameters
    ----------
    obs_dim    : int  — observation vector length
    action_dim : int  — number of discrete actions (= MAX_QUEUE)
    hidden     : int  — neurons per hidden layer
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Replay Buffer
# =============================================================================


class ReplayBuffer:
    """
    Circular experience replay buffer.

    Stores tuples of:
        (obs, action, reward, next_obs, done, next_mask)

    where ``next_mask`` is the action mask for the next state,
    used to mask invalid actions in the target-network computation.

    Parameters
    ----------
    capacity : int  — maximum number of transitions stored
    """

    def __init__(self, capacity: int):
        self._buf: deque = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        self._buf.append((obs, action, reward, next_obs, done, next_mask))

    def sample(
        self, batch_size: int
    ) -> Tuple[
        np.ndarray,  # obs
        np.ndarray,  # actions
        np.ndarray,  # rewards
        np.ndarray,  # next_obs
        np.ndarray,  # dones
        np.ndarray,  # next_masks
    ]:
        batch = random.sample(self._buf, batch_size)
        obs, actions, rewards, next_obs, dones, next_masks = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(next_masks, dtype=bool),
        )

    def __len__(self) -> int:
        return len(self._buf)


# =============================================================================
# DQN Agent
# =============================================================================


class DQNAgent:
    """
    Deep Q-Network agent with epsilon-greedy exploration and action masking.

    Parameters
    ----------
    obs_dim           : int    — observation vector length (42 for NetworkSchedulingEnv)
    action_dim        : int    — number of discrete actions (= MAX_QUEUE = 10)
    lr                : float  — Adam learning rate
    gamma             : float  — discount factor
    epsilon_start     : float  — initial exploration probability
    epsilon_min       : float  — minimum exploration probability
    epsilon_decay     : float  — multiplicative per-episode decay factor
    buffer_size       : int    — replay buffer capacity
    batch_size        : int    — mini-batch size for gradient updates
    target_update_freq: int    — gradient steps between hard target updates
    hidden            : int    — hidden layer width of Q-network
    device            : str or None — 'cpu' | 'cuda' | None (auto-detect)
    seed              : int    — for reproducibility
    """

    # Q-value assigned to invalid actions so they are never selected
    NEGATIVE_INF: float = -1e9

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 50,
        hidden: int = 128,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device selection
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Reproducibility
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Networks
        self.q_network = QNetwork(obs_dim, action_dim, hidden).to(
            self.device
        )
        self.target_network = QNetwork(obs_dim, action_dim, hidden).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self._grad_step_count: int = 0

    # ------------------------------------------------------------------
    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        training: bool = True,
    ) -> int:
        """
        Select an action using epsilon-greedy with action masking.

        Invalid actions (``action_mask[i] == False``) are never
        selected: during random exploration, only valid indices are
        sampled; during greedy exploitation, invalid Q-values are set
        to ``NEGATIVE_INF`` before ``argmax``.

        Parameters
        ----------
        obs         : np.ndarray  shape (obs_dim,)
        action_mask : np.ndarray  shape (action_dim,), dtype=bool
        training    : bool  — if False, always greedy (ε=0)

        Returns
        -------
        int  — chosen action index
        """
        valid_indices = np.where(action_mask)[0]
        assert len(valid_indices) > 0, (
            "select_action() called with no valid actions in mask."
        )

        # Epsilon-greedy exploration — only sample from valid actions
        if training and random.random() < self.epsilon:
            return int(np.random.choice(valid_indices))

        # Greedy: mask invalids → argmax
        obs_t = torch.tensor(
            obs[np.newaxis, :], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            q_values = self.q_network(obs_t).squeeze(0).cpu().numpy()

        masked_q = q_values.copy()
        masked_q[~action_mask] = self.NEGATIVE_INF

        return int(np.argmax(masked_q))

    # ------------------------------------------------------------------
    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        """Push one transition into the replay buffer."""
        self.buffer.push(obs, action, reward, next_obs, done, next_mask)

    # ------------------------------------------------------------------
    def learn(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient update on the Q-network.

        Returns the Huber loss value, or ``None`` if the buffer has
        fewer entries than ``batch_size``.
        """
        if len(self.buffer) < self.batch_size:
            return None

        obs_b, act_b, rew_b, nobs_b, done_b, nmask_b = (
            self.buffer.sample(self.batch_size)
        )

        obs_t  = torch.tensor(obs_b,  device=self.device)
        act_t  = torch.tensor(act_b,  device=self.device).unsqueeze(1)
        rew_t  = torch.tensor(rew_b,  device=self.device).unsqueeze(1)
        nobs_t = torch.tensor(nobs_b, device=self.device)
        done_t = torch.tensor(done_b, device=self.device).unsqueeze(1)
        nmask_t = torch.tensor(nmask_b, device=self.device)  # (B, action_dim)

        # Current Q-values for taken actions
        q_vals = self.q_network(obs_t).gather(1, act_t)

        # Target Q-values with invalid-action masking on next state
        with torch.no_grad():
            next_q = self.target_network(nobs_t)
            # Apply mask: set invalid next-state actions to -inf
            next_q = next_q.masked_fill(~nmask_t, self.NEGATIVE_INF)
            next_q_max = next_q.max(dim=1, keepdim=True)[0]
            # Bellman target
            targets = rew_t + self.gamma * next_q_max * (1.0 - done_t)

        # Huber loss
        loss = F.smooth_l1_loss(q_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for training stability
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        self._grad_step_count += 1

        # Hard target network update
        if self._grad_step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )

        return float(loss.item())

    # ------------------------------------------------------------------
    def decay_epsilon(self) -> None:
        """
        Decay epsilon by the configured multiplicative factor.
        Call once per episode after all gradient updates.
        """
        self.epsilon = max(
            self.epsilon_min, self.epsilon * self.epsilon_decay
        )

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """
        Persist model weights, optimiser state, epsilon, and step count.

        Parameters
        ----------
        path : str  — file path (e.g. 'rl/checkpoints/dqn_best.pt')
        """
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "grad_step_count": self._grad_step_count,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    # ------------------------------------------------------------------
    def load(self, path: str) -> None:
        """
        Load model weights and training state from a checkpoint.

        Parameters
        ----------
        path : str  — file path of the checkpoint
        """
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=True
        )
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self._grad_step_count = checkpoint.get("grad_step_count", 0)
        self.q_network.eval()
