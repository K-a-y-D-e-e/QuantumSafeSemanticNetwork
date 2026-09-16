import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SemanticCompressionEnv(gym.Env):

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
