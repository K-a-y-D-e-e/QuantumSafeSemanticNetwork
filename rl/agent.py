from pathlib import Path
from typing import Optional, Union

from stable_baselines3 import PPO


DEFAULT_PPO_CHECKPOINT = "rl/checkpoints/ppo_semantic_agent"


class RLAgent:
    """
    Stable-Baselines3 PPO wrapper for semantic compression control.
    """

    def __init__(
        self,
        environment,
        seed: int = 42,
        learning_rate: float = 3e-4,
        verbose: int = 1,
    ):
        self.seed = seed
        self.model = PPO(
            "MlpPolicy",
            environment,
            verbose=verbose,
            seed=seed,
            learning_rate=learning_rate,
        )

    def train(self, timesteps: int = 10000):
        self.model.learn(total_timesteps=timesteps)

    def predict(self, state, deterministic: bool = True) -> int:
        action, _ = self.model.predict(state, deterministic=deterministic)
        return int(action)

    def save(self, path: str = DEFAULT_PPO_CHECKPOINT):
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(checkpoint))

    def load(
        self,
        path: str = DEFAULT_PPO_CHECKPOINT,
        environment=None,
    ):
        self.model = PPO.load(path, env=environment)
