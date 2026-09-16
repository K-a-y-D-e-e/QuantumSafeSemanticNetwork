from stable_baselines3 import PPO


class RLAgent:

    def __init__(self, environment):
        self.model = PPO(
            "MlpPolicy",
            environment,
            verbose=1
        )

    def train(self, timesteps=10000):
        self.model.learn(total_timesteps=timesteps)

    def predict(self, state):
        action, _ = self.model.predict(
            state,
            deterministic=True
        )

        return int(action)

    def save(self, path="rl/ppo_semantic_agent"):
        self.model.save(path)

    def load(self, path="rl/ppo_semantic_agent"):
        self.model = PPO.load(path)
