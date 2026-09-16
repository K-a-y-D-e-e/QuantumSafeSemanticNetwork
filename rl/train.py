from environment import SemanticCompressionEnv
from agent import RLAgent


def main():

    print("=" * 60)
    print("TRAINING RL AGENT")
    print("=" * 60)

    env = SemanticCompressionEnv()

    agent = RLAgent(env)

    agent.train(timesteps=10000)

    agent.save()

    print("=" * 60)
    print("RL TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
