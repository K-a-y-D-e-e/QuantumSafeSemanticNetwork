"""
Communication controller connecting rule-based orchestration with semantic RL.
"""

from orchestration.agent import AIOrchestrationAgent


COMPRESSION_LEVELS = [0.25, 0.50, 0.75, 1.00]


class CommunicationController:
    """
    Combines orchestration objective selection with PPO compression actions.

    Orchestration is rule-based; only the compression policy is RL-trained.
    """

    def __init__(self, rl_agent, orchestrator=None):
        self.orchestrator = orchestrator or AIOrchestrationAgent()
        self.rl_agent = rl_agent

    def select_compression(self, state, apply_objective_bias=True):
        """
        Select orchestration objective and compression level for ``state``.

        Parameters
        ----------
        state : array-like, shape (7,)
            SemanticCompressionEnv observation vector.
        apply_objective_bias : bool
            When True, adjust the RL action using rule-based objective bias.

        Returns
        -------
        objective : str
        compression : float
            Retained latent fraction in {0.25, 0.50, 0.75, 1.00}.
        action : int
            Final discrete action after optional objective bias.
        """
        task_criticality = float(state[0])
        network_load = float(state[2])
        latency = float(state[3])
        security_requirement = float(state[5])

        objective = self.orchestrator.decide_objective(
            task_criticality,
            network_load,
            latency,
            security_requirement,
        )

        raw_action = self.rl_agent.predict(state)
        action = raw_action

        if apply_objective_bias:
            action = self.orchestrator.bias_action_for_objective(
                raw_action,
                objective,
            )

        action = int(max(0, min(3, action)))
        compression = COMPRESSION_LEVELS[action]

        return objective, compression, action
