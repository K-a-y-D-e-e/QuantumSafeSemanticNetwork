class AIOrchestrationAgent:
    """
    Rule-based orchestration agent (not learned).

    Selects one of four communication objectives from network/task context.
    Objectives adjust reward weights during RL training and apply a small
    rule-based action bias at inference time in CommunicationController.
    """

    OBJECTIVES = [
        "LATENCY_CRITICAL",
        "QUALITY_CRITICAL",
        "BANDWIDTH_EFFICIENT",
        "BALANCED",
    ]

    # Reward weight multipliers per objective.
    # Keys: quality, latency, bandwidth, deadline, criticality
    REWARD_WEIGHTS = {
        "LATENCY_CRITICAL": {
            "quality": 0.6,
            "latency": 1.8,
            "bandwidth": 1.4,
            "deadline": 1.2,
            "criticality": 0.8,
        },
        "QUALITY_CRITICAL": {
            "quality": 2.0,
            "latency": 0.5,
            "bandwidth": 0.4,
            "deadline": 1.0,
            "criticality": 1.5,
        },
        "BANDWIDTH_EFFICIENT": {
            "quality": 0.5,
            "latency": 0.8,
            "bandwidth": 2.0,
            "deadline": 0.8,
            "criticality": 0.6,
        },
        "BALANCED": {
            "quality": 1.0,
            "latency": 1.0,
            "bandwidth": 1.0,
            "deadline": 1.0,
            "criticality": 1.0,
        },
    }

    def __init__(self):
        self.objectives = list(self.OBJECTIVES)

    def decide_objective(
        self,
        task_criticality,
        network_load,
        latency,
        security_requirement,
    ):
        # High-priority task + high latency
        if task_criticality > 0.8 and latency > 0.7:
            return "LATENCY_CRITICAL"

        # Important task
        if task_criticality > 0.8:
            return "QUALITY_CRITICAL"

        # Congested network
        if network_load > 0.7:
            return "BANDWIDTH_EFFICIENT"

        return "BALANCED"

    def get_reward_weights(self, objective):
        """Return reward weight multipliers for the given objective."""
        if objective not in self.REWARD_WEIGHTS:
            objective = "BALANCED"
        return dict(self.REWARD_WEIGHTS[objective])

    @staticmethod
    def bias_action_for_objective(action, objective):
        """
        Apply a transparent rule-based action adjustment at inference time.

        Actions map to retained latent fractions:
        0=25%, 1=50%, 2=75%, 3=100%.
        """
        action = int(max(0, min(3, action)))

        if objective == "BANDWIDTH_EFFICIENT":
            return min(action, 1)
        if objective == "QUALITY_CRITICAL":
            return max(action, 2)
        if objective == "LATENCY_CRITICAL":
            return min(action, 2)

        return action
