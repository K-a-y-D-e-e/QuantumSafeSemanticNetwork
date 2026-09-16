class AIOrchestrationAgent:

    def __init__(self):

        self.objectives = [
            "LATENCY_CRITICAL",
            "QUALITY_CRITICAL",
            "BANDWIDTH_EFFICIENT",
            "BALANCED"
        ]

    def decide_objective(
        self,
        task_criticality,
        network_load,
        latency,
        security_requirement
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
