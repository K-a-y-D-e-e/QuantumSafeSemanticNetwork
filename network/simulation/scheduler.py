class PriorityScheduler:
    PRIORITY_ORDER = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2
    }

    def schedule(self, packets):
        return sorted(
            packets,
            key=lambda packet: self.PRIORITY_ORDER[
                packet["priority"]
            ]
        )