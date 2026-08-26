class DeadlineAwareScheduler:

    PRIORITY_WEIGHT = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2
    }

    def schedule(self, packets, current_time_us):
        if not packets:
            return []

        def scheduling_key(packet):
            deadline_us = packet["deadline_ms"] * 1000

            remaining_time = (
                deadline_us
                - current_time_us
            )

            return (
                remaining_time,
                self.PRIORITY_WEIGHT[
                    packet["priority"]
                ]
            )

        return sorted(
            packets,
            key=scheduling_key
        )