class NetworkFlow:
    def __init__(
        self,
        flow_id,
        source,
        destination,
        priority,
        deadline_ms
    ):
        self.flow_id = flow_id
        self.source = source
        self.destination = destination
        self.priority = priority
        self.deadline_ms = deadline_ms

    def create_packet(self, payload, arrival_time_us=0):
        return {
            "flow_id": self.flow_id,
            "source": self.source.node_id,
            "destination": self.destination.node_id,
            "priority": self.priority,
            "deadline_ms": self.deadline_ms,
            "payload": payload,
            "size_bytes": len(payload),
            "arrival_time_us": arrival_time_us
        }