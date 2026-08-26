class TrafficSimulator:
    def __init__(self, link, scheduler):
        self.link = link
        self.scheduler = scheduler

    def simulate(self, packets):
        packets = sorted(
            packets,
            key=lambda packet: packet["arrival_time_us"]
        )

        current_time_us = 0
        results = []

        waiting_packets = []

        for packet in packets:

            arrival_time = packet["arrival_time_us"]

            if arrival_time > current_time_us:
                current_time_us = arrival_time

            waiting_packets.append(packet)

            scheduled_packets = self.scheduler.schedule(
                waiting_packets
            )

            packet = scheduled_packets[0]
            waiting_packets.remove(packet)

            transmission_delay = (
                self.link.transmission_delay_us(
                    packet["size_bytes"]
                )
            )

            propagation_delay = (
                self.link.propagation_delay_us
            )

            start_time_us = current_time_us

            completion_time_us = (
                start_time_us
                + transmission_delay
                + propagation_delay
            )

            latency_us = (
                completion_time_us
                - packet["arrival_time_us"]
            )

            deadline_us = (
                packet["deadline_ms"] * 1000
            )

            deadline_met = latency_us <= deadline_us

            results.append({
                "flow_id": packet["flow_id"],
                "priority": packet["priority"],
                "arrival_time_us": packet["arrival_time_us"],
                "start_time_us": start_time_us,
                "completion_time_us": completion_time_us,
                "latency_us": latency_us,
                "deadline_us": deadline_us,
                "deadline_met": deadline_met
            })

            current_time_us = completion_time_us

        return results