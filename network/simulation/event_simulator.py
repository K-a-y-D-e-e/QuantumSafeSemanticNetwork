class EventDrivenSimulator:
    def __init__(self, link, scheduler):
        self.link = link
        self.scheduler = scheduler

    def simulate(self, packets):
        events = sorted(
            packets,
            key=lambda packet: packet["arrival_time_us"]
        )

        waiting_queue = []
        results = []

        current_time_us = 0
        event_index = 0

        while event_index < len(events) or waiting_queue:

            # Add all packets that have arrived
            # by the current simulation time.
            while (
                event_index < len(events)
                and events[event_index]["arrival_time_us"]
                <= current_time_us
            ):
                waiting_queue.append(
                    events[event_index]
                )
                event_index += 1

            # If no packets are waiting,
            # jump to the next packet arrival.
            if not waiting_queue:
                current_time_us = (
                    events[event_index]["arrival_time_us"]
                )
                continue

            # Ask the scheduler to select
            # the next packet.
            if (
                self.scheduler.__class__.__name__
                == "DeadlineAwareScheduler"
            ):
                scheduled = self.scheduler.schedule(
                    waiting_queue,
                    current_time_us
                )
            else:
                scheduled = self.scheduler.schedule(
                    waiting_queue
                )

            packet = scheduled[0]
            waiting_queue.remove(packet)

            # Calculate transmission delay.
            transmission_delay = (
                self.link.transmission_delay_us(
                    packet["size_bytes"]
                )
            )

            # Calculate propagation delay.
            propagation_delay = (
                self.link.propagation_delay_us
            )

            start_time_us = current_time_us

            # Calculate completion time.
            completion_time_us = (
                start_time_us
                + transmission_delay
                + propagation_delay
            )

            # End-to-end latency is measured
            # from packet arrival until completion.
            latency_us = (
                completion_time_us
                - packet["arrival_time_us"]
            )

            # Convert deadline from milliseconds
            # to microseconds.
            deadline_us = (
                packet["deadline_ms"] * 1000
            )

            deadline_met = (
                latency_us <= deadline_us
            )

            results.append({
                "flow_id": packet["flow_id"],
                "priority": packet["priority"],
                "arrival_time_us":
                    packet["arrival_time_us"],
                "start_time_us":
                    start_time_us,
                "completion_time_us":
                    completion_time_us,
                "latency_us":
                    latency_us,
                "deadline_us":
                    deadline_us,
                "deadline_met":
                    deadline_met
            })

            # The link remains occupied until
            # the packet has finished transmitting.
            current_time_us = completion_time_us

        return results