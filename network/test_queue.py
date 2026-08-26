from nodes.node import NetworkNode
from flows.flow import NetworkFlow
from simulation.link import NetworkLink
from simulation.queue import NetworkQueue
from simulation.scheduler import PriorityScheduler


print("=" * 60)
print("NETWORK QUEUEING AND CONTENTION TEST")
print("=" * 60)


# Nodes
node_a = NetworkNode("Node-A")
node_b = NetworkNode("Node-B")


# 10 Mbps link
link = NetworkLink(
    bandwidth_mbps=10,
    propagation_delay_us=100
)


# Create flows
high_flow = NetworkFlow(
    flow_id=17,
    source=node_a,
    destination=node_b,
    priority="HIGH",
    deadline_ms=1
)

medium_flow = NetworkFlow(
    flow_id=18,
    source=node_a,
    destination=node_b,
    priority="MEDIUM",
    deadline_ms=5
)

low_flow = NetworkFlow(
    flow_id=19,
    source=node_a,
    destination=node_b,
    priority="LOW",
    deadline_ms=20
)


# Create packets
packets = [
    low_flow.create_packet(b"L" * 1000),
    medium_flow.create_packet(b"M" * 1000),
    high_flow.create_packet(b"H" * 1000)
]


# Schedule packets
scheduler = PriorityScheduler()
scheduled_packets = scheduler.schedule(packets)


# Queue packets
queue = NetworkQueue()

for packet in scheduled_packets:
    queue.enqueue(packet)


print("\nScheduled transmission order:")

for packet in scheduled_packets:
    print(
        f"Flow {packet['flow_id']} | "
        f"Priority: {packet['priority']} | "
        f"Deadline: {packet['deadline_ms']} ms"
    )


print("\nTransmission analysis:")

elapsed_time_us = 0

while queue.size() > 0:

    packet = queue.dequeue()

    transmission_delay = link.transmission_delay_us(
        packet["size_bytes"]
    )

    propagation_delay = link.propagation_delay_us

    queueing_delay = elapsed_time_us

    total_delay = (
        queueing_delay
        + transmission_delay
        + propagation_delay
    )

    deadline_us = packet["deadline_ms"] * 1000

    deadline_met = total_delay <= deadline_us

    print(
        f"\nFlow {packet['flow_id']}"
    )

    print(
        f"Priority:          {packet['priority']}"
    )

    print(
        f"Queueing delay:     {queueing_delay:.2f} µs"
    )

    print(
        f"Transmission delay: {transmission_delay:.2f} µs"
    )

    print(
        f"Propagation delay:  {propagation_delay:.2f} µs"
    )

    print(
        f"Total latency:      {total_delay:.2f} µs"
    )

    print(
        f"Deadline:           {deadline_us:.2f} µs"
    )

    print(
        f"Deadline met:       {deadline_met}"
    )

    elapsed_time_us += (
        transmission_delay
        + propagation_delay
    )


print("\nTest complete.")