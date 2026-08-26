from nodes.node import NetworkNode
from flows.flow import NetworkFlow
from simulation.scheduler import PriorityScheduler


print("=" * 60)
print("PRIORITY SCHEDULING TEST")
print("=" * 60)


node_a = NetworkNode("Node-A")
node_b = NetworkNode("Node-B")


# Create three deterministic flows
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
    low_flow.create_packet(b"LOW_PRIORITY_DATA"),
    high_flow.create_packet(b"MOVE_FORWARD"),
    medium_flow.create_packet(b"SENSOR_UPDATE")
]


# Schedule packets
scheduler = PriorityScheduler()
scheduled_packets = scheduler.schedule(packets)


print("\nScheduled transmission order:")

for position, packet in enumerate(scheduled_packets, start=1):
    print(
        f"{position}. "
        f"Flow {packet['flow_id']} | "
        f"Priority: {packet['priority']} | "
        f"Deadline: {packet['deadline_ms']} ms"
    )