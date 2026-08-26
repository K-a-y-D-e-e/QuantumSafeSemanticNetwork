from nodes.node import NetworkNode
from flows.flow import NetworkFlow
from simulation.link import NetworkLink
from simulation.scheduler import PriorityScheduler
from simulation.event_simulator import EventDrivenSimulator


print("=" * 60)
print("EVENT-DRIVEN PRIORITY SCHEDULING TEST")
print("=" * 60)


# Nodes
node_a = NetworkNode("Node-A")
node_b = NetworkNode("Node-B")


# Network link
link = NetworkLink(
    bandwidth_mbps=10,
    propagation_delay_us=100
)


# Scheduler
scheduler = PriorityScheduler()


# Flows
low_flow = NetworkFlow(
    flow_id=19,
    source=node_a,
    destination=node_b,
    priority="LOW",
    deadline_ms=20
)

medium_flow = NetworkFlow(
    flow_id=18,
    source=node_a,
    destination=node_b,
    priority="MEDIUM",
    deadline_ms=5
)

high_flow = NetworkFlow(
    flow_id=17,
    source=node_a,
    destination=node_b,
    priority="HIGH",
    deadline_ms=1
)


# Packets arrive at different times
packets = [
    low_flow.create_packet(
        b"L" * 1000,
        arrival_time_us=0
    ),

    medium_flow.create_packet(
        b"M" * 1000,
        arrival_time_us=200
    ),

    high_flow.create_packet(
        b"H" * 1000,
        arrival_time_us=400
    )
]


print("\nPacket arrivals:")

for packet in sorted(
    packets,
    key=lambda packet: packet["arrival_time_us"]
):
    print(
        f"Flow {packet['flow_id']} | "
        f"Priority: {packet['priority']} | "
        f"Arrival: {packet['arrival_time_us']} µs"
    )


# Run simulation
simulator = EventDrivenSimulator(
    link=link,
    scheduler=scheduler
)

results = simulator.simulate(packets)


print("\nTransmission order:")

for position, result in enumerate(
    results,
    start=1
):
    print(
        f"{position}. "
        f"Flow {result['flow_id']} | "
        f"Priority: {result['priority']} | "
        f"Start: {result['start_time_us']:.2f} µs"
    )


print("\nDetailed results:")

for result in results:

    print(
        f"\nFlow {result['flow_id']}"
    )

    print(
        f"Priority:           "
        f"{result['priority']}"
    )

    print(
        f"Arrival time:       "
        f"{result['arrival_time_us']:.2f} µs"
    )

    print(
        f"Transmission start: "
        f"{result['start_time_us']:.2f} µs"
    )

    print(
        f"Completion time:    "
        f"{result['completion_time_us']:.2f} µs"
    )

    print(
        f"End-to-end latency: "
        f"{result['latency_us']:.2f} µs"
    )

    print(
        f"Deadline:            "
        f"{result['deadline_us']:.2f} µs"
    )

    print(
        f"Deadline met:       "
        f"{result['deadline_met']}"
    )


print("\nSimulation complete.")