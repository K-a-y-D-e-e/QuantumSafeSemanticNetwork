from nodes.node import NetworkNode
from flows.flow import NetworkFlow
from simulation.link import NetworkLink
from simulation.scheduler import PriorityScheduler
from simulation.event_simulator import EventDrivenSimulator
from simulation.analyzer import NetworkAnalyzer


print("=" * 60)
print("NETWORK LATENCY, JITTER AND DEADLINE ANALYSIS")
print("=" * 60)


# Nodes
node_a = NetworkNode("Node-A")
node_b = NetworkNode("Node-B")


# Link
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


# Dynamic traffic
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


# Run event-driven simulation
simulator = EventDrivenSimulator(
    link=link,
    scheduler=scheduler
)

results = simulator.simulate(packets)


# Analyze results
analysis = NetworkAnalyzer.analyze(results)


print("\nPer-flow results:")

for result in results:

    queueing_delay = (
        result["start_time_us"]
        - result["arrival_time_us"]
    )

    print(
        f"\nFlow {result['flow_id']}"
    )

    print(
        f"Priority:          "
        f"{result['priority']}"
    )

    print(
        f"Queueing delay:    "
        f"{queueing_delay:.2f} µs"
    )

    print(
        f"End-to-end latency:"
        f" {result['latency_us']:.2f} µs"
    )

    print(
        f"Deadline:           "
        f"{result['deadline_us']:.2f} µs"
    )

    print(
        f"Deadline met:      "
        f"{result['deadline_met']}"
    )


print("\n" + "=" * 60)
print("NETWORK PERFORMANCE SUMMARY")
print("=" * 60)

print(
    f"Average latency:        "
    f"{analysis['average_latency_us']:.2f} µs"
)

print(
    f"Minimum latency:        "
    f"{analysis['minimum_latency_us']:.2f} µs"
)

print(
    f"Maximum latency:        "
    f"{analysis['maximum_latency_us']:.2f} µs"
)

print(
    f"Average queueing delay:  "
    f"{analysis['average_queueing_delay_us']:.2f} µs"
)

print(
    f"Average jitter:          "
    f"{analysis['jitter_us']:.2f} µs"
)

print(
    f"Deadline misses:         "
    f"{analysis['deadline_misses']}"
    f"/{analysis['total_flows']}"
)

print("=" * 60)
print("Analysis complete.")