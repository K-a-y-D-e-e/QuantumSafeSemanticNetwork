from nodes.node import NetworkNode
from flows.flow import NetworkFlow
from simulation.link import NetworkLink
from simulation.scheduler import PriorityScheduler
from simulation.deadline_scheduler import DeadlineAwareScheduler
from simulation.event_simulator import EventDrivenSimulator
from simulation.analyzer import NetworkAnalyzer


print("=" * 60)
print("SCHEDULER CONFLICT EXPERIMENT")
print("=" * 60)


# ------------------------------------------------------------
# Network
# ------------------------------------------------------------

node_a = NetworkNode("Node-A")
node_b = NetworkNode("Node-B")

link = NetworkLink(
    bandwidth_mbps=10,
    propagation_delay_us=100
)


# ------------------------------------------------------------
# Flows
# ------------------------------------------------------------

low_flow = NetworkFlow(
    flow_id=19,
    source=node_a,
    destination=node_b,
    priority="LOW",
    deadline_ms=2
)

medium_flow = NetworkFlow(
    flow_id=18,
    source=node_a,
    destination=node_b,
    priority="MEDIUM",
    deadline_ms=10
)

high_flow = NetworkFlow(
    flow_id=17,
    source=node_a,
    destination=node_b,
    priority="HIGH",
    deadline_ms=20
)


# ------------------------------------------------------------
# Traffic
# ------------------------------------------------------------

packets = [
    low_flow.create_packet(
        b"L" * 1000,
        arrival_time_us=0
    ),

    medium_flow.create_packet(
        b"M" * 1000,
        arrival_time_us=100
    ),

    high_flow.create_packet(
        b"H" * 1000,
        arrival_time_us=200
    )
]


print("\nTraffic configuration:")

for packet in sorted(
    packets,
    key=lambda packet: packet["arrival_time_us"]
):
    print(
        f"Flow {packet['flow_id']} | "
        f"Priority: {packet['priority']} | "
        f"Deadline: {packet['deadline_ms']} ms | "
        f"Arrival: {packet['arrival_time_us']} µs"
    )


# ------------------------------------------------------------
# Strict Priority Scheduler
# ------------------------------------------------------------

priority_scheduler = PriorityScheduler()

priority_simulator = EventDrivenSimulator(
    link=link,
    scheduler=priority_scheduler
)

priority_results = priority_simulator.simulate(
    packets
)

priority_analysis = NetworkAnalyzer.analyze(
    priority_results
)


# ------------------------------------------------------------
# Deadline-Aware / EDF Scheduler
# ------------------------------------------------------------

deadline_scheduler = DeadlineAwareScheduler()

deadline_simulator = EventDrivenSimulator(
    link=link,
    scheduler=deadline_scheduler
)

deadline_results = deadline_simulator.simulate(
    packets
)

deadline_analysis = NetworkAnalyzer.analyze(
    deadline_results
)


# ------------------------------------------------------------
# Strict Priority Results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STRICT PRIORITY RESULTS")
print("=" * 60)

print("\nTransmission order:")

for position, result in enumerate(
    priority_results,
    start=1
):
    print(
        f"{position}. "
        f"Flow {result['flow_id']} | "
        f"{result['priority']}"
    )

print("\nPer-flow results:")

for result in priority_results:

    print(
        f"\nFlow {result['flow_id']}"
    )

    print(
        f"Priority:          "
        f"{result['priority']}"
    )

    print(
        f"Latency:            "
        f"{result['latency_us']:.2f} µs"
    )

    print(
        f"Deadline:           "
        f"{result['deadline_us']:.2f} µs"
    )

    print(
        f"Deadline met:      "
        f"{result['deadline_met']}"
    )

print(
    f"\nDeadline misses: "
    f"{priority_analysis['deadline_misses']}/"
    f"{priority_analysis['total_flows']}"
)

print(
    f"Average latency: "
    f"{priority_analysis['average_latency_us']:.2f} µs"
)

print(
    f"Average jitter:   "
    f"{priority_analysis['jitter_us']:.2f} µs"
)


# ------------------------------------------------------------
# EDF Results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DEADLINE-AWARE / EDF RESULTS")
print("=" * 60)

print("\nTransmission order:")

for position, result in enumerate(
    deadline_results,
    start=1
):
    print(
        f"{position}. "
        f"Flow {result['flow_id']} | "
        f"{result['priority']}"
    )

print("\nPer-flow results:")

for result in deadline_results:

    print(
        f"\nFlow {result['flow_id']}"
    )

    print(
        f"Priority:          "
        f"{result['priority']}"
    )

    print(
        f"Latency:            "
        f"{result['latency_us']:.2f} µs"
    )

    print(
        f"Deadline:           "
        f"{result['deadline_us']:.2f} µs"
    )

    print(
        f"Deadline met:      "
        f"{result['deadline_met']}"
    )

print(
    f"\nDeadline misses: "
    f"{deadline_analysis['deadline_misses']}/"
    f"{deadline_analysis['total_flows']}"
)

print(
    f"Average latency: "
    f"{deadline_analysis['average_latency_us']:.2f} µs"
)

print(
    f"Average jitter:   "
    f"{deadline_analysis['jitter_us']:.2f} µs"
)


# ------------------------------------------------------------
# Comparison
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SCHEDULER COMPARISON")
print("=" * 60)

print(
    f"Strict Priority deadline misses: "
    f"{priority_analysis['deadline_misses']}"
)

print(
    f"EDF deadline misses:              "
    f"{deadline_analysis['deadline_misses']}"
)

print(
    f"\nStrict Priority average latency: "
    f"{priority_analysis['average_latency_us']:.2f} µs"
)

print(
    f"EDF average latency:              "
    f"{deadline_analysis['average_latency_us']:.2f} µs"
)

print(
    f"\nStrict Priority average jitter: "
    f"{priority_analysis['jitter_us']:.2f} µs"
)

print(
    f"EDF average jitter:              "
    f"{deadline_analysis['jitter_us']:.2f} µs"
)

print("=" * 60)
print("Experiment complete.")