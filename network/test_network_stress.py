from nodes.node import NetworkNode
from flows.flow import NetworkFlow
from simulation.link import NetworkLink
from simulation.scheduler import PriorityScheduler
from simulation.deadline_scheduler import DeadlineAwareScheduler
from simulation.event_simulator import EventDrivenSimulator
from simulation.analyzer import NetworkAnalyzer


print("=" * 70)
print("NETWORK STRESS TEST")
print("=" * 70)


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
# Flow configuration
# ------------------------------------------------------------

flow_configs = [
    (1, "LOW", 4, 0, 1000),
    (2, "HIGH", 2, 100, 1500),
    (3, "MEDIUM", 5, 150, 1200),
    (4, "HIGH", 3, 200, 1500),
    (5, "LOW", 8, 250, 800),
    (6, "MEDIUM", 4, 300, 1200),
    (7, "HIGH", 6, 350, 1000),
    (8, "LOW", 10, 400, 1500),
    (9, "MEDIUM", 3, 450, 800),
    (10, "HIGH", 7, 500, 1000),
]


packets = []


for flow_id, priority, deadline_ms, arrival_us, size_bytes in flow_configs:

    flow = NetworkFlow(
        flow_id=flow_id,
        source=node_a,
        destination=node_b,
        priority=priority,
        deadline_ms=deadline_ms
    )

    packet = flow.create_packet(
        b"X" * size_bytes,
        arrival_time_us=arrival_us
    )

    packets.append(packet)


# ------------------------------------------------------------
# Traffic configuration
# ------------------------------------------------------------

print("\nTraffic configuration:")

for packet in packets:
    print(
        f"Flow {packet['flow_id']:2d} | "
        f"Priority: {packet['priority']:6s} | "
        f"Deadline: {packet['deadline_ms']:2d} ms | "
        f"Arrival: {packet['arrival_time_us']:4d} µs | "
        f"Size: {packet['size_bytes']:4d} bytes"
    )


# ------------------------------------------------------------
# Strict Priority
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
# Deadline-Aware / EDF
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

print("\n" + "=" * 70)
print("STRICT PRIORITY RESULTS")
print("=" * 70)

print("\nTransmission order:")

for position, result in enumerate(
    priority_results,
    start=1
):
    print(
        f"{position:2d}. "
        f"Flow {result['flow_id']:2d} | "
        f"{result['priority']:6s} | "
        f"Latency: {result['latency_us']:8.2f} µs | "
        f"Deadline: {result['deadline_us']:8.2f} µs | "
        f"Met: {result['deadline_met']}"
    )


print("\nSummary:")

print(
    f"Average latency:       "
    f"{priority_analysis['average_latency_us']:.2f} µs"
)

print(
    f"Maximum latency:       "
    f"{priority_analysis['maximum_latency_us']:.2f} µs"
)

print(
    f"Minimum latency:       "
    f"{priority_analysis['minimum_latency_us']:.2f} µs"
)

print(
    f"Average queueing:      "
    f"{priority_analysis['average_queueing_delay_us']:.2f} µs"
)

print(
    f"Average jitter:        "
    f"{priority_analysis['jitter_us']:.2f} µs"
)

print(
    f"Deadline misses:       "
    f"{priority_analysis['deadline_misses']}/"
    f"{priority_analysis['total_flows']}"
)


# ------------------------------------------------------------
# EDF Results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DEADLINE-AWARE / EDF RESULTS")
print("=" * 70)

print("\nTransmission order:")

for position, result in enumerate(
    deadline_results,
    start=1
):
    print(
        f"{position:2d}. "
        f"Flow {result['flow_id']:2d} | "
        f"{result['priority']:6s} | "
        f"Latency: {result['latency_us']:8.2f} µs | "
        f"Deadline: {result['deadline_us']:8.2f} µs | "
        f"Met: {result['deadline_met']}"
    )


print("\nSummary:")

print(
    f"Average latency:       "
    f"{deadline_analysis['average_latency_us']:.2f} µs"
)

print(
    f"Maximum latency:       "
    f"{deadline_analysis['maximum_latency_us']:.2f} µs"
)

print(
    f"Minimum latency:       "
    f"{deadline_analysis['minimum_latency_us']:.2f} µs"
)

print(
    f"Average queueing:      "
    f"{deadline_analysis['average_queueing_delay_us']:.2f} µs"
)

print(
    f"Average jitter:        "
    f"{deadline_analysis['jitter_us']:.2f} µs"
)

print(
    f"Deadline misses:       "
    f"{deadline_analysis['deadline_misses']}/"
    f"{deadline_analysis['total_flows']}"
)


# ------------------------------------------------------------
# Comparison
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STRESS TEST COMPARISON")
print("=" * 70)

print(
    f"{'Metric':30s}"
    f"{'Strict Priority':>20s}"
    f"{'EDF':>15s}"
)

print("-" * 70)

print(
    f"{'Average latency (µs)':30s}"
    f"{priority_analysis['average_latency_us']:>20.2f}"
    f"{deadline_analysis['average_latency_us']:>15.2f}"
)

print(
    f"{'Maximum latency (µs)':30s}"
    f"{priority_analysis['maximum_latency_us']:>20.2f}"
    f"{deadline_analysis['maximum_latency_us']:>15.2f}"
)

print(
    f"{'Average queueing (µs)':30s}"
    f"{priority_analysis['average_queueing_delay_us']:>20.2f}"
    f"{deadline_analysis['average_queueing_delay_us']:>15.2f}"
)

print(
    f"{'Average jitter (µs)':30s}"
    f"{priority_analysis['jitter_us']:>20.2f}"
    f"{deadline_analysis['jitter_us']:>15.2f}"
)

print(
    f"{'Deadline misses':30s}"
    f"{priority_analysis['deadline_misses']:>20d}"
    f"{deadline_analysis['deadline_misses']:>15d}"
)

print("=" * 70)
print("Stress test complete.")