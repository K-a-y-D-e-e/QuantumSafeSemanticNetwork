from nodes.node import NetworkNode
from flows.flow import NetworkFlow
from simulation.simulator import NetworkSimulator


print("=" * 60)
print("DETERMINISTIC NETWORK FOUNDATION TEST")
print("=" * 60)


# Create nodes
node_a = NetworkNode("Node-A")
node_b = NetworkNode("Node-B")


# Create simulator
simulator = NetworkSimulator()

simulator.add_node(node_a)
simulator.add_node(node_b)


# Create deterministic flow
flow = NetworkFlow(
    flow_id=17,
    source=node_a,
    destination=node_b,
    priority="HIGH",
    deadline_ms=1
)

simulator.add_flow(flow)


# Create packet
packet = flow.create_packet(
    b"MOVE_FORWARD"
)


# Transmit packet
latency = simulator.transmit(packet)


print("\nFlow configuration:")
print(f"Flow ID:       {packet['flow_id']}")
print(f"Source:        {packet['source']}")
print(f"Destination:   {packet['destination']}")
print(f"Priority:      {packet['priority']}")
print(f"Deadline:      {packet['deadline_ms']} ms")
print(f"Payload:       {packet['payload']}")


print("\nTransmission:")
print(f"Latency:       {latency:.2f} µs")


print("\nDelivery:")
print(
    f"Packets received by Node-B: "
    f"{len(node_b.received_packets)}"
)


print("\nTest complete.")