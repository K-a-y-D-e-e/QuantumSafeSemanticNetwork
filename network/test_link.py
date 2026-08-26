from nodes.node import NetworkNode
from flows.flow import NetworkFlow
from simulation.link import NetworkLink


print("=" * 60)
print("NETWORK LINK DELAY TEST")
print("=" * 60)


# Create nodes
node_a = NetworkNode("Node-A")
node_b = NetworkNode("Node-B")


# Create a 10 Mbps link with 100 µs propagation delay
link = NetworkLink(
    bandwidth_mbps=10,
    propagation_delay_us=100
)


# Create a flow
flow = NetworkFlow(
    flow_id=17,
    source=node_a,
    destination=node_b,
    priority="HIGH",
    deadline_ms=1
)


# Create a 1000-byte packet
payload = b"A" * 1000

packet = flow.create_packet(payload)


# Calculate delays
transmission_delay = link.transmission_delay_us(
    packet["size_bytes"]
)

total_delay = link.total_delay_us(
    packet["size_bytes"]
)


print("\nLink configuration:")
print(f"Bandwidth:             {link.bandwidth_mbps} Mbps")
print(
    f"Propagation delay:     "
    f"{link.propagation_delay_us} µs"
)

print("\nPacket:")
print(f"Flow ID:               {packet['flow_id']}")
print(f"Packet size:           {packet['size_bytes']} bytes")
print(f"Priority:              {packet['priority']}")
print(f"Deadline:              {packet['deadline_ms']} ms")

print("\nDelay calculation:")
print(
    f"Transmission delay:    "
    f"{transmission_delay:.2f} µs"
)

print(
    f"Propagation delay:     "
    f"{link.propagation_delay_us:.2f} µs"
)

print(
    f"Total link delay:      "
    f"{total_delay:.2f} µs"
)

print("\nDeadline analysis:")

deadline_us = packet["deadline_ms"] * 1000

print(f"Deadline:              {deadline_us:.2f} µs")
print(f"Total delay:           {total_delay:.2f} µs")

if total_delay <= deadline_us:
    print("Deadline met:          True")
else:
    print("Deadline met:          False")