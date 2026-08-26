import time


class NetworkSimulator:
    def __init__(self):
        self.nodes = []
        self.flows = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_flow(self, flow):
        self.flows.append(flow)

    def transmit(self, packet):
        start_time = time.perf_counter_ns()

        source = next(
            node for node in self.nodes
            if node.node_id == packet["source"]
        )

        destination = next(
            node for node in self.nodes
            if node.node_id == packet["destination"]
        )

        source.send(packet)
        destination.receive(packet)

        end_time = time.perf_counter_ns()

        latency_us = (
            end_time - start_time
        ) / 1000

        return latency_us