class NetworkNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.sent_packets = []
        self.received_packets = []

    def send(self, packet):
        self.sent_packets.append(packet)

    def receive(self, packet):
        self.received_packets.append(packet)