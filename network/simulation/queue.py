class NetworkQueue:
    def __init__(self):
        self.packets = []

    def enqueue(self, packet):
        self.packets.append(packet)

    def dequeue(self):
        if not self.packets:
            return None

        return self.packets.pop(0)

    def size(self):
        return len(self.packets)