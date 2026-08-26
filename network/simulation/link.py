class NetworkLink:
    def __init__(
        self,
        bandwidth_mbps,
        propagation_delay_us
    ):
        self.bandwidth_mbps = bandwidth_mbps
        self.propagation_delay_us = propagation_delay_us

    def transmission_delay_us(self, packet_size_bytes):
        packet_size_bits = packet_size_bytes * 8
        bandwidth_bits_per_second = self.bandwidth_mbps * 1_000_000

        delay_seconds = (
            packet_size_bits / bandwidth_bits_per_second
        )

        return delay_seconds * 1_000_000

    def total_delay_us(self, packet_size_bytes):
        transmission = self.transmission_delay_us(
            packet_size_bytes
        )

        return transmission + self.propagation_delay_us