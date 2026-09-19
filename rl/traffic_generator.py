"""
Traffic generator for the RL network scheduling environment.

Generates deterministic, reproducible packet workloads that are
compatible with the existing packet-dict format produced by
NetworkFlow.create_packet() and consumed by EventDrivenSimulator.

The generator uses Python's built-in ``random.Random`` with an
explicit seed so that training workloads (seed) and evaluation
workloads (eval_seed) are independent and reproducible.

Usage
-----
    from rl.traffic_generator import TrafficGenerator

    # Training workloads
    gen = TrafficGenerator(seed=42, num_flows=10)
    packets = gen.generate()          # one episode workload
    batch   = gen.generate_batch(5)  # five independent workloads

    # Evaluation workloads (different seed → unseen traffic)
    eval_gen = TrafficGenerator(seed=999, num_flows=10)
    eval_packets = eval_gen.generate()
"""

import random
from typing import Any, Dict, List

# Priority labels used by the existing scheduler and flow model
PRIORITIES: List[str] = ["HIGH", "MEDIUM", "LOW"]

# Normalised priority values used by the observation encoder
PRIORITY_NORM: Dict[str, float] = {
    "HIGH": 1.0,
    "MEDIUM": 0.5,
    "LOW": 0.0,
}


class TrafficGenerator:
    """
    Generates randomised packet workloads for RL training and evaluation.

    Each call to ``generate()`` produces a fresh list of packet dicts
    drawn from the generator's internal RNG.  The RNG state is *not*
    reset between calls, so successive ``generate()`` calls yield
    different (but reproducible) workloads.

    Call ``reset_seed()`` to start over from the original seed.

    Parameters
    ----------
    seed : int
        Seed for the internal random number generator.
    num_flows : int
        Number of packets (flows) per workload.
    min_size_bytes, max_size_bytes : int
        Packet payload size range in bytes.
    min_deadline_ms, max_deadline_ms : int
        Flow deadline range in milliseconds.
    max_arrival_spread_us : int
        Packets arrive uniformly at random in [0, max_arrival_spread_us] µs.
    """

    def __init__(
        self,
        seed: int = 42,
        num_flows: int = 10,
        min_size_bytes: int = 500,
        max_size_bytes: int = 2000,
        min_deadline_ms: int = 2,
        max_deadline_ms: int = 15,
        max_arrival_spread_us: int = 1000,
    ):
        self.seed = seed
        self.num_flows = num_flows
        self.min_size_bytes = min_size_bytes
        self.max_size_bytes = max_size_bytes
        self.min_deadline_ms = min_deadline_ms
        self.max_deadline_ms = max_deadline_ms
        self.max_arrival_spread_us = max_arrival_spread_us

        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    def reset_seed(self) -> None:
        """Reset the internal RNG to the original seed value."""
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------
    def generate(self) -> List[Dict[str, Any]]:
        """
        Generate one workload of ``num_flows`` packets.

        Returns
        -------
        list of dict
            Each dict matches the packet format produced by
            ``NetworkFlow.create_packet()``, i.e.:

            .. code-block:: python

                {
                    "flow_id":        int,
                    "source":         str,
                    "destination":    str,
                    "priority":       "HIGH" | "MEDIUM" | "LOW",
                    "deadline_ms":    int,
                    "payload":        bytes,
                    "size_bytes":     int,
                    "arrival_time_us": int,
                }
        """
        packets: List[Dict[str, Any]] = []

        for flow_id in range(1, self.num_flows + 1):
            priority = self._rng.choice(PRIORITIES)
            deadline_ms = self._rng.randint(
                self.min_deadline_ms, self.max_deadline_ms
            )
            size_bytes = self._rng.randint(
                self.min_size_bytes, self.max_size_bytes
            )
            arrival_time_us = self._rng.randint(
                0, self.max_arrival_spread_us
            )

            packets.append(
                {
                    "flow_id": flow_id,
                    "source": "Node-A",
                    "destination": "Node-B",
                    "priority": priority,
                    "deadline_ms": deadline_ms,
                    "payload": b"X" * size_bytes,
                    "size_bytes": size_bytes,
                    "arrival_time_us": arrival_time_us,
                }
            )

        return packets

    # ------------------------------------------------------------------
    def generate_batch(self, n: int) -> List[List[Dict[str, Any]]]:
        """
        Generate ``n`` independent workloads by calling ``generate()``
        ``n`` times in sequence (RNG state advances between calls).

        Parameters
        ----------
        n : int
            Number of workloads to generate.

        Returns
        -------
        list of list of dict
        """
        return [self.generate() for _ in range(n)]
