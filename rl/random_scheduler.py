"""
Random scheduling baseline for the network scheduler comparison experiment.

Selects a uniformly random packet from the waiting queue on each call.
Used as a lower-bound baseline alongside Priority, EDF, and DQN schedulers.

Design note
-----------
The ``RandomScheduler.schedule()`` signature matches the existing
``PriorityScheduler.schedule(packets)`` interface so that it can be
used directly inside ``EventDrivenSimulator`` without modification.
"""

import random
from typing import Any, Dict, List


class RandomScheduler:
    """
    Selects a uniformly random packet from the waiting queue.

    Parameters
    ----------
    seed : int
        Seed for the internal random number generator.
        Use the same ``eval_seed`` as the evaluation workload generator
        to ensure the random policy is itself reproducible.

    Example
    -------
    Compatible with ``EventDrivenSimulator``::

        from network.simulation.event_simulator import EventDrivenSimulator
        from rl.random_scheduler import RandomScheduler

        sched = RandomScheduler(seed=999)
        sim   = EventDrivenSimulator(link=link, scheduler=sched)
        results = sim.simulate(packets)
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def schedule(
        self, packets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Return the packet list in a uniformly random order.

        ``EventDrivenSimulator`` always takes ``scheduled[0]``, so
        placing a randomly chosen packet first is equivalent to random
        selection.

        Parameters
        ----------
        packets : list of dict
            The current waiting queue as packet dicts.

        Returns
        -------
        list of dict
            Packets in random order (first element is the chosen packet).
        """
        if not packets:
            return []

        shuffled = list(packets)
        self._rng.shuffle(shuffled)
        return shuffled
