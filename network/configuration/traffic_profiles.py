"""
Traffic priority and deadline profiles.

These profiles define the characteristics of traffic used in
deterministic network-condition and workload evaluation.
"""


# ---------------------------------------------------------------------------
# Traffic constants
# ---------------------------------------------------------------------------

DEFAULT_PACKET_SIZE_BYTES = 1000


# ---------------------------------------------------------------------------
# Priority profiles
# ---------------------------------------------------------------------------

PRIORITY_PROFILES = {
    "HIGH": {
        "priority": "HIGH",
        "description": "Latency- and deadline-sensitive traffic",
    },
    "NORMAL": {
        "priority": "NORMAL",
        "description": "Non-critical application traffic",
    },
}


# ---------------------------------------------------------------------------
# Deadline profiles
# ---------------------------------------------------------------------------

DEADLINE_PROFILES = {
    "CRITICAL": {
        "deadline_ms": 5,
    },
    "TIGHT": {
        "deadline_ms": 10,
    },
    "RELAXED": {
        "deadline_ms": 50,
    },
}


# ---------------------------------------------------------------------------
# Profile access functions
# ---------------------------------------------------------------------------

def get_priority_profile(profile_name):
    """
    Return a priority profile by name.
    """

    if profile_name not in PRIORITY_PROFILES:
        raise ValueError(
            f"Unknown priority profile: {profile_name}"
        )

    return PRIORITY_PROFILES[profile_name].copy()


def get_deadline_profile(profile_name):
    """
    Return a deadline profile by name.
    """

    if profile_name not in DEADLINE_PROFILES:
        raise ValueError(
            f"Unknown deadline profile: {profile_name}"
        )

    return DEADLINE_PROFILES[profile_name].copy()


# ---------------------------------------------------------------------------
# Traffic-profile builder
# ---------------------------------------------------------------------------

def build_traffic_profile(
    priority_profile,
    deadline_profile,
    packet_size_bytes=DEFAULT_PACKET_SIZE_BYTES,
):
    """
    Build one complete traffic profile.

    Args:
        priority_profile: HIGH or NORMAL.
        deadline_profile: CRITICAL, TIGHT, or RELAXED.
        packet_size_bytes: Packet size in bytes.

    Returns:
        dict: Complete traffic profile.
    """

    if packet_size_bytes <= 0:
        raise ValueError(
            "Packet size must be greater than zero."
        )

    priority = get_priority_profile(priority_profile)
    deadline = get_deadline_profile(deadline_profile)

    return {
        "priority": priority["priority"],
        "deadline_class": deadline_profile,
        "deadline_ms": deadline["deadline_ms"],
        "packet_size_bytes": packet_size_bytes,
    }
