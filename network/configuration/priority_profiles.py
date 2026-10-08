"""
Priority and per-priority traffic profiles.

This module defines how total offered traffic is divided between
HIGH-priority and NORMAL-priority traffic.

Each priority class has its own deadline condition.
"""


# ---------------------------------------------------------------------------
# Priority traffic profiles
# ---------------------------------------------------------------------------

PRIORITY_TRAFFIC_PROFILES = {
    "HIGH": {
        "priority": "HIGH",
        "deadline_class": "TIGHT",
        "deadline_ms": 10,
        "description": "Latency- and deadline-sensitive traffic",
    },

    "NORMAL": {
        "priority": "NORMAL",
        "deadline_class": "RELAXED",
        "deadline_ms": 50,
        "description": "Non-critical application traffic",
    },
}


# ---------------------------------------------------------------------------
# Priority-mix profiles
# ---------------------------------------------------------------------------

PRIORITY_MIX_PROFILES = {
    "HIGH_ONLY": {
        "high_ratio": 1.00,
        "normal_ratio": 0.00,
    },

    "NORMAL_ONLY": {
        "high_ratio": 0.00,
        "normal_ratio": 1.00,
    },

    "BALANCED": {
        "high_ratio": 0.50,
        "normal_ratio": 0.50,
    },

    "HIGH_DOMINANT": {
        "high_ratio": 0.70,
        "normal_ratio": 0.30,
    },

    "NORMAL_DOMINANT": {
        "high_ratio": 0.30,
        "normal_ratio": 0.70,
    },
}


# ---------------------------------------------------------------------------
# Profile access
# ---------------------------------------------------------------------------

def get_priority_traffic_profile(priority):
    """
    Return the traffic condition for one priority class.
    """

    if priority not in PRIORITY_TRAFFIC_PROFILES:
        raise ValueError(
            f"Unknown priority: {priority}"
        )

    return PRIORITY_TRAFFIC_PROFILES[priority].copy()


def get_priority_mix_profile(profile_name):
    """
    Return a priority-mix profile by name.
    """

    if profile_name not in PRIORITY_MIX_PROFILES:
        raise ValueError(
            f"Unknown priority-mix profile: {profile_name}"
        )

    return PRIORITY_MIX_PROFILES[profile_name].copy()


# ---------------------------------------------------------------------------
# Traffic-rate calculation
# ---------------------------------------------------------------------------

def calculate_priority_loads(
    offered_load_mbps,
    priority_mix,
):
    """
    Split total offered load between HIGH and NORMAL traffic.
    """

    if offered_load_mbps < 0:
        raise ValueError(
            "Offered load cannot be negative."
        )

    profile = get_priority_mix_profile(priority_mix)

    high_load_mbps = (
        offered_load_mbps * profile["high_ratio"]
    )

    normal_load_mbps = (
        offered_load_mbps * profile["normal_ratio"]
    )

    return {
        "high_ratio": profile["high_ratio"],
        "normal_ratio": profile["normal_ratio"],
        "high_load_mbps": high_load_mbps,
        "normal_load_mbps": normal_load_mbps,
        "total_load_mbps": (
            high_load_mbps + normal_load_mbps
        ),
    }


# ---------------------------------------------------------------------------
# Packet-rate calculation
# ---------------------------------------------------------------------------

def calculate_priority_packet_rates(
    high_load_mbps,
    normal_load_mbps,
    packet_size_bytes=1000,
):
    """
    Calculate packet generation rates for HIGH and NORMAL traffic.
    """

    if high_load_mbps < 0:
        raise ValueError(
            "HIGH traffic load cannot be negative."
        )

    if normal_load_mbps < 0:
        raise ValueError(
            "NORMAL traffic load cannot be negative."
        )

    if packet_size_bytes <= 0:
        raise ValueError(
            "Packet size must be greater than zero."
        )

    packet_size_bits = packet_size_bytes * 8

    high_packet_rate_pps = (
        high_load_mbps * 1_000_000
        / packet_size_bits
    )

    normal_packet_rate_pps = (
        normal_load_mbps * 1_000_000
        / packet_size_bits
    )

    return {
        "high_packet_rate_pps": high_packet_rate_pps,
        "normal_packet_rate_pps": normal_packet_rate_pps,
        "total_packet_rate_pps": (
            high_packet_rate_pps
            + normal_packet_rate_pps
        ),
    }


# ---------------------------------------------------------------------------
# Complete priority workload
# ---------------------------------------------------------------------------

def build_priority_workload(
    offered_load_mbps,
    priority_mix,
    packet_size_bytes=1000,
):
    """
    Build a complete HIGH/NORMAL priority workload.

    The total offered traffic rate is preserved.
    """

    loads = calculate_priority_loads(
        offered_load_mbps,
        priority_mix,
    )

    rates = calculate_priority_packet_rates(
        loads["high_load_mbps"],
        loads["normal_load_mbps"],
        packet_size_bytes,
    )

    high_profile = get_priority_traffic_profile(
        "HIGH"
    )

    normal_profile = get_priority_traffic_profile(
        "NORMAL"
    )

    return {
        "priority_mix": priority_mix,

        "high": {
            "priority": high_profile["priority"],
            "deadline_class": high_profile[
                "deadline_class"
            ],
            "deadline_ms": high_profile[
                "deadline_ms"
            ],
            "load_ratio": loads["high_ratio"],
            "load_mbps": loads["high_load_mbps"],
            "packet_rate_pps": rates[
                "high_packet_rate_pps"
            ],
        },

        "normal": {
            "priority": normal_profile["priority"],
            "deadline_class": normal_profile[
                "deadline_class"
            ],
            "deadline_ms": normal_profile[
                "deadline_ms"
            ],
            "load_ratio": loads["normal_ratio"],
            "load_mbps": loads["normal_load_mbps"],
            "packet_rate_pps": rates[
                "normal_packet_rate_pps"
            ],
        },

        "packet_size_bytes": packet_size_bytes,

        "total_load_mbps": loads[
            "total_load_mbps"
        ],

        "total_packet_rate_pps": rates[
            "total_packet_rate_pps"
        ],
    }
