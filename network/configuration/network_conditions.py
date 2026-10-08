"""
Network-condition profiles for deterministic network evaluation.

This module defines the controlled network conditions used by the
Network Layer experiments.

The profiles are configuration data only. They do not perform
simulation themselves.
"""


# ---------------------------------------------------------------------------
# Bandwidth profiles
# ---------------------------------------------------------------------------

BANDWIDTH_PROFILES = {
    "10M": {
        "bandwidth_mbps": 10,
    },
    "25M": {
        "bandwidth_mbps": 25,
    },
    "50M": {
        "bandwidth_mbps": 50,
    },
    "100M": {
        "bandwidth_mbps": 100,
    },
}


# ---------------------------------------------------------------------------
# Latency profiles
# ---------------------------------------------------------------------------

LATENCY_PROFILES = {
    "LOW": {
        "latency_ms": 1,
    },
    "MEDIUM": {
        "latency_ms": 10,
    },
    "HIGH": {
        "latency_ms": 50,
    },
}


# ---------------------------------------------------------------------------
# Packet-loss profiles
# ---------------------------------------------------------------------------

PACKET_LOSS_PROFILES = {
    "NONE": {
        "packet_loss_percent": 0,
    },
    "LOW": {
        "packet_loss_percent": 1,
    },
    "MEDIUM": {
        "packet_loss_percent": 3,
    },
    "HIGH": {
        "packet_loss_percent": 5,
    },
}


# ---------------------------------------------------------------------------
# Queue configuration
# ---------------------------------------------------------------------------

QUEUE_CONFIG = {
    "capacity_packets": 100,
}


# ---------------------------------------------------------------------------
# Network-condition helper functions
# ---------------------------------------------------------------------------

def get_bandwidth_profile(profile_name):
    """
    Return a bandwidth profile by name.
    """
    if profile_name not in BANDWIDTH_PROFILES:
        raise ValueError(
            f"Unknown bandwidth profile: {profile_name}"
        )

    return BANDWIDTH_PROFILES[profile_name].copy()


def get_latency_profile(profile_name):
    """
    Return a latency profile by name.
    """
    if profile_name not in LATENCY_PROFILES:
        raise ValueError(
            f"Unknown latency profile: {profile_name}"
        )

    return LATENCY_PROFILES[profile_name].copy()


def get_packet_loss_profile(profile_name):
    """
    Return a packet-loss profile by name.
    """
    if profile_name not in PACKET_LOSS_PROFILES:
        raise ValueError(
            f"Unknown packet-loss profile: {profile_name}"
        )

    return PACKET_LOSS_PROFILES[profile_name].copy()


def build_network_condition(
    bandwidth_profile,
    latency_profile,
    packet_loss_profile,
):
    """
    Build one complete network-condition configuration.

    Returns:
        dict: A deterministic network-condition configuration.
    """

    bandwidth = get_bandwidth_profile(bandwidth_profile)
    latency = get_latency_profile(latency_profile)
    packet_loss = get_packet_loss_profile(packet_loss_profile)

    return {
        "bandwidth_mbps": bandwidth["bandwidth_mbps"],
        "latency_ms": latency["latency_ms"],
        "packet_loss_percent": packet_loss["packet_loss_percent"],
        "queue_capacity_packets": QUEUE_CONFIG["capacity_packets"],
    }
