"""
Traffic-load profiles for deterministic network evaluation.

Load is defined as a ratio of offered traffic rate to link capacity.

LOW     = 20%
MEDIUM  = 50%
HIGH    = 80%
HEAVY   = 95%
"""


# ---------------------------------------------------------------------------
# Load profiles
# ---------------------------------------------------------------------------

LOAD_PROFILES = {
    "LOW": {
        "load_ratio": 0.20,
    },
    "MEDIUM": {
        "load_ratio": 0.50,
    },
    "HIGH": {
        "load_ratio": 0.80,
    },
    "HEAVY": {
        "load_ratio": 0.95,
    },
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PACKET_SIZE_BYTES = 1000


# ---------------------------------------------------------------------------
# Profile access
# ---------------------------------------------------------------------------

def get_load_profile(profile_name):
    """
    Return a load profile by name.
    """

    if profile_name not in LOAD_PROFILES:
        raise ValueError(
            f"Unknown load profile: {profile_name}"
        )

    return LOAD_PROFILES[profile_name].copy()


# ---------------------------------------------------------------------------
# Traffic-rate calculations
# ---------------------------------------------------------------------------

def calculate_offered_load_mbps(bandwidth_mbps, load_profile):
    """
    Calculate offered traffic rate from link bandwidth and load ratio.

    Formula:

        offered_load = bandwidth x load_ratio

    Args:
        bandwidth_mbps: Link capacity in Mbps.
        load_profile: Load profile name.

    Returns:
        float: Offered traffic rate in Mbps.
    """

    if bandwidth_mbps <= 0:
        raise ValueError(
            "Bandwidth must be greater than zero."
        )

    profile = get_load_profile(load_profile)

    return bandwidth_mbps * profile["load_ratio"]


def calculate_packet_rate_pps(
    offered_load_mbps,
    packet_size_bytes=DEFAULT_PACKET_SIZE_BYTES,
):
    """
    Calculate packet generation rate.

    Formula:

        packets/sec =
            offered_rate_bits/sec /
            packet_size_bits

    Args:
        offered_load_mbps: Offered traffic rate in Mbps.
        packet_size_bytes: Packet size in bytes.

    Returns:
        float: Packet generation rate in packets per second.
    """

    if offered_load_mbps < 0:
        raise ValueError(
            "Offered load cannot be negative."
        )

    if packet_size_bytes <= 0:
        raise ValueError(
            "Packet size must be greater than zero."
        )

    offered_rate_bps = offered_load_mbps * 1_000_000
    packet_size_bits = packet_size_bytes * 8

    return offered_rate_bps / packet_size_bits


# ---------------------------------------------------------------------------
# Complete load configuration
# ---------------------------------------------------------------------------

def build_load_configuration(
    bandwidth_mbps,
    load_profile,
    packet_size_bytes=DEFAULT_PACKET_SIZE_BYTES,
):
    """
    Build a complete deterministic traffic-load configuration.
    """

    profile = get_load_profile(load_profile)

    offered_load_mbps = calculate_offered_load_mbps(
        bandwidth_mbps,
        load_profile,
    )

    packet_rate_pps = calculate_packet_rate_pps(
        offered_load_mbps,
        packet_size_bytes,
    )

    return {
        "load_level": load_profile,
        "load_ratio": profile["load_ratio"],
        "bandwidth_mbps": bandwidth_mbps,
        "offered_load_mbps": offered_load_mbps,
        "packet_size_bytes": packet_size_bytes,
        "packet_rate_pps": packet_rate_pps,
    }
