"""
Validation utilities for deterministic network configurations.
"""

import math


EXPECTED_BANDWIDTHS = {
    "10M": 10,
    "25M": 25,
    "50M": 50,
    "100M": 100,
}

EXPECTED_LATENCIES = {
    "LOW": 1,
    "MEDIUM": 10,
    "HIGH": 50,
}

EXPECTED_PACKET_LOSS = {
    "NONE": 0,
    "LOW": 1,
    "MEDIUM": 3,
    "HIGH": 5,
}

EXPECTED_LOAD_RATIOS = {
    "LOW": 0.20,
    "MEDIUM": 0.50,
    "HIGH": 0.80,
    "HEAVY": 0.95,
}

EXPECTED_CRYPTO = {
    "CLASSICAL": 128,
    "PQC": 7533,
}

EXPECTED_APPLICATION_OVERHEAD = 16

FLOAT_TOLERANCE = 1e-9


def assert_close(actual, expected, field_name):
    """Validate two numeric values within tolerance."""

    if not math.isclose(
        actual,
        expected,
        rel_tol=FLOAT_TOLERANCE,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError(
            f"{field_name} mismatch: "
            f"expected {expected}, got {actual}"
        )


def validate_scenario(scenario):
    """
    Validate one complete scenario.

    Raises:
        ValueError: If any configuration invariant is violated.

    Returns:
        True: If validation succeeds.
    """

    required_sections = {
        "scenario_id",
        "network",
        "traffic",
        "crypto",
        "evaluation",
    }

    missing = required_sections - set(scenario)

    if missing:
        raise ValueError(
            f"Missing scenario sections: {sorted(missing)}"
        )

    if not scenario["scenario_id"]:
        raise ValueError(
            "scenario_id cannot be empty."
        )

    network = scenario["network"]
    traffic = scenario["traffic"]
    crypto = scenario["crypto"]

    # ---------------------------------------------------------------
    # Network validation
    # ---------------------------------------------------------------

    bandwidth_profile = network["bandwidth_profile"]

    if bandwidth_profile not in EXPECTED_BANDWIDTHS:
        raise ValueError(
            f"Invalid bandwidth profile: "
            f"{bandwidth_profile}"
        )

    assert_close(
        network["bandwidth_mbps"],
        EXPECTED_BANDWIDTHS[bandwidth_profile],
        "bandwidth_mbps",
    )

    latency_profile = network["latency_profile"]

    if latency_profile not in EXPECTED_LATENCIES:
        raise ValueError(
            f"Invalid latency profile: "
            f"{latency_profile}"
        )

    assert_close(
        network["latency_ms"],
        EXPECTED_LATENCIES[latency_profile],
        "latency_ms",
    )

    packet_loss_profile = network[
        "packet_loss_profile"
    ]

    if packet_loss_profile not in EXPECTED_PACKET_LOSS:
        raise ValueError(
            f"Invalid packet-loss profile: "
            f"{packet_loss_profile}"
        )

    assert_close(
        network["packet_loss_percent"],
        EXPECTED_PACKET_LOSS[
            packet_loss_profile
        ],
        "packet_loss_percent",
    )

    # ---------------------------------------------------------------
    # Load validation
    # ---------------------------------------------------------------

    load_profile = traffic["load_profile"]

    if load_profile not in EXPECTED_LOAD_RATIOS:
        raise ValueError(
            f"Invalid load profile: {load_profile}"
        )

    expected_ratio = EXPECTED_LOAD_RATIOS[
        load_profile
    ]

    assert_close(
        traffic["load_ratio"],
        expected_ratio,
        "load_ratio",
    )

    bandwidth = network["bandwidth_mbps"]
    offered_load = traffic[
        "offered_load_mbps"
    ]

    expected_offered_load = (
        bandwidth * expected_ratio
    )

    assert_close(
        offered_load,
        expected_offered_load,
        "offered_load_mbps",
    )

    if offered_load > bandwidth:
        raise ValueError(
            "Offered load exceeds link bandwidth."
        )

    # ---------------------------------------------------------------
    # Priority workload validation
    # ---------------------------------------------------------------

    high = traffic["high"]
    normal = traffic["normal"]

    if high["priority"] != "HIGH":
        raise ValueError(
            "HIGH traffic has invalid priority."
        )

    if normal["priority"] != "NORMAL":
        raise ValueError(
            "NORMAL traffic has invalid priority."
        )

    if high["deadline_ms"] <= 0:
        raise ValueError(
            "HIGH deadline must be positive."
        )

    if normal["deadline_ms"] <= 0:
        raise ValueError(
            "NORMAL deadline must be positive."
        )

    priority_load_total = (
        high["load_mbps"]
        + normal["load_mbps"]
    )

    assert_close(
        priority_load_total,
        offered_load,
        "HIGH + NORMAL load",
    )

    priority_packet_total = (
        high["packet_rate_pps"]
        + normal["packet_rate_pps"]
    )

    assert_close(
        priority_packet_total,
        traffic["total_packet_rate_pps"],
        "HIGH + NORMAL packet rate",
    )

    assert_close(
        traffic["total_load_mbps"],
        offered_load,
        "total_load_mbps",
    )

    # ---------------------------------------------------------------
    # Crypto validation
    # ---------------------------------------------------------------

    crypto_profile = crypto["profile"]

    if crypto_profile not in EXPECTED_CRYPTO:
        raise ValueError(
            f"Invalid crypto profile: "
            f"{crypto_profile}"
        )

    assert_close(
        crypto["session_setup_overhead_bytes"],
        EXPECTED_CRYPTO[crypto_profile],
        "session_setup_overhead_bytes",
    )

    assert_close(
        crypto["application_data_overhead_bytes"],
        EXPECTED_APPLICATION_OVERHEAD,
        "application_data_overhead_bytes",
    )

    return True


def validate_scenario_matrix(scenarios):
    """
    Validate an entire scenario matrix.

    Returns:
        dict: Validation summary.
    """

    if not scenarios:
        raise ValueError(
            "Scenario matrix cannot be empty."
        )

    scenario_ids = [
        scenario["scenario_id"]
        for scenario in scenarios
    ]

    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError(
            "Duplicate scenario IDs detected."
        )

    classical_count = 0
    pqc_count = 0

    for scenario in scenarios:

        validate_scenario(scenario)

        if scenario["crypto"]["profile"] == "CLASSICAL":
            classical_count += 1

        elif scenario["crypto"]["profile"] == "PQC":
            pqc_count += 1

    return {
        "scenario_count": len(scenarios),
        "classical_count": classical_count,
        "pqc_count": pqc_count,
        "status": "VALID",
    }
