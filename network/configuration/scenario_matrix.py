"""
Controlled deterministic scenario matrix generator.

This module generates reproducible network scenarios for evaluation
of network conditions, traffic load, priority behavior, deadlines,
packet loss, and classical-vs-PQC cryptographic overhead.

The matrix is intentionally controlled rather than a full Cartesian
product of every available parameter.
"""

from network.configuration.scenario_generator import (
    generate_scenario,
)


# ---------------------------------------------------------------------------
# Controlled experimental matrix
# ---------------------------------------------------------------------------

MATRIX_PROFILES = [
    {
        "name": "BASELINE",
        "bandwidth_profile": "10M",
        "latency_profile": "LOW",
        "packet_loss_profile": "NONE",
        "load_profile": "LOW",
        "priority_mix": "BALANCED",
        "deadline_profile": "RELAXED",
    },
    {
        "name": "MEDIUM_LOAD",
        "bandwidth_profile": "25M",
        "latency_profile": "MEDIUM",
        "packet_loss_profile": "LOW",
        "load_profile": "MEDIUM",
        "priority_mix": "BALANCED",
        "deadline_profile": "TIGHT",
    },
    {
        "name": "HIGH_LOAD",
        "bandwidth_profile": "50M",
        "latency_profile": "HIGH",
        "packet_loss_profile": "MEDIUM",
        "load_profile": "HIGH",
        "priority_mix": "HIGH_DOMINANT",
        "deadline_profile": "TIGHT",
    },
    {
        "name": "HEAVY_LOAD",
        "bandwidth_profile": "100M",
        "latency_profile": "HIGH",
        "packet_loss_profile": "HIGH",
        "load_profile": "HEAVY",
        "priority_mix": "HIGH_DOMINANT",
        "deadline_profile": "CRITICAL",
    },
]


# ---------------------------------------------------------------------------
# Cryptographic configurations
# ---------------------------------------------------------------------------

CRYPTO_PROFILES = [
    "CLASSICAL",
    "PQC",
]


# ---------------------------------------------------------------------------
# Traffic priority configurations
# ---------------------------------------------------------------------------

TRAFFIC_PRIORITY_PROFILE = "HIGH"


# ---------------------------------------------------------------------------
# Matrix generation
# ---------------------------------------------------------------------------

def generate_scenario_matrix():
    """
    Generate the controlled scenario matrix.

    Each network-condition profile is evaluated with both:

        CLASSICAL
        PQC

    Returns:
        list[dict]: Generated scenario configurations.
    """

    scenarios = []

    scenario_number = 1

    for matrix_profile in MATRIX_PROFILES:

        for crypto_profile in CRYPTO_PROFILES:

            scenario_id = (
                f"SC_{scenario_number:03d}_"
                f"{matrix_profile['name']}_"
                f"{crypto_profile}"
            )

            scenario = generate_scenario(
                scenario_id=scenario_id,

                bandwidth_profile=(
                    matrix_profile["bandwidth_profile"]
                ),

                latency_profile=(
                    matrix_profile["latency_profile"]
                ),

                packet_loss_profile=(
                    matrix_profile["packet_loss_profile"]
                ),

                load_profile=(
                    matrix_profile["load_profile"]
                ),

                priority_profile=(
                    TRAFFIC_PRIORITY_PROFILE
                ),

                deadline_profile=(
                    matrix_profile["deadline_profile"]
                ),

                crypto_profile=crypto_profile,

                priority_mix=(
                    matrix_profile["priority_mix"]
                ),
            )

            scenarios.append(scenario)

            scenario_number += 1

    return scenarios


# ---------------------------------------------------------------------------
# Matrix summary
# ---------------------------------------------------------------------------

def summarize_scenario_matrix(scenarios):
    """
    Generate a compact summary of the scenario matrix.
    """

    summary = {
        "scenario_count": len(scenarios),
        "classical_count": 0,
        "pqc_count": 0,
        "load_levels": {},
        "bandwidths_mbps": {},
    }

    for scenario in scenarios:

        crypto_profile = scenario["crypto"]["profile"]

        if crypto_profile == "CLASSICAL":
            summary["classical_count"] += 1

        elif crypto_profile == "PQC":
            summary["pqc_count"] += 1

        load_level = scenario["traffic"]["load_profile"]

        summary["load_levels"][load_level] = (
            summary["load_levels"].get(load_level, 0) + 1
        )

        bandwidth = scenario["network"]["bandwidth_mbps"]

        summary["bandwidths_mbps"][bandwidth] = (
            summary["bandwidths_mbps"].get(bandwidth, 0) + 1
        )

    return summary
