"""
Deterministic network scenario generator.

This module combines network-condition, load, priority, deadline,
and cryptographic profiles into a single validated scenario.
"""

from network.configuration.network_conditions import (
    build_network_condition,
)

from network.configuration.load_profiles import (
    build_load_configuration,
)

from network.configuration.traffic_profiles import (
    build_traffic_profile,
)

from network.configuration.priority_profiles import (
    build_priority_workload,
)

from network.configuration.crypto_profiles import (
    get_crypto_profile,
    calculate_session_setup_overhead,
    get_application_data_overhead,
)


# ---------------------------------------------------------------------------
# Scenario defaults
# ---------------------------------------------------------------------------

DEFAULT_RANDOM_SEED = 42
DEFAULT_SIMULATION_DURATION_S = 10


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------

def generate_scenario(
    scenario_id,
    bandwidth_profile,
    latency_profile,
    packet_loss_profile,
    load_profile,
    priority_profile,
    deadline_profile,
    crypto_profile,
    priority_mix="BALANCED",
    random_seed=DEFAULT_RANDOM_SEED,
    simulation_duration_s=DEFAULT_SIMULATION_DURATION_S,
):
    """
    Generate one complete deterministic network scenario.
    """

    if not scenario_id:
        raise ValueError(
            "scenario_id cannot be empty."
        )

    if random_seed < 0:
        raise ValueError(
            "random_seed cannot be negative."
        )

    if simulation_duration_s <= 0:
        raise ValueError(
            "simulation_duration_s must be greater than zero."
        )

    # ---------------------------------------------------------------
    # Build network condition
    # ---------------------------------------------------------------

    network = build_network_condition(
        bandwidth_profile,
        latency_profile,
        packet_loss_profile,
    )

    # ---------------------------------------------------------------
    # Build total traffic load
    # ---------------------------------------------------------------

    load = build_load_configuration(
        network["bandwidth_mbps"],
        load_profile,
    )

    # ---------------------------------------------------------------
    # Build legacy/general traffic profile
    #
    # This remains in the scenario for compatibility with existing
    # configuration consumers.
    # ---------------------------------------------------------------

    traffic = build_traffic_profile(
        priority_profile,
        deadline_profile,
    )

    # ---------------------------------------------------------------
    # Build explicit HIGH/NORMAL workload
    # ---------------------------------------------------------------

    priority_workload = build_priority_workload(
        load["offered_load_mbps"],
        priority_mix,
        load["packet_size_bytes"],
    )

    # ---------------------------------------------------------------
    # Validate and obtain crypto profile
    # ---------------------------------------------------------------

    crypto = get_crypto_profile(
        crypto_profile
    )

    session_setup_overhead = (
        calculate_session_setup_overhead(
            crypto_profile
        )
    )

    application_data_overhead = (
        get_application_data_overhead(
            crypto_profile
        )
    )

    # ---------------------------------------------------------------
    # Construct final scenario
    # ---------------------------------------------------------------

    scenario = {
        "scenario_id": scenario_id,

        "network": {
            "bandwidth_profile": bandwidth_profile,
            "bandwidth_mbps": network[
                "bandwidth_mbps"
            ],

            "latency_profile": latency_profile,
            "latency_ms": network[
                "latency_ms"
            ],

            "packet_loss_profile": packet_loss_profile,
            "packet_loss_percent": network[
                "packet_loss_percent"
            ],

            "queue_capacity_packets": network[
                "queue_capacity_packets"
            ],
        },

        "traffic": {
            "load_profile": load_profile,
            "load_ratio": load[
                "load_ratio"
            ],

            "offered_load_mbps": load[
                "offered_load_mbps"
            ],

            "packet_size_bytes": load[
                "packet_size_bytes"
            ],

            "packet_rate_pps": load[
                "packet_rate_pps"
            ],

            # General/legacy traffic condition.
            "priority_profile": priority_profile,
            "deadline_class": traffic[
                "deadline_class"
            ],
            "deadline_ms": traffic[
                "deadline_ms"
            ],

            # Explicit mixed-priority workload.
            "priority_mix": priority_workload[
                "priority_mix"
            ],

            "high": priority_workload[
                "high"
            ],

            "normal": priority_workload[
                "normal"
            ],

            "total_load_mbps": priority_workload[
                "total_load_mbps"
            ],

            "total_packet_rate_pps": priority_workload[
                "total_packet_rate_pps"
            ],
        },

        "crypto": {
            "profile": crypto[
                "profile"
            ],

            "key_exchange": crypto[
                "key_exchange"
            ],

            "authentication": crypto[
                "authentication"
            ],

            "bulk_cipher": crypto[
                "bulk_cipher"
            ],

            "session_setup_overhead_bytes": (
                session_setup_overhead
            ),

            "application_data_overhead_bytes": (
                application_data_overhead
            ),

            "computational_overhead_us": (
                crypto[
                    "computational_overhead_us"
                ]
            ),
        },

        "evaluation": {
            "random_seed": random_seed,
            "simulation_duration_s": (
                simulation_duration_s
            ),
            "packet_generation_mode": (
                "DETERMINISTIC"
            ),
        },
    }

    return scenario
