
"""
Isolated Classical vs PQC communication-overhead evaluation.

This is a model-based evaluation, not an NS-3 experiment.
Configured cryptographic processing values are not re-measured here.
"""

import csv
import json
import sys
from pathlib import Path

# Allow running this script directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from network.configuration.crypto_profiles import (
    CRYPTO_PROFILES,
    calculate_session_setup_overhead,
    get_application_data_overhead,
)
from network.simulation.link import NetworkLink


OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"

# Illustrative link conditions; these are explicit test assumptions,
# not claims about a real network.
LINK_CONDITIONS = [
    {"name": "LOW_BANDWIDTH", "bandwidth_mbps": 1, "propagation_delay_us": 100},
    {"name": "MEDIUM_BANDWIDTH", "bandwidth_mbps": 10, "propagation_delay_us": 100},
    {"name": "HIGH_BANDWIDTH", "bandwidth_mbps": 100, "propagation_delay_us": 100},
]

APPLICATION_PAYLOAD_BYTES = 1000
DEADLINE_US = 2000


def evaluate_profile(profile_name, condition):
    """Calculate modeled size and delay metrics for one profile/link pair."""
    profile = CRYPTO_PROFILES[profile_name]
    link = NetworkLink(
        bandwidth_mbps=condition["bandwidth_mbps"],
        propagation_delay_us=condition["propagation_delay_us"],
    )

    setup_bytes = calculate_session_setup_overhead(profile_name)
    tag_bytes = get_application_data_overhead(profile_name)

    # Model the setup material as one aggregate transmission for comparison.
    # This is not a complete handshake packetization or protocol simulation.
    setup_transmission_us = link.transmission_delay_us(setup_bytes)
    setup_link_delay_us = link.total_delay_us(setup_bytes)

    # Both profiles use AES-GCM, so the encrypted application packet
    # includes the same 16-byte authentication tag.
    encrypted_packet_bytes = APPLICATION_PAYLOAD_BYTES + tag_bytes
    application_transmission_us = link.transmission_delay_us(
        encrypted_packet_bytes
    )
    application_link_delay_us = link.total_delay_us(
        encrypted_packet_bytes
    )

    crypto_costs = profile["computational_overhead_us"]

    # Report configured costs separately; do not silently add CPU time
    # to link transmission time.
    return {
        "profile": profile_name,
        "condition": condition["name"],
        "bandwidth_mbps": condition["bandwidth_mbps"],
        "propagation_delay_us": condition["propagation_delay_us"],
        "setup_material_bytes": setup_bytes,
        "setup_transmission_us": round(setup_transmission_us, 4),
        "setup_link_delay_us": round(setup_link_delay_us, 4),
        "application_payload_bytes": APPLICATION_PAYLOAD_BYTES,
        "application_tag_bytes": tag_bytes,
        "encrypted_packet_bytes": encrypted_packet_bytes,
        "application_transmission_us": round(application_transmission_us, 4),
        "application_link_delay_us": round(application_link_delay_us, 4),
        "configured_key_exchange_us": crypto_costs["key_exchange"],
        "configured_signing_us": crypto_costs["signing"],
        "configured_verification_us": crypto_costs["verification"],
        "application_deadline_us": DEADLINE_US,
        "application_deadline_met": application_link_delay_us <= DEADLINE_US,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for condition in LINK_CONDITIONS:
        for profile_name in ("CLASSICAL", "PQC"):
            rows.append(evaluate_profile(profile_name, condition))

    csv_path = OUTPUT_DIR / "pqc_overhead_results.csv"
    json_path = OUTPUT_DIR / "pqc_overhead_results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)

    print("CLASSICAL vs PQC — modeled communication overhead")
    print("=" * 58)

    for row in rows:
        print(
            f'{row["condition"]:16} | {row["profile"]:9} | '
            f'setup={row["setup_material_bytes"]:5} B | '
            f'setup link delay={row["setup_link_delay_us"]:10.2f} us | '
            f'application={row["encrypted_packet_bytes"]:4} B | '
            f'application link delay={row["application_link_delay_us"]:8.2f} us | '
            f'deadline met={row["application_deadline_met"]}'
        )

    print("\nConfigured crypto processing costs are reported separately.")
    print("This does not measure live cryptographic execution or NS-3 performance.")
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
