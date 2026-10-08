"""
Command-line validation for the network configuration matrix.

This module validates:

1. Scenario configuration structure
2. Network conditions
3. Traffic load
4. Priority and deadline conditions
5. Cryptographic configuration
6. Classical/PQC experimental pairing
7. Exported JSON artifacts
"""

import json
from pathlib import Path

from network.configuration.scenario_matrix import (
    generate_scenario_matrix,
)

from network.configuration.config_validator import (
    validate_scenario,
    validate_scenario_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIRECTORY = (
    PROJECT_ROOT / "configs" / "network"
)


def validate_classical_pqc_pairs(scenarios):
    """
    Validate that each Classical/PQC pair has identical
    experimental conditions apart from cryptographic settings.
    """

    if len(scenarios) % 2 != 0:
        raise ValueError(
            "Scenario count must be even for Classical/PQC pairing."
        )

    pair_count = 0

    for index in range(0, len(scenarios), 2):

        classical = scenarios[index]
        pqc = scenarios[index + 1]

        if classical["crypto"]["profile"] != "CLASSICAL":
            raise ValueError(
                f"Expected Classical scenario at index {index}: "
                f"{classical['scenario_id']}"
            )

        if pqc["crypto"]["profile"] != "PQC":
            raise ValueError(
                f"Expected PQC scenario at index {index + 1}: "
                f"{pqc['scenario_id']}"
            )

        if classical["network"] != pqc["network"]:
            raise ValueError(
                "Classical/PQC network conditions differ: "
                f"{classical['scenario_id']} / "
                f"{pqc['scenario_id']}"
            )

        if classical["traffic"] != pqc["traffic"]:
            raise ValueError(
                "Classical/PQC traffic conditions differ: "
                f"{classical['scenario_id']} / "
                f"{pqc['scenario_id']}"
            )

        pair_count += 1

    return pair_count


def validate_exported_json(scenarios):
    """
    Validate that every generated scenario has a matching JSON file.
    """

    missing_files = []

    for scenario in scenarios:

        scenario_id = scenario["scenario_id"]

        file_path = (
            CONFIG_DIRECTORY
            / f"{scenario_id}.json"
        )

        if not file_path.exists():
            missing_files.append(
                str(file_path)
            )
            continue

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            exported = json.load(file)

        if exported != scenario:
            raise ValueError(
                f"Exported JSON does not match generated "
                f"scenario: {scenario_id}"
            )

    matrix_file = (
        CONFIG_DIRECTORY
        / "scenario_matrix.json"
    )

    if not matrix_file.exists():
        missing_files.append(
            str(matrix_file)
        )
    else:
        with matrix_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            matrix = json.load(file)

        if matrix["scenario_count"] != len(scenarios):
            raise ValueError(
                "scenario_matrix.json count mismatch."
            )

        if matrix["scenarios"] != scenarios:
            raise ValueError(
                "scenario_matrix.json does not match "
                "generated scenario matrix."
            )

    if missing_files:
        raise ValueError(
            "Missing exported configuration files:\n"
            + "\n".join(missing_files)
        )

    return True


def run_validation():
    """
    Execute the complete configuration validation pipeline.
    """

    print("[1] Generating scenario matrix")

    scenarios = generate_scenario_matrix()

    print(
        "    PASS -",
        len(scenarios),
        "scenarios generated",
    )

    print("[2] Validating scenario schema and parameters")

    matrix_result = validate_scenario_matrix(
        scenarios
    )

    print(
        "    PASS -",
        matrix_result,
    )

    print("[3] Validating individual scenarios")

    for scenario in scenarios:
        validate_scenario(scenario)

    print(
        "    PASS - all",
        len(scenarios),
        "scenarios valid",
    )

    print("[4] Validating Classical/PQC pairs")

    pair_count = validate_classical_pqc_pairs(
        scenarios
    )

    print(
        "    PASS -",
        pair_count,
        "controlled pairs",
    )

    print("[5] Validating exported JSON artifacts")

    validate_exported_json(
        scenarios
    )

    print(
        "    PASS - exported JSON matches "
        "generated configurations"
    )

    print()
    print("Configuration validation completed successfully.")
    print("Status: VALID")

    return True


if __name__ == "__main__":
    run_validation()
