"""
Scenario configuration exporter.

Generates JSON configuration artifacts from the controlled
network scenario matrix.

Output:

    configs/
    network/
        SC_001_....json
        ...
        scenario_matrix.json
"""

import json
from pathlib import Path

from network.configuration.scenario_matrix import (
    generate_scenario_matrix,
)


# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "configs" / "network"
)


# ---------------------------------------------------------------------------
# JSON writer
# ---------------------------------------------------------------------------

def write_json_file(file_path, data):
    """
    Write a Python object to a formatted JSON file.
    """

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            sort_keys=False,
        )

        file.write("\n")


# ---------------------------------------------------------------------------
# Scenario export
# ---------------------------------------------------------------------------

def export_scenarios(output_directory=OUTPUT_DIRECTORY):
    """
    Generate and export all controlled network scenarios.

    Returns:
        dict: Export summary.
    """

    scenarios = generate_scenario_matrix()

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    exported_files = []

    # ---------------------------------------------------------------
    # Export individual scenario files
    # ---------------------------------------------------------------

    for scenario in scenarios:

        scenario_id = scenario["scenario_id"]

        file_path = (
            output_directory
            / f"{scenario_id}.json"
        )

        write_json_file(
            file_path,
            scenario,
        )

        exported_files.append(
            file_path
        )

    # ---------------------------------------------------------------
    # Export complete scenario matrix
    # ---------------------------------------------------------------

    matrix_file = (
        output_directory
        / "scenario_matrix.json"
    )

    write_json_file(
        matrix_file,
        {
            "scenario_count": len(scenarios),
            "scenarios": scenarios,
        },
    )

    exported_files.append(
        matrix_file
    )

    # ---------------------------------------------------------------
    # Return export summary
    # ---------------------------------------------------------------

    return {
        "scenario_count": len(scenarios),
        "output_directory": str(output_directory),
        "files": [
            str(path)
            for path in exported_files
        ],
    }


# ---------------------------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    summary = export_scenarios()

    print("Network scenario export completed.")
    print(
        "Scenario count:",
        summary["scenario_count"],
    )
    print(
        "Output directory:",
        summary["output_directory"],
    )

    print("Generated files:")

    for file_path in summary["files"]:
        print(
            " -",
            file_path,
        )
