
import csv
import json
import subprocess
import sys
from pathlib import Path

from network.configuration.crypto_profiles import (
    calculate_session_setup_overhead,
    get_application_data_overhead,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "evaluation" / "pqc_overhead.py"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def test_crypto_profile_overheads():
    assert calculate_session_setup_overhead("CLASSICAL") == 128
    assert calculate_session_setup_overhead("PQC") == 7533

    assert get_application_data_overhead("CLASSICAL") == 16
    assert get_application_data_overhead("PQC") == 16


def test_evaluation_generates_results():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr

    csv_path = RESULTS_DIR / "pqc_overhead_results.csv"
    json_path = RESULTS_DIR / "pqc_overhead_results.json"

    assert csv_path.exists()
    assert json_path.exists()

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    with json_path.open("r", encoding="utf-8") as file:
        json_rows = json.load(file)

    assert len(rows) == 6
    assert len(json_rows) == 6

    assert {row["profile"] for row in rows} == {"CLASSICAL", "PQC"}
    assert {row["condition"] for row in rows} == {
        "LOW_BANDWIDTH",
        "MEDIUM_BANDWIDTH",
        "HIGH_BANDWIDTH",
    }


def test_application_packet_size_is_equal_for_both_profiles():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    with (RESULTS_DIR / "pqc_overhead_results.csv").open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    for condition in ("LOW_BANDWIDTH", "MEDIUM_BANDWIDTH", "HIGH_BANDWIDTH"):
        pair = [row for row in rows if row["condition"] == condition]
        assert len(pair) == 2
        assert {int(row["encrypted_packet_bytes"]) for row in pair} == {1016}
        assert (
            float(next(row for row in pair if row["profile"] == "CLASSICAL")[
                "application_link_delay_us"
            ])
            ==
            float(next(row for row in pair if row["profile"] == "PQC")[
                "application_link_delay_us"
            ])
        )
