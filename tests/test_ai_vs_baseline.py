"""
Unit and integration tests for AI-enabled vs conventional baseline evaluation.
Verifies deliverables, schema consistency, trade-off sanity, and NS-3 export format.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

RUN_DIR = Path("rl/results/ai_vs_baseline")
PLOTS_DIR = RUN_DIR / "plots"


def test_deliverable_files_exist():
    required = [
        "policy_comparison.csv",
        "combined_results.csv",
        "semantic_results.csv",
        "scheduling_results.csv",
        "experiment_config.json",
        "report.md",
        "ns3_workload_A.csv",
        "ns3_workload_B.csv",
    ]
    for fname in required:
        fpath = RUN_DIR / fname
        assert fpath.exists(), f"Missing required file: {fpath}"
        assert fpath.stat().st_size > 0, f"File is empty: {fpath}"


def test_plots_exist():
    required_plots = [
        "quality_vs_latency.png",
        "deadline_miss_rate.png",
        "semantic_quality_comparison.png",
        "latency_comparison.png",
        "payload_vs_quality.png",
    ]
    for pname in required_plots:
        ppath = PLOTS_DIR / pname
        assert ppath.exists(), f"Missing plot: {ppath}"
        assert ppath.stat().st_size > 1000, f"Plot empty or corrupted: {ppath}"


def test_policy_comparison_schema():
    df = pd.read_csv(RUN_DIR / "policy_comparison.csv")
    assert len(df) == 7, f"Expected 7 configurations, got {len(df)}"
    expected_configs = ["A", "B", "C", "D", "E", "F", "G"]
    assert list(df["config"]) == expected_configs

    # Verify quality bounded in [0, 1]
    assert (df["sem_quality_mean"] >= 0.0).all()
    assert (df["sem_quality_mean"] <= 1.0).all()

    # Verify deadline satisfaction rates in [0, 1]
    assert (df["net_deadline_miss_rate"] >= 0.0).all()
    assert (df["net_deadline_miss_rate"] <= 1.0).all()
    assert (df["net_deadline_sat_rate"] >= 0.0).all()
    assert (df["net_deadline_sat_rate"] <= 1.0).all()


def test_ns3_workload_exports():
    for cfg in ["A", "B"]:
        df = pd.read_csv(RUN_DIR / f"ns3_workload_{cfg}.csv")
        assert len(df) > 0, f"ns3_workload_{cfg}.csv is empty"
        assert "avg_latency_us" in df.columns
        assert "deadline_miss_rate" in df.columns
        assert "avg_payload_bytes" in df.columns
        assert "sem_quality" in df.columns
        assert (df["note"].str.contains("NOT NS-3 measurement")).all()
