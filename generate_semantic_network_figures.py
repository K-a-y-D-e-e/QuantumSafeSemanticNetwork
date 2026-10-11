import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Generate report-ready figures from semantic/network baseline CSV results."
    )
    parser.add_argument(
        "--results-dir",
        default="rl/results",
        help="Directory containing semantic_network_baselines_rawppo_summary.csv and uncertainty CSVs",
    )
    parser.add_argument(
        "--output-dir",
        default="rl/results/figures",
        help="Directory where figures and a copied metrics table will be saved",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = results_dir / "semantic_network_baselines_rawppo_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing input summary: {summary_path}")

    df = pd.read_csv(summary_path)
    required = {
        "condition", "mean_retained_fraction", "mean_reconstruction_mse",
        "mean_semantic_quality", "mean_packet_bytes", "mean_latency_us",
        "mean_queueing_us", "deadline_miss_rate", "deadline_misses",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Summary CSV is missing columns: {sorted(missing)}")

    order = [
        "Fixed_25pct", "Fixed_50pct", "Fixed_75pct", "Fixed_100pct",
        "PPO_Raw", "PPO_Orchestrated",
    ]
    df["condition"] = pd.Categorical(df["condition"], categories=order, ordered=True)
    df = df.sort_values("condition").copy()
    df["mean_latency_ms"] = df["mean_latency_us"] / 1000.0
    df["mean_queueing_ms"] = df["mean_queueing_us"] / 1000.0
    df["deadline_miss_percent"] = df["deadline_miss_rate"] * 100.0
    df.to_csv(output_dir / "final_metrics_table.csv", index=False)

    # Figure 1: semantic quality vs. modeled latency.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for _, r in df.iterrows():
        ax.scatter(r["mean_latency_ms"], r["mean_semantic_quality"], s=70)
        ax.annotate(str(r["condition"]), (r["mean_latency_ms"], r["mean_semantic_quality"]),
                    xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean modeled latency (ms)")
    ax.set_ylabel("Mean semantic quality (higher is better)")
    ax.set_title("Semantic Quality vs. Modeled Network Latency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_quality_vs_latency.png", dpi=220)
    plt.close(fig)

    # Figure 2: retention vs. packet size.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for _, r in df.iterrows():
        ax.scatter(r["mean_retained_fraction"] * 100, r["mean_packet_bytes"], s=70)
        ax.annotate(str(r["condition"]), (r["mean_retained_fraction"] * 100, r["mean_packet_bytes"]),
                    xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean retained latent fraction (%)")
    ax.set_ylabel("Mean modeled packet size (bytes)")
    ax.set_title("Semantic Retention vs. Modeled Packet Size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_retention_vs_packet_size.png", dpi=220)
    plt.close(fig)

    # Figure 3: raw PPO vs orchestrated PPO on four key metrics.
    pair = df[df["condition"].isin(["PPO_Raw", "PPO_Orchestrated"])].copy()
    pair = pair.set_index("condition").reindex(["PPO_Raw", "PPO_Orchestrated"])
    metrics = [
        ("mean_reconstruction_mse", "Reconstruction MSE", False),
        ("mean_semantic_quality", "Semantic quality", True),
        ("mean_latency_ms", "Mean latency (ms)", False),
        ("deadline_miss_percent", "Deadline misses (%)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (col, title, _) in zip(axes.flat, metrics):
        vals = pair[col].to_numpy()
        bars = ax.bar(["Raw PPO", "Orchestrated PPO"], vals)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.bar_label(bars, fmt="%.3f" if col in ("mean_reconstruction_mse", "mean_semantic_quality") else "%.2f")
        ax.tick_params(axis="x", labelrotation=10)
    fig.suptitle("Raw PPO vs. Orchestrated PPO", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_raw_vs_orchestrated.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: trade-off comparison across all six conditions.
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    panels = [
        ("mean_semantic_quality", "Semantic quality (↑)"),
        ("mean_latency_ms", "Mean latency (ms, ↓)"),
        ("deadline_miss_percent", "Deadline misses (%, ↓)"),
    ]
    labels = ["25% fixed", "50% fixed", "75% fixed", "100% fixed", "Raw PPO", "Orchestrated"]
    for ax, (col, title) in zip(axes, panels):
        vals = df[col].to_numpy()
        bars = ax.bar(labels, vals)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax.bar_label(bars, fmt="%.3f" if col == "mean_semantic_quality" else "%.2f", fontsize=8)
    fig.suptitle("Semantic Quality and Network Trade-offs", y=1.04)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_all_conditions_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # If uncertainty summary is present, copy it into the output bundle for handoff.
    uncertainty = results_dir / "semantic_network_baselines_rawppo_uncertainty_paired_differences.csv"
    if uncertainty.exists():
        pd.read_csv(uncertainty).to_csv(
            output_dir / "paired_uncertainty_results.csv", index=False
        )

    print("Generated report figures and metrics table:")
    for filename in [
        "fig1_quality_vs_latency.png",
        "fig2_retention_vs_packet_size.png",
        "fig3_raw_vs_orchestrated.png",
        "fig4_all_conditions_comparison.png",
        "final_metrics_table.csv",
    ]:
        print(output_dir / filename)
    if uncertainty.exists():
        print(output_dir / "paired_uncertainty_results.csv")
    print("\nNote: all network latency/queueing and packet sizes are modeled in the Python simulation.")
    print("These figures do not represent NS-3 validation or measured PQC overhead.")


if __name__ == "__main__":
    main()
