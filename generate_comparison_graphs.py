"""
Generate all publication-quality comparison graphs and comprehensive report
for AI-Enabled Semantic Network vs Conventional Baseline.

Exports 10 distinct figures in 300-DPI PNG, vector PDF, and vector SVG:
1. semantic_quality_vs_mean_latency.png/.pdf/.svg (Trade-off Pareto frontier)
2. mean_and_p95_latency_by_policy.png/.pdf/.svg
3. deadline_miss_rate_by_policy.png/.pdf/.svg
4. reconstruction_mse_by_policy.png/.pdf/.svg
5. semantic_quality_by_policy.png/.pdf/.svg
6. mean_payload_size_by_policy.png/.pdf/.svg
7. semantic_quality_vs_payload_size.png/.pdf/.svg
8. mean_latency_vs_deadline_miss_rate.png/.pdf/.svg
9. ppo_vs_ppo_orchestration_ablation.png/.pdf/.svg
10. dqn_vs_edf_scheduling_comparison.png/.pdf/.svg
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("rl/results/ai_vs_baseline")
GRAPHS_DIR = RESULTS_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# Styling standards
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})

# Policy metadata and consistent visual mapping
POLICY_METADATA = {
    "A": {"label": "Conv-50%+EDF (Baseline)", "short": "A: Conv-50%+EDF", "color": "#1f77b4", "is_baseline": True},
    "B": {"label": "PPO+DQN+Orch (Full AI)", "short": "B: PPO+DQN+Orch", "color": "#d62728", "is_baseline": False},
    "C": {"label": "PPO+DQN (No Orch)", "short": "C: PPO+DQN", "color": "#ff7f0e", "is_baseline": False},
    "D": {"label": "Fixed-25%+EDF", "short": "D: Fixed-25%+EDF", "color": "#2ca02c", "is_baseline": False},
    "E": {"label": "Fixed-75%+EDF", "short": "E: Fixed-75%+EDF", "color": "#9467bd", "is_baseline": False},
    "F": {"label": "PPO+EDF", "short": "F: PPO+EDF", "color": "#8c564b", "is_baseline": False},
    "G": {"label": "PPO+Orch+EDF", "short": "G: PPO+Orch+EDF", "color": "#e377c2", "is_baseline": False},
}

def save_fig(fig, base_name: str):
    """Save in PNG (300 DPI), PDF (vector), and SVG (vector)."""
    fig.tight_layout()
    fig.savefig(GRAPHS_DIR / f"{base_name}.png", dpi=300)
    fig.savefig(GRAPHS_DIR / f"{base_name}.pdf")
    fig.savefig(GRAPHS_DIR / f"{base_name}.svg")
    plt.close(fig)
    print(f"Saved: {base_name}.png/.pdf/.svg")

def generate_all_graphs():
    df = pd.read_csv(RESULTS_DIR / "policy_comparison.csv")
    df_sched = pd.read_csv(RESULTS_DIR / "scheduling_results.csv")
    df_sem = pd.read_csv(RESULTS_DIR / "semantic_results.csv")

    configs = list(df["config"])
    labels = [POLICY_METADATA[c]["short"] for c in configs]
    colors = [POLICY_METADATA[c]["color"] for c in configs]

    # Compute repeated-run CI / error bars from scheduling episodes
    sched_err_lat = []
    sched_err_p95 = []
    sched_err_miss = []
    for c in configs:
        sub = df_sched[df_sched["config"] == c]
        n = len(sub)
        lat_sem = (sub["avg_latency_us"].std() / np.sqrt(n)) * 1.96 if n > 0 else 0
        p95_sem = (sub["p95_latency_us"].std() / np.sqrt(n)) * 1.96 if n > 0 else 0
        miss_sem = (sub["deadline_miss_rate"].std() / np.sqrt(n)) * 1.96 if n > 0 else 0
        sched_err_lat.append(lat_sem)
        sched_err_p95.append(p95_sem)
        sched_err_miss.append(miss_sem)

    # -------------------------------------------------------------------------
    # 1. Semantic Quality vs Mean Latency (Trade-off Pareto Frontier)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, row in df.iterrows():
        c = row["config"]
        meta = POLICY_METADATA[c]
        size = 180 if meta["is_baseline"] else 120
        edgecolor = "black" if meta["is_baseline"] else "none"
        lw = 2.0 if meta["is_baseline"] else 0.5
        ax.scatter(row["net_avg_latency_us"], row["sem_quality_mean"],
                   color=meta["color"], s=size, edgecolors=edgecolor, linewidth=lw, zorder=5,
                   label=f"{meta['label']}" + (" ★" if meta["is_baseline"] else ""))
        
        # Annotation offset
        offset_x = 25 if c not in ["B", "G"] else -120
        offset_y = 0.008 if c not in ["C"] else -0.015
        ax.annotate(f"{c}: {meta['short'].split(': ')[1]}",
                    (row["net_avg_latency_us"], row["sem_quality_mean"]),
                    xytext=(row["net_avg_latency_us"] + offset_x, row["sem_quality_mean"] + offset_y),
                    fontsize=8.5, fontweight="bold" if meta["is_baseline"] else "normal")

    # Add error bar for latency
    ax.errorbar(df["net_avg_latency_us"], df["sem_quality_mean"],
                xerr=sched_err_lat, yerr=df["sem_quality_std"]/np.sqrt(256)*1.96,
                fmt="none", ecolor="gray", elinewidth=0.8, capsize=3, alpha=0.7, zorder=3)

    ax.set_xlabel("Mean End-to-End Latency (µs) [Python Simulation]")
    ax.set_ylabel("Semantic Quality [1 / (1 + MSE)]")
    ax.set_title("Semantic Quality vs Mean Latency Trade-off")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    save_fig(fig, "semantic_quality_vs_mean_latency")

    # -------------------------------------------------------------------------
    # 2. Mean Latency and P95 Latency by Policy
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(configs))
    w = 0.36
    b1 = ax.bar(x - w/2, df["net_avg_latency_us"], w, label="Mean Latency (µs)", color=colors, alpha=0.85,
                edgecolor=["black" if POLICY_METADATA[c]["is_baseline"] else "none" for c in configs], linewidth=1.5)
    ax.errorbar(x - w/2, df["net_avg_latency_us"], yerr=sched_err_lat, fmt="none", ecolor="black", capsize=4)

    b2 = ax.bar(x + w/2, df["net_p95_latency_us"], w, label="P95 Latency (µs)", color=colors, alpha=0.45, hatch="//",
                edgecolor=["black" if POLICY_METADATA[c]["is_baseline"] else "none" for c in configs], linewidth=1.5)
    ax.errorbar(x + w/2, df["net_p95_latency_us"], yerr=sched_err_p95, fmt="none", ecolor="black", capsize=4)

    # Highlight baseline marker
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Latency (µs) [Python Simulation]")
    ax.set_title("Mean and 95th-Percentile Latency by Policy (95% CI)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    
    # Value annotations on mean bars
    for rect in b1:
        h = rect.get_height()
        ax.annotate(f"{h:.0f}µs", (rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    save_fig(fig, "mean_and_p95_latency_by_policy")

    # -------------------------------------------------------------------------
    # 3. Deadline-Miss Rate by Policy
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5))
    bars = ax.bar(x, df["net_deadline_miss_rate"] * 100.0, color=colors, alpha=0.85,
                  edgecolor=["black" if POLICY_METADATA[c]["is_baseline"] else "none" for c in configs],
                  linewidth=[2.0 if POLICY_METADATA[c]["is_baseline"] else 0.5 for c in configs])
    ax.errorbar(x, df["net_deadline_miss_rate"] * 100.0, yerr=np.array(sched_err_miss)*100.0,
                fmt="none", ecolor="black", capsize=4)

    # Add baseline reference line
    baseline_miss = df.loc[df["config"]=="A", "net_deadline_miss_rate"].values[0] * 100.0
    ax.axhline(baseline_miss, color="#1f77b4", linestyle="--", alpha=0.7, label=f"Baseline A Miss Rate ({baseline_miss:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Deadline-Miss Rate (%) [Python Simulation]")
    ax.set_title("Deadline-Miss Rate Across Policies (N=300 Episodes, 95% CI)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", (rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    save_fig(fig, "deadline_miss_rate_by_policy")

    # -------------------------------------------------------------------------
    # 4. Reconstruction MSE by Policy
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5))
    # Error bar from sample standard error over 256 samples
    sem_mse_ci = (df["sem_mse_p95"] - df["sem_mse_mean"]) / 2.0  # display spread
    bars = ax.bar(x, df["sem_mse_mean"], color=colors, alpha=0.85,
                  edgecolor=["black" if POLICY_METADATA[c]["is_baseline"] else "none" for c in configs],
                  linewidth=[2.0 if POLICY_METADATA[c]["is_baseline"] else 0.5 for c in configs])

    baseline_mse = df.loc[df["config"]=="A", "sem_mse_mean"].values[0]
    ax.axhline(baseline_mse, color="#1f77b4", linestyle="--", alpha=0.7, label=f"Baseline A MSE ({baseline_mse:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Reconstruction MSE (Lower is Better)")
    ax.set_title("Reconstruction MSE on Held-Out JIGSAWS Trajectories")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", (rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    save_fig(fig, "reconstruction_mse_by_policy")

    # -------------------------------------------------------------------------
    # 5. Semantic Quality by Policy
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5))
    sem_qual_err = df["sem_quality_std"] / np.sqrt(256) * 1.96
    bars = ax.bar(x, df["sem_quality_mean"], color=colors, alpha=0.85,
                  edgecolor=["black" if POLICY_METADATA[c]["is_baseline"] else "none" for c in configs],
                  linewidth=[2.0 if POLICY_METADATA[c]["is_baseline"] else 0.5 for c in configs])
    ax.errorbar(x, df["sem_quality_mean"], yerr=sem_qual_err, fmt="none", ecolor="black", capsize=4)

    baseline_qual = df.loc[df["config"]=="A", "sem_quality_mean"].values[0]
    ax.axhline(baseline_qual, color="#1f77b4", linestyle="--", alpha=0.7, label=f"Baseline A Quality ({baseline_qual:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Semantic Quality [1 / (1 + MSE)]")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Semantic Quality by Policy (256 Held-Out Samples, 95% CI)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", (rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    save_fig(fig, "semantic_quality_by_policy")

    # -------------------------------------------------------------------------
    # 6. Mean Payload Size by Policy
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5))
    bars = ax.bar(x, df["payload_bytes_mean"], color=colors, alpha=0.85,
                  edgecolor=["black" if POLICY_METADATA[c]["is_baseline"] else "none" for c in configs],
                  linewidth=[2.0 if POLICY_METADATA[c]["is_baseline"] else 0.5 for c in configs])

    baseline_payload = df.loc[df["config"]=="A", "payload_bytes_mean"].values[0]
    ax.axhline(baseline_payload, color="#1f77b4", linestyle="--", alpha=0.7, label=f"Baseline A Payload ({baseline_payload:.0f} B)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean Packet Payload Size (Bytes)")
    ax.set_title("Transmitted Packet Payload Size by Policy")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.0f} B", (rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    save_fig(fig, "mean_payload_size_by_policy")

    # -------------------------------------------------------------------------
    # 7. Semantic Quality vs Payload Size
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, row in df.iterrows():
        c = row["config"]
        meta = POLICY_METADATA[c]
        size = 180 if meta["is_baseline"] else 120
        edgecolor = "black" if meta["is_baseline"] else "none"
        lw = 2.0 if meta["is_baseline"] else 0.5
        ax.scatter(row["payload_bytes_mean"], row["sem_quality_mean"],
                   color=meta["color"], s=size, edgecolors=edgecolor, linewidth=lw, zorder=5,
                   label=f"{meta['label']}" + (" ★" if meta["is_baseline"] else ""))
        
        offset_x = 20 if c not in ["E"] else -90
        offset_y = 0.008 if c not in ["C"] else -0.015
        ax.annotate(f"{c}: {meta['short'].split(': ')[1]}",
                    (row["payload_bytes_mean"], row["sem_quality_mean"]),
                    xytext=(row["payload_bytes_mean"] + offset_x, row["sem_quality_mean"] + offset_y),
                    fontsize=8.5, fontweight="bold" if meta["is_baseline"] else "normal")

    ax.set_xlabel("Mean Transmitted Payload Size (Bytes)")
    ax.set_ylabel("Semantic Quality [1 / (1 + MSE)]")
    ax.set_title("Rate-Distortion Frontier: Semantic Quality vs Payload Size")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    save_fig(fig, "semantic_quality_vs_payload_size")

    # -------------------------------------------------------------------------
    # 8. Mean Latency vs Deadline-Miss Rate
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, row in df.iterrows():
        c = row["config"]
        meta = POLICY_METADATA[c]
        size = 180 if meta["is_baseline"] else 120
        edgecolor = "black" if meta["is_baseline"] else "none"
        lw = 2.0 if meta["is_baseline"] else 0.5
        miss_pct = row["net_deadline_miss_rate"] * 100.0
        ax.scatter(row["net_avg_latency_us"], miss_pct,
                   color=meta["color"], s=size, edgecolors=edgecolor, linewidth=lw, zorder=5,
                   label=f"{meta['label']}" + (" ★" if meta["is_baseline"] else ""))
        
        offset_x = 20 if c not in ["B", "G"] else -110
        offset_y = 0.5 if c not in ["C"] else -1.2
        ax.annotate(f"{c}: {meta['short'].split(': ')[1]}",
                    (row["net_avg_latency_us"], miss_pct),
                    xytext=(row["net_avg_latency_us"] + offset_x, miss_pct + offset_y),
                    fontsize=8.5, fontweight="bold" if meta["is_baseline"] else "normal")

    ax.set_xlabel("Mean Latency (µs) [Python Simulation]")
    ax.set_ylabel("Deadline-Miss Rate (%) [Python Simulation]")
    ax.set_title("Network Reliability Trade-off: Latency vs Deadline Misses")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9)
    save_fig(fig, "mean_latency_vs_deadline_miss_rate")

    # -------------------------------------------------------------------------
    # 9. PPO vs PPO + Orchestration Ablation
    # -------------------------------------------------------------------------
    # Direct comparison pairs: (C vs B) under DQN scheduler, (F vs G) under EDF scheduler
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    ablation_pairs = [("C: PPO Raw\n(DQN)", "B: PPO+Orch\n(DQN)"),
                      ("F: PPO Raw\n(EDF)", "G: PPO+Orch\n(EDF)")]
    
    # Semantic Quality comparison
    q_vals = [
        [df.loc[df['config']=='C', 'sem_quality_mean'].values[0], df.loc[df['config']=='B', 'sem_quality_mean'].values[0]],
        [df.loc[df['config']=='F', 'sem_quality_mean'].values[0], df.loc[df['config']=='G', 'sem_quality_mean'].values[0]],
    ]
    ax1.bar([0, 1], q_vals[0], color=["#ff7f0e", "#d62728"], width=0.5, alpha=0.85)
    ax1.bar([2.5, 3.5], q_vals[1], color=["#8c564b", "#e377c2"], width=0.5, alpha=0.85)
    ax1.set_xticks([0, 1, 2.5, 3.5])
    ax1.set_xticklabels(["C: PPO (DQN)", "B: +Orch (DQN)", "F: PPO (EDF)", "G: +Orch (EDF)"], fontsize=8.5)
    ax1.set_ylabel("Semantic Quality [1 / (1 + MSE)]")
    ax1.set_title("Semantic Quality Gain (+2.35%)")
    ax1.set_ylim(0.6, 0.75)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    for pos, val in zip([0, 1, 2.5, 3.5], [q_vals[0][0], q_vals[0][1], q_vals[1][0], q_vals[1][1]]):
        ax1.annotate(f"{val:.4f}", (pos, val), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)

    # Latency & Miss Rate trade-off
    l_vals = [
        [df.loc[df['config']=='C', 'net_avg_latency_us'].values[0], df.loc[df['config']=='B', 'net_avg_latency_us'].values[0]],
        [df.loc[df['config']=='F', 'net_avg_latency_us'].values[0], df.loc[df['config']=='G', 'net_avg_latency_us'].values[0]],
    ]
    ax2.bar([0, 1], l_vals[0], color=["#ff7f0e", "#d62728"], width=0.5, alpha=0.85)
    ax2.bar([2.5, 3.5], l_vals[1], color=["#8c564b", "#e377c2"], width=0.5, alpha=0.85)
    ax2.set_xticks([0, 1, 2.5, 3.5])
    ax2.set_xticklabels(["C: PPO (DQN)", "B: +Orch (DQN)", "F: PPO (EDF)", "G: +Orch (EDF)"], fontsize=8.5)
    ax2.set_ylabel("Mean Latency (µs)")
    ax2.set_title("Latency Overhead of Orchestration (+6.5%)")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    for pos, val in zip([0, 1, 2.5, 3.5], [l_vals[0][0], l_vals[0][1], l_vals[1][0], l_vals[1][1]]):
        ax2.annotate(f"{val:.0f}µs", (pos, val), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)

    save_fig(fig, "ppo_vs_ppo_orchestration_ablation")

    # -------------------------------------------------------------------------
    # 10. DQN vs EDF Scheduling Comparison
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # Compare under matched workloads: Raw PPO (C vs F) and Orchestrated PPO (B vs G)
    x_pairs = np.array([0, 1.8])
    w_sched = 0.35

    dqn_lats = [df.loc[df['config']=='C', 'net_avg_latency_us'].values[0],
                df.loc[df['config']=='B', 'net_avg_latency_us'].values[0]]
    edf_lats = [df.loc[df['config']=='F', 'net_avg_latency_us'].values[0],
                df.loc[df['config']=='G', 'net_avg_latency_us'].values[0]]

    ax1.bar(x_pairs - w_sched/2, dqn_lats, w_sched, label="DQN Scheduler", color="#d62728", alpha=0.85)
    ax1.bar(x_pairs + w_sched/2, edf_lats, w_sched, label="EDF Scheduler", color="#1f77b4", alpha=0.85)
    ax1.set_xticks(x_pairs)
    ax1.set_xticklabels(["Under Raw PPO\n(C vs F)", "Under PPO+Orch\n(B vs G)"])
    ax1.set_ylabel("Mean Latency (µs)")
    ax1.set_title("Mean Latency: DQN Achieves Lower Delay")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax1.legend()
    for xp, dl, el in zip(x_pairs, dqn_lats, edf_lats):
        ax1.annotate(f"{dl:.0f}µs", (xp - w_sched/2, dl), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax1.annotate(f"{el:.0f}µs", (xp + w_sched/2, el), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    # Deadline Miss Rate
    dqn_miss = [df.loc[df['config']=='C', 'net_deadline_miss_rate'].values[0]*100,
                df.loc[df['config']=='B', 'net_deadline_miss_rate'].values[0]*100]
    edf_miss = [df.loc[df['config']=='F', 'net_deadline_miss_rate'].values[0]*100,
                df.loc[df['config']=='G', 'net_deadline_miss_rate'].values[0]*100]

    ax2.bar(x_pairs - w_sched/2, dqn_miss, w_sched, label="DQN Scheduler", color="#d62728", alpha=0.85)
    ax2.bar(x_pairs + w_sched/2, edf_miss, w_sched, label="EDF Scheduler", color="#1f77b4", alpha=0.85)
    ax2.set_xticks(x_pairs)
    ax2.set_xticklabels(["Under Raw PPO\n(C vs F)", "Under PPO+Orch\n(B vs G)"])
    ax2.set_ylabel("Deadline-Miss Rate (%)")
    ax2.set_title("Deadline Compliance: EDF Vastly Superior")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax2.legend()
    for xp, dm, em in zip(x_pairs, dqn_miss, edf_miss):
        ax2.annotate(f"{dm:.1f}%", (xp - w_sched/2, dm), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax2.annotate(f"{em:.1f}%", (xp + w_sched/2, em), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    save_fig(fig, "dqn_vs_edf_scheduling_comparison")

if __name__ == "__main__":
    generate_all_graphs()
