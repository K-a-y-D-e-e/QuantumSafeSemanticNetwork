"""
Analyze semantic_network_baselines_packets.csv using episode-level uncertainty.

Reports per-condition mean, episode-to-episode SD, and bootstrap 95% CI.
Also reports paired episode-level differences for PPO_Orchestrated versus each
fixed-retention condition. This avoids treating the 1,000 packets as fully
independent observations when traffic is grouped into 100 episodes.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import math
import random
import statistics

METRICS = {
    "network_latency_us": "latency_ms",
    "queueing_delay_us": "queueing_ms",
    "reconstruction_mse": "mse",
    "semantic_quality": "quality",
    "modeled_packet_bytes": "packet_bytes",
    "deadline_miss": "deadline_miss_rate",
}
CONDITIONS = [
    "Fixed_25pct", "Fixed_50pct", "Fixed_75pct",
    "Fixed_100pct", "PPO_Orchestrated",
]


def as_float(value):
    return float(value) if value not in (None, "") else float("nan")


def as_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def bootstrap_ci(values, rng, reps=10000):
    """Percentile bootstrap CI for the mean of episode-level values."""
    n = len(values)
    if not n:
        return float("nan"), float("nan")
    means = []
    for _ in range(reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return means[int(0.025 * reps)], means[min(reps - 1, int(0.975 * reps))]


def summarize(values, rng):
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    lo, hi = bootstrap_ci(values, rng)
    return mean, sd, lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="rl/results/semantic_network_baselines_packets.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="rl/results/semantic_network_baselines_uncertainty",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if args.bootstrap_reps < 1000:
        raise ValueError("Use at least 1000 bootstrap replicates.")

    episode_packets = defaultdict(list)
    with input_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"episode", "condition", "network_latency_us",
                    "queueing_delay_us", "reconstruction_mse",
                    "semantic_quality", "modeled_packet_bytes", "deadline_met"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        for row in reader:
            condition = row["condition"]
            if condition not in CONDITIONS:
                continue
            row["episode_num"] = int(row["episode"])
            row["deadline_miss"] = 0.0 if as_bool(row["deadline_met"]) else 1.0
            for column in METRICS:
                if column != "deadline_miss":
                    row[column] = as_float(row[column])
            row["deadline_miss"] = float(row["deadline_miss"])
            episode_packets[(row["episode_num"], condition)].append(row)

    if not episode_packets:
        raise ValueError("No recognized condition rows found in the input CSV.")

    # Each point is an episode mean, so every episode has equal weight.
    episode_values = defaultdict(dict)
    for (episode, condition), rows in episode_packets.items():
        for metric in METRICS:
            vals = [r[metric] for r in rows]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            episode_values[condition].setdefault(metric, {})[episode] = (
                statistics.fmean(vals) if vals else float("nan")
            )

    rng = random.Random(args.seed)
    summary_rows = []
    print("\nEPISODE-LEVEL MEANS (95% bootstrap CI; latency/queue in ms)")
    print(f"{'Condition':20s} {'Metric':18s} {'Mean':>12s} {'Episode SD':>12s} {'95% CI':>25s}")
    print("-" * 92)
    for condition in CONDITIONS:
        if condition not in episode_values:
            continue
        for metric, label in METRICS.items():
            vals_by_ep = episode_values[condition].get(metric, {})
            vals = [v for _, v in sorted(vals_by_ep.items()) if not math.isnan(v)]
            if not vals:
                continue
            scale = 0.001 if metric in {"network_latency_us", "queueing_delay_us"} else 1.0
            mean, sd, lo, hi = summarize(vals, rng)
            mean, sd, lo, hi = [x * scale for x in (mean, sd, lo, hi)]
            row = {
                "condition": condition, "metric": label,
                "episodes": len(vals), "mean": mean, "episode_sd": sd,
                "ci95_low": lo, "ci95_high": hi,
            }
            summary_rows.append(row)
            print(f"{condition:20s} {label:18s} {mean:12.4f} {sd:12.4f} [{lo:10.4f}, {hi:10.4f}]")

    print("\nPAIRED EPISODE DIFFERENCES: PPO_Orchestrated minus fixed retention")
    print("Negative latency/MSE/miss differences favor PPO; positive quality favors PPO.")
    print(f"{'Comparison':28s} {'Metric':18s} {'Mean diff':>12s} {'95% CI':>25s}")
    print("-" * 88)
    paired_rows = []
    ppo = episode_values.get("PPO_Orchestrated", {})
    for fixed in CONDITIONS:
        if fixed == "PPO_Orchestrated" or fixed not in episode_values:
            continue
        for metric in ["network_latency_us", "queueing_delay_us",
                       "reconstruction_mse", "semantic_quality", "deadline_miss"]:
            pmap = ppo.get(metric, {})
            fmap = episode_values[fixed].get(metric, {})
            common = sorted(set(pmap) & set(fmap))
            diffs = [pmap[e] - fmap[e] for e in common
                     if not math.isnan(pmap[e]) and not math.isnan(fmap[e])]
            if not diffs:
                continue
            lo, hi = bootstrap_ci(diffs, rng, args.bootstrap_reps)
            scale = 0.001 if metric in {"network_latency_us", "queueing_delay_us"} else 1.0
            mean, lo, hi = [x * scale for x in (statistics.fmean(diffs), lo, hi)]
            label = METRICS[metric]
            paired_rows.append({
                "comparison": f"PPO_minus_{fixed}", "metric": label,
                "paired_episodes": len(diffs), "mean_difference": mean,
                "ci95_low": lo, "ci95_high": hi,
            })
            print(f"{('PPO - ' + fixed):28s} {label:18s} {mean:12.4f} [{lo:10.4f}, {hi:10.4f}]")

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(str(prefix) + "_summary.csv")
    paired_path = Path(str(prefix) + "_paired_differences.csv")
    for path, rows in [(summary_path, summary_rows), (paired_path, paired_rows)]:
        if not rows:
            continue
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nSummary with confidence intervals: {summary_path}")
    print(f"Paired differences:               {paired_path}")
    print("Interpret CIs descriptively; they do not correct for every possible experimental bias.")


if __name__ == "__main__":
    main()
