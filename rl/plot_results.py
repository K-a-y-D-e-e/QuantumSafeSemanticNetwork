"""
Plotting script for RL training curves and scheduler comparison results.

Generates:
    1. rl/results/training_curves.png  -- Reward, Latency, Misses vs Episode
    2. rl/results/comparison_chart.png -- Bar chart comparing Random/Priority/EDF/DQN
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(log_csv_path: str, output_image_path: str) -> None:
    """Plot reward, latency, and deadline misses over training episodes."""
    episodes = []
    rewards = []
    latencies = []
    misses = []
    epsilons = []

    with open(log_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            latencies.append(float(row["avg_latency_us"]))
            misses.append(int(row["deadline_misses"]))
            epsilons.append(float(row["epsilon"]))

    if not episodes:
        print(f"No data in {log_csv_path}")
        return

    # Window for rolling average
    window = max(5, len(episodes) // 20)
    
    def moving_average(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Reward
    axes[0].plot(episodes, rewards, color='steelblue', alpha=0.3, label='Raw Reward')
    if len(episodes) >= window:
        ma_rewards = moving_average(rewards, window)
        ma_episodes = episodes[window-1:]
        axes[0].plot(ma_episodes, ma_rewards, color='darkblue', linewidth=2, label=f'{window}-Ep Rolling Mean')
    axes[0].set_ylabel('Reward', fontsize=11, fontweight='bold')
    axes[0].set_title('DQN Training Curves', fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper left')

    # 2. Latency
    axes[1].plot(episodes, latencies, color='coral', alpha=0.3, label='Raw Latency')
    if len(episodes) >= window:
        ma_lat = moving_average(latencies, window)
        axes[1].plot(ma_episodes, ma_lat, color='darkred', linewidth=2, label=f'{window}-Ep Rolling Mean')
    axes[1].set_ylabel('Avg Latency (us)', fontsize=11, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(loc='upper right')

    # 3. Deadline Misses
    axes[2].plot(episodes, misses, color='purple', alpha=0.3, label='Raw Misses')
    if len(episodes) >= window:
        ma_misses = moving_average(misses, window)
        axes[2].plot(ma_episodes, ma_misses, color='indigo', linewidth=2, label=f'{window}-Ep Rolling Mean')
    axes[2].set_xlabel('Episode', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Deadline Misses', fontsize=11, fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image_path, dpi=200)
    plt.close()
    print(f"Training curves saved to: {output_image_path}")


def plot_comparison(results_csv_path: str, output_image_path: str) -> None:
    """Plot bar chart comparing Random, Priority, EDF, and DQN schedulers."""
    schedulers = []
    latencies = []
    miss_rates = []

    with open(results_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedulers.append(row["scheduler"])
            latencies.append(float(row["avg_latency_us"]))
            miss_rates.append(float(row["deadline_miss_rate"]) * 100.0)

    if not schedulers:
        print(f"No data in {results_csv_path}")
        return

    x = np.arange(len(schedulers))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 6))

    color1 = 'tab:blue'
    rects1 = ax1.bar(x - width/2, latencies, width, label='Avg Latency (us)', color=color1, alpha=0.85)
    ax1.set_ylabel('Average Latency (us)', color=color1, fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(schedulers, fontsize=11, fontweight='bold')

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    rects2 = ax2.bar(x + width/2, miss_rates, width, label='Deadline Miss Rate (%)', color=color2, alpha=0.85)
    ax2.set_ylabel('Deadline Miss Rate (%)', color=color2, fontsize=11, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Bar value labels
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}%',
                     xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)

    plt.title('Scheduler Comparison: Random vs Priority vs EDF vs DQN', fontsize=13, fontweight='bold')
    plt.tight_layout()
    Path(output_image_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image_path, dpi=200)
    plt.close()
    print(f"Comparison chart saved to: {output_image_path}")


def main():
    log_csv = "rl/logs/training_log.csv"
    curves_png = "rl/results/training_curves.png"
    if Path(log_csv).exists():
        plot_training_curves(log_csv, curves_png)

    res_csv = "rl/results/comparison_results.csv"
    comp_png = "rl/results/comparison_chart.png"
    if Path(res_csv).exists():
        plot_comparison(res_csv, comp_png)


if __name__ == "__main__":
    main()
