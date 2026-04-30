import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_results(csv_file, plot_file, logger):
    df = pd.read_csv(csv_file)
    if df.empty: return

    # Dynamically find all strategies tested
    strategies = [col.replace("Time_", "") for col in df.columns if col.startswith("Time_")]
    ratio_cols = [f"Ratio_{s}" for s in strategies]
    time_cols = [f"Time_{s}" for s in strategies]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Use a dynamic colormap
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(strategies)))

    df.plot(x="Dataset", y=ratio_cols, kind="bar", ax=axes[0], color=colors)
    axes[0].set_title("Compression Ratio (Lower is Better)")
    axes[0].tick_params(axis='x', rotation=45 if len(df) > 1 else 0)
    axes[0].legend(strategies)

    df.plot(x="Dataset", y=time_cols, kind="bar", ax=axes[1], color=colors)
    axes[1].set_title("Execution Time (Seconds)")
    axes[1].tick_params(axis='x', rotation=45 if len(df) > 1 else 0)
    axes[1].legend(strategies)

    plt.tight_layout()
    plt.savefig(plot_file)
    logger.debug(f"Saved bar plot to {plot_file}")
    plt.close()

def plot_runs_variance(dataset_name, all_times_dict, all_ratios_dict, runs_dir):
    # Updated to handle dictionaries of lists (one list per strategy)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(all_times_dict)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    for idx, (strat, times) in enumerate(all_times_dict.items()):
        if not times: continue
        runs_x = list(range(1, len(times) + 1))
        marker = markers[idx % len(markers)]

        axes[0].plot(runs_x, all_ratios_dict[strat], marker=marker, color=colors[idx], label=strat)
        axes[1].plot(runs_x, times, marker=marker, color=colors[idx], label=strat)

    axes[0].set_title("Compression Ratio Variance")
    axes[0].legend()
    axes[1].set_title("Execution Time Variance")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(runs_dir, f"{dataset_name}_variance_plot.pdf"))
    plt.close()

def plot_parameter_analysis(csv_file, param_name, plot_file):
    df = pd.read_csv(csv_file)
    if df.empty: return

    strategies = [col.replace("Time_", "") for col in df.columns if col.startswith("Time_")]
    avg_df = df.groupby(param_name).mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(strategies)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']

    for idx, strat in enumerate(strategies):
        marker = markers[idx % len(markers)]
        color = colors[idx]

        # Style baseline as a dashed line to stand out
        line_style ='-'
        line_width = 2.5

        axes[0].plot(avg_df[param_name], avg_df[f'Ratio_{strat}'], marker=marker, linestyle=line_style, color=color, linewidth=line_width, markersize=8, label=strat)
        axes[1].plot(avg_df[param_name], avg_df[f'Time_{strat}'], marker=marker, linestyle=line_style, color=color, linewidth=line_width, markersize=8, label=strat)

    axes[0].set_title(f"Average Compression Ratio vs {param_name.upper()}", fontsize=14, fontweight='bold')
    axes[0].set_xlabel(f"Parameter: {param_name.upper()}", fontsize=12)
    axes[0].set_ylabel("Compression Ratio (Lower is Better)", fontsize=12)
    axes[0].set_xticks(avg_df[param_name])
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, linestyle=':', alpha=0.7)
    axes[0].legend(fontsize=11)

    axes[1].set_title(f"Average Execution Time vs {param_name.upper()}", fontsize=14, fontweight='bold')
    axes[1].set_xlabel(f"Parameter: {param_name.upper()}", fontsize=12)
    axes[1].set_ylabel("Execution Time in Seconds (Lower is Better)", fontsize=12)
    axes[1].set_xticks(avg_df[param_name])
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, linestyle=':', alpha=0.7)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()