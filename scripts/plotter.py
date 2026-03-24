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

    if not ratio_cols or not time_cols:
        logger.warning("[!] No valid Time or Ratio data found to plot. Skipping plot generation.")
        return

    valid_ratio_cols = [c for c in ratio_cols if c in df.columns]
    valid_time_cols = [c for c in time_cols if c in df.columns]

    if not valid_ratio_cols and not valid_time_cols:
        logger.warning("[!] No valid columns found in dataframe. Skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Use a dynamic colormap
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(strategies)))

    if valid_ratio_cols:
        df.plot(x="Dataset", y=valid_ratio_cols, kind="bar", ax=axes[0], color=colors[:len(valid_ratio_cols)])
        axes[0].set_title("Compression Ratio (Lower is Better)")
        axes[0].tick_params(axis='x', rotation=45 if len(df) > 1 else 0)
        axes[0].legend([c.replace("Ratio_", "") for c in valid_ratio_cols])

    if valid_time_cols:
        df.plot(x="Dataset", y=valid_time_cols, kind="bar", ax=axes[1], color=colors[:len(valid_time_cols)])
        axes[1].set_title("Execution Time (Seconds)")
        axes[1].tick_params(axis='x', rotation=45 if len(df) > 1 else 0)
        axes[1].legend([c.replace("Time_", "") for c in valid_time_cols])

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
    plt.savefig(os.path.join(runs_dir, f"{dataset_name}.pdf"))
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
        line_style ='-'
        line_width = 2.5

        if f'Ratio_{strat}' in avg_df.columns and not avg_df[f'Ratio_{strat}'].isnull().all():
            axes[0].plot(avg_df[param_name], avg_df[f'Ratio_{strat}'], marker=marker,
                         linestyle=line_style, color=color, linewidth=line_width, markersize=8, label=strat)

        if f'Time_{strat}' in avg_df.columns and not avg_df[f'Time_{strat}'].isnull().all():
            axes[1].plot(avg_df[param_name], avg_df[f'Time_{strat}'], marker=marker,
                         linestyle=line_style, color=color, linewidth=line_width, markersize=8, label=strat)

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

# In get_pareto_front_2d()
def get_pareto_front_2d(df, x_col, y_col):
    """
    Calculates the Pareto front
    """
    sorted_df = df.sort_values(by=[x_col, y_col])

    pareto_indices = []
    min_y = float('inf')

    for index, row in sorted_df.iterrows():
        if row[y_col] < min_y:
            pareto_indices.append(index)
            min_y = row[y_col]

    return df.loc[pareto_indices].copy()

def plot_pareto_front(csv_file, plot_file):
    df = pd.read_csv(filepath_or_buffer=str(csv_file)) # Fixed IDE warning
    if df.empty: return

    # Identify unique datasets to create subplots
    datasets = df['Dataset'].unique()
    n_datasets = len(datasets)

    # Dynamically calculate grid size (2 columns wide)
    cols = 2 if n_datasets > 1 else 1
    rows = (n_datasets + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))

    # Ensure axes is always a flattened array for easy iteration
    if n_datasets == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Create a consistent color map for algorithms
    algorithms = df['Algorithm'].unique()
    cmap = plt.get_cmap('tab10')
    colors = {algo: cmap(i / max(1, len(algorithms) - 1)) for i, algo in enumerate(algorithms)}

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        ds_df = df[df['Dataset'] == dataset]

        # 1. Scatter plot all LHS samples (The "Cloud")
        for algo in algorithms:
            algo_df = ds_df[ds_df['Algorithm'] == algo]
            if algo_df.empty: continue

            ax.scatter(algo_df['Time'], algo_df['Ratio'],
                       color=colors[algo], label=algo, alpha=0.4, s=40, edgecolors='none')

        # 2. Calculate and plot the Global Pareto Front
        pareto_df = get_pareto_front_2d(ds_df, 'Time', 'Ratio')

        if not pareto_df.empty:
            # Draw the line connecting the optimal points
            ax.plot(pareto_df['Time'], pareto_df['Ratio'],
                    color='red', linestyle='--', linewidth=2, label='Global Pareto Front', alpha=0.8)

            # Highlight the Pareto-optimal configurations with distinct stars
            ax.scatter(pareto_df['Time'], pareto_df['Ratio'],
                       color='gold', edgecolor='red', zorder=5, s=150, marker='*', label='Optimal Config')

        # 3. Formatting
        ax.set_title(f"Optimization Landscape: {dataset}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Execution Time (Seconds) ↓", fontsize=11)
        ax.set_ylabel("Compression Ratio ↓", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)

        # Deduplicate the legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

    # Clean up any empty subplots (if n_datasets is an odd number)
    for j in range(len(datasets), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()