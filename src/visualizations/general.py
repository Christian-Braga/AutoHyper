import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# Set a consistent, elegant color palette for all visualizations
PALETTE = "viridis"
HIGHLIGHT_COLOR = "#FF5722"  # Orange for highlighting best configs
FIGURE_SIZE = (10, 6)
STYLE = "whitegrid"


def plot_performance_distribution(hpo_results, save_path=None):
    """
    Visualizes the performance distribution of all configurations, highlighting the best ones.

    Args:
        hpo_results (dict): Structured output from the HPO process
        save_path (str, optional): Path to save the figure. If None, the plot is displayed.

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    sns.set_style(STYLE)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Extract data
    configs = hpo_results["configs_summary"]
    scores = [config["raw_score"] for config in configs]

    # Create DataFrame for easier plotting
    df = pd.DataFrame(
        {
            "Configuration Index": range(len(configs)),
            "Performance Score": scores,
            "Is Best": [False] * len(configs),
        }
    )

    # Mark best configurations
    df.loc[0, "Is Best"] = True  # Best weighted
    best_perf_idx = next(
        i
        for i, c in enumerate(configs)
        if json.dumps(c["params"], sort_keys=True)
        == json.dumps(hpo_results["best_by_performance"]["params"], sort_keys=True)
    )
    df.loc[best_perf_idx, "Is Best"] = True

    # Sort by performance for better visualization
    df = df.sort_values("Performance Score", ascending=False).reset_index(drop=True)

    # Plot distribution with KDE
    sns.histplot(
        df["Performance Score"],
        kde=True,
        color=sns.color_palette(PALETTE)[2],
        ax=ax,
        alpha=0.7,
    )

    # Highlight best configurations
    for _, row in df[df["Is Best"]].iterrows():
        ax.axvline(
            x=row["Performance Score"],
            color=HIGHLIGHT_COLOR,
            linestyle="--",
            linewidth=2,
        )

    # Add annotations
    best_weighted = hpo_results["best_config"]["performance_score"]
    best_perf = hpo_results["best_by_performance"]["performance_score"]

    ax.annotate(
        f"Best weighted: {best_weighted:.4f}",
        xy=(best_weighted, 0),
        xytext=(best_weighted, 0.2),
        textcoords="axes fraction",
        color=HIGHLIGHT_COLOR,
        arrowprops=dict(facecolor=HIGHLIGHT_COLOR, shrink=0.05, width=2, headwidth=8),
        fontsize=11,
        fontweight="bold",
    )

    if best_weighted != best_perf:
        ax.annotate(
            f"Best performance: {best_perf:.4f}",
            xy=(best_perf, 0),
            xytext=(best_perf, 0.4),
            textcoords="axes fraction",
            color=HIGHLIGHT_COLOR,
            arrowprops=dict(
                facecolor=HIGHLIGHT_COLOR, shrink=0.05, width=2, headwidth=8
            ),
            fontsize=11,
            fontweight="bold",
        )

    # Add statistical info
    stats_text = (
        f"Mean: {np.mean(scores):.4f}\n"
        f"Median: {np.median(scores):.4f}\n"
        f"Std Dev: {np.std(scores):.4f}"
    )

    ax.text(
        0.02,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_title(
        "Performance Distribution of All Configurations", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Performance Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


def plot_param_importance(hpo_results, save_path=None):
    """
    Creates a visualization showing the importance of different hyperparameters
    by analyzing their frequency in top-performing configurations.

    Args:
        hpo_results (dict): Structured output from the HPO process
        save_path (str, optional): Path to save the figure. If None, the plot is displayed.

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    sns.set_style(STYLE)

    # Extract top 25% of configurations by performance
    configs = hpo_results["configs_summary"]
    top_quartile = len(configs) // 4
    top_configs = sorted(configs, key=lambda x: x["raw_score"], reverse=True)[
        :top_quartile
    ]

    # Extract all parameter keys
    all_params = set()
    for config in configs:
        all_params.update(config["params"].keys())

    # Initialize parameter value frequency dictionaries
    param_value_counts = {param: {} for param in all_params}

    # Count parameter values in top configurations
    for config in top_configs:
        for param, value in config["params"].items():
            value_str = str(value)
            if value_str not in param_value_counts[param]:
                param_value_counts[param][value_str] = 0
            param_value_counts[param][value_str] += 1

    # Create a data structure for visualization
    viz_data = []
    for param, values in param_value_counts.items():
        # Calculate entropy to measure parameter importance
        total = sum(values.values())
        entropy = -sum(
            (count / total) * np.log2(count / total) for count in values.values()
        )
        importance = 1 - (
            entropy / np.log2(max(len(values), 1))
        )  # Normalized importance

        # Add data about the most frequent value
        most_common_value = (
            max(values.items(), key=lambda x: x[1])[0] if values else "N/A"
        )
        most_common_count = max(values.values()) if values else 0

        viz_data.append(
            {
                "Parameter": param,
                "Importance": importance,
                "Most Common Value": most_common_value,
                "Value Count": most_common_count,
                "Dominance": most_common_count / total if total > 0 else 0,
            }
        )

    # Sort by importance
    viz_data = sorted(viz_data, key=lambda x: x["Importance"], reverse=True)

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [2, 1]}
    )

    # Bar chart of parameter importance
    df = pd.DataFrame(viz_data)
    sns.barplot(
        x="Importance",
        y="Parameter",
        data=df,
        palette=sns.color_palette(PALETTE, len(df)),
        ax=ax1,
    )

    ax1.set_title("Hyperparameter Importance", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Importance Score (based on value distribution)", fontsize=12)
    ax1.set_ylabel("Hyperparameter", fontsize=12)

    # Add most common values as text
    for i, row in enumerate(viz_data):
        ax1.text(
            row["Importance"] + 0.02,
            i,
            f"Most common: {row['Most Common Value']} ({row['Dominance']:.0%})",
            va="center",
            fontsize=9,
        )

    # Heatmap showing parameter-value frequencies for top parameters
    top_params = [row["Parameter"] for row in viz_data[: min(5, len(viz_data))]]
    heatmap_data = {}

    for param in top_params:
        values = sorted(
            param_value_counts[param].items(), key=lambda x: x[1], reverse=True
        )
        # Take top 5 values or all if fewer
        values = values[: min(5, len(values))]
        heatmap_data[param] = {val: count for val, count in values}

    # Convert to DataFrame for heatmap with parameters as rows and values as columns
    heatmap_df = pd.DataFrame(
        {
            param: {val: count for val, count in param_value_counts[param].items()}
            for param in top_params
        }
    ).T.fillna(0)

    # Keep only columns with non-zero values
    heatmap_df = heatmap_df.loc[:, (heatmap_df != 0).any(axis=0)]

    # If we have data for the heatmap
    if not heatmap_df.empty and len(heatmap_df.columns) > 0:
        # Normalize by row for better visualization
        heatmap_norm = heatmap_df.div(heatmap_df.sum(axis=1), axis=0)

        cmap = sns.color_palette(PALETTE, as_cmap=True)
        sns.heatmap(
            heatmap_norm,
            annot=heatmap_df,
            fmt=".0f",
            cmap=cmap,
            linewidths=0.5,
            ax=ax2,
            cbar_kws={"label": "Normalized Frequency"},
        )

        ax2.set_title(
            "Top Parameter Value Distributions", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Parameter Values", fontsize=12)
        ax2.set_ylabel("", fontsize=12)
    else:
        ax2.text(
            0.5,
            0.5,
            "Insufficient data for heatmap",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax2.axis("off")

    # Highlight best configuration parameters
    best_config_text = "Best configuration:\n"
    for param, value in hpo_results["best_config"]["params"].items():
        best_config_text += f"• {param}: {value}\n"

    plt.figtext(
        0.5,
        0.01,
        best_config_text,
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.2),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


def plot_rank_correlation(hpo_results, save_path=None):
    """
    Visualizes the relationship between performance rank, frequency rank,
    and weighted rank for all configurations.

    Args:
        hpo_results (dict): Structured output from the HPO process
        save_path (str, optional): Path to save the figure. If None, the plot is displayed.

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    sns.set_style(STYLE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Extract data
    configs = hpo_results["configs_summary"]
    df = pd.DataFrame(
        {
            "Performance Rank": [c["performance_rank"] for c in configs],
            "Frequency Rank": [c["frequency_rank"] for c in configs],
            "Weighted Rank": [c["weighted_rank"] for c in configs],
            "Performance Score": [c["raw_score"] for c in configs],
            "Frequency": [c["frequency"] for c in configs],
            "Weighted Score": [c["weighted_score"] for c in configs],
            "Is Best Overall": [False] * len(configs),
            "Is Best Performance": [False] * len(configs),
            "Is Most Frequent": [False] * len(configs),
        }
    )

    # Mark best configurations
    df.loc[0, "Is Best Overall"] = True  # First config is best weighted

    best_perf_idx = next(
        i
        for i, c in enumerate(configs)
        if json.dumps(c["params"], sort_keys=True)
        == json.dumps(hpo_results["best_by_performance"]["params"], sort_keys=True)
    )
    df.loc[best_perf_idx, "Is Best Performance"] = True

    most_freq_idx = next(
        i
        for i, c in enumerate(configs)
        if json.dumps(c["params"], sort_keys=True)
        == json.dumps(hpo_results["most_frequent_config"]["params"], sort_keys=True)
    )
    df.loc[most_freq_idx, "Is Most Frequent"] = True

    # Scatter plot of Performance vs Frequency Rank with weighted rank as color
    scatter = ax1.scatter(
        df["Performance Rank"],
        df["Frequency Rank"],
        c=df["Weighted Rank"],
        cmap=PALETTE,
        s=100,
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, label="Weighted Rank")
    cbar.set_label("Weighted Rank", fontsize=12)

    # Mark special configurations
    ax1.scatter(
        df[df["Is Best Overall"]]["Performance Rank"],
        df[df["Is Best Overall"]]["Frequency Rank"],
        s=200,
        color=HIGHLIGHT_COLOR,
        marker="*",
        edgecolor="k",
        linewidth=1.5,
        label="Best Overall",
    )

    if not df[df["Is Best Performance"]]["Is Best Overall"].iloc[0]:
        ax1.scatter(
            df[df["Is Best Performance"]]["Performance Rank"],
            df[df["Is Best Performance"]]["Frequency Rank"],
            s=150,
            color="green",
            marker="X",
            edgecolor="k",
            linewidth=1.5,
            label="Best Performance",
        )

    if not df[df["Is Most Frequent"]]["Is Best Overall"].iloc[0]:
        ax1.scatter(
            df[df["Is Most Frequent"]]["Performance Rank"],
            df[df["Is Most Frequent"]]["Frequency Rank"],
            s=150,
            color="purple",
            marker="D",
            edgecolor="k",
            linewidth=1.5,
            label="Most Frequent",
        )

    # Set labels and title
    ax1.set_xlabel("Performance Rank", fontsize=12)
    ax1.set_ylabel("Frequency Rank", fontsize=12)
    ax1.set_title("Rank Correlation Plot", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right")

    # Add grid
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Add weighting information
    weights = hpo_results["weighting_factors"]
    weight_text = (
        f"Weighting Factors:\n"
        f"Performance: {weights['performance_weight']:.2f}\n"
        f"Frequency: {weights['frequency_weight']:.2f}"
    )

    ax1.text(
        0.02,
        0.98,
        weight_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Second subplot: Scores comparison
    # Extract top N configurations
    top_n = min(10, len(configs))
    top_configs = pd.DataFrame(
        {
            "Config": range(1, top_n + 1),
            "Performance": [c["raw_score"] for c in configs[:top_n]],
            "Frequency (Normalized)": [
                c["frequency_normalized"] for c in configs[:top_n]
            ],
            "Weighted Score": [c["weighted_score"] for c in configs[:top_n]],
        }
    )

    # Melt for grouped bar chart
    melted = pd.melt(
        top_configs,
        id_vars=["Config"],
        value_vars=["Performance", "Frequency (Normalized)", "Weighted Score"],
    )

    # Plot grouped bar chart
    sns.barplot(
        x="Config",
        y="value",
        hue="variable",
        data=melted,
        ax=ax2,
        palette=[
            sns.color_palette(PALETTE)[0],
            sns.color_palette(PALETTE)[3],
            HIGHLIGHT_COLOR,
        ],
    )

    # Set labels and title
    ax2.set_xlabel("Configuration Rank", fontsize=12)
    ax2.set_ylabel("Score Value", fontsize=12)
    ax2.set_title(
        "Top 10 Configurations: Score Comparison", fontsize=14, fontweight="bold"
    )
    ax2.legend(title="Metric")

    # Annotate best configuration
    best_config_params = hpo_results["best_config"]["params"]
    param_text = "\n".join(
        [f"{k}: {v}" for k, v in list(best_config_params.items())[:3]]
    )  # Show first 3 params
    if len(best_config_params) > 3:
        param_text += f"\n+ {len(best_config_params) - 3} more params"

    ax2.annotate(
        f"Best Config:\n{param_text}",
        xy=(0, 0),
        xytext=(0.02, 0.02),
        textcoords="axes fraction",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


def plot_cross_validation_analysis(hpo_results, save_path=None):
    """
    Visualizes cross-validation metrics across folds for the top configurations.

    Args:
        hpo_results (dict): Structured output from the HPO process
        save_path (str, optional): Path to save the figure. If None, the plot is displayed.

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    sns.set_style(STYLE)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]}
    )

    # Extract fold metrics and top configurations
    fold_metrics = hpo_results["fold_metrics"]
    top_configs = hpo_results["configs_summary"][:5]  # Top 5 configurations

    # Prepare data for boxplot
    config_names = []
    scores = []
    configs = []

    for i, config in enumerate(top_configs):
        config_id = f"Config {i + 1}"
        config_names.append(config_id)
        configs.append(config)

        # Find metrics for this configuration across folds
        config_scores = []
        for fold in fold_metrics:
            for fold_config in fold["configs"]:
                if json.dumps(fold_config["config"], sort_keys=True) == json.dumps(
                    config["params"], sort_keys=True
                ):
                    config_scores.append(fold_config["score"])
                    break

        # Add scores for this config
        for score in config_scores:
            scores.append({"Configuration": config_id, "Score": score})

    # Create DataFrame for seaborn
    df = pd.DataFrame(scores)

    # Box plot of scores by configuration
    if not df.empty:
        sns.boxplot(
            x="Configuration",
            y="Score",
            data=df,
            ax=ax1,
            palette=sns.color_palette(PALETTE, len(config_names)),
        )

        # Add individual points
        sns.stripplot(
            x="Configuration",
            y="Score",
            data=df,
            ax=ax1,
            color="black",
            alpha=0.5,
            jitter=True,
            size=4,
        )

        # Calculate and display statistics
        for i, config_id in enumerate(config_names):
            config_df = df[df["Configuration"] == config_id]
            if not config_df.empty:
                config_scores = config_df["Score"].values
                mean = np.mean(config_scores)
                std = np.std(config_scores)
                cv = (
                    std / mean if mean != 0 else float("inf")
                )  # Coefficient of variation

                # Add text with statistics
                ax1.text(
                    i,
                    df["Score"].min() - 0.05 * (df["Score"].max() - df["Score"].min()),
                    f"Mean: {mean:.4f}\nStd: {std:.4f}\nCV: {cv:.2f}",
                    ha="center",
                    va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                )
    else:
        ax1.text(
            0.5,
            0.5,
            "No cross-validation data available",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax1.axis("off")

    ax1.set_title(
        "Cross-Validation Score Distribution by Configuration",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Configuration", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)

    # Second subplot: Parameter comparison for top configs
    # Extract parameter keys (common across all configs)
    all_params = set()
    for config in configs:
        all_params.update(config["params"].keys())

    # Create a heatmap data structure
    heatmap_data = []
    for param in all_params:
        row = {"Parameter": param}
        for i, config in enumerate(configs):
            config_id = f"Config {i + 1}"
            row[config_id] = config["params"].get(param, "N/A")
        heatmap_data.append(row)

    # Create DataFrame for the heatmap
    heatmap_df = pd.DataFrame(heatmap_data)

    # Set Parameter as index
    if not heatmap_df.empty:
        heatmap_df.set_index("Parameter", inplace=True)

        # Create a custom colormap for categorical data
        categorical_colors = sns.color_palette(
            PALETTE, min(20, len(heatmap_df.stack().unique()))
        )

        # Plot a "categorical heatmap" - this is just for visualization, not true heat
        sns.heatmap(
            np.zeros_like(heatmap_df, dtype=float),
            cmap="Blues",
            cbar=False,
            ax=ax2,
            linewidths=1,
            linecolor="white",
        )

        # Add text annotations with parameter values
        for i, param in enumerate(heatmap_df.index):
            for j, config in enumerate(heatmap_df.columns):
                value = heatmap_df.loc[param, config]
                # Different colors for the best overall config
                text_color = "black"
                bbox_props = None

                if j == 0:  # Best config
                    bbox_props = dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)

                ax2.text(
                    j + 0.5,
                    i + 0.5,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=text_color,
                    bbox=bbox_props,
                )

        # Add legend for best configuration
        legend_elements = [
            Patch(
                facecolor="yellow",
                alpha=0.3,
                edgecolor="black",
                label="Best Configuration",
            )
        ]
        ax2.legend(handles=legend_elements, loc="upper right")
    else:
        ax2.text(
            0.5,
            0.5,
            "No configuration data available",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax2.axis("off")

    ax2.set_title(
        "Parameter Comparison for Top Configurations", fontsize=14, fontweight="bold"
    )

    # Add fold execution time information
    if fold_metrics:
        fold_times = [fold["time_seconds"] for fold in fold_metrics]
        time_text = (
            f"CV Info:\n"
            f"Folds: {len(fold_metrics)}\n"
            f"Avg Fold Time: {np.mean(fold_times):.2f}s\n"
            f"Total Time: {sum(fold_times):.2f}s"
        )

        plt.figtext(
            0.02,
            0.02,
            time_text,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    return fig


# Example usage:
# hpo_results = {...}  # Your structured output
# plot_performance_distribution(hpo_results, "performance_dist.png")
# plot_param_importance(hpo_results, "param_importance.png")
# plot_rank_correlation(hpo_results, "rank_correlation.png")
# plot_cross_validation_analysis(hpo_results, "cv_analysis.png")
