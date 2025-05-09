import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D


# -- PLot N.1 --
def plot_performance_vs_frequency(hpo_results, figsize=(10, 6), save_path=None):
    """
    Creates a scatter plot comparing performance scores with selection frequency.

    Parameters:
    -----------
    hpo_results : dict
        The output dictionary from the HPO tuning process
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, the plot will be saved to this path

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Extract data from HPO results
    configs = hpo_results["configs_summary"]

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(
        [
            {
                "raw_score": c["raw_score"],
                "frequency": c["frequency"],
                "weighted_score": c["weighted_score"],
                "perf_rank": c["performance_rank"],
                "freq_rank": c["frequency_rank"],
                "weighted_rank": c["weighted_rank"],
                # Extract the first two key parameters for labeling
                "params": ", ".join(
                    [f"{k}:{v}" for k, v in list(c["params"].items())[:2]]
                ),
            }
            for c in configs
        ]
    )

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot
    scatter = ax.scatter(
        df["raw_score"],
        df["frequency"],
        s=100,
        c=df["weighted_score"],
        cmap="viridis",
        alpha=0.7,
        edgecolors="w",
    )

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Weighted Score")

    # Highlight special configurations
    best_config = next(c for c in configs if c["weighted_rank"] == 1)
    best_perf = next(c for c in configs if c["performance_rank"] == 1)
    most_freq = next(c for c in configs if c["frequency_rank"] == 1)

    # Plot best weighted score
    ax.scatter(
        best_config["raw_score"],
        best_config["frequency"],
        s=200,
        facecolors="none",
        edgecolors="red",
        linewidth=2,
        label="Best Weighted Score",
    )

    # Plot best performance (if different)
    if best_perf != best_config:
        ax.scatter(
            best_perf["raw_score"],
            best_perf["frequency"],
            s=200,
            facecolors="none",
            edgecolors="blue",
            linewidth=2,
            label="Best Performance",
        )

    # Plot most frequent (if different)
    if most_freq != best_config and most_freq != best_perf:
        ax.scatter(
            most_freq["raw_score"],
            most_freq["frequency"],
            s=200,
            facecolors="none",
            edgecolors="green",
            linewidth=2,
            label="Most Frequent",
        )

    # Add labels for the top configurations
    for i, row in df.sort_values("weighted_score", ascending=False).head(3).iterrows():
        ax.annotate(
            row["params"],
            xy=(row["raw_score"], row["frequency"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Set labels and title
    ax.set_xlabel("Performance Score")
    ax.set_ylabel("Selection Frequency")
    ax.set_title("Performance vs. Frequency of Hyperparameter Configurations")

    # Add grid and legend
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.show(fig)


# -- PLot N.2 --
def plot_parameter_importance(hpo_results, figsize=(10, 6), save_path=None):
    """
    Creates a bar plot showing the relative importance of each hyperparameter.

    Parameters:
    -----------
    hpo_results : dict
        The output dictionary from the HPO tuning process
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, the plot will be saved to this path

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Extract data from HPO results
    configs = hpo_results["configs_summary"]

    # Get all unique parameters
    all_params = set()
    for config in configs:
        all_params.update(config["params"].keys())

    # Compute parameter importance for each parameter
    param_importance = {}

    for param in all_params:
        # Group configs by parameter value
        param_values = {}
        for config in configs:
            if param in config["params"]:
                value = config["params"][param]
                if value not in param_values:
                    param_values[value] = []
                param_values[value].append(config["raw_score"])

        # Calculate variance of means across parameter values
        if len(param_values) > 1:
            means = [np.mean(scores) for scores in param_values.values()]
            param_importance[param] = np.std(means)
        else:
            param_importance[param] = 0

    # Normalize importance scores
    max_importance = max(param_importance.values())
    if max_importance > 0:
        for param in param_importance:
            param_importance[param] /= max_importance

    # Create DataFrame for plotting
    df = pd.DataFrame(
        {
            "Parameter": list(param_importance.keys()),
            "Importance": list(param_importance.values()),
        }
    ).sort_values("Importance", ascending=False)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create bar plot
    bars = ax.barh(
        df["Parameter"], df["Importance"], color=plt.cm.viridis(df["Importance"])
    )

    # Add values to the bars
    for i, v in enumerate(df["Importance"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")

    # Set labels and title
    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("Hyperparameter")
    ax.set_title("Hyperparameter Importance")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.show(fig)


# -- PLot N.3 --
def plot_parameter_space(
    hpo_results, param_x, param_y, param_size=None, figsize=(10, 8), save_path=None
):
    """
    Creates a parameter space visualization showing how different combinations of parameters
    affect performance.

    Parameters:
    -----------
    hpo_results : dict
        The output dictionary from the HPO tuning process
    param_x : str
        The parameter to plot on the x-axis
    param_y : str
        The parameter to plot on the y-axis
    param_size : str, optional
        If provided, this parameter will be represented by the size of the points
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, the plot will be saved to this path

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Extract data from HPO results
    configs = hpo_results["configs_summary"]

    # Filter configs that have the required parameters
    valid_configs = []
    for config in configs:
        if param_x in config["params"] and param_y in config["params"]:
            if param_size is None or param_size in config["params"]:
                valid_configs.append(config)

    if not valid_configs:
        raise ValueError(
            f"No configurations found with parameters {param_x} and {param_y}"
        )

    # Extract parameter values and scores
    x_values = [config["params"][param_x] for config in valid_configs]
    y_values = [config["params"][param_y] for config in valid_configs]
    scores = [config["raw_score"] for config in valid_configs]
    frequencies = [config["frequency"] for config in valid_configs]

    if param_size:
        size_values = [config["params"][param_size] for config in valid_configs]
        # Normalize sizes for plotting
        size_norm = [
            (v - min(size_values)) / (max(size_values) - min(size_values) + 1e-10) * 300
            + 50
            for v in size_values
        ]
    else:
        size_norm = [100] * len(valid_configs)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot
    scatter = ax.scatter(
        x_values,
        y_values,
        s=size_norm,
        c=scores,
        cmap="viridis",
        alpha=0.7,
        edgecolors="w",
    )

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Performance Score")

    # Add labels
    for i, (x, y, score, freq) in enumerate(
        zip(x_values, y_values, scores, frequencies)
    ):
        if freq > np.median(frequencies) or score > np.median(scores):
            ax.annotate(
                f"Score: {score:.3f}\nFreq: {freq}",
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

    # Set labels and title
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    title = f"Parameter Space: {param_x} vs {param_y}"
    if param_size:
        title += f" (Size: {param_size})"
    ax.set_title(title)

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.show(fig)


# -- PLot N.4 --
def plot_cv_stability(hpo_results, top_n=5, figsize=(12, 6), save_path=None):
    """
    Creates a plot showing the stability of hyperparameter configurations across CV folds.

    Parameters:
    -----------
    hpo_results : dict
        The output dictionary from the HPO tuning process
    top_n : int, optional
        Number of top configurations to include
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, the plot will be saved to this path

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Extract fold metrics
    fold_metrics = hpo_results["fold_metrics"]

    # Group configurations across folds
    config_metrics = {}

    for fold in fold_metrics:
        fold_num = fold.get("fold", 0)
        for config in fold["configs"]:
            # Create a key for the configuration
            config_key = tuple(sorted(config["params"].items()))

            if config_key not in config_metrics:
                config_metrics[config_key] = {
                    "params": dict(config_key),
                    "scores": [],
                    "folds": [],
                }

            config_metrics[config_key]["scores"].append(config["score"])
            config_metrics[config_key]["folds"].append(fold_num)

    # Calculate statistics for each config
    for config_key, metrics in config_metrics.items():
        metrics["mean"] = np.mean(metrics["scores"])
        metrics["std"] = np.std(metrics["scores"])
        metrics["cv"] = metrics["std"] / metrics["mean"] if metrics["mean"] > 0 else 0
        metrics["stability"] = 1 / (metrics["cv"] + 1e-10)  # Higher is more stable

    # Sort by mean performance and select top N
    top_configs = sorted(
        config_metrics.values(), key=lambda x: x["mean"], reverse=True
    )[:top_n]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Mean and std of scores
    config_labels = [
        ", ".join([f"{k}:{v}" for k, v in list(c["params"].items())[:2]])
        for c in top_configs
    ]
    means = [c["mean"] for c in top_configs]
    stds = [c["std"] for c in top_configs]

    y_pos = np.arange(len(config_labels))

    ax1.barh(
        y_pos,
        means,
        xerr=stds,
        align="center",
        alpha=0.7,
        color=plt.cm.viridis(np.linspace(0, 1, len(config_labels))),
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(config_labels)
    ax1.set_xlabel("Mean Score (with Std Dev)")
    ax1.set_title("Performance Stability Across Folds")
    ax1.grid(True, linestyle="--", alpha=0.7, axis="x")

    # Plot 2: Coefficient of variation (lower is more stable)
    stability = [c["stability"] for c in top_configs]

    ax2.barh(
        y_pos,
        stability,
        align="center",
        alpha=0.7,
        color=plt.cm.viridis(np.linspace(0, 1, len(config_labels))),
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(config_labels)
    ax2.set_xlabel("Stability Score (1/CV)")
    ax2.set_title("Configuration Stability (Higher is Better)")
    ax2.grid(True, linestyle="--", alpha=0.7, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.show(fig)


# -- PLot N.5 --
def plot_rank_comparison(hpo_results, top_n=10, figsize=(12, 8), save_path=None):
    """
    Creates a plot comparing different ranking methods (performance, frequency, weighted).

    Parameters:
    -----------
    hpo_results : dict
        The output dictionary from the HPO tuning process
    top_n : int, optional
        Number of top configurations to include
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, the plot will be saved to this path

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Extract data from HPO results
    configs = hpo_results["configs_summary"][:top_n]

    # Prepare data for plotting
    config_labels = []
    perf_ranks = []
    freq_ranks = []
    weighted_ranks = []

    for config in configs:
        # Create config label from first two parameters
        params = config["params"]
        label = ", ".join([f"{k}:{v}" for k, v in list(params.items())[:2]])
        config_labels.append(label)

        perf_ranks.append(config["performance_rank"])
        freq_ranks.append(config["frequency_rank"])
        weighted_ranks.append(config["weighted_rank"])

    # Create DataFrame for plotting
    df = pd.DataFrame(
        {
            "Config": config_labels,
            "Performance Rank": perf_ranks,
            "Frequency Rank": freq_ranks,
            "Weighted Rank": weighted_ranks,
        }
    )

    # Melt DataFrame for grouped bar plot
    df_melted = pd.melt(
        df,
        id_vars=["Config"],
        value_vars=["Performance Rank", "Frequency Rank", "Weighted Rank"],
        var_name="Ranking Method",
        value_name="Rank",
    )

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create grouped bar plot
    sns.barplot(
        x="Config",
        y="Rank",
        hue="Ranking Method",
        data=df_melted,
        palette="viridis",
        ax=ax,
    )

    # Set labels and title
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Rank (Lower is Better)")
    ax.set_title("Comparison of Different Ranking Methods")

    # Improve x-axis labels
    plt.xticks(rotation=45, ha="right")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")

    # Add weighting factor information
    weight_info = hpo_results.get("weighting_factors", {})
    if weight_info:
        perf_weight = weight_info.get("performance_weight", 0)
        freq_weight = weight_info.get("frequency_weight", 0)
        ax.text(
            0.01,
            0.01,
            f"Weighting: Performance ({perf_weight:.1%}), Frequency ({freq_weight:.1%})",
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.5"),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.show(fig)


# -- PLot N.6 --
def plot_fold_performance(hpo_results, top_n=4, figsize=(10, 6), save_path=None):
    """
    Creates a line plot showing how top configurations perform across different folds.

    Parameters:
    -----------
    hpo_results : dict
        The output dictionary from the HPO tuning process
    top_n : int, optional
        Number of top configurations to include
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, the plot will be saved to this path

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Extract fold metrics and configs
    fold_metrics = hpo_results["fold_metrics"]
    top_configs = hpo_results["configs_summary"][:top_n]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create color map for configurations
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_configs)))

    # Track unique folds
    all_folds = set()

    # Plot each top configuration's performance across folds
    for i, config in enumerate(top_configs):
        config_params = config["params"]
        config_label = ", ".join(
            [f"{k}:{v}" for k, v in list(config_params.items())[:2]]
        )

        # Find this config in each fold
        fold_scores = []
        fold_numbers = []

        for fold in fold_metrics:
            fold_num = fold.get("fold", len(fold_scores) + 1)
            all_folds.add(fold_num)

            # Look for matching config in this fold
            for fold_config in fold["configs"]:
                # Check if this is the same config
                is_match = True
                for k, v in config_params.items():
                    if k not in fold_config["params"] or fold_config["params"][k] != v:
                        is_match = False
                        break

                if is_match:
                    fold_scores.append(fold_config["score"])
                    fold_numbers.append(fold_num)
                    break

        # Sort by fold number
        fold_data = sorted(zip(fold_numbers, fold_scores))
        if fold_data:
            fold_numbers, fold_scores = zip(*fold_data)

            # Plot the line
            ax.plot(
                fold_numbers,
                fold_scores,
                "o-",
                color=colors[i],
                label=config_label,
                linewidth=2,
            )

    # Set labels and title
    ax.set_xlabel("Fold Number")
    ax.set_ylabel("Performance Score")
    ax.set_title("Performance of Top Configurations Across Folds")

    # Set x-ticks to integers
    ax.set_xticks(sorted(list(all_folds)))

    # Add grid and legend
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="best")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.show(fig)


# -- PLot N.7 --
def plot_weighted_score_components(
    hpo_results, top_n=8, figsize=(12, 6), save_path=None
):
    """
    Creates a stacked bar chart showing how performance and frequency components
    contribute to the final weighted score.

    Parameters:
    -----------
    hpo_results : dict
        The output dictionary from the HPO tuning process
    top_n : int, optional
        Number of top configurations to include
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, the plot will be saved to this path

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Extract data from HPO results
    configs = hpo_results["configs_summary"][:top_n]
    weights = hpo_results.get(
        "weighting_factors", {"performance_weight": 0.7, "frequency_weight": 0.3}
    )

    perf_weight = weights.get("performance_weight", 0.7)
    freq_weight = weights.get("frequency_weight", 0.3)

    # Prepare data for plotting
    config_labels = []
    perf_components = []
    freq_components = []
    weighted_scores = []

    for config in configs:
        # Create config label from first two parameters
        params = config["params"]
        label = ", ".join([f"{k}:{v}" for k, v in list(params.items())[:2]])
        config_labels.append(label)

        # Get normalized raw score (assume between 0 and 1)
        raw_score = config["raw_score"]

        # Get normalized frequency
        freq_normalized = config.get(
            "frequency_normalized",
            config["frequency"] / sum(c["frequency"] for c in configs),
        )

        # Calculate components
        perf_component = raw_score * perf_weight
        freq_component = freq_normalized * freq_weight

        perf_components.append(perf_component)
        freq_components.append(freq_component)
        weighted_scores.append(config["weighted_score"])

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create stacked bar chart
    bar_width = 0.6
    bars1 = ax.bar(
        config_labels, perf_components, bar_width, label="Performance Component"
    )
    bars2 = ax.bar(
        config_labels,
        freq_components,
        bar_width,
        bottom=perf_components,
        label="Frequency Component",
    )

    # Add weighted score line
    ax.plot(config_labels, weighted_scores, "ro-", linewidth=2, label="Weighted Score")

    # Set labels and title
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score Component")
    ax.set_title(
        f"Weighted Score Components (Perf Weight: {perf_weight:.1%}, Freq Weight: {freq_weight:.1%})"
    )

    # Improve x-axis labels
    plt.xticks(rotation=45, ha="right")

    # Add grid and legend
    ax.grid(True, linestyle="--", alpha=0.7, axis="y")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.show(fig)


# -- PLot N.8 --
def plot_parameter_interaction(
    hpo_results, param1, param2, figsize=(10, 8), save_path=None
):
    """
    Creates a heatmap showing how the interaction between two parameters affects performance.

    Parameters:
    -----------
    hpo_results : dict
        The output dictionary from the HPO tuning process
    param1 : str
        Name of the first parameter
    param2 : str
        Name of the second parameter
    figsize : tuple, optional
        Figure size (width, height) in inches
    save_path : str, optional
        If provided, the plot will be saved to this path

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure object
    """
    # Extract data from HPO results
    configs = hpo_results["configs_summary"]

    # Filter configs that have both parameters
    valid_configs = [
        c for c in configs if param1 in c["params"] and param2 in c["params"]
    ]

    if not valid_configs:
        raise ValueError(
            f"No configurations found with both parameters {param1} and {param2}"
        )

    # Get unique values for each parameter
    param1_values = sorted(set(c["params"][param1] for c in valid_configs))
    param2_values = sorted(set(c["params"][param2] for c in valid_configs))

    # Create a grid of scores
    score_grid = np.zeros((len(param2_values), len(param1_values)))
    count_grid = np.zeros((len(param2_values), len(param1_values)))

    for config in valid_configs:
        p1_idx = param1_values.index(config["params"][param1])
        p2_idx = param2_values.index(config["params"][param2])

        score_grid[p2_idx, p1_idx] += config["raw_score"]
        count_grid[p2_idx, p1_idx] += 1

    # Calculate average scores
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_grid = np.divide(score_grid, count_grid)
        avg_grid = np.nan_to_num(avg_grid)  # Replace NaNs with zeros

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create custom colormap from red to green
    cmap = LinearSegmentedColormap.from_list(
        "RdYlGn", ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]
    )

    # Create heatmap
    im = ax.imshow(avg_grid, cmap=cmap, aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Average Performance Score")

    # Set labels and title
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_title(f"Parameter Interaction: {param1} vs {param2}")

    # Set ticks
    ax.set_xticks(np.arange(len(param1_values)))
    ax.set_yticks(np.arange(len(param2_values)))
    ax.set_xticklabels(param1_values)
    ax.set_yticklabels(param2_values)

    # Rotate tick labels if needed
    plt.xticks(rotation=45, ha="right")

    # Add text annotations
    for i in range(len(param2_values)):
        for j in range(len(param1_values)):
            if count_grid[i, j] > 0:
                text = ax.text(
                    j,
                    i,
                    f"{avg_grid[i, j]:.3f}\n(n={int(count_grid[i, j])})",
                    ha="center",
                    va="center",
                    color="black" if avg_grid[i, j] < 0.85 else "white",
                    fontsize=8,
                )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.show(fig)
