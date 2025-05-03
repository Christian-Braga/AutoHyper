# > Libraries
import itertools
import json
import time
import random
from collections import Counter
from typing import Optional


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.base import clone
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split


# > Hyperparameter Optimizer
class HPO:
    """A class to perform HPO on a given model with different HPO techniques."""

    def __init__(self, model, data_features, data_target, hp_values: dict, task: str):
        self.model = model
        self.features = data_features
        self.target = data_target
        self.hp = hp_values
        self.task = task

        self.available_tasks = ["regression", "classification"]
        if self.task not in self.available_tasks:
            raise Exception(
                f"the selected task is not available, the only available tasks are: {self.available_tasks[0]}, {self.available_tasks[1]}"
            )
        self.available_methods = ["grid_search", "random_search"]

    # > Support Functions

    def _evaluation_metrics(self, y_true, y_pred):
        """Compute different metrics for evaluating performances."""
        metrics = {}

        if self.task == "regression":
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            metrics["R2"] = r2
            metrics["mse"] = mse

        elif self.task == "classification":
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["accuracy"] = accuracy
            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1"] = f1

        else:
            raise ValueError(f"Unknown task type: {self.task}")

        return metrics

    def _inner_cross_validation(self, configuration, X, y, n_splits):
        """Evaluate a single hyperparameter configuration using K-Fold cross-validation. Returns the average performance metric."""
        model = clone(self.model)
        model.set_params(**configuration)

        inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        scores = []

        for train_idx, test_idx in inner_cv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            metrics = self._evaluation_metrics(y_true=y_val, y_pred=y_pred)

            if self.task == "regression":
                scores.append(metrics["R2"])
            elif self.task == "classification":
                scores.append(metrics["accuracy"])
            else:
                raise ValueError(f"Unknown task type: {self.task}")

        average_score = sum(scores) / n_splits
        return average_score

    # > Main functions

    # Grid search
    def grid_search(self, X, y, n_splits, n_trials: Optional[int] = None):
        # Create the grid combination
        hp_keys = list(self.hp.keys())
        hp_values = list(self.hp.values())
        param_combinations = list(itertools.product(*hp_values))
        hyperparameters_configs = [
            dict(zip(hp_keys, combo)) for combo in param_combinations
        ]

        store_metrics = []

        for configuration in hyperparameters_configs:
            avg_score = self._inner_cross_validation(configuration, X, y, n_splits)
            store_metrics.append({"configuration": configuration, "score": avg_score})

        # Find the best configuration
        best_configuration = max(store_metrics, key=lambda x: x["score"])[
            "configuration"
        ]

        return best_configuration

    # Random search
    def random_search(self, X, y, n_splits, n_trials):
        store_metrics = []

        # Start the random sampling loop
        for trial in range(n_trials):
            configuration = {}

            # Create the configuration
            for param, values in self.hp.items():
                if all(isinstance(v, int) for v in values):
                    low, high = min(values), max(values)
                    configuration[param] = random.randint(low, high)
                elif all(isinstance(v, float) for v in values):
                    low, high = min(values), max(values)
                    configuration[param] = random.uniform(low, high)
                else:
                    # Fallback for categorical or mixed types
                    configuration[param] = random.choice(values)

            avg_score = self._inner_cross_validation(configuration, X, y, n_splits)
            store_metrics.append({"configuration": configuration, "score": avg_score})

        # Find the best configuration
        best_configuration = max(store_metrics, key=lambda x: x["score"])[
            "configuration"
        ]
        return best_configuration

    # Plot results of the tuning loop
    def plot_results(self):
        pass

    def hp_tuning(
        self,
        hpo_method: str,
        outer_k: int,
        inner_k: int,
        n_trials: Optional[int] = None,
        shuffle=True,
    ):
        """
        Perform hyperparameter optimization using nested cross-validation.
        Returns a simplified, non-redundant output optimized for data visualization.
        """
        # Data
        X = self.features
        y = self.target

        # Set up outer CV loop
        outer_cv = KFold(n_splits=outer_k, random_state=42, shuffle=shuffle)
        results_outer_cv = []

        # Store all configurations and their performances
        all_configs = {}

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
            X_outer_train, X_outer_test = X.iloc[train_idx], X.iloc[test_idx]
            y_outer_train, y_outer_test = y.iloc[train_idx], y.iloc[test_idx]

            # Start time
            start_time = time.time()

            # Select hpo method
            if hpo_method not in self.available_methods:
                raise Exception(
                    f"The selected HPO technique is not available, the available methods are: {self.available_methods}"
                )
            elif hpo_method == "grid_search":
                hpo_technique = self.grid_search
            elif hpo_method == "random_search":
                hpo_technique = self.random_search
                if n_trials is None:
                    raise ValueError(
                        "To use the random search method you need to specify the number of hyperparameter configurations you want to try"
                    )

            # Call the HPO method and obtain the best configuration in the inner_cv loop
            best_config_inner_cv = hpo_technique(
                X=X_outer_train, y=y_outer_train, n_splits=inner_k, n_trials=n_trials
            )

            # End timing
            elapsed_time = time.time() - start_time

            # Evaluate on the outer test set the best configuration found by the hpo method
            model = clone(self.model)
            model.set_params(**best_config_inner_cv)
            model.fit(X_outer_train, y_outer_train)
            y_pred = model.predict(X_outer_test)

            # Compute metrics
            results = self._evaluation_metrics(y_true=y_outer_test, y_pred=y_pred)
            fold_result = {
                "fold": fold_idx,
                "metrics": results,
                "time_seconds": elapsed_time,
                "config": best_config_inner_cv,
            }
            results_outer_cv.append(fold_result)

            # Track this configuration's performance
            config_str = json.dumps(best_config_inner_cv, sort_keys=True)
            if config_str not in all_configs:
                all_configs[config_str] = {
                    "config": best_config_inner_cv,
                    "performances": [],
                    "count": 0,
                }

            # Add this fold's metrics
            all_configs[config_str]["performances"].append(results)
            all_configs[config_str]["count"] += 1

        # Process configuration performances
        for config_str, data in all_configs.items():
            performances = data["performances"]
            n_appearances = len(performances)

            # Calculate average metrics for each configuration
            if self.task == "regression":
                data["avg_r2"] = (
                    sum(perf["R2"] for perf in performances) / n_appearances
                )
                data["avg_mse"] = (
                    sum(perf["mse"] for perf in performances) / n_appearances
                )
                data["score"] = data["avg_r2"]
            elif self.task == "classification":
                data["avg_accuracy"] = (
                    sum(perf["accuracy"] for perf in performances) / n_appearances
                )
                data["avg_precision"] = (
                    sum(perf["precision"] for perf in performances) / n_appearances
                )
                data["avg_recall"] = (
                    sum(perf["recall"] for perf in performances) / n_appearances
                )
                data["avg_f1"] = (
                    sum(perf["f1"] for perf in performances) / n_appearances
                )
                data["score"] = data["avg_f1"]
            else:
                raise ValueError(f"Unknown task type: {self.task}")

        # Calculate the frequency-weighted scores
        max_count = max(data["count"] for data in all_configs.values())
        for data in all_configs.values():
            # Normalize frequency to [0, 1]
            frequency_normalized = data["count"] / max_count

            # Calculate weighted score (70% performance, 30% frequency)
            performance_weight = 0.7
            frequency_weight = 0.3
            data["weighted_score"] = (data["score"] * performance_weight) + (
                frequency_normalized * frequency_weight
            )

            # Store raw components for analysis
            data["frequency_normalized"] = frequency_normalized

        # Sort configurations by weighted score (descending)
        sorted_configs = sorted(
            all_configs.values(),
            key=lambda x: x["weighted_score"],
            reverse=True,
        )

        # Sort configurations by raw performance score (for comparison)
        performance_sorted_configs = sorted(
            all_configs.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        # Best configuration based on weighted metrics
        best_config = sorted_configs[0]["config"]

        # Get most frequent configuration
        config_counter = Counter(
            json.dumps(fold["config"], sort_keys=True) for fold in results_outer_cv
        )
        most_frequent_config_str = config_counter.most_common(1)[0][0]
        most_frequent_config = json.loads(most_frequent_config_str)
        most_frequent_count = config_counter.most_common(1)[0][1]

        # Generate all ranking information

        # Rank by frequency
        configs_by_frequency = sorted(
            list(all_configs.values()), key=lambda x: x["count"], reverse=True
        )
        for idx, config in enumerate(configs_by_frequency):
            config["frequency_rank"] = idx + 1

        # Rank by pure performance
        for idx, config in enumerate(performance_sorted_configs):
            config["performance_rank"] = idx + 1

        # Rank by weighted score
        for idx, config in enumerate(sorted_configs):
            config["weighted_rank"] = idx + 1

        # Calculate metrics for visualization
        metrics_by_fold = []
        for fold_result in results_outer_cv:
            fold_metrics = fold_result["metrics"].copy()
            fold_metrics["fold"] = fold_result["fold"]
            fold_metrics["config"] = json.dumps(fold_result["config"], sort_keys=True)
            metrics_by_fold.append(fold_metrics)

        # Calculate overall metrics
        overall_metrics = {}
        if self.task == "regression":
            overall_metrics["r2_mean"] = (
                sum(fold["metrics"]["R2"] for fold in results_outer_cv) / outer_k
            )
            overall_metrics["r2_std"] = np.std(
                [fold["metrics"]["R2"] for fold in results_outer_cv]
            )
            overall_metrics["mse_mean"] = (
                sum(fold["metrics"]["mse"] for fold in results_outer_cv) / outer_k
            )
            overall_metrics["mse_std"] = np.std(
                [fold["metrics"]["mse"] for fold in results_outer_cv]
            )
        elif self.task == "classification":
            metrics_keys = ["accuracy", "precision", "recall", "f1"]
            for key in metrics_keys:
                values = [fold["metrics"][key] for fold in results_outer_cv]
                overall_metrics[f"{key}_mean"] = sum(values) / outer_k
                overall_metrics[f"{key}_std"] = np.std(values)

        # Prepare simplified results with visualization-friendly structure
        return {
            "best_config": {
                "params": best_config,
                "weighted_score": sorted_configs[0]["weighted_score"],
                "performance_score": sorted_configs[0]["score"],
                "performance_rank": sorted_configs[0]["performance_rank"],
                "frequency": sorted_configs[0]["count"],
                "frequency_rank": sorted_configs[0]["frequency_rank"],
                "weighted_rank": 1,  # Always 1 since it's sorted by weighted score
            },
            "most_frequent_config": {
                "params": most_frequent_config,
                "frequency": most_frequent_count,
                "frequency_rank": 1,  # Always 1 since it's the most frequent
            },
            "best_by_performance": {
                "params": performance_sorted_configs[0]["config"],
                "performance_score": performance_sorted_configs[0]["score"],
                "performance_rank": 1,  # Always 1 since it's sorted by performance
                "weighted_rank": next(
                    c["weighted_rank"]
                    for c in sorted_configs
                    if json.dumps(c["config"], sort_keys=True)
                    == json.dumps(
                        performance_sorted_configs[0]["config"], sort_keys=True
                    )
                ),
                "frequency": performance_sorted_configs[0]["count"],
                "frequency_rank": performance_sorted_configs[0]["frequency_rank"],
            },
            "overall_metrics": overall_metrics,
            "configs_summary": [
                {
                    "params": c["config"],
                    "raw_score": c["score"],
                    "weighted_score": c["weighted_score"],
                    "performance_rank": c["performance_rank"],
                    "weighted_rank": c["weighted_rank"],
                    "frequency": c["count"],
                    "frequency_rank": c["frequency_rank"],
                    "frequency_normalized": c["frequency_normalized"],
                }
                for c in sorted_configs
            ],
            "fold_metrics": metrics_by_fold,
            "execution_time": sum(fold["time_seconds"] for fold in results_outer_cv),
            "weighting_factors": {"performance_weight": 0.7, "frequency_weight": 0.3},
        }


# Test the class
if __name__ == "__main__":
    # Load the California Housing Dataset and convert it into a pandas DataFrame
    X, y = fetch_california_housing(return_X_y=True)
    X = pd.DataFrame(
        X,
        columns=[
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ],
    )
    y = pd.Series(y, name="target")

    # attribute values
    model = xgb.XGBRegressor()
    data_features = X
    data_target = y
    hp_values = {
        "max_depth": [1, 3, 5],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [10, 30, 50],
    }
    task = "regression"

    # instance of HPO
    test_hpo = HPO(
        model=model,
        data_features=data_features,
        data_target=data_target,
        hp_values=hp_values,
        task=task,
    )

    # test run
    print(test_hpo.hp_tuning(hpo_method="random_search", outer_k=5, inner_k=3))
