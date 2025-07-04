# > Libraries
import os
import sys

# Ensure parent dir is in path for 'utils' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
from collections import Counter
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
from sklearn.model_selection import KFold

from utils.logger import get_logger


# HPO methods
from .hpo_methods import GridSearch
from .hpo_methods import RandomSearch
from .hpo_methods import EvolutionaryAlgorithm


# > Hyperparameter Optimizer
class HPO:
    """A class to perform HPO on a given model with different HPO techniques."""

    def __init__(self, model, data_features, data_target, hp_values: dict, task: str):
        # Model
        self.model = model

        # Data
        self.features = data_features
        self.target = data_target

        # Hyperparameters
        self.hp = hp_values

        # Tasks
        self.task = task
        self.available_tasks = ["regression", "classification"]
        if self.task not in self.available_tasks:
            raise Exception(
                f"the selected task is not available, the only available tasks are: {self.available_tasks[0]}, {self.available_tasks[1]}"
            )

        # HPO methods
        self.available_methods = [
            "grid_search",
            "random_search",
            "evolutionary_algorithm",
        ]
        self.grid_search_method = GridSearch(hyperparameters=self.hp)
        self.random_search_method = RandomSearch(hyperparameters=self.hp)
        self.ea = EvolutionaryAlgorithm(
            model=model, hyperparameters=self.hp, task=self.task
        )

        # Logger
        self.logger = get_logger("HPO")
        self.logger.info("Initialized HPO class")

        # Ouptut
        self.structured_output = None

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

        self.logger.debug("Starting Inner CV loop")

        model = clone(self.model)
        model.set_params(**configuration)

        inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        scores = []

        try:
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

        except Exception as exc:
            self.logger.error(f"Error in inner CV loop : {exc}", exc_info=True)

        average_score = sum(scores) / n_splits
        self.logger.debug(
            f"Average score for the configuration {configuration} : {average_score:.4f} in the Inner CV loop"
        )

        return average_score

    # > Main functions

    # Grid search
    def grid_search(self, X, y, n_splits):
        self.logger.info("Starting Grid Search...")

        hyperparameters_configs = self.grid_search_method.grid_combinations()

        store_metrics = []

        for i, configuration in enumerate(hyperparameters_configs, 1):
            self.logger.debug(
                f"Evaluating config {i}/{len(hyperparameters_configs)}: {configuration}"
            )

            avg_score = self._inner_cross_validation(configuration, X, y, n_splits)
            store_metrics.append({"configuration": configuration, "score": avg_score})

        # Find the best configuration
        best_configuration = max(store_metrics, key=lambda x: x["score"])[
            "configuration"
        ]

        self.logger.info(
            f"Best config found from the Inner CV loop: {best_configuration}"
        )

        return best_configuration

    # Random search
    def random_search(self, X, y, n_splits, n_trials):
        """
        Perform random search hyperparameter optimization.

        Parameters:
        -----------
        X : pandas.DataFrame
            Features for training and validation.
        y : pandas.Series
            Target variable.
        n_splits : int
            Number of cross-validation splits for the inner CV loop.
        n_trials : int
            Number of random hyperparameter configurations to try.

        Returns:
        --------
        dict
            Best hyperparameter configuration found by random search.
        """
        self.logger.info("Starting Random Search...")

        # Handle error in n_trials setup
        if n_trials <= 0:
            self.logger.warning(
                "Number of trials is set to 0 or less. No configurations will be evaluated."
            )
            return None

        # Generate Random configurations
        configurations = self.random_search_method.random_configurations(
            n_trials=n_trials
        )

        store_metrics = []

        # Evaluate each configuration using the inner cross-validation
        for i, configuration in enumerate(configurations, 1):
            self.logger.debug(
                f"Evaluating random config {i}/{n_trials}: {configuration}"
            )

            avg_score = self._inner_cross_validation(configuration, X, y, n_splits)
            store_metrics.append({"configuration": configuration, "score": avg_score})

        if not store_metrics:
            self.logger.error("No valid configurations were successfully evaluated.")
            return None

        # Find the best configuration
        best_configuration = max(store_metrics, key=lambda x: x["score"])[
            "configuration"
        ]

        self.logger.info(
            f"Best config found from the Inner CV loop: {best_configuration}"
        )

        return best_configuration

    # Evolutionary Algorithm
    def evolutionary_algorithm(
        self,
        X,
        y,
        n_splits_cv,
        parents_selection_mechanism,
        parents_selection_ratio,
        max_generations,
        n_new_configs=None,
    ):
        best_config_EA = self.ea.evolution_process(
            X=X,
            y=y,
            n_splits_cv=n_splits_cv,
            parents_selection_mechanism=parents_selection_mechanism,
            parents_selection_ratio=parents_selection_ratio,
            max_generations=max_generations,
            n_new_configs=n_new_configs,
        )
        return best_config_EA

    # Main Hyperparameters tuning function
    def hp_tuning(
        self,
        hpo_method: str,
        outer_k: int,
        inner_k: int,
        n_trials: Optional[int] = None,
        parents_selection_mechanism: Optional[str] = None,
        parents_selection_ratio: Optional[float] = None,
        max_generations: Optional[int] = None,
        n_new_configs: Optional[int] = None,
        shuffle=True,
    ):
        """
        Perform hyperparameter optimization using nested cross-validation.
        Returns a simplified, non-redundant output optimized for data visualization.
        """

        self.logger.info(f"{'#' * 48} HPO {'#' * 48}")
        self.logger.info("Starting Nested Cross-Validation for HPO")
        self.logger.info("=" * 96)
        self.logger.info(
            f"Number of outer folds: {outer_k}, Number of inner folds: {inner_k}, Method: {hpo_method}"
        )
        self.logger.info(f"Target task: {self.task}")
        self.logger.info("=" * 96)

        # Data
        X = self.features
        y = self.target

        # Set up outer CV loop
        outer_cv = KFold(n_splits=outer_k, random_state=42, shuffle=shuffle)
        results_outer_cv = []

        # Store all configurations and their performances
        all_configs = {}

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
            self.logger.info(f"OUTER FOLD {fold_idx + 1}/{outer_k} STARTED")
            X_outer_train, X_outer_test = X.iloc[train_idx], X.iloc[test_idx]
            y_outer_train, y_outer_test = y.iloc[train_idx], y.iloc[test_idx]

            # Start time
            start_time = time.time()

            # Select hpo method
            if hpo_method not in self.available_methods:
                error_msg = f"The selected HPO technique is not available, the available methods are: {self.available_methods}"
                self.logger.error(error_msg)
                raise Exception(error_msg)

            elif hpo_method == "grid_search":
                hpo_technique = self.grid_search
                best_config_inner_cv = hpo_technique(
                    X=X_outer_train,
                    y=y_outer_train,
                    n_splits=inner_k,
                )
                # Remove 'fitness' key if present
                best_config_inner_cv.pop("fitness", None)

            elif hpo_method == "evolutionary_algorithm":
                hpo_technique = self.evolutionary_algorithm
                best_config_inner_cv = hpo_technique(
                    X=X_outer_train,
                    y=y_outer_train,
                    n_splits_cv=inner_k,
                    parents_selection_mechanism=parents_selection_mechanism,
                    parents_selection_ratio=parents_selection_ratio,
                    max_generations=max_generations,
                    n_new_configs=n_new_configs,
                )

                if "config" in best_config_inner_cv:
                    best_config_inner_cv = best_config_inner_cv["config"]

            elif hpo_method == "random_search":
                hpo_technique = self.random_search
                best_config_inner_cv = hpo_technique(
                    X=X_outer_train,
                    y=y_outer_train,
                    n_splits=inner_k,
                    n_trials=n_trials,
                )

                best_config_inner_cv.pop("fitness", None)

                if n_trials is None:
                    error_msg = "To use the random search method you need to specify the number of hyperparameter configurations you want to try"
                    self.logger.error(error_msg)

                    raise ValueError(error_msg)

            # End timing
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Inner CV execution time: {elapsed_time:.2f} seconds")

            # Evaluate on the outer test set the best configuration found by the hpo method

            model = clone(self.model)
            model.set_params(**best_config_inner_cv)
            model.fit(X_outer_train, y_outer_train)
            y_pred = model.predict(X_outer_test)

            # Compute metrics
            results = self._evaluation_metrics(y_true=y_outer_test, y_pred=y_pred)

            # Log metrics based on task type
            if self.task == "regression":
                self.logger.info(
                    f"Outer fold {fold_idx + 1} results - R2: {results['R2']:.4f}, MSE: {results['mse']:.4f} for congiguration: {best_config_inner_cv}"
                )
            elif self.task == "classification":
                self.logger.info(
                    f"Outer fold {fold_idx + 1} results - Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f} for congiguration: {best_config_inner_cv}"
                )

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
            self.logger.info(f"OUTER FOLD {fold_idx + 1}/{outer_k} COMPLETED")
            self.logger.info("=" * 96)

        self.logger.info("Outer CV completed, Results:")

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
        self.logger.info(
            f"Best weighted configuration: {best_config} with weight score {sorted_configs[0]['weighted_score']:.4f}"
        )

        # Get most frequent configuration
        config_counter = Counter(
            json.dumps(fold["config"], sort_keys=True) for fold in results_outer_cv
        )
        most_frequent_config_str = config_counter.most_common(1)[0][0]
        most_frequent_config = json.loads(most_frequent_config_str)
        most_frequent_count = config_counter.most_common(1)[0][1]
        self.logger.info(
            f"Most frequent configuration: {most_frequent_config} (appeared {most_frequent_count} times)"
        )

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

        self.logger.info(
            f"Best by performance: {performance_sorted_configs[0]['config']} with performance score {performance_sorted_configs[0]['score']:.4f}"
        )

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

        total_time = sum(fold["time_seconds"] for fold in results_outer_cv)
        self.logger.info(f"Total execution time: {total_time:.2f} seconds")
        self.logger.info("=" * 80)
        self.logger.info("Hyperparameter optimization complete :)")
        self.logger.info("=" * 80)

        # Prepare simplified results with visualization-friendly structure
        self.structured_output = {
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

        return self.structured_output


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
    model = RandomForestRegressor()
    data_features = X
    data_target = y
    hp_values = {
        "max_depth": [1, 3],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [10, 30],
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
    print(
        test_hpo.hp_tuning(
            hpo_method="grid_search",
            outer_k=5,
            inner_k=3,
        )
    )
