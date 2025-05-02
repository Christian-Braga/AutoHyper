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
        # logica da rivedere beene in particolare controllare che la selezione della best config sia ottimale
        # valutare come semplificarla per ridurre redundancy, fallo bene
        # valutare che vada bene anche per data visualization

        """Main function to perform HPO using nested cross validation with improved configuration selection"""
        # Data
        X = self.features
        y = self.target

        # Set up outer CV loop
        outer_cv = KFold(n_splits=outer_k, random_state=42, shuffle=shuffle)
        results_outer_cv = []

        # Store all configurations and their performances
        all_configs = {}

        for train_idx, test_idx in outer_cv.split(X):
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
                if n_trials == None:
                    raise ValueError(
                        "To use the random search method you need to specify the number of hyperparameter configurations you want to try"
                    )

            # Call the HPO method and obtain the best configuration in the inner_cv loop
            best_config_inner_cv = hpo_technique(
                X=X_outer_train, y=y_outer_train, n_splits=inner_k, n_trials=n_trials
            )

            # End timing
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Evaluate on the outer test set the best configuration found by the hpo method
            model = clone(self.model)
            model.set_params(**best_config_inner_cv)
            model.fit(X_outer_train, y_outer_train)
            y_pred = model.predict(X_outer_test)

            # Compute metrics
            results = self._evaluation_metrics(y_true=y_outer_test, y_pred=y_pred)
            fold_result = {
                "metrics": results,
                "hpo_time_seconds": elapsed_time,
                "best_config_inner_cv": best_config_inner_cv,
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

        # Compute average metrics for each configuration
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
                # Lower MSE is better, so we sort by negative MSE (or directly by R2)
                data["score"] = data["avg_r2"]  # or use -data["avg_mse"]
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
                # F1 score is often a good balanced metric for classification
                data["score"] = data["avg_f1"]
            else:
                raise ValueError(f"Unknown task type: {self.task}")

        # Sort configurations by performance score
        sorted_configs = sorted(
            all_configs.values(),
            key=lambda x: x["score"],
            reverse=True,  # Higher score is better
        )

        # Best configuration based on average metrics
        best_config = sorted_configs[0]["config"]

        # Also track most frequent configuration for comparison
        config_counter = Counter(
            json.dumps(fold["best_config_inner_cv"], sort_keys=True)
            for fold in results_outer_cv
        )
        max_frequency = max(config_counter.values())
        most_frequent_configs = [
            json.loads(config)
            for config, freq in config_counter.items()
            if freq == max_frequency
        ]

        # Calculate aggregate performance metrics across all folds
        if self.task == "regression":
            r2_scores = [fold["metrics"]["R2"] for fold in results_outer_cv]
            mse_scores = [fold["metrics"]["mse"] for fold in results_outer_cv]
            avg_r2 = sum(r2_scores) / outer_k
            avg_mse = sum(mse_scores) / outer_k
            overall_metrics = {"best_model_R2": avg_r2, "best_model_mse": avg_mse}
        elif self.task == "classification":
            accuracy_scores = [fold["metrics"]["accuracy"] for fold in results_outer_cv]
            precision_scores = [
                fold["metrics"]["precision"] for fold in results_outer_cv
            ]
            recall_scores = [fold["metrics"]["recall"] for fold in results_outer_cv]
            f1_scores = [fold["metrics"]["f1"] for fold in results_outer_cv]
            overall_metrics = {
                "best_model_accuracy": sum(accuracy_scores) / outer_k,
                "best_model_precision": sum(precision_scores) / outer_k,
                "best_model_recall": sum(recall_scores) / outer_k,
                "best_model_f1": sum(f1_scores) / outer_k,
            }

        # Prepare final results
        final_result = {
            "best_config_by_performance": best_config,
            "best_config_performance": sorted_configs[0],
            "most_frequent_configs": most_frequent_configs,
            "most_frequent_config_count": max_frequency,
            "all_configurations": sorted_configs,
            "overall_metrics": overall_metrics,  # Average metrics across all folds
            "outer_cv_results": results_outer_cv,  # Individual fold results
        }

        # Add overall metrics at the top level for backwards compatibility
        final_result.update(overall_metrics)

        return final_result


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
