# > Libraries
import itertools
import json
import time
from collections import Counter

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
    def grid_search(self, X, y, n_splits):
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
    def random_search(self, X, y, n_splits):
        # the logic is that i will simply sample at random into the range of the configurations defined by the user
        # the problem is that i need to delegate the use the correct setting of the hp configuration in input, in particular
        # abourt the integer and float values of the hp, so also the sampling need to respect this (sample only interger for some hp,
        # sample also float for other hpo)
        # so the sampled configuration is configured in the standard format, given into the inner search loop, and the output putted
        # into the outer loop
        pass

    # Plot results of the tuning loop
    def plot_results(self):
        pass

    def hp_tuning(self, hpo_method: str, outer_k: int, inner_k: int, shuffle=True):
        """Main function to perform HPO using nested cross validation with improved configuration selection"""
        # Data
        X = self.features
        y = self.target

        # Set up outer CV loop
        outer_cv = KFold(n_splits=outer_k, random_state=42, shuffle=shuffle)
        fold_results = []

        # Dictionary to track configurations and their performance
        config_tracker = {}

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
            X_outer_train, X_outer_test = X.iloc[train_idx], X.iloc[test_idx]
            y_outer_train, y_outer_test = y.iloc[train_idx], y.iloc[test_idx]

            # Start time
            start_time = time.time()

            # Select HPO method
            if hpo_method not in self.available_methods:
                raise Exception(
                    f"The selected HPO technique is not available, the available methods are: {self.available_methods}"
                )
            elif hpo_method == "grid_search":
                hpo_technique = self.grid_search
            elif hpo_method == "random_search":
                hpo_technique = self.random_search

            # Call the HPO method and obtain the best configuration in the inner_cv loop
            best_config_inner_cv = hpo_technique(
                X=X_outer_train, y=y_outer_train, n_splits=inner_k
            )

            # End timing
            elapsed_time = time.time() - start_time

            # Evaluate on the outer test set the best configuration found by the hpo method
            model = clone(self.model)
            model.set_params(**best_config_inner_cv)
            model.fit(X_outer_train, y_outer_train)
            y_pred = model.predict(X_outer_test)

            # Compute metrics
            metrics = self._evaluation_metrics(y_true=y_outer_test, y_pred=y_pred)

            # Store results for this fold
            fold_result = {
                "fold": fold_idx,
                "metrics": metrics,
                "time": elapsed_time,
                "config": best_config_inner_cv,
            }
            fold_results.append(fold_result)

            # Track this configuration's performance
            config_str = json.dumps(best_config_inner_cv, sort_keys=True)
            if config_str not in config_tracker:
                config_tracker[config_str] = {
                    "config": best_config_inner_cv,
                    "metrics": [],
                    "folds": [],
                }

            # Add this fold's metrics and fold index
            config_tracker[config_str]["metrics"].append(metrics)
            config_tracker[config_str]["folds"].append(fold_idx)

        # Process all tracked configurations
        for config_str, data in config_tracker.items():
            metrics_list = data["metrics"]
            n_appearances = len(metrics_list)

            # Calculate average metrics based on task type
            if self.task == "regression":
                data["avg_metrics"] = {
                    "r2": sum(m["R2"] for m in metrics_list) / n_appearances,
                    "mse": sum(m["mse"] for m in metrics_list) / n_appearances,
                }
                # For regression, use R2 as primary score (higher is better)
                data["primary_score"] = data["avg_metrics"]["r2"]

            elif self.task == "classification":
                data["avg_metrics"] = {
                    "accuracy": sum(m["accuracy"] for m in metrics_list)
                    / n_appearances,
                    "precision": sum(m["precision"] for m in metrics_list)
                    / n_appearances,
                    "recall": sum(m["recall"] for m in metrics_list) / n_appearances,
                    "f1": sum(m["f1"] for m in metrics_list) / n_appearances,
                }
                # For classification, use F1 as primary score (higher is better)
                data["primary_score"] = data["avg_metrics"]["f1"]

            else:
                raise ValueError(f"Unknown task type: {self.task}")

            # Add frequency information
            data["frequency"] = n_appearances
            data["frequency_pct"] = (n_appearances / outer_k) * 100

        # Rank configurations by performance
        configs_by_performance = sorted(
            list(config_tracker.values()),
            key=lambda x: x["primary_score"],
            reverse=True,  # Higher score is better
        )

        # Rank configurations by frequency
        configs_by_frequency = sorted(
            list(config_tracker.values()),
            key=lambda x: x["frequency"],
            reverse=True,  # Higher frequency is better
        )

        # Calculate overall metrics across all folds
        if self.task == "regression":
            overall_metrics = {
                "r2_mean": np.mean([fold["metrics"]["R2"] for fold in fold_results]),
                "r2_std": np.std([fold["metrics"]["R2"] for fold in fold_results]),
                "mse_mean": np.mean([fold["metrics"]["mse"] for fold in fold_results]),
                "mse_std": np.std([fold["metrics"]["mse"] for fold in fold_results]),
            }
        elif self.task == "classification":
            overall_metrics = {
                "accuracy_mean": np.mean(
                    [fold["metrics"]["accuracy"] for fold in fold_results]
                ),
                "accuracy_std": np.std(
                    [fold["metrics"]["accuracy"] for fold in fold_results]
                ),
                "f1_mean": np.mean([fold["metrics"]["f1"] for fold in fold_results]),
                "f1_std": np.std([fold["metrics"]["f1"] for fold in fold_results]),
            }

        # Build the final result structure
        final_result = {
            "best_config": configs_by_performance[0]["config"],
            "best_config_metrics": configs_by_performance[0]["avg_metrics"],
            "best_config_frequency": configs_by_performance[0]["frequency"],
            "overall_metrics": overall_metrics,
            "hpo_method": hpo_method,
            "configs_ranked_by_performance": configs_by_performance[
                :5
            ],  # Top 5 by performance
            "configs_ranked_by_frequency": configs_by_frequency[
                :5
            ],  # Top 5 by frequency
        }

        return final_result

    # Hyperparameters tuning loop
    # def hp_tuning(self, hpo_method: list, outer_k: int, inner_k: int, shuffle=True):
    #     """Main function to perform HPO using nested cross validation"""
    #     # Data
    #     X = self.features
    #     y = self.target

    #     # Set up outer CV loop
    #     outer_cv = KFold(n_splits=outer_k, random_state=42, shuffle=True)
    #     results_outer_cv = []

    #     for train_idx, test_idx in outer_cv.split(X):
    #         X_outer_train, X_outer_test = X.iloc[train_idx], X.iloc[test_idx]
    #         y_outer_train, y_outer_test = y.iloc[train_idx], y.iloc[test_idx]

    #         # Start time
    #         start_time = time.time()

    #         # Select hpo method
    #         if hpo_method not in self.available_methods:
    #             raise Exception(
    #                 f"the selected hpo techniques is not available, the available methods are: {self.available_methods}"
    #             )
    #         elif hpo_method == "grid_search":
    #             hpo_technique = self.grid_search
    #         elif hpo_method == "random_search":
    #             hpo_technique = self.random_search

    #         # Call the HPO method and obtain the best configuration in the inner_cv loop (OUTPUT OF EACH METHOD A DICT WITH THE BEST CONFIG)
    #         best_config_inner_cv = hpo_technique(
    #             X=X_outer_train, y=y_outer_train, n_splits=inner_k
    #         )

    #         # End timing
    #         end_time = time.time()
    #         elapsed_time = end_time - start_time

    #         # Evaluate on the outer test set the best configuration found by the hpo method
    #         model = clone(self.model)
    #         model.set_params(**best_config_inner_cv)
    #         model.fit(X_outer_train, y_outer_train)
    #         y_pred = model.predict(X_outer_test)

    #         # Compute metrics
    #         results = self._evaluation_metrics(y_true=y_outer_test, y_pred=y_pred)
    #         results_outer_cv.append(
    #             {
    #                 "metrics": results,
    #                 "hpo_time_seconds": elapsed_time,
    #                 "best_config_inner_cv": best_config_inner_cv,
    #             }
    #         )

    #     # HPO output
    #     final_result = {}

    #     # computation of the unbiased model performance
    #     config_counter = Counter()

    #     # Prepare metrics collectors
    #     if self.task == "regression":
    #         r2_scores = []
    #         mse_scores = []
    #     elif self.task == "classification":
    #         accuracy_scores = []
    #         precision_scores = []
    #         recall_scores = []
    #         f1_scores = []
    #     else:
    #         raise ValueError(f"Unknown task type: {self.task}")

    #     # Loop over folds
    #     for fold in results_outer_cv:
    #         # Count configuration
    #         config_str = json.dumps(fold["best_config_inner_cv"], sort_keys=True)
    #         config_counter[config_str] += 1

    #         if self.task == "regression":
    #             r2_scores.append(fold["metrics"]["R2"])
    #             mse_scores.append(fold["metrics"]["mse"])
    #         elif self.task == "classification":
    #             accuracy_scores.append(fold["metrics"]["accuracy"])
    #             precision_scores.append(fold["metrics"]["precision"])
    #             recall_scores.append(fold["metrics"]["recall"])
    #             f1_scores.append(fold["metrics"]["f1"])

    #     max_frequency = max(config_counter.values())
    #     most_frequent_configs = [
    #         json.loads(config)
    #         for config, freq in config_counter.items()
    #         if freq == max_frequency
    #     ]

    #     final_result["most_frequent_configs"] = most_frequent_configs
    #     final_result["most_frequent_config_count"] = max_frequency
    #     final_result["outer_cv_results"] = results_outer_cv

    #     if self.task == "regression":
    #         avg_r2 = sum(r2_scores) / outer_k
    #         avg_mse = sum(mse_scores) / outer_k
    #         final_result["best_model_R2"] = avg_r2
    #         final_result["best_model_mse"] = avg_mse

    #     elif self.task == "classification":
    #         avg_accuracy = sum(accuracy_scores) / outer_k
    #         avg_precision = sum(precision_scores) / outer_k
    #         avg_recall = sum(recall_scores) / outer_k
    #         avg_f1 = sum(f1_scores) / outer_k
    #         final_result["best_model_accuracy"] = avg_accuracy
    #         final_result["best_model_precision"] = avg_precision
    #         final_result["best_model_recall"] = avg_recall
    #         final_result["best_model_f1"] = avg_f1

    #     return final_result


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
    print(test_hpo.hp_tuning(hpo_method="grid_search", outer_k=5, inner_k=3))
