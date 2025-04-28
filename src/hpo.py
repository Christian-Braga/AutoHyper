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
            mse = np.sqrt(mean_squared_error(y_true, y_pred))
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

        # Inner cross validation loop
        inner_cv = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        store_metrics = []

        # loop for each configuration
        for configuration in hyperparameters_configs:
            model = clone(self.model)
            model.set_params(**configuration)
            inner_loop_results = []

            # perform kfold cv for each configuration
            for train_idx, test_idx in inner_cv.split(X):
                X_inner_train, X_inner_val = X.iloc[train_idx], X.iloc[test_idx]
                y_inner_train, y_inner_val = y.iloc[train_idx], y.iloc[test_idx]

                model.fit(X_inner_train, y_inner_train)
                y_pred_inner_cv = model.predict(X_inner_val)
                results = self._evaluation_metrics(
                    y_true=y_inner_val, y_pred=y_pred_inner_cv
                )
                config_dict = {}
                config_dict["configuration"] = configuration
                config_dict["results"] = results
                inner_loop_results.append(config_dict)

            # compute the average performance over all the folds for each config
            if self.task == "regression":
                r2_scores = []
            elif self.task == "classification":
                accuracy_scores = []
            else:
                raise ValueError(f"Unknown task type: {self.task}")

            # Loop over folds
            for fold in inner_loop_results:
                configuration_results = {}

                if self.task == "regression":
                    r2_scores.append(fold["results"]["R2"])
                    avg_r2 = sum(r2_scores) / n_splits
                    configuration_results["configuration"] = fold["configuration"]
                    configuration_results["results"] = avg_r2
                    store_metrics.append(configuration_results)

                elif self.task == "classification":
                    accuracy_scores.append(fold["results"]["accuracy"])
                    avg_accuracy = sum(accuracy_scores) / n_splits
                    configuration_results["configuration"] = fold["configuration"]
                    configuration_results["results"] = avg_accuracy
                    store_metrics.append(configuration_results)

        # Find the best configuration
        best_configuration = store_metrics[0]["configuration"]
        best_config_results = store_metrics[0]["results"]
        for config_metrics in store_metrics:
            if config_metrics["results"] > best_config_results:
                best_configuration = config_metrics["configuration"]
                best_config_results = config_metrics["results"]
        return best_configuration

    # Random search
    def random_search(self):
        pass

    # Plot results of the tuning loop
    def plot_results(self):
        pass

    # Hyperparameters tuning loop
    def hp_tuning(self, hpo_method: list, outer_k: int, inner_k: int, shuffle=True):
        """Main function to perform HPO using nested cross validation"""
        # Data
        X = self.features
        y = self.target

        # Set up outer CV loop
        outer_cv = KFold(n_splits=outer_k, random_state=42, shuffle=True)
        results_outer_cv = []

        for train_idx, test_idx in outer_cv.split(X):
            X_outer_train, X_outer_test = X.iloc[train_idx], X.iloc[test_idx]
            y_outer_train, y_outer_test = y.iloc[train_idx], y.iloc[test_idx]

            # Start time
            start_time = time.time()

            # Select hpo method
            if hpo_method not in self.available_methods:
                raise Exception(
                    f"the selected hpo techniques is not available, the available methods are: {self.available_methods}"
                )
            elif hpo_method == "grid_search":
                hpo_technique = self.grid_search
            elif hpo_method == "random_search":
                hpo_technique = self.random_search

            # Call the HPO method and obtain the best configuration in the inner_cv loop (OUTPUT OF EACH METHOD A DICT WITH THE BEST CONFIG)
            best_config_inner_cv = hpo_technique(
                X=X_outer_train, y=y_outer_train, n_splits=inner_k
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
            results_outer_cv.append(
                {
                    "metrics": results,
                    "hpo_time_seconds": elapsed_time,
                    "best_config_inner_cv": best_config_inner_cv,
                }
            )

        # HPO output
        final_result = {}

        # computation of the unbiased model performance
        config_counter = Counter()

        # Prepare metrics collectors
        if self.task == "regression":
            r2_scores = []
            mse_scores = []
        elif self.task == "classification":
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
        else:
            raise ValueError(f"Unknown task type: {self.task}")

        # Loop over folds
        for fold in results_outer_cv:
            # Count configuration
            config_str = json.dumps(fold["best_config_inner_cv"], sort_keys=True)
            config_counter[config_str] += 1

            if self.task == "regression":
                r2_scores.append(fold["metrics"]["R2"])
                mse_scores.append(fold["metrics"]["mse"])
            elif self.task == "classification":
                accuracy_scores.append(fold["metrics"]["accuracy"])
                precision_scores.append(fold["metrics"]["precision"])
                recall_scores.append(fold["metrics"]["recall"])
                f1_scores.append(fold["metrics"]["f1"])

        max_frequency = max(config_counter.values())
        most_frequent_configs = [
            json.loads(config)
            for config, freq in config_counter.items()
            if freq == max_frequency
        ]

        final_result["most_frequent_configs"] = most_frequent_configs
        final_result["most_frequent_config_count"] = max_frequency
        final_result["outer_cv_results"] = results_outer_cv

        if self.task == "regression":
            avg_r2 = sum(r2_scores) / outer_k
            avg_mse = sum(mse_scores) / outer_k
            final_result["best_model_R2"] = avg_r2
            final_result["best_model_mse"] = avg_mse

        elif self.task == "classification":
            avg_accuracy = sum(accuracy_scores) / outer_k
            avg_precision = sum(precision_scores) / outer_k
            avg_recall = sum(recall_scores) / outer_k
            avg_f1 = sum(f1_scores) / outer_k
            final_result["best_model_accuracy"] = avg_accuracy
            final_result["best_model_precision"] = avg_precision
            final_result["best_model_recall"] = avg_recall
            final_result["best_model_f1"] = avg_f1

        return final_result
