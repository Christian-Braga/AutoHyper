# > Libraries
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.base import clone
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split


# > Hyperparameter Optimizer
class HPO:
    """A class to perform HPO on a given model with different HPO techniques."""

    def __init__(self, model, data_features, data_target, hp_values: dict):
        self.model = model
        self.features = data_features
        self.target = data_target
        self.hp = hp_values

        self.available_methods = ["grid_search", "random_search"]

    # > Support Functions

    def _evaluation_metrics(self, predicted_target, true_target):
        """Compute different metrics for evaluating performances."""
        r2 = r2_score(predicted_target, true_target)
        mse = mean_squared_error(predicted_target, true_target)
        return r2, mse

    # > Main functions

    # Grid search
    def grid_search(self):
        hp_keys = list(self.hp.keys())
        hp_values = list(self.hp.values())
        param_combinations = list(itertools.product(*hp_values))
        hyperparameters = [dict(zip(hp_keys, combo)) for combo in param_combinations]
        return hyperparameters

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
            best_config_inner_cv = hpo_technique(X=X_outer_train, y=y_outer_train)

            # Evaluate on the outer test set the best configuration found by the hpo method
            model = clone(self.model)
            model.set_params(**best_config_inner_cv)
            model.fit(X_outer_train, y_outer_train)
            y_pred = model.predict(X_outer_test)

            # Compute metrics
            results = self._evaluation_metrics(
                y_true=y_outer_test, y_pred=y_pred
            )  # eval metrics outputs a dict
            results_outer_cv.append(results)

            # HPO output
            # write the code to output the best configurations and the unbiased performance of the best model

    # Best model training
    def train_best_model(self):
        """A function to train and evaluate the best model obtained from the tuning loop."""
