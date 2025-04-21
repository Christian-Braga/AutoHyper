# > Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split


# > The dataset

# Load the california housing dataset
data = fetch_california_housing()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# create the df
df = X.copy()
df["target"] = y


class HPO:
    """A class to perform HPO on a given model with different HPO techniques."""

    def __init__(self, model, dataset: pd.DataFrame, hp_values: dict):
        self.model = model
        self.data = dataset
        self.hp = hp_values

    def _nested_CV(self):
        """Perform nested resempling."""
        pass

    def _evaluation_metrics(self):
        """Compute different metrics for evaluating performances."""
        pass
