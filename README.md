# AutoHyper

![AutoHyper Logo](logo/autohyper_logo.png)


**AutoHyper** is a Python package designed to facilitate hyperparameter optimization (HPO) for supervised learning models on **tabular data**.  
It serves as a **lightweight, modular, and fully customizable** alternative, giving you **fine-grained control** over the entire tuning and validation process.

---

## Key Features

AutoHyper is designed to:

- 1. Provide a **clear and consistent interface** for different HPO strategies:
  at the moment `grid search`, `random search`, and `evolutionary algorithms`.
  (For more detailed examples and explanations, see the [Docs.ipynb](https://github.com/Christian-Braga/AutoHyper/blob/main/Docs.ipynb) notebook included in the repository.
)

- 2. Leverage **nested cross-validation** to deliver **robust and unbiased** estimates of out-of-sample model performance.

- 3.  Incorporate a **custom selection mechanism** that combines performance and robustness using a weighted scoring function, ensuring the best configurations are both **accurate and consistently effective** across multiple resampling iterations.

- 4. Return **structured outputs**, ideal for quantitative comparison and visual inspection of configurations.

- 5. Offer **detailed logging** and **configuration ranking** based on a composite score of performance and frequency.

---

## Supported Models & Tasks

AutoHyper is compatible with any supervised model following the [scikit-learn](https://scikit-learn.org/stable/) API, meaning the model must implement:

- `fit(X, y)`
- `predict(X)`
- `set_params(**kwargs)`

Examples:
- Regressors: `RandomForestRegressor`, `XGBRegressor`, `SVR`, etc.
- Classifiers: `LogisticRegression`, `RandomForestClassifier`, `SVC`, etc.

To use deep learning models: (at the moment)
- Use [SciKeras](https://github.com/adriangb/scikeras) to wrap Keras models
- Use [skorch](https://skorch.readthedocs.io/) to wrap PyTorch models

### Supported Tasks

- **Regression**: using metrics like `R²`, `MSE`
- **Classification**: using metrics like `Accuracy`, `Precision`, `Recall`, `F1`

> Currently supports only **tabular datasets** (`pandas.DataFrame` + `Series`).

---

### Resempling Strategy: Nested Resampling

AutoHyper applies **nested cross-validation** to avoid the risk of overfitting during hyperparameter search.

Standard CV may result in overly optimistic performance estimates because the same validation data is used both for tuning and evaluating. Nested CV avoids this by separating:

- **Inner loop**: performs the hyperparameter search
- **Outer loop**: evaluates the selected configuration on unseen test folds

This results in **robust and unbiased** performance estimation.

---

##  Evaluation & Ranking

AutoHyper selects the best configuration using a **weighted scoring function** that balances performance and robustness:

```math
Weighted Score = 0.7 × Performance Score + 0.3 × Normalized Frequency
```

## Final Output Includes

-  **Best configuration** based on the **weighted score**
-  **Most frequently selected configuration** across outer folds
-  **Best raw performer** by average metric
-  **Summary of all evaluated configurations**, including scores and rankings
-  **Execution time** per outer fold


## Installation

```bash
# install directly from GitHub
pip install git+https://github.com/Christian-Braga/AutoHyper.git
```
This will install the latest version of the package along with all required dependencies.

After installation, you can import the main optimization class like this:

```python
from autohyper import HPO
```


## Usage Example

For more detailed examples and explanations, see the [Docs.ipynb](https://github.com/Christian-Braga/AutoHyper/blob/main/Docs.ipynb) notebook included in the repository.

```python
from autohyper import HPO
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load data
X, y = fetch_california_housing(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model and hyperparameter space
model = RandomForestRegressor()
hp_values = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, 30],
}

# Create HPO instance
hpo = HPO(
    model=model,
    data_features=X_train,
    data_target=y_train,
    hp_values=hp_values,
    task="regression",
)

# Run hyperparameter optimization using random search
results = hpo.hp_tuning(
    hpo_method="random_search",
    outer_k=5,
    inner_k=3,
    n_trials=20,
)
```



## Future Improvment
* Review the structure of the library to improve modularity, ease of extension and user experience.
* Introduce **multiprocessing** to parallelize computations and improve optimization runtime
* Implement a **Bayesian Optimization** mechanism with different acquisition functions (using **Random Forest** as surrogate model – still considering whether to implement **Gaussian Process** as well)
* Implement **visualization tools** to analyze optimization results (e.g., **Parallel Coordinate Plot**) and methods to investigate hyperparameter importance (e.g., **Ablation Study**)
* Implement **multi-objective optimization** mechanisms (e.g., **Hypervolume Indicator**, **Pareto Front**)
* Evaluate the integration of **Meta-Learning strategies**
