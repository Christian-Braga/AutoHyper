# Package Name: AutoHyper

**AutoHyper** is a **lightweight** and **highly-customizable** framework for hyperparameter optimization on tabular-data machine-learning models and neural networks.

Current optimization strategies:

- 🔍 **Grid Search**
- 🎲 **Random Search**
- 🧬 **Evolutionary Algorithm**

---

## Installation

```bash
# install directly from GitHub
pip install git+https://github.com/<Christian-Braga>/AutoHyper.git
```

If you have cloned the repository locally, make sure to install all required dependencies by running:

```bash
pip install -r requirements.txt
```

After installation, you can import the main optimization class like this:

```bash
from autohyper import HPO
```

## Project Layout

```bash
AutoHyper/
├── src/
│   └── autohyper/
│       ├── __init__.py            # exposes HPO class
│       ├── hpo.py                 # main HPO class
│       ├── hpo_methods/           # grid, random, evolutionary search
│       └── visualizations/        # optional plotting modules
├── utils/
│   ├── __init__.py
│   └── logger.py                  # logging utilities
├── Tests/
│   └── 01_basic_HPO.ipynb         # example/test notebook
├── setup.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Usage Example

```python
from autohyper import HPO
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load data
X, y = load_boston(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Define model and search space
model = RandomForestRegressor()
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
}

# Initialize HPO
hpo = HPO(
    model=model,
    X_train=X_train, y_train=y_train,
    X_val=X_val,   y_val=y_val,
)

# Run grid search
best_model, best_params = hpo.grid_search(param_grid)
print("Best hyperparameters:", best_params)
#CORREGGI TUTTO QUESTO E' SBAGLIATO
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests if you:

- Add new optimization strategies

- Improve documentation or examples

- Fix bugs or enhance performance

Please follow the standard GitHub workflow: fork → branch → pull request.

## Future Improvment
