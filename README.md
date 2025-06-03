# * Package Name: AutoHyper

**AutoHyper** is a **lightweight** and **highly-customizable** framework for hyperparameter optimization on tabular-data machine-learning models and neural networks.

Current optimization strategies:

- 🔍 **Grid Search**
- 🎲 **Random Search**
- 🧬 **Evolutionary Algorithm**

---

## * Installation

```bash
# install directly from GitHub
pip install git+https://github.com/<Christian-Braga>/AutoHyper.git
```

After installation, you can import the main optimization class like this:

```bash
from autohyper import HPO
```

## * Installing Dependencies

If you have cloned the repository locally, make sure to install all required dependencies by running:

```bash
pip install -r requirements.txt
```

After installation, you can import the main optimization class like this:

```bash
from autohyper import HPO
```

## * Project Layout

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
