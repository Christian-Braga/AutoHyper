import itertools


# ** Grid Search Method for Hyperparameter Optimization **
class GridSearch:
    """
    A class to perform grid search over a given hyperparameter space.

    This class generates all possible combinations of hyperparameter values
    from a user-defined grid and returns them in dictionary format. It is
    typically used as part of hyperparameter optimization pipelines, where
    each configuration can be evaluated via cross-validation.

    Attributes:
    -----------
    hps : dict
        A dictionary containing the hyperparameter grid. Each key is a
        hyperparameter name, and the value is a list of candidate values.

    Methods:
    --------
    grid_combinations():
        Returns a list of dictionaries, each representing a unique combination
        of hyperparameters to be evaluated.
    """

    def __init__(self, hyperparameters: dict):
        self.hps = hyperparameters

    def grid_combinations(self):
        """
        Generate all possible hyperparameter combinations from the defined grid.

        Returns:
        --------
        List[dict]
            A list where each element is a dictionary mapping hyperparameter names
            to specific values, representing a single configuration.
        """
        # Create the grid combination
        hp_keys = list(self.hps.keys())
        hp_values = list(self.hps.values())
        param_combinations = list(itertools.product(*hp_values))

        return [dict(zip(hp_keys, combo)) for combo in param_combinations]
