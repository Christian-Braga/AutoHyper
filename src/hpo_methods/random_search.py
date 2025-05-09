# ** Random Search Method for Hyperparameter Optimization **
class RandomSearch:
    """Random search HPO method.

    This class implements the random search algorithm for hyperparameter optimization.
    It generates random hyperparameter configurations within the specified search space.
    """

    def __init__(self, hyperparameters):
        """
        Initialize the random search method.

        Parameters:
        -----------
        hyperparameters : dict
            Dictionary of hyperparameter names and their possible values.
        """
        self.hyperparameters = hyperparameters

    def random_configurations(self, n_trials):
        """
        Generate random hyperparameter configurations.

        Parameters:
        -----------
        n_trials : int
            Number of random configurations to generate.

        Returns:
        --------
        list
            List of randomly generated hyperparameter configurations.
        """
        import random

        if n_trials <= 0:
            return []

        configurations = []

        for _ in range(n_trials):
            configuration = {}

            # Create the configuration
            for param, values in self.hyperparameters.items():
                if all(isinstance(v, int) for v in values):
                    low, high = min(values), max(values)
                    configuration[param] = random.randint(low, high)
                elif all(isinstance(v, float) for v in values):
                    low, high = min(values), max(values)
                    configuration[param] = random.uniform(low, high)
                else:
                    # Fallback for categorical or mixed types
                    configuration[param] = random.choice(values)

            configurations.append(configuration)

        return configurations
