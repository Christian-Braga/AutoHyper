# * Evolutionary Algorithm * #

# EA non è lanciato per ogni split dell' inner cv, Invece, ogni volta che
# l'EA valuta un individuo, usa l'intero inner CV per stimare la fitness di quell’individuo.
# quindi l'inner cv loop è inserito nella fitness function dell' EA.


# elementi da implementare nel EA:

# * 1
# funzione per generare la popolazione iniziale a partire da una configurazione di Hyperparametri (a random)
# deve consentire di generare n configurazioni decisa a random dall'utente o fissare se l'utente non vuole generarne di nuove ma partire
# da quelle che ha

# * 2
# fitness function, utilizzata per valutare la bontà di una configurazione, questa coinvolge quindi la inner cross validation e misura l'accuratezza
# della configurazione

# * 3
# selection process
# funzione per fare parent selection e penso anche survival selection? é la stessa?

# * 4
# funzione con metodi di generazione offsprings mutation and recombination

# * 5
# funzione principale algoritmo evolutivo

# * 6
# funzione per visualization

# Scrivo questa classe in maniera il più possibile indipendente da hpo poi eventualmente la adatto per ciò che serve

import pandas as pd
import itertools
from typing import Optional
from sklearn.model_selection import cross_val_score


class EvolutionaryAlgorithm:
    """To DO."""

    def __init__(self, model, hyperparameters: dict, task: str):
        self.model = model
        self.hp = hyperparameters
        self.task = task

    def _initialization_population(self):
        # Create the initial population
        hp_keys = list(self.hp.keys())
        hp_values = list(self.hp.values())
        param_combinations = list(itertools.product(*hp_values))
        return [dict(zip(hp_keys, combo)) for combo in param_combinations]

    def _fitness_computation(
        self, configuration: dict, X: pd.DataFrame, y: pd.DataFrame, n_splits_cv: int
    ):
        # the fitness function is performed by doing k-fold CV
        model_instance = self.model(**configuration)
        if self.task == "classification":
            scoring = "accuracy"
        elif self.task == "regression":
            scoring = "r2"
        else:
            raise ValueError(f"Unknown task type: {self.task}")
        scores = cross_val_score(model_instance, X, y, cv=n_splits_cv, scoring=scoring)

        return {"config": configuration, "fitness": scores.mean()}

        pass

    def _selection_mechanisms(self):
        pass

    def _generation_mechanisms(self):
        pass

    def _visualization(self):
        pass

    def evolution_process(
        self,
        task: str,
        X: pd.DataFrame,
        y: pd.DataFrame,
        n_splits_cv: int,
        selection_mechanism: str,
        generation_mechanism: str,
    ):
        pass
