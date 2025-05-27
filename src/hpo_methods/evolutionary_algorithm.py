# * Evolutionary Algorithm * #

# TO DO
# -> RIVEDERE BENE MECCANISMO DI CREAZIONE CONFIGURAZIONI RANDOM, GUARDANDO L'OUTPUT NON SEMBRA CI SIA DIVERSITÁ
# INOLTRE DEVO AGGIUNGERE IL CONTROLLO ANCHE CHE LE CONFIGURAZIONI NON SIANO UGUALI (FORSE DA CONTROLLARE ANCHE
# NELLA LOGICA DI Random search)
# -> COMPLETARE MECCANISMO DI PARENT SELECTION CON I SUOI METODI

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
from utils.logger import get_logger
import random
import math


class EvolutionaryAlgorithm:
    """To DO."""

    def __init__(self, model, hyperparameters: dict, task: str):
        self.model = model
        self.hp = hyperparameters
        self.task = (task,)

        # Set-up Logger
        self.logger = get_logger("HPO")

    # *  Population Initialization

    def _initialization_population(self, n_new_configs):
        # Create the initial population

        # unpack the given hyperparameters without creating a new ones
        hp_keys = list(self.hp.keys())
        hp_values = list(self.hp.values())
        param_combinations = list(itertools.product(*hp_values))
        configurations = [dict(zip(hp_keys, combo)) for combo in param_combinations]

        # create random configuration if requested
        # check valid input
        if n_new_configs is not None:
            if isinstance(n_new_configs, int) and n_new_configs < 0:
                self.logger.error(
                    f"Invalid value for n_new_configs: {n_new_configs}. It must be a non-negative integer or None."
                )
                raise ValueError(
                    "n_new_configs must be a non-negative integer or None."
                )

            # generate random configuration
            for _ in range(n_new_configs):
                configuration = {}

                for param, values in self.hp.items():
                    if all(isinstance(v, int) for v in values):
                        configuration[param] = random.randint(min(values), max(values))
                    elif all(isinstance(v, float) for v in values):
                        configuration[param] = random.uniform(min(values), max(values))
                    else:
                        configuration[param] = random.choice(values)

                configurations.append(configuration)

        print("Popolazione iniziale:")
        for c in configurations:
            print(c)
        return configurations

    # * Fitness Computation Function

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

    # * Parent Selection Function

    # > Method for Parent Selection: Neutral Selection
    def _neutral_selection(self, population: list, parents_selection_rateo: float):
        number_of_parents = math.ceil(parents_selection_rateo * len(population))
        return random.sample(population, k=number_of_parents)

    # > Method for Parent Selection: Fitness Proportional Selection
    def _fitness_proportional_selection(self):
        pass

    # > Method for Parent Selection: Tournament Selection
    def _tournament_selection(self):
        pass

    # * Survival Selection Function

    def _survival_selection(self):
        pass

    # * Offsprings Generation Function

    def _generation_mechanisms(self):
        pass

    # * Visualization Function

    def _visualization(self):
        pass

    # * Main Method: Evolution Process

    def evolution_process(
        self,
        task: str,
        X: pd.DataFrame,
        y: pd.DataFrame,
        n_splits_cv: int,
        parents_selection_mechanism: str,
        generation_mechanism: str,
        parents_selection_rateo: float,
        n_new_configs: Optional[int] = None,
    ):
        # Step 1: Initialization of the population
        population = self._initialization_population(n_new_configs=n_new_configs)

        # Step 2: Parent Selection
        if parents_selection_mechanism == "neutral_selection":
            parents = self._neutral_selection(
                population=population, parents_selection_rateo=parents_selection_rateo
            )
        return parents

        # Step 3: Offsprings Generation

        # Step 4: Survival Selection


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    # Carica dataset
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target

    # Definizione del modello e degli iperparametri da ottimizzare
    model = DecisionTreeClassifier
    hyperparameters = {"max_depth": [2, 3, 4, 5], "min_samples_split": [2, 4, 6]}

    # Crea istanza dell'algoritmo evolutivo
    ea = EvolutionaryAlgorithm(
        model=model, hyperparameters=hyperparameters, task="classification"
    )

    # Avvia processo evolutivo (inizialmente testiamo solo parent selection)
    parents = ea.evolution_process(
        task="classification",
        X=X,
        y=y,
        n_splits_cv=5,
        parents_selection_mechanism="neutral_selection",
        generation_mechanism=None,
        parents_selection_rateo=0.4,  # ad esempio il 40% della popolazione iniziale
        n_new_configs=10,  # 10 configurazioni casuali in aggiunta a quelle generate da product
    )

    print("Genitori selezionati:")
    for p in parents:
        print(p)
