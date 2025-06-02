# * Evolutionary Algorithm * #

# TO DO
# -> COMPLETARE MECCANISMO DI TOURNAMENT SELECTION POI FARE GENERATION POI TESTARE IL TUTTO CON METODO DI VISUALIZZAZIONE

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
import numpy as np
from numpy.random import choice
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
        self.task = task

        # Set-up Logger
        self.logger = get_logger("HPO")

    # *  Population Initialization

    def _initialization_population(self, n_new_configs):
        # Create the initial population

        # unpack the given hyperparameters without creating a new ones
        self.logger.debug("Create the HP configuration")

        hp_keys = list(self.hp.keys())
        hp_values = list(self.hp.values())
        param_combinations = list(itertools.product(*hp_values))
        configurations = [dict(zip(hp_keys, combo)) for combo in param_combinations]

        self.logger.debug(f"{len(configurations)} Standard configurations created")

        # Create set
        seen_configs = set(frozenset(cfg.items()) for cfg in configurations)

        # generate random configuration if requested
        if n_new_configs is not None:
            max_attempts = n_new_configs * 10
            attempts = 0
            generated = 0

            while generated < n_new_configs and attempts < max_attempts:
                configuration = {}

                for param, values in self.hp.items():
                    if all(isinstance(v, int) for v in values):
                        configuration[param] = random.randint(min(values), max(values))
                    elif all(isinstance(v, float) for v in values):
                        configuration[param] = round(
                            random.uniform(min(values), max(values)), 2
                        )
                    else:
                        configuration[param] = random.choice(values)

                frozen = frozenset(configuration.items())

                if frozen not in seen_configs:
                    configurations.append(configuration)
                    seen_configs.add(frozen)
                    generated += 1
                attempts += 1

            if generated < n_new_configs:
                self.logger.warning(
                    f"Only {generated}/{n_new_configs} unique configurations could be generated after {attempts} attempts."
                )

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

        self.logger.debug(
            f"{n_new_configs} new random configurations created. Total number of element in the initial population: {len(configurations)}"
        )

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
        self.logger.debug("Parent Selection Method: Neutral selection")

        number_of_parents = math.ceil(parents_selection_rateo * len(population))
        return random.sample(population, k=number_of_parents)

    # > Method for Parent Selection: Fitness Proportional Selection
    def _fitness_proportional_selection(
        self,
        population: list,
        parents_selection_rateo: float,
        X: pd.DataFrame,
        y: pd.DataFrame,
        n_splits_cv: int,
    ):
        self.logger.debug("Parent Selection Method: Fitness Proportional Selection")
        # compute the fitness
        configurations_fitness: list[dict] = []
        for cfg in population:
            res = self._fitness_computation(cfg, X, y, n_splits_cv)
            configurations_fitness.append(res)

        if not configurations_fitness:  # popolazione vuota
            raise RuntimeError("Empty population!")

        fitness_vals = np.array(
            [d["fitness"] for d in configurations_fitness], dtype=float
        )
        tot_fit = fitness_vals.sum()

        if tot_fit == 0:
            self.logger.warning("All fitness = 0 -> neutral selection fallback")
            return self._neutral_selection(population, parents_selection_rateo)

        # compute the probabilities of each config to be sampled
        probabilities = fitness_vals / tot_fit

        k = math.ceil(parents_selection_rateo * len(configurations_fitness))
        k = min(k, len(configurations_fitness))

        # selection
        selected = choice(
            configurations_fitness,
            size=k,
            replace=False,
            p=probabilities,
        )

        return [entry["config"] for entry in selected]

    # > Method for Parent Selection: Tournament Selection
    def _tournament_selection(
        self,
        population: list,
        parents_selection_rateo: float,
        X: pd.DataFrame,
        y: pd.DataFrame,
        n_splits_cv: int,
    ):
        self.logger.debug("Parent Selection Method: Tournament Selection")
        # group dimesion fixed at the 10% of the population
        N = len(population)
        k = max(2, min(int(round(0.1 * N)), N))
        parents = []

        # sampling at random k individual from the population
        for configuration in range(math.ceil(parents_selection_rateo * N)):
            tournament_group = random.sample(population, k)

            # compute the fitness of the sampled group
            fitness_results = []
            for cfg in tournament_group:
                fitness_results.append(
                    self._fitness_computation(
                        configuration=cfg, X=X, y=y, n_splits_cv=n_splits_cv
                    )
                )

            # exctract the better configuratrion and add it to the parents group
            best = max(fitness_results, key=lambda d: d["fitness"])
            parents.append(best["config"])

        return parents

    # * Offsprings Generation Functions

    # VALUTA SE DEPRECARE CROSSOVER, NON HA SENSO PER COME L'HO FATTO ORA
    def _crossover(
        self, parents: list[dict], existing_population: list[dict]
    ) -> list[dict]:
        offspring = []
        hp_keys = list(self.hp.keys())

        # Existing configs
        seen_configs = set(frozenset(cfg.items()) for cfg in existing_population)

        max_attempts = len(parents) * 10
        attempts = 0

        while len(offspring) < len(parents) and attempts < max_attempts:
            p1, p2 = random.sample(parents, k=2)
            child = {
                param: p1[param] if random.random() < 0.5 else p2[param]
                for param in hp_keys
            }

            frozen = frozenset(child.items())
            if frozen not in seen_configs:
                offspring.append(child)
                seen_configs.add(frozen)
            attempts += 1

        if len(offspring) < len(parents):
            self.logger.warning(
                f"Only {len(offspring)} unique offspring generated from {len(parents)} parents after {attempts} attempts."
            )

        return offspring

    def _mutation(self, parents: list[dict], population: list[dict]) -> list[dict]:
        mutated_offspring = []
        seen_configs = set(frozenset(c.items()) for c in population)

        hp_keys = list(self.hp.keys())

        for parent in parents:
            max_attempts = 50
            attempts = 0
            mutated = None

            while attempts < max_attempts:
                new_config = {}

                for param in hp_keys:
                    # Estrai i valori del parametro dalla popolazione
                    param_values = [c[param] for c in population]

                    if all(isinstance(v, float) for v in param_values):
                        low = min(param_values)
                        high = max(param_values)
                        new_config[param] = round(random.uniform(low, high), 2)

                    elif all(isinstance(v, int) for v in param_values):
                        min_val = min(param_values)
                        max_val = max(param_values)
                        delta = max(1, int(0.2 * (max_val - min_val)))
                        new_min = max(1, min_val - delta)
                        new_max = max_val + delta
                        new_config[param] = random.randint(new_min, new_max)

                    else:
                        new_config[param] = random.choice(self.hp[param])

                frozen = frozenset(new_config.items())
                if frozen not in seen_configs:
                    mutated = new_config
                    seen_configs.add(frozen)
                    break

                attempts += 1

            if mutated is not None:
                mutated_offspring.append(mutated)
            else:
                self.logger.warning(
                    "Mutation failed to generate a unique configuration after max attempts. Skipping."
                )

        return mutated_offspring

    # * Survival Selection Function

    def _survival_selection(self):
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

        elif parents_selection_mechanism == "fitness_proportional_selection":
            parents = self._fitness_proportional_selection(
                population=population,
                parents_selection_rateo=parents_selection_rateo,
                X=X,
                y=y,
                n_splits_cv=n_splits_cv,
            )

        elif parents_selection_mechanism == "tournament_selection":
            parents = self._tournament_selection(
                population=population,
                parents_selection_rateo=parents_selection_rateo,
                X=X,
                y=y,
                n_splits_cv=n_splits_cv,
            )

        # Step 3: Offsprings Generation

        if generation_mechanism == "crossover":
            offsprings = self._crossover(
                parents=parents, existing_population=population
            )
            return offsprings

        elif generation_mechanism == "mutation":
            offsprings = self._mutation(parents=parents, population=population)
            return offsprings

        # Step 4: Survival Selection


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    # Carica il dataset
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target

    # Definizione del modello e spazio degli iperparametri
    model = DecisionTreeClassifier
    hyperparameters = {"max_depth": [1, 3, 4], "min_samples_split": [3, 5]}

    # Istanza dell'algoritmo evolutivo
    ea = EvolutionaryAlgorithm(
        model=model, hyperparameters=hyperparameters, task="classification"
    )

    # 1. Genera la popolazione iniziale
    population = ea._initialization_population(n_new_configs=5)
    print("\n🔵 Popolazione iniziale:")
    for i, config in enumerate(population, 1):
        print(f"[{i}] {config}")

    # 2. Seleziona i genitori
    parents = ea._tournament_selection(
        population=population, parents_selection_rateo=0.4, X=X, y=y, n_splits_cv=3
    )
    print("\n🟡 Genitori selezionati (tournament):")
    for i, config in enumerate(parents, 1):
        print(f"[{i}] {config}")

    # 3. Genera offsprings tramite mutation
    offsprings = ea._mutation(parents=parents, population=population)
    print("\n🟢 Offspring generati tramite mutation:")
    for i, config in enumerate(offsprings, 1):
        print(f"[{i}] {config}")
