# * Evolutionary Algorithm * #

import pandas as pd
import numpy as np
from numpy.random import choice
import itertools
from typing import Optional, Dict, List, Tuple
from sklearn.model_selection import cross_val_score
from utils.logger import get_logger
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class EvolutionaryAlgorithm:
    """
    Evolutionary Algorithm for Hyperparameter Optimization.
    """

    # Initialization
    def __init__(self, model, hyperparameters: dict, task: str):
        self.model = model
        self.hp = hyperparameters
        self.task = task

        # Evaluation cache
        self._fitness_cache: Dict[frozenset, Dict] = {}

        # Tracking for visualization
        self.generation_stats: List[Dict] = []
        self.best_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.parameter_evolution: defaultdict = defaultdict(list)

        self.logger = get_logger("HPO")

    # Validator
    def _is_valid_config(self, cfg: Dict) -> bool:
        """
        Ritorna True se il modello accetta i parametri di cfg.
        Funziona anche con estitor non clonabili (es. XGBoost, LightGBM…).
        """
        try:
            # 1. Prendi la classe dell’estimator originale
            ModelCls = self.model.__class__

            # 2. Parti dai parametri di default dell’istanza originale
            base_params = getattr(self.model, "get_params", lambda: {})()

            # 3. Crea una nuova istanza fondendo i parametri base con quelli di cfg
            ModelCls(**{**base_params, **cfg})
            return True
        except (TypeError, ValueError) as e:
            self.logger.debug(f"Invalid config {cfg} -> {type(e).__name__}: {e}")
            return False

    # Population Initialization
    def _initialization_population(self, n_new_configs: Optional[int]) -> List[Dict]:
        hp_keys = list(self.hp.keys())
        hp_values = list(self.hp.values())

        # combinazioni cartesiane filtrate con validator
        param_combinations = list(itertools.product(*hp_values))
        configurations = [
            cfg
            for cfg in (dict(zip(hp_keys, combo)) for combo in param_combinations)
            if self._is_valid_config(cfg)
        ]
        self.logger.info(f"Initial population size: {len(configurations)}")

        seen_configs = {frozenset(c.items()) for c in configurations}

        # extra random configs
        if n_new_configs:
            if n_new_configs < 0:
                raise ValueError("n_new_configs must be non-negative or None")

            max_attempts = n_new_configs * 10
            generated = attempts = 0

            while generated < n_new_configs and attempts < max_attempts:
                conf = {}
                for p, vals in self.hp.items():
                    if all(isinstance(v, int) for v in vals):
                        conf[p] = random.randint(min(vals), max(vals))
                    elif all(isinstance(v, float) for v in vals):
                        conf[p] = round(random.uniform(min(vals), max(vals)), 2)
                    else:
                        conf[p] = random.choice(vals)

                frozen = frozenset(conf.items())
                if self._is_valid_config(conf) and frozen not in seen_configs:
                    configurations.append(conf)
                    seen_configs.add(frozen)
                    generated += 1
                attempts += 1

            if generated < n_new_configs:
                self.logger.warning(
                    f"Only {generated}/{n_new_configs} unique random configs generated."
                )

        return configurations

    # Fitness & Evaluation
    def _fitness_computation(
        self, configuration: Dict, X: pd.DataFrame, y: pd.DataFrame, n_splits_cv: int
    ) -> Dict:
        key = frozenset(configuration.items())
        if key in self._fitness_cache:
            return self._fitness_cache[key]

        model_instance = self.model.__class__(
            **{**self.model.get_params(), **configuration}
        )

        model_instance.set_params(**configuration)

        scoring = "accuracy" if self.task == "classification" else "r2"
        scores = cross_val_score(model_instance, X, y, cv=n_splits_cv, scoring=scoring)

        res = {"config": configuration, "fitness": scores.mean()}
        self._fitness_cache[key] = res
        return res

    def _evaluate_population(
        self, pop: List[Dict], X: pd.DataFrame, y: pd.DataFrame, cv: int
    ) -> List[Dict]:
        evaluated = []
        for cfg in pop:
            if not self._is_valid_config(cfg):
                self.logger.warning(f"Invalid config skipped: {cfg}")
                continue
            evaluated.append(self._fitness_computation(cfg, X, y, cv))

        if not evaluated:
            self.logger.error("All configurations in the population are invalid.")
            raise RuntimeError("Population evaluation failed: no valid configurations.")

        return evaluated

    # Parent Selection
    def _neutral_selection(self, pop: List[Dict], ratio: float) -> List[Dict]:
        k = math.ceil(ratio * len(pop))
        return random.sample(pop, k)

    def _fitness_proportional_selection(
        self, eval_pop: List[Dict], ratio: float
    ) -> List[Dict]:
        fitness = np.array([d["fitness"] for d in eval_pop])
        if fitness.sum() == 0:
            return self._neutral_selection([d["config"] for d in eval_pop], ratio)

        probs = fitness / fitness.sum()
        k = min(math.ceil(ratio * len(eval_pop)), len(eval_pop))
        selected = choice(eval_pop, size=k, replace=False, p=probs)
        return [e["config"] for e in selected]

    def _tournament_selection(self, eval_pop: List[Dict], ratio: float) -> List[Dict]:
        N = len(eval_pop)
        tour_size = max(2, min(int(0.1 * N), N))
        num_parents = math.ceil(ratio * N)
        parents = []
        for _ in range(num_parents):
            group = random.sample(eval_pop, tour_size)
            best = max(group, key=lambda d: d["fitness"])
            parents.append(best["config"])
        return parents

    def _mutation(self, parents: List[Dict], population: List[Dict]) -> List[Dict]:
        offspring, seen = [], {frozenset(c.items()) for c in population}
        hp_keys = list(self.hp.keys())

        for parent in parents:
            attempts = 0
            while attempts < 100:
                child = parent.copy()
                for p in hp_keys:
                    if random.random() < 0.5:
                        pop_vals = [c[p] for c in population]
                        parent_val = parent[p]

                        if all(isinstance(v, float) for v in pop_vals):
                            min_val = min(pop_vals)
                            max_val = max(pop_vals)
                            delta = 0.2 * (max_val - min_val or 1.0)
                            new_val = parent_val + random.uniform(-delta, delta)
                            new_val = min(
                                max(min_val, new_val), max_val
                            )  # clamp to [min_val, max_val]
                            child[p] = round(new_val, 4)

                        elif all(isinstance(v, int) for v in pop_vals):
                            min_val = min(pop_vals)
                            max_val = max(pop_vals)
                            delta = max(1, int(0.2 * (max_val - min_val)))
                            new_val = parent_val + random.randint(-delta, delta)
                            new_val = min(max(min_val, new_val), max_val)
                            child[p] = new_val

                        else:
                            # Categorical: pick a different value from the space
                            choices = [v for v in self.hp[p] if v != parent_val]
                            if choices:
                                child[p] = random.choice(choices)

                frozen = frozenset(child.items())
                if frozen not in seen and self._is_valid_config(child):
                    offspring.append(child)
                    seen.add(frozen)
                    break  # valid child trovato, esci dal while
                attempts += 1
            else:
                self.logger.debug(f"No valid upward mutation for parent {parent}")
        return offspring

    # Survival (μ + λ)
    def _survival_selection(
        self, eval_pop: List[Dict], eval_off: List[Dict]
    ) -> Tuple[List[Dict], Dict]:
        mu = len(eval_pop)
        combined = sorted(eval_pop + eval_off, key=lambda d: d["fitness"], reverse=True)
        survivors = [e["config"] for e in combined[:mu]]
        return survivors, combined[0]

    # Stats tracking
    def _calc_diversity(self, eval_pop: List[Dict]) -> float:
        if len(eval_pop) <= 1:
            return 0.0
        return np.std([d["fitness"] for d in eval_pop])

    def _track_stats(self, gen: int, eval_pop: List[Dict]):
        if not eval_pop:
            self.logger.warning(
                f"Generation {gen}: population is empty, skipping stats tracking."
            )
            return

        fit = [d["fitness"] for d in eval_pop]
        stats = {
            "generation": gen,
            "best_fitness": max(fit),
            "mean_fitness": np.mean(fit),
            "diversity": self._calc_diversity(eval_pop),
        }
        self.generation_stats.append(stats)
        self.best_fitness_history.append(stats["best_fitness"])
        self.diversity_history.append(stats["diversity"])

        for p in self.hp.keys():
            vals = [d["config"][p] for d in eval_pop]
            if all(isinstance(v, (int, float)) for v in vals):
                self.parameter_evolution[p].append(
                    {"generation": gen, "mean": np.mean(vals), "std": np.std(vals)}
                )

    # Main evolution loop
    def evolution_process(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        n_splits_cv: int,
        parents_selection_mechanism: str,
        parents_selection_ratio: float,
        n_new_configs: Optional[int] = None,
        max_generations: int = 10,
    ) -> Dict:
        pop = self._initialization_population(n_new_configs)
        eval_pop = self._evaluate_population(pop, X, y, n_splits_cv)
        self._track_stats(-1, eval_pop)
        best = max(eval_pop, key=lambda d: d["fitness"])

        for gen in range(max_generations):
            # parent selection
            if parents_selection_mechanism == "neutral_selection":
                parents = self._neutral_selection(
                    [d["config"] for d in eval_pop], parents_selection_ratio
                )
            elif parents_selection_mechanism == "fitness_proportional_selection":
                parents = self._fitness_proportional_selection(
                    eval_pop, parents_selection_ratio
                )
            elif parents_selection_mechanism == "tournament_selection":
                parents = self._tournament_selection(eval_pop, parents_selection_ratio)
            else:
                raise ValueError("Unknown parent selection mechanism.")

            # offspring generation (only mutation shown)
            offspring = self._mutation(parents, [d["config"] for d in eval_pop])

            eval_off = self._evaluate_population(offspring, X, y, n_splits_cv)

            # survival
            pop, best_curr = self._survival_selection(eval_pop, eval_off)
            eval_pop = self._evaluate_population(pop, X, y, n_splits_cv)

            if best_curr["fitness"] > best["fitness"]:
                best = best_curr

            self._track_stats(gen, eval_pop)

        return best


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    # Carica il dataset
    data = load_iris(as_frame=True)
    X = data.data
    y = data.target

    # Definizione del modello e spazio degli iperparametri
    model = DecisionTreeClassifier()
    hyperparameters = {
        "max_depth": [1, 3, 4],
        "min_samples_split": [2, 3, 4, 5],  # rimosso 1
    }

    # Istanza dell'algoritmo evolutivo
    ea = EvolutionaryAlgorithm(
        model=model, hyperparameters=hyperparameters, task="classification"
    )

    # Esegui l'evoluzione
    best_result = ea.evolution_process(
        X=X,
        y=y,
        n_splits_cv=5,
        parents_selection_mechanism="tournament_selection",
        parents_selection_ratio=0.5,
        max_generations=20,
    )
    print(best_result)
