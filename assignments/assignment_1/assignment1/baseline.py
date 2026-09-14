"""Evaluation-budget-matched random-search baseline."""

import networkx as nx

from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from .config import DEFAULT_CONFIG, ExperimentConfig
from .evaluation import fitness_of_genome
from .initialization import random_genome


def random_search(
    targets: list[nx.DiGraph],
    evaluation_budget: int,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> tuple[TreeGenome, list[float]]:
    """Return the best random genome and best-so-far fitness trajectory."""
    if evaluation_budget < 1:
        raise ValueError("evaluation_budget must be positive")

    best_genome: TreeGenome | None = None
    best_fitness = float("inf")
    best_so_far: list[float] = []
    for _ in range(evaluation_budget):
        genome = random_genome(config.max_modules)
        fitness = fitness_of_genome(genome, targets)
        if fitness < best_fitness:
            best_genome = genome
            best_fitness = fitness
        best_so_far.append(best_fitness)

    if best_genome is None:
        raise RuntimeError("random search produced no genome")
    return best_genome, best_so_far
