"""Creation of random tree-genome individuals and populations."""

import random

import numpy as np

from ariel.ec import Individual, Population
from ariel.ec.genotypes.tree.operators import random_tree
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from .config import DEFAULT_CONFIG, ExperimentConfig


def seed_random_generators(seed: int) -> None:
    """Seed every random generator used by the tree implementation."""
    random.seed(seed)
    np.random.seed(seed)


def random_genome(max_modules: int) -> TreeGenome:
    """Create a random genome with at most ``max_modules`` total nodes.

    ARIEL's ``random_tree`` argument counts non-core additions, despite its
    name and docstring. Subtracting one keeps the core inside our module cap.
    """
    if max_modules < 1:
        msg = "max_modules must include at least the core module"
        raise ValueError(msg)
    return random_tree(max_modules=max_modules - 1)


def make_individual(
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> Individual:
    """Create one unevaluated individual containing a serialized TreeGenome."""
    individual = Individual()
    individual.genotype = random_genome(config.max_modules).to_dict()
    return individual


def make_population(
    pop_size: int | None = None,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> Population:
    """Create a random population."""
    size = config.pop_size if pop_size is None else pop_size
    return Population([make_individual(config) for _ in range(size)])
