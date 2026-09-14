"""Target loading and the assignment's required fitness evaluation."""

from pathlib import Path
from typing import Any, cast

import networkx as nx

from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.ec import Population
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from ..tree_edit_distance import mean_plus_std_tree_edit_distance
from .config import TARGET_DIR


def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """Load all target body JSON files in deterministic filename order.

    This mirrors the starter's small I/O helper without importing the starter
    module, whose top-level demo setup constructs the unused NDE representation.
    The target decoder and target files themselves remain the provided ones.
    """
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(path) for path in paths]


def fitness_of_genome(
    genome: TreeGenome,
    targets: list[nx.DiGraph],
) -> float:
    """Decode only for evaluation and return mean distance plus one std."""
    return mean_plus_std_tree_edit_distance(genome.to_networkx(), targets)


def evaluate(
    population: Population,
    targets: list[nx.DiGraph],
) -> Population:
    """Evaluate only individuals whose fitness is currently stale."""
    for individual in population.unevaluated:
        genotype = cast(dict[str, Any], individual.genotype)
        genome = TreeGenome.from_dict(genotype)
        individual.fitness = fitness_of_genome(genome, targets)
    return population
