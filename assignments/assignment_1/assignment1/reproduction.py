"""Tree crossover, mutation, validation, and offspring creation."""

import random
from typing import Any, cast

from ariel.ec import Individual, Population
from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    validate_tree_depth,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict

from .config import DEFAULT_CONFIG, ExperimentConfig


def valid_genome(
    genome: TreeGenome,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> bool:
    """Check ARIEL structure rules plus this experiment's size/depth caps."""
    if not 1 <= len(genome.nodes) <= config.max_modules:
        return False
    if not validate_tree_depth(genome, config.max_depth):
        return False
    try:
        validate_genome_dict(genome.to_dict())
    except ValueError:
        return False
    return True


def crossover(
    parent_a: TreeGenome,
    parent_b: TreeGenome,
) -> tuple[TreeGenome, TreeGenome]:
    """Produce two children with ARIEL subtree crossover."""
    return crossover_subtree(parent_a, parent_b)


def mutate(
    genome: TreeGenome,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> TreeGenome:
    """Apply the configured ARIEL tree mutation strategy."""
    original = TreeGenome.from_dict(genome.to_dict())
    mutated = TreeGenome.from_dict(genome.to_dict())

    if config.mutation_strategy == "point_only":
        mutate_replace_node(mutated)
    else:
        operator = random.choice(
            (
                mutate_replace_node,
                mutate_subtree_replacement,
                mutate_shrink,
                mutate_hoist,
            ),
        )
        if operator is mutate_subtree_replacement:
            operator(mutated, max_modules=config.max_modules)
        else:
            operator(mutated)

    return mutated if valid_genome(mutated, config) else original


def reproduction(
    population: Population,
    config: ExperimentConfig = DEFAULT_CONFIG,
) -> Population:
    """Append one population's worth of valid, unevaluated offspring."""
    parents = [
        individual
        for individual in population.alive
        if individual.tags.get("selected", False)
    ]
    if not parents:
        raise ValueError("parent selection produced no parents")

    offspring: list[Individual] = []
    while len(offspring) < config.pop_size:
        parent_a = random.choice(parents)
        parent_b = random.choice(parents)
        genome_a = TreeGenome.from_dict(
            cast(dict[str, Any], parent_a.genotype),
        )
        genome_b = TreeGenome.from_dict(
            cast(dict[str, Any], parent_b.genotype),
        )

        if random.random() < config.crossover_probability:
            child_genomes = crossover(genome_a, genome_b)
        else:
            child_genomes = (
                TreeGenome.from_dict(genome_a.to_dict()),
                TreeGenome.from_dict(genome_b.to_dict()),
            )

        for child_genome, fallback in zip(
            child_genomes,
            (genome_a, genome_b),
            strict=True,
        ):
            if not valid_genome(child_genome, config):
                child_genome = TreeGenome.from_dict(fallback.to_dict())
            if random.random() < config.mutation_probability:
                child_genome = mutate(child_genome, config)

            child = Individual()
            child.genotype = child_genome.to_dict()
            child.tags = {"selected": False}
            offspring.append(child)
            if len(offspring) == config.pop_size:
                break

    population.extend(offspring)
    return population
