"""Parent and survivor selection interfaces for a minimisation EA."""

import random

from ariel.ec import Population


def parent_selection(
    population: Population,
    tournament_size: int,
) -> Population:
    """Tag tournament winners as parents; lower fitness must win."""
    candidates = population.alive.to_list()
    if not candidates:
        raise ValueError("cannot select parents from an empty population")
    if tournament_size < 1:
        raise ValueError("tournament_size must be positive")

    selected_ids: set[int] = set()
    sample_size = min(tournament_size, len(candidates))
    for _ in range(len(candidates)):
        contestants = random.sample(candidates, sample_size)
        winner = min(contestants, key=lambda individual: individual.fitness)
        selected_ids.add(id(winner))

    for individual in candidates:
        individual.tags = {"selected": id(individual) in selected_ids}
    return population


def survivor_selection(
    population: Population,
    pop_size: int,
) -> Population:
    """Keep the ``pop_size`` lowest-fitness living individuals."""
    ranked = sorted(
        population.alive,
        key=lambda individual: individual.fitness,
    )
    survivor_ids = {id(individual) for individual in ranked[:pop_size]}
    for individual in population.alive:
        if id(individual) not in survivor_ids:
            individual.alive = False
    return population
