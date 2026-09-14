"""Minimal ARIEL EA for the Assignment 1 tree-genome experiment."""

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, cast

from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from .config import DEFAULT_CONFIG, OUTPUT_DIR, ExperimentConfig
from .evaluation import evaluate, load_targets
from .initialization import make_population, seed_random_generators
from .reproduction import reproduction
from .selection import parent_selection, survivor_selection


@dataclass(slots=True)
class RunResult:
    """Result needed by the experiment runner."""

    best: Individual
    history: list[dict[str, float | int]]


def population_statistics(
    population: Population,
    generation: int,
) -> dict[str, float | int]:
    """Summarize the living population at one generation."""
    fitnesses = [individual.fitness for individual in population.alive]
    return {
        "generation": generation,
        "best": min(fitnesses),
        "mean": fmean(fitnesses),
        "std": pstdev(fitnesses),
    }


def record_statistics(
    population: Population,
    history: list[dict[str, float | int]],
) -> Population:
    """EA operation that records statistics after survivor selection."""
    history.append(population_statistics(population, len(history)))
    return population


def run_ea(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    quiet: bool = True,
) -> RunResult:
    """Run one reproducible minimising EA and persist its ARIEL database."""
    seed_random_generators(config.seed)
    targets = load_targets()
    population = evaluate(make_population(config=config), targets)
    history = [population_statistics(population, 0)]

    operations = [
        EAOperation(parent_selection, config.tournament_size),
        EAOperation(reproduction, config),
        EAOperation(evaluate, targets),
        EAOperation(survivor_selection, config.pop_size),
        EAOperation(record_statistics, history),
    ]
    ea = EA(
        population,
        operations,
        num_steps=config.num_generations,
        is_maximisation=config.is_maximisation,
        quiet=quiet,
        db_file_path=output_dir / "database.db",
        db_handling="delete",
    )
    ea.run()
    return RunResult(best=ea.get_solution("best"), history=history)


def main() -> None:
    """Run one default EA and save its best tree genome."""
    config = DEFAULT_CONFIG
    output_dir = OUTPUT_DIR / "single_run"
    result = run_ea(config, output_dir, quiet=False)
    genome = TreeGenome.from_dict(
        cast(dict[str, Any], result.best.genotype),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    genome.save_json(str(output_dir / "best_genome.json"))

    print(f"best fitness: {result.best.fitness:.4f} (lower is better)")
    print(f"best modules: {len(genome.nodes)}")
    print(f"results: {output_dir}")


if __name__ == "__main__":
    main()
