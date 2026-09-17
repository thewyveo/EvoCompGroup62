"""Assignment 1: evolving tree-genome robot bodies towards a set of target bodies."""

import copy
import csv
import json
import random
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, cast

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from rich.console import Console
from scipy.stats import mannwhitneyu

from ariel.body_phenotypes.robogen_lite.config import IDX_OF_CORE
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import load_graph_from_json
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
    remove_subtree,
    validate_tree_depth,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from tree_edit_distance import mean_plus_std_tree_edit_distance

console = Console()

HERE = Path(__file__).resolve().parent
TARGET_DIR = HERE / "target_bodies"
OUTPUT_DIR = HERE / "__data__" / "assignment1"

POP_SIZE = 50
NUM_GENERATIONS = 100
MAX_MODULES = 20
MAX_DEPTH = 12
TOURNAMENT_SIZE = 3
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 1.0
MAX_VARIATION_ATTEMPTS = 10
EVALUATION_BUDGET = POP_SIZE * (NUM_GENERATIONS + 1)
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
STRATEGIES = ("point_only", "mixed")
APPROACHES = (*STRATEGIES, "random_search")
LABELS = {
    "point_only": "EA: point mutation",
    "mixed": "EA: mixed mutation",
    "random_search": "Random search",
}


def load_targets() -> list[nx.DiGraph]:
    """Load all target body JSON files in deterministic filename order."""
    paths = sorted(TARGET_DIR.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no target bodies found in {TARGET_DIR}")
    return [load_graph_from_json(path) for path in paths]


TARGETS = load_targets()


def seed_everything(seed: int) -> None:
    """Seed every random generator used by the tree implementation."""
    random.seed(seed)
    np.random.seed(seed)


def clone(genome: TreeGenome) -> TreeGenome:
    """Return an independent deep copy.

    ``TreeGenome.from_dict`` reuses the edge list it is given, so a child built
    from ``parent.to_dict()`` would share edges with its parent.
    """
    return TreeGenome.from_dict(copy.deepcopy(genome.to_dict()))


def genome_of(individual: Individual) -> TreeGenome:
    """Deserialize an individual's stored genotype into an independent genome."""
    return TreeGenome.from_dict(copy.deepcopy(cast(dict[str, Any], individual.genotype)))


def is_valid(genome: TreeGenome) -> bool:
    """Check ARIEL structure rules plus the module and depth caps."""
    if not 1 <= len(genome.nodes) <= MAX_MODULES:
        return False
    if not validate_tree_depth(genome, MAX_DEPTH):
        return False
    try:
        validate_genome_dict(genome.to_dict())
    except ValueError:
        return False
    return True


def canonical_form(genome: TreeGenome) -> str:
    """Return an id-independent string that identifies the body structure.

    Two genomes share a canonical form exactly when they describe the same body
    (module types, rotations, and attachment faces), regardless of node ids.
    """
    children: dict[int, list[tuple[str, int]]] = {}
    for edge in genome.edges:
        children.setdefault(edge["parent"], []).append((edge["face"], edge["child"]))

    def encode(node: int) -> str:
        attrs = genome.nodes[node]
        parts = sorted(f"{face}:{encode(child)}" for face, child in children.get(node, []))
        return f"{attrs['type']}/{attrs['rotation']}({','.join(parts)})"

    return encode(IDX_OF_CORE)


def fitness_of(genome: TreeGenome) -> float:
    """Mean tree edit distance to the targets plus one std. Lower is better."""
    return mean_plus_std_tree_edit_distance(genome.to_networkx(), TARGETS)


def random_genome() -> TreeGenome:
    """Create a random genome with a uniformly drawn size in [1, MAX_MODULES].

    ARIEL's ``random_tree`` argument counts non-core modules and almost always
    fills that budget, so the size is drawn first to avoid every random body
    being exactly ``MAX_MODULES`` large.
    """
    return random_tree(max_modules=random.randint(0, MAX_MODULES - 1))


def make_individual() -> Individual:
    """Create one unevaluated individual with a random tree genome."""
    ind = Individual()
    ind.genotype = random_genome().to_dict()
    return ind


def evaluate(population: Population) -> Population:
    """Evaluate only individuals whose fitness is currently stale."""
    for ind in population.unevaluated:
        ind.fitness = fitness_of(genome_of(ind))
    return population


def parent_selection(population: Population) -> Population:
    """Run one size-3 tournament per population slot; lower fitness wins.

    Tournaments are drawn with replacement, so a strong individual can win
    several times. Each individual is tagged with its number of wins
    (``selected``), which is how many copies it gets in the mating pool.
    """
    candidates = population.alive.to_list()
    wins = [0] * len(candidates)
    for _ in range(len(candidates)):
        contestants = random.sample(range(len(candidates)), min(TOURNAMENT_SIZE, len(candidates)))
        wins[min(contestants, key=lambda i: candidates[i].fitness)] += 1
    for ind, count in zip(candidates, wins, strict=True):
        ind.tags = {"selected": count, "mutate": False}
    return population


def subtree_crossover(
    parent_a: TreeGenome,
    parent_b: TreeGenome,
) -> tuple[TreeGenome, TreeGenome, bool]:
    """Perform ARIEL subtree crossover, retrying until it produces valid changed offspring.

    ARIEL's patched ``crossover_subtree`` performs the actual subtree exchange.
    We retain our own retry logic because the experiment additionally imposes
    MAX_MODULES and MAX_DEPTH constraints and tracks whether crossover had an
    observable effect.
    """
    parent_forms = (canonical_form(parent_a), canonical_form(parent_b))

    for _ in range(MAX_VARIATION_ATTEMPTS):
        child_a, child_b = crossover_subtree(parent_a, parent_b)

        if not (is_valid(child_a) and is_valid(child_b)):
            continue

        child_forms = (canonical_form(child_a), canonical_form(child_b))

        if child_forms != parent_forms:
            return child_a, child_b, True

    return clone(parent_a), clone(parent_b), False


def crossover(population: Population, stats: dict[str, int]) -> Population:
    """Pair up the mating pool and append ``POP_SIZE`` offspring tagged for mutation."""
    pool = [ind for ind in population.alive for _ in range(int(ind.tags.get("selected", 0)))]
    random.shuffle(pool)

    offspring: list[Individual] = []
    idx = 0
    while len(offspring) < POP_SIZE:
        g_a = genome_of(pool[idx % len(pool)])
        g_b = genome_of(pool[(idx + 1) % len(pool)])
        idx += 2

        if random.random() < CROSSOVER_PROBABILITY:
            g_a, g_b, success = subtree_crossover(g_a, g_b)
            stats["crossover_pairs"] += 1
            stats["crossover_success"] += int(success)

        for genome in (g_a, g_b)[: POP_SIZE - len(offspring)]:
            child = Individual()
            child.genotype = genome.to_dict()
            child.tags = {"selected": 0, "mutate": True}
            offspring.append(child)

    population.extend(offspring)
    return population


def apply_mutation(genome: TreeGenome, strategy: str) -> None:
    """Apply one mutation operator of the given strategy in place."""
    if strategy == "point_only":
        mutate_replace_node(genome)
        return
    operator = random.choice((mutate_replace_node, mutate_subtree_replacement, mutate_shrink, mutate_hoist))
    if operator is mutate_subtree_replacement:
        mutate_subtree_replacement(genome, max_modules=MAX_MODULES)
    else:
        operator(genome)


def mutate_genome(genome: TreeGenome, strategy: str) -> tuple[TreeGenome, bool]:
    """Return a mutated copy that changes the body and respects the caps.

    ARIEL's operators silently do nothing when their result is invalid, and
    subtree replacement often exceeds the module cap, so the operator is
    re-drawn until it has an effect. On failure an unchanged copy is returned.
    """
    original_form = canonical_form(genome)
    for _ in range(MAX_VARIATION_ATTEMPTS):
        mutated = clone(genome)
        apply_mutation(mutated, strategy)
        if is_valid(mutated) and canonical_form(mutated) != original_form:
            return mutated, True
    return clone(genome), False


def mutate(population: Population, strategy: str, stats: dict[str, int]) -> Population:
    """Mutate the offspring tagged by crossover."""
    for ind in population.where(lambda ind: bool(ind.tags.get("mutate", False))):
        if random.random() < MUTATION_PROBABILITY:
            genome, success = mutate_genome(genome_of(ind), strategy)
            ind.genotype = genome.to_dict()
            stats["mutations"] += 1
            stats["mutation_success"] += int(success)
        ind.tags = {"mutate": False}
        ind.requires_eval = True
    return population


def survivor_selection(population: Population) -> Population:
    """Keep the ``POP_SIZE`` lowest-fitness living individuals (mu + lambda)."""
    ranked = sorted(population.alive, key=lambda ind: ind.fitness)
    survivors = {id(ind) for ind in ranked[:POP_SIZE]}
    for ind in population.alive:
        if id(ind) not in survivors:
            ind.alive = False
    return population


def new_stats() -> dict[str, int]:
    """Return zeroed per-generation operator counters."""
    return {"crossover_pairs": 0, "crossover_success": 0, "mutations": 0, "mutation_success": 0}


def population_statistics(population: Population, generation: int, stats: dict[str, int]) -> dict[str, float]:
    """Summarize fitness, diversity, body size, and operator success of the living population."""
    alive = population.alive.to_list()
    fitnesses = [ind.fitness for ind in alive]
    genomes = [genome_of(ind) for ind in alive]
    return {
        "generation": generation,
        "best": min(fitnesses),
        "mean": fmean(fitnesses),
        "std": pstdev(fitnesses),
        "unique_fraction": len({canonical_form(g) for g in genomes}) / len(alive),
        "mean_modules": fmean(len(g.nodes) for g in genomes),
        "crossover_success_rate": stats["crossover_success"] / max(stats["crossover_pairs"], 1),
        "mutation_success_rate": stats["mutation_success"] / max(stats["mutations"], 1),
    }


def record_statistics(population: Population, history: list[dict[str, float]], stats: dict[str, int]) -> Population:
    """Append this generation's statistics to ``history`` and reset the counters."""
    history.append(population_statistics(population, len(history), stats))
    stats.update(new_stats())
    return population


def run_ea(strategy: str, seed: int) -> dict[str, Any]:
    """Run one EA and save its best genome and per-generation history."""
    seed_everything(seed)
    run_dir = OUTPUT_DIR / strategy / f"seed_{seed}"

    initial = Population([make_individual() for _ in range(POP_SIZE)])
    initial = evaluate(initial)
    stats = new_stats()
    history = [population_statistics(initial, 0, stats)]

    ops: list[EAOperation] = [
        EAOperation(parent_selection),
        EAOperation(crossover, stats),
        EAOperation(mutate, strategy, stats),
        EAOperation(evaluate),
        EAOperation(survivor_selection),
        EAOperation(record_statistics, history, stats),
    ]

    ea = EA(
        initial,
        ops,
        num_steps=NUM_GENERATIONS,
        is_maximisation=False,
        quiet=True,
        db_file_path=run_dir / "database.db",
        db_handling="delete",
    )
    ea.run()

    best = ea.get_solution("best")
    genome = genome_of(best)
    genome.save_json(str(run_dir / "best_genome.json"))
    return save_record(run_dir, strategy, seed, best.fitness, genome, history)


def run_random_search(seed: int) -> dict[str, Any]:
    """Sample ``EVALUATION_BUDGET`` random bodies, logging best-so-far every ``POP_SIZE`` evaluations."""
    seed_everything(seed)
    run_dir = OUTPUT_DIR / "random_search" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_genome, best_fitness = None, float("inf")
    history = []
    for evaluation in range(1, EVALUATION_BUDGET + 1):
        genome = random_genome()
        fitness = fitness_of(genome)
        if fitness < best_fitness:
            best_genome, best_fitness = genome, fitness
        if evaluation % POP_SIZE == 0:
            history.append({"generation": len(history), "best": best_fitness})

    best_genome.save_json(str(run_dir / "best_genome.json"))
    return save_record(run_dir, "random_search", seed, best_fitness, best_genome, history)


def save_record(
    run_dir: Path,
    approach: str,
    seed: int,
    final_best: float,
    genome: TreeGenome,
    history: list[dict[str, float]],
) -> dict[str, Any]:
    """Write one run's result to ``results.json`` and return it."""
    record = {
        "approach": approach,
        "seed": seed,
        "evaluation_budget": EVALUATION_BUDGET,
        "final_best": final_best,
        "best_modules": len(genome.nodes),
        "history": history,
    }
    with (run_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
    return record


def final_values(records: list[dict[str, Any]], approach: str) -> list[float]:
    """Final best fitness of every run of one approach."""
    return [float(r["final_best"]) for r in records if r["approach"] == approach]


def vargha_delaney_a12(x: list[float], y: list[float]) -> float:
    """Probability that a run from ``x`` ends better (lower) than one from ``y``."""
    return sum((a < b) + 0.5 * (a == b) for a in x for b in y) / (len(x) * len(y))


def write_summary(records: list[dict[str, Any]]) -> None:
    """Write final-fitness statistics and Mann-Whitney U tests to ``summary.json`` and ``summary.csv``."""
    summary = []
    for approach in APPROACHES:
        values = final_values(records, approach)
        summary.append(
            {
                "approach": approach,
                "runs": len(values),
                "mean_final_best": fmean(values),
                "std_final_best": pstdev(values),
                "min_final_best": min(values),
                "max_final_best": max(values),
            },
        )

    tests = []
    for first, second in (("mixed", "point_only"), ("mixed", "random_search"), ("point_only", "random_search")):
        x, y = final_values(records, first), final_values(records, second)
        test = mannwhitneyu(x, y, alternative="two-sided")
        tests.append(
            {
                "comparison": f"{first} vs {second}",
                "U": float(test.statistic),
                "p_value": float(test.pvalue),
                "A12": vargha_delaney_a12(x, y),
            },
        )

    settings = {
        "pop_size": POP_SIZE,
        "num_generations": NUM_GENERATIONS,
        "evaluation_budget": EVALUATION_BUDGET,
        "max_modules": MAX_MODULES,
        "max_depth": MAX_DEPTH,
        "tournament_size": TOURNAMENT_SIZE,
        "crossover_probability": CROSSOVER_PROBABILITY,
        "mutation_probability": MUTATION_PROBABILITY,
        "max_variation_attempts": MAX_VARIATION_ATTEMPTS,
        "seeds": list(SEEDS),
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({"settings": settings, "summary": summary, "tests": tests}, file, indent=2)
    with (OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    for row in summary:
        console.log(f"{row['approach']:>14}: {row['mean_final_best']:.3f} +/- {row['std_final_best']:.3f}")
    for row in tests:
        console.log(f"{row['comparison']:>30}: p = {row['p_value']:.4f}, A12 = {row['A12']:.2f}")


def band(axis: plt.Axes, records: list[dict[str, Any]], approach: str, metric: str) -> None:
    """Plot the mean of ``metric`` over runs with a one-standard-deviation band."""
    values = np.array([[row[metric] for row in r["history"]] for r in records if r["approach"] == approach])
    mean, std = values.mean(axis=0), values.std(axis=0)
    generations = np.arange(values.shape[1])
    axis.plot(generations, mean, label=LABELS[approach])
    axis.fill_between(generations, mean - std, mean + std, alpha=0.18)


def plot_results(records: list[dict[str, Any]]) -> None:
    """Save ``convergence.png`` and ``population_dynamics.png``."""
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    for approach in APPROACHES:
        band(axis, records, approach, "best")
    axis.set_xlabel(f"Generation ({POP_SIZE} evaluations each)")
    axis.set_ylabel("Best fitness (lower is better)")
    axis.set_title(f"Convergence: mean ± std over {len(SEEDS)} independent runs")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "convergence.png", dpi=300)
    plt.close(figure)

    panels = (
        ("mean", "Mean population fitness"),
        ("unique_fraction", "Distinct bodies (fraction)"),
        ("mean_modules", "Mean modules per body"),
    )
    figure, axes = plt.subplots(1, len(panels), figsize=(12.0, 3.6))
    for axis, (metric, title) in zip(axes, panels, strict=True):
        for approach in STRATEGIES:
            band(axis, records, approach, metric)
        axis.set_xlabel("Generation")
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "population_dynamics.png", dpi=300)
    plt.close(figure)


def main() -> None:
    """Run both EA variants and random search for every seed, then summarize and plot."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in SEEDS:
        for strategy in STRATEGIES:
            console.log(f"EA {strategy}, seed {seed}")
            records.append(run_ea(strategy, seed))
        console.log(f"random search, seed {seed}")
        records.append(run_random_search(seed))

    write_summary(records)
    plot_results(records)


if __name__ == "__main__":
    main()
