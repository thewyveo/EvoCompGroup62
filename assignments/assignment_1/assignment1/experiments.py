"""Multi-seed orchestration for two EA variants and random search."""

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, cast

from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from .baseline import random_search
from .config import DEFAULT_CONFIG, OUTPUT_DIR, ExperimentConfig
from .evaluation import load_targets
from .initialization import seed_random_generators
from .main import run_ea

FINAL_SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)
RESEARCH_QUESTION = (
    "How does mutation strategy affect convergence when evolving a bounded "
    "direct tree genome toward a set of target robot morphologies?"
)


def run_ea_variant(
    name: str,
    config: ExperimentConfig,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Run and persist one configured EA variant."""
    run_dir = output_dir / name / f"seed_{config.seed}"
    result = run_ea(config, run_dir)
    genome = TreeGenome.from_dict(
        cast(dict[str, Any], result.best.genotype),
    )
    genome.save_json(str(run_dir / "best_genome.json"))

    record: dict[str, Any] = {
        "approach": name,
        "seed": config.seed,
        "evaluation_budget": config.evaluation_budget,
        "config": asdict(config),
        "final_best": result.best.fitness,
        "best_modules": len(genome.nodes),
        "history": result.history,
    }
    with (run_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
    return record


def run_random_baseline(
    config: ExperimentConfig,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    """Run random search with exactly the EA's candidate-evaluation budget."""
    seed_random_generators(config.seed)
    genome, best_per_evaluation = random_search(
        load_targets(),
        config.evaluation_budget,
        config,
    )
    history = [
        {
            "generation": generation,
            "best": best_per_evaluation[
                ((generation + 1) * config.pop_size) - 1
            ],
        }
        for generation in range(config.num_generations + 1)
    ]

    run_dir = output_dir / "random_search" / f"seed_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    genome.save_json(str(run_dir / "best_genome.json"))
    record: dict[str, Any] = {
        "approach": "random_search",
        "seed": config.seed,
        "evaluation_budget": config.evaluation_budget,
        "config": asdict(config),
        "final_best": history[-1]["best"],
        "best_modules": len(genome.nodes),
        "history": history,
    }
    with (run_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
    return record


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute final-fitness statistics across independent runs."""
    rows: list[dict[str, Any]] = []
    for approach in ("point_only", "mixed", "random_search"):
        values = [
            float(record["final_best"])
            for record in records
            if record["approach"] == approach
        ]
        rows.append(
            {
                "approach": approach,
                "runs": len(values),
                "mean_final_best": fmean(values),
                "std_final_best": pstdev(values),
                "min_final_best": min(values),
                "max_final_best": max(values),
            },
        )
    return rows


def run_all_experiments(
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Run both one-factor EA variants and the matched random baseline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for seed in FINAL_SEEDS:
        point_config = replace(
            DEFAULT_CONFIG,
            seed=seed,
            mutation_strategy="point_only",
        )
        mixed_config = replace(
            DEFAULT_CONFIG,
            seed=seed,
            mutation_strategy="mixed",
        )
        records.append(run_ea_variant("point_only", point_config, output_dir))
        records.append(run_ea_variant("mixed", mixed_config, output_dir))
        records.append(run_random_baseline(point_config, output_dir))

    summary = summarize(records)
    metadata = {
        "research_question": RESEARCH_QUESTION,
        "variant_point_only": "One mutate_replace_node operation per offspring.",
        "variant_mixed": (
            "One uniformly selected operation per offspring: replace node, "
            "replace subtree, shrink, or hoist."
        ),
        "controlled_variables": (
            "All other settings, targets, fitness, budgets, and paired seeds "
            "are identical. Crossover is disabled."
        ),
        "summary": summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    with (output_dir / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    from .plotting import plot_convergence

    plot_convergence(output_dir)


if __name__ == "__main__":
    run_all_experiments()
