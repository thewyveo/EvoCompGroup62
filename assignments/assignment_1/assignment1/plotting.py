"""Convergence plotting across independent experiment runs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR


def plot_convergence(
    results_dir: Path = OUTPUT_DIR,
    output_file: Path | None = None,
) -> None:
    """Plot per-generation mean fitness with one-standard-deviation bands."""
    destination = output_file or results_dir / "convergence.png"
    labels = {
        "point_only": "EA: point mutation",
        "mixed": "EA: mixed mutation",
        "random_search": "Random search",
    }

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    for approach, label in labels.items():
        paths = sorted((results_dir / approach).glob("seed_*/results.json"))
        if not paths:
            raise FileNotFoundError(f"no results found for {approach}")
        histories = []
        for path in paths:
            with path.open(encoding="utf-8") as file:
                record = json.load(file)
            histories.append(
                [float(row["best"]) for row in record["history"]],
            )

        values = np.asarray(histories, dtype=float)
        generations = np.arange(values.shape[1])
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        axis.plot(generations, mean, label=label)
        axis.fill_between(generations, mean - std, mean + std, alpha=0.18)

    axis.set_xlabel("Generation")
    axis.set_ylabel("Best fitness (lower is better)")
    axis.set_title("Convergence across five independent runs")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=300)
    plt.close(figure)


if __name__ == "__main__":
    plot_convergence()
