"""Reproducible settings and paths for Assignment 1."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MutationStrategy = Literal["mixed", "point_only"]

ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
TARGET_DIR = ASSIGNMENT_DIR / "target_bodies"
OUTPUT_DIR = ASSIGNMENT_DIR / "__data__" / "assignment1"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Configuration for one EA run.

    These values are shared by both final mutation variants.
    """

    pop_size: int = 50
    num_generations: int = 100
    max_modules: int = 20
    max_depth: int = 12
    seed: int = 42
    tournament_size: int = 3
    crossover_probability: float = 0.0
    mutation_probability: float = 1.0
    mutation_strategy: MutationStrategy = "mixed"
    is_maximisation: bool = False

    @property
    def evaluation_budget(self) -> int:
        """Return the initial evaluations plus one population per generation."""
        return self.pop_size * (self.num_generations + 1)


DEFAULT_CONFIG = ExperimentConfig()
