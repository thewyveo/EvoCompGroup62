# Assignment 2

Research question: **For a (μ+λ) neuroevolution EA evolving the controller of a
fixed body, does a self-adaptive mutation step size reach a lower final
distance to the target than a fixed step size at the same evaluation budget?**

- `fixed`: Gaussian mutation of all 114 weights with σ = 0.1.
- `selfadapt`: σ is an extra gene, mutated log-normally
  (σ' = max(σ·exp(τ·N(0,1)), 1e-3), τ = 1/√114) before it mutates the weights.
- `random`: equal-budget random search from the same initialisation.

Everything else is identical: body (`john_set.gecko`, 6 hinges), world
(`SimpleFlatWorld`), spawn (0, 0) → target (2, 0), 15 s simulation, fitness
(planar distance to target, lower is better), 13 → 6 → 6 tanh controller
(inputs: bias, 1 Hz clock, target bearing, normalised distance, 6 joint
angles), μ = λ = 30, tournament k = 3, uniform crossover p = 0.9,
(μ+λ) survival, 400 generations (12,030 evaluations), seeds 0–9. For a given
seed, `fixed` and `selfadapt` start from the same initial population.

## Results

Final best distance to the target over 10 runs (`__data__/A2_2026/summary.json`):

| Approach | Mean ± std [m] | Median [m] | Runs < 0.1 m |
|---|---|---|---|
| random search | 1.361 ± 0.090 | 1.363 | 0/10 |
| fixed σ | 0.366 ± 0.237 | 0.458 | 3/10 |
| self-adaptive σ | 0.235 ± 0.253 | 0.229 | 4/10 |

One-sided exact Mann-Whitney U tests with Vargha-Delaney A12:

| Comparison | p | A12 |
|---|---|---|
| self-adaptive < fixed (gen 400) | 0.095 | 0.68 |
| fixed < random (gen 400) | 5.4e-6 | 1.00 |
| self-adaptive < random (gen 400) | 5.4e-6 | 1.00 |
| self-adaptive < fixed (gen 150, exploratory) | 0.124 | 0.66 |

- Self-adaptive σ is ahead on average from generation 8 on, but not
  significantly, so H1 is not supported.
- Both EAs beat random search in every run.
- Outcomes are bimodal: runs either reach the target or walk a straight but
  too slow gait and stop 0.2–0.66 m short.
- In the self-adaptive runs, σ fell below 0.01 in 4 runs, and all 4 got
  stuck; the 4 runs that reached the target kept σ between 0.015 and 0.044.
- Over the last 50 generations the mean best-so-far still improved by only
  0.02 m (fixed) and 0.01 m (self-adaptive), so the curves have flattened.

## Files

| File | Contents |
|---|---|
| `A2_2026.py` | Controller, our EA, random-search baseline, statistics, figures |
| `__data__/A2_2026/{fixed,selfadapt,random}/seed_*/` | Final experiment: `stats.csv` (per-generation statistics), `best_genome.json`, `config.json` |
| `__data__/A2_2026/summary.{json,csv}` | Final best fitness per approach, Mann-Whitney U tests, A12 |
| `__data__/A2_2026/pilot/` | Pilot runs (5 seeds × 150 generations, one 300-generation self-adaptive run) used to choose the 400-generation budget |
| `figures/` | Figures, see below |
| `A2_template_2026.py`, `Assignment2.html` | Course material |

The ARIEL SQLite databases (`database.db`) are written next to each run but
not committed (19–50 MB each); all figures and statistics come from
`stats.csv` and `best_genome.json`.

## Figures

| Figure | Shows |
|---|---|
| `fitness_over_generations.png` | Best-so-far distance to target per generation, mean ± std over 10 runs |
| `final_best.png` | Final best distance of every run (box plot + individual runs) |
| `sigma.png` | Self-adaptive step size σ per run over generations |
| `sigma_vs_final.png` | Final σ vs. final distance per run: links σ collapse to runs that got stuck |
| `trajectory.png` | Replayed path of the best and median controller of each approach |

## Running

From the repository root:

```bash
# one call per configuration; defaults are the final settings (seeds 0-9, 400 generations)
uv run python assignments/assignment_2/A2_2026.py run --algo fixed
uv run python assignments/assignment_2/A2_2026.py run --algo selfadapt
uv run python assignments/assignment_2/A2_2026.py run --algo random

# statistics + figures, then replay best/median controllers for the trajectory figure
uv run python assignments/assignment_2/A2_2026.py plot --no-show
uv run python assignments/assignment_2/A2_2026.py trajectory --no-show

# watch one evolved controller
uv run python assignments/assignment_2/A2_2026.py show --algo selfadapt --seed 0
```

A run takes about 8–10 minutes with 11 worker processes (`--workers`).
Evaluation is deterministic, so a seed reproduces its run.

## Note on `src/ariel`

The A2 template imports `set_seed` from `ariel.ec`, which our fork did not
have yet. `src/ariel/ec/__init__.py`, `src/ariel/ec/generators.py` and
`src/ariel/utils/runners.py` were therefore synced to the course repository's
"assignment 2" commit (`a435b19` in AndrzejSzczepura/EvolutionaryComputing2026);
`src/` is identical to that commit, with no changes of our own.
