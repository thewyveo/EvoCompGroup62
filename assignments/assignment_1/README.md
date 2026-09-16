# Assignment 1

Research question: **For a direct tree genome evolved with subtree crossover,
does mixed structural mutation converge to better compromise bodies than
node-replacement mutation?**

- `point_only`: one `mutate_replace_node` operation per offspring.
- `mixed`: one uniformly chosen operation per offspring:
  `mutate_replace_node`, `mutate_subtree_replacement`, `mutate_shrink`, or
  `mutate_hoist`.

Everything else is identical: tree representation, targets, fitness,
population, subtree crossover, tournament selection, elitist survival,
module/depth limits, evaluation budget, and seeds.

## Files

| File | Contents |
|---|---|
| `A1_evolution.py` | Our EA, random-search baseline, statistics, and plots |
| `__data__/assignment1/` | Results of the final experiment |
| `A1_template_2026.py`, `tree_edit_distance.py`, `target_bodies/`, `Assignment1.html` | Course material |

## Running

From the repository root:

```bash
uv run python assignments/assignment_1/A1_evolution.py
```

This runs ten seeds of each EA variant and ten equal-budget random-search runs
(about 5 minutes). Results go to `assignments/assignment_1/__data__/assignment1`:
per-run ARIEL databases, `results.json` and `best_genome.json`; `summary.json`
and `summary.csv` (including Mann-Whitney U tests); `convergence.png` and
`population_dynamics.png`.
