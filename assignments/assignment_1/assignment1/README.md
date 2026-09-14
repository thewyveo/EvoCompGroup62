# Assignment 1

Research question: **How does mutation strategy affect convergence when
evolving a bounded direct tree genome toward a set of target robot
morphologies?**

- `point_only`: one `mutate_replace_node` operation per offspring.
- `mixed`: one uniformly chosen operation per offspring:
  `mutate_replace_node`, `mutate_subtree_replacement`, `mutate_shrink`, or
  `mutate_hoist`.

Everything else is fixed: tree representation, targets, fitness, population,
tournament selection, module/depth limits, evaluation budget, and paired
seeds. Crossover is disabled so mutation is the only variation mechanism.
Fitness is the provided mean tree-edit distance plus one population standard
deviation, and lower is better.

From the repository root:

```bash
uv run python -m assignments.assignment_1.assignment1.experiments
```

This performs five runs of each EA variant and five equal-budget random-search
runs. Results are written under `assignments/assignment_1/__data__/assignment1`:
ARIEL SQLite databases, per-run JSON, best genomes, `summary.csv`,
`summary.json`, and `convergence.png`.
