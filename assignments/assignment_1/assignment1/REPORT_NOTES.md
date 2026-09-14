# Report content

## Research question and hypothesis

**RQ:** How does mutation strategy affect convergence when evolving a bounded
direct tree genome toward a set of target robot morphologies?

**Hypothesis:** Mixed structural mutation will reach lower fitness than
node-replacement mutation because it can alter both labels and topology.

## Methods

Both variants used ARIEL `TreeGenome`, population 50, 100 generations,
20 modules maximum, depth 12 maximum, tournament size 3, no crossover, one
mutation per offspring, and elitist `(mu + lambda)` survival. Variant
`point_only` used `mutate_replace_node`. Variant `mixed` selected uniformly
from `mutate_replace_node`, `mutate_subtree_replacement`, `mutate_shrink`, and
`mutate_hoist`. Fitness was the provided mean weighted tree-edit distance to
the five targets plus one population standard deviation; lower was better.

Each variant used paired seeds 11, 22, 33, 44, and 55. Each run evaluated 50
initial individuals plus 50 offspring for each of 100 generations: 5,050
evaluations. Random search used the same seeds, representation, constraints,
fitness, and 5,050-evaluation budget. The plotted line is mean best-so-far
fitness across runs and the band is plus/minus one population standard
deviation across runs.

## Results and discussion

Final best fitness (mean plus/minus population standard deviation):

- Point mutation: 13.8403 +/- 0.1818
- Mixed mutation: 12.8598 +/- 0.4736
- Random search: 16.6439 +/- 0.1171

Mixed mutation beat point mutation for every paired seed. Its mean final
fitness was 0.9805 lower, although its between-run spread was larger. Both EAs
clearly outperformed equal-budget random search. The convergence plot shows
that point mutation plateaued earlier, while mixed mutation continued
improving. These observations support the hypothesis: operators that can
replace, shrink, or hoist subtrees provide broader structural exploration than
node replacement (which can only prune structure as a side effect). The
conclusion is limited to this target set, budget, and five runs.

## Conclusion

Mutation strategy materially affected convergence. Under otherwise identical
conditions, mixed structural mutation produced better final compromises
across the target morphologies than node-replacement mutation.

## Bibliography leads

- Koza, J. R. (1992). *Genetic Programming*. MIT Press.
- Zhang, K. and Shasha, D. (1989). Simple fast algorithms for the editing
  distance between trees and related problems. *SIAM Journal on Computing*.
