from itertools import combinations
from tree_edit_distance import tree_edit_distance
import networkx as nx
from pathlib import Path
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import load_graph_from_json

HERE = Path(__file__).resolve().parent
TARGET_DIR = HERE / "target_bodies"

def load_targets() -> list[nx.DiGraph]:
    """Load all target body JSON files in deterministic filename order."""
    paths = sorted(TARGET_DIR.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no target bodies found in {TARGET_DIR}")
    return [load_graph_from_json(path) for path in paths]


TARGETS = load_targets()

pairwise_distances = [
    tree_edit_distance(a, b)
    for a, b in combinations(TARGETS, 2)
]

print(pairwise_distances)
print("Mean pairwise target distance:", sum(pairwise_distances) / len(pairwise_distances))