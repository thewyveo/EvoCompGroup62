import json
from pathlib import Path

root = Path("assignments/assignment_1/__data__/assignment1/point_only")

best = None

for result_file in root.glob("seed_*/results.json"):
    data = json.loads(result_file.read_text())

    if best is None or data["final_best"] < best["fitness"]:
        best = {
            "fitness": data["final_best"],
            "seed": data["seed"],
            "genome": result_file.parent / "best_genome.json",
        }

print("Best fitness:", best["fitness"])
print("Seed:", best["seed"])
print("Genome:", best["genome"])