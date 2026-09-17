import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "__data__" / "assignment1"

for approach in ("point_only", "mixed", "random_search"):
    sizes = []

    for results_file in sorted((OUTPUT_DIR / approach).glob("seed_*/results.json")):
        with results_file.open("r", encoding="utf-8") as f:
            result = json.load(f)

        sizes.append(result["best_modules"])

    print(f"{approach}:")
    print("  sizes:", sizes)

    if sizes:
        print(f"  range: {min(sizes)}--{max(sizes)}")
    else:
        print("  NO RESULTS FOUND")