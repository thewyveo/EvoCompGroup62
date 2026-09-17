import argparse
import json
from pathlib import Path

import mujoco

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer

"""
uv run python assignments/assignment_1/insights/render.py \
assignments/assignment_1/__data__/assignment1/point_only/seed_44/best_genome.json
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("genome", type=Path, help="Path to best_genome.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rendered_best.png"),
    )
    args = parser.parse_args()

    # Load saved TreeGenome
    genome_dict = json.loads(args.genome.read_text())
    genome = TreeGenome.from_dict(genome_dict)

    print(f"Loaded genome with {len(genome.nodes)} modules")

    # TreeGenome -> NetworkX -> MuJoCo robot
    graph = genome.to_networkx()
    robot_spec = construct_mjspec_from_graph(graph).spec

    # Put robot into a flat world
    world = SimpleFlatWorld()
    world.spawn(
        robot_spec,
        position=(0.0, 0.0, 0.1),
        correct_collision_with_floor=True,
    )

    # Compile MuJoCo model
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Let MuJoCo establish the initial state
    mujoco.mj_forward(model, data)

    # Render
    args.output.parent.mkdir(parents=True, exist_ok=True)

    single_frame_renderer(
        model,
        data,
        show=False,
        save=True,
        save_path=args.output,
    )

    print(f"Saved render to: {args.output}")


if __name__ == "__main__":
    main()