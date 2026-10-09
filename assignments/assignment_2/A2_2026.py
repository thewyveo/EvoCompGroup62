"""EC A2 template code - neuroevolution for targeted locomotion with ARIEL.

WHAT THIS FILE IS
-----------------
A *demo*, not a solution. It spawns a robot, drives it with a neural network
whose weights are RANDOM, runs the simulation, and reports how close the robot
ended up to a target.

There is deliberately NO evolution in here. Building the EA (representation,
initialisation, parent selection, variation, survivor selection) is the assignment.
See "YOUR JOB" at the bottom of this file.

THE ASSIGNMENT IN A NUTSHELL
------------------------------
Evolve the weights of a neural network controller so that a robot moves from
SPAWN_POS to TARGET_POSITION within the simulation time.

    fitness = distance between the robot's final position and TARGET_POSITION

HOW TO RUN
----------

Change MODE below to switch between an interactive viewer, a headless run,
a rendered video, or a single frame.
"""

# Standard library
from pathlib import Path
from typing import Literal
import argparse
import csv
import json
import multiprocessing as mp
import random
import time

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.ec import set_seed
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from ariel.ec import (
    EA,
    Crossover,
    EAOperation,
    FloatMutator,
    FloatsGenerator,
    Individual,
    Population,
    config,
)


# Type aliases
type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
SEED = 42
RNG = np.random.default_rng(SEED)

# ariel.ec's own generators/mutators/crossover draw from a separate,
# package-level RNG. Reseed it too if you build your EA on ariel.ec,
# or every one of your "multiple seeds" runs the same variation operators.
set_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]  # where the robot starts
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]  # where it should end up
SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation
MODE: ViewerTypes = "simple"  # see run_experiment() for the options


# ============================================================================ #
#  1. THE BODY AND THE WORLD
# ============================================================================ #
def build_world() -> SimpleFlatWorld:
    """Create the environment the robot lives in.

    YOU MAY CHANGE THIS. Options include: SimpleFlatWorld, RuggedTerrainWorld,
    CraterTerrainWorld, AmphitheatreTerrainWorld, OlympicArena, ...
    (SimpleTiltedWorld is not supported for this task.)

    Whatever you pick, keep it FIXED for all runs you compare against each
    other, and say in your report which one you used. A controller evolved on
    flat ground and one evolved on rugged terrain are not comparable numbers.
    """
    return SimpleFlatWorld()


def build_robot() -> CoreModule:
    """Create the robot body.

    YOU MAY CHANGE THIS. Options include the prebuilt bodies in
    `ariel.body_phenotypes.robogen_lite.prebuilt_robots` (gecko, spider, ...).

    Two consequences of this choice, and they matter:
      * The body determines `model.nu` (the number of hinges you must send
        commands to) - that is the OUTPUT size of your controller.
      * The body determines the size of `data.qpos` - if you feed qpos to your
        network, that is (part of) your INPUT size.
    Change the body and your genotype length changes with it. Keep the body
    FIXED within an experiment.
    """
    return john_set.gecko()


# ============================================================================ #
#  2. THE CONTROLLER CONTRACT
# ============================================================================ #
#
# MuJoCo calls the controller every physics step with (model, data); its job
# is to write into data.ctrl.
#
#   INPUTS   : whatever you read from `data` (qpos, qvel, time, ...), plus any
#              task info you already know, e.g. the vector to TARGET_POSITION.
#              INPUT SIZE is your choice, but must stay CONSTANT.
#   OUTPUTS  : exactly `model.nu` values, one per actuated hinge.
#   RANGE    : hinges accept [-pi/2, +pi/2] radians. A tanh output gives
#              [-1, 1] - rescale: actions * (np.pi / 2).
#   WRITING  : DIRECT (data.ctrl[:] = actions) commands the angle straight -
#              fast, but can destabilise the sim on large jumps. DELTA
#              (data.ctrl[:] += actions * alpha, alpha ~ 0.05, then clip) is
#              smoother but accumulates, so clipping is required. Pick one,
#              justify it, use it everywhere.
#   NaN      : blown-up weights silently write NaN into data.ctrl. Assert
#              against it while developing.
#
# ============================================================================ #

# Controller architecture - decide before writing your EA.
HIDDEN_SIZE: int = 6
GAIT_FREQ: float = 1.0


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    weights: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Map robot state to hinge commands: in -> hidden -> actions.

    In this demo `weights` is drawn at RANDOM. In your assignment, `weights`
    is what the evolutionary algorithm produces: an individual's genotype,
    reshaped into these matrices. You are free to change the architecture
    itself (layers, activations, ...) - just keep input/output sizes correct.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model. Use `model.nu` for the number of hinges.
    data : mj.MjData
        The MuJoCo data. This is where you read the robot's state from.
    weights : list of ndarray
        [w1, w2] - the layer weight matrices.

    Returns
    -------
    npt.NDArray[np.float64]
        `model.nu` action values, already scaled to [-pi/2, pi/2].
    """
    w1, w2 = weights

    # --- INPUTS ---------------------------------------------------------- #
    # Bare qpos - the simplest choice, not necessarily a good one. See
    # YOUR JOB below.
    t = data.time
    w, x, y, z = data.qpos[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))  # heading
    to_target = np.asarray(TARGET_POSITION[:2]) - data.qpos[0:2]
    bearing = np.arctan2(to_target[1], to_target[0]) - yaw  # robot frame
    start_dist = np.linalg.norm(np.subtract(TARGET_POSITION[:2], SPAWN_POS[:2]))
    phase = 2 * np.pi * GAIT_FREQ * t
    inputs = np.concatenate((
        [1.0, np.sin(phase), np.cos(phase), np.sin(2 * phase)],
        [np.cos(bearing), np.sin(bearing)],
        [np.linalg.norm(to_target) / start_dist],
        data.qpos[7:],
    ))

    # --- FORWARD PASS ----------------------------------------------------- #
    layer1 = np.tanh(inputs @ w1)
    outputs = np.tanh(layer1 @ w2)  # in [-1, 1]

    # --- RESCALE TO THE HINGE RANGE --------------------------------------- #
    return outputs * (np.pi / 2)  # in [-pi/2, pi/2]


def make_random_weights(
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Draw a random parameter set for `nn_controller`.

    THIS IS THE FUNCTION YOUR EA REPLACES. Instead of sampling weights from a
    normal distribution, your EA will search for them.

    Note the total parameter count printed by main(): that is the length of the
    flat vector an individual's genotype has to encode. Reshaping a flat
    genotype back into these matrices is on you.
    """
    genome = globals().get("CURRENT_GENOME")

    if genome is None:
        return [
            RNG.normal(scale=0.5, size=(input_size, HIDDEN_SIZE)),
            RNG.normal(scale=0.5, size=(HIDDEN_SIZE, output_size)),
        ]

    g = np.asarray(genome)
    split = input_size * HIDDEN_SIZE

    return [
        g[:split].reshape(input_size, HIDDEN_SIZE),
        g[split : split + HIDDEN_SIZE * output_size].reshape(HIDDEN_SIZE, output_size)
    ]


# ============================================================================ #
#  3. POSITION AND FITNESS
# ============================================================================ #
#
# The robot is spawned with a free joint, so data.qpos[0:3] IS the core's
# (x, y, z) world position. Read it before and after stepping - no tracker or
# bookkeeping needed. (`data.geom("robot1_core").xpos` works too.)
#
# ============================================================================ #


def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position."""
    return np.asarray(data.qpos[0:3]).copy()


def fitness_function(
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
) -> float:
    """Score one evaluation. LOWER IS BETTER.

    The plain version: how far is the robot from the target when time runs out?

    `initial_position` is unused here on purpose - it is passed in because the
    moment you want a less naive fitness you will need it. Some things worth
    thinking about (and, ideally, comparing in your report):
      * Distance *reduced* rather than distance remaining, so a robot that
        starts closer is not rewarded for standing still.
      * Penalising a robot that falls over or leaves the arena.
      * Whether the z-axis should count at all - a robot that jumps is not
        closer to the target in any way you care about.
    See `ariel.simulation.tasks.targeted_locomotion` for some worked variants.
    """
    target = np.asarray(TARGET_POSITION)
    return float(np.linalg.norm(final_position[:2] - target[:2]))


# ============================================================================ #
#  4. RUNNING ONE EVALUATION
# ============================================================================ #


def run_experiment(mode: ViewerTypes = MODE) -> float:
    """Set up the world, run one simulation, and return the fitness.

    This is the function your EA calls once per individual, with `mode` set
    to "simple" (headless).

    Returns
    -------
    float
        The fitness of this run. Lower is better.
    """
    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    # --- World and robot --------------------------------------------------- #
    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Compile the world into a model. USE AS IS.
    model = world.spec.compile()
    data = mj.MjData(model)

    # Put the simulation in a clean, known state before reading anything.
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # --- Wire up the controller -------------------------------------------- #
    # Sizes are read from the compiled model, never hardcoded - they depend on
    # the body you chose in build_robot().
    input_size = len(data.qpos)
    output_size = model.nu

    weights = make_random_weights(input_size, output_size)

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        actions = nn_controller(m, d, weights)

        # DIRECT application (see the controller contract above).
        d.ctrl[:] = actions

        # DELTA application - comment out the line above and use these instead:
        # delta = 0.05
        # d.ctrl[:] += actions * delta
        # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    # --- Record the starting point ----------------------------------------- #
    initial_position = get_core_position(data)

    # --- Run ---------------------------------------------------------------- #
    if mode != "no_control":
        mj.set_mjcb_control(control_callback)

    match mode:
        case "launcher":
            # Interactive window. Great for seeing what your robot does,
            # useless inside an evolutionary loop.
            viewer.launch(model=model, data=data)
        case "simple":
            # Headless. THIS is the one your EA uses.
            simple_runner(model, data, duration=SIM_DURATION)
        case "video":
            # Render to an mp4 - for the figures in your report.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(
                model,
                data,
                duration=SIM_DURATION,
                video_recorder=recorder,
            )
        case "frame":
            # A single image of the scene. Useful to check your spawn position
            # and that the robot is not clipping through the floor.
            single_frame_renderer(model, data, steps=1, show=True)
        case "no_control":
            # No controller attached: drag the hinges around by hand.
            viewer.launch(model=model, data=data)

    # Detach the callback again so the next run starts clean.
    mj.set_mjcb_control(None)

    # --- Score -------------------------------------------------------------- #
    final_position = get_core_position(data)
    fitness = fitness_function(initial_position, final_position)

    console.log(f"start  : {np.round(initial_position, 3)}")
    console.log(f"end    : {np.round(final_position, 3)}")
    console.log(f"target : {np.round(TARGET_POSITION, 3)}")
    console.log(f"fitness: {fitness:.4f}   (lower is better)")

    return fitness


def main() -> None:
    """Run a single demo evaluation with a randomly-weighted controller."""
    # A quick look at the size of the problem you are about to search.
    mj.set_mjcb_control(None)
    world = build_world()
    robot = build_robot()
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mj.MjData(model)

    input_size = len(data.qpos)
    output_size = model.nu
    num_weights = (
        input_size * HIDDEN_SIZE
        + HIDDEN_SIZE * output_size
    )
    console.log(f"controller inputs (len(data.qpos)) : {input_size}")
    console.log(f"controller outputs (model.nu)      : {output_size}")
    console.log(f"genotype length (total weights)    : {num_weights}")

    run_experiment(MODE)


if __name__ == "__main__":
    main()


# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above runs one robot with random weights. It will score badly, and
# it will score badly in a slightly different way every time you change SEED.
# Your task is to replace "random" with "evolved".
#
# Build a proper EA on top of `ariel.ec`. You are expected to use that module -
# it gives you the population/individual data model, the operators, and free
# persistence of every generation to a SQLite database, which you will want
# when it is time to plot convergence curves for the report.
#
#     from ariel.ec import EA, EAOperation, Individual, Population
#
# For a complete, runnable example of how those pieces fit together (a one-max
# EA with parent selection, crossover, mutation and survivor selection written
# as separate steps), read:
#
#     examples/new_EC_engine_example.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- EXPERIMENTAL RIGOUR ---------------------------------------------------
#
#   One run proves nothing. Repeat every configuration over several
#     independent seeds and report mean and spread.
#   Log best/mean/worst fitness per generation. The database `ariel.ec`
#     writes makes this straightforward.
#   Compare against a baseline. Random search with the same evaluation
#     budget is a simple, but reasonable choice; and it is nearly free to run.
#   Keep body, world, SIM_DURATION and fitness function identical across
#     everything you compare. Change one thing at a time.
#
# ============================================================================ #

type Algo = Literal["fixed", "selfadapt", "random"]

SIGMA_INIT: float = 0.1
SIGMA_MIN: float = 1e-3
W_INIT_STD: float = 0.5
W_MAX: float = 5.0
TOURNAMENT_K: int = 3
P_CROSSOVER: float = 0.9
FAIL_FITNESS: float = 1e3

CURRENT_GENOME: list[float] | None = None
POOL = None


def seed_everything(seed: int) -> None:
    global RNG
    RNG = np.random.default_rng(seed)
    set_seed(seed)
    random.seed(seed)


def problem_size() -> tuple[int, int]:
    mj.set_mjcb_control(None)
    world = build_world()
    world.spawn(build_robot().spec, position=SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    return len(data.qpos), model.nu


def genotype_length() -> int:
    input_size, output_size = problem_size()
    return input_size * HIDDEN_SIZE + HIDDEN_SIZE * output_size + 1


def random_genome(n_genes: int) -> list[float]:
    weights = FloatsGenerator.normal(mean=0.0, std=W_INIT_STD, size=n_genes - 1)
    return [*np.clip(weights, -W_MAX, W_MAX).tolist(), SIGMA_INIT]


def evaluate_genome(genome: list[float], mode: ViewerTypes = "simple") -> float:
    global CURRENT_GENOME
    CURRENT_GENOME = list(genome)
    was_quiet = console.quiet
    console.quiet = mode == "simple"
    try:
        fitness = run_experiment(mode)
    finally:
        console.quiet = was_quiet
        CURRENT_GENOME = None
    return fitness if np.isfinite(fitness) else FAIL_FITNESS


def evaluate_genomes(genomes: list[list[float]]) -> list[float]:
    if POOL is None:
        return [evaluate_genome(g) for g in genomes]
    return POOL.map(evaluate_genome, genomes)


class RunLogger:
    FIELDS = ("generation", "evaluations", "best", "mean", "median", "worst",
              "std", "best_so_far", "mean_sigma", "seconds")

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "stats.csv"
        self.generation = 0
        self.evaluations = 0
        self.best_so_far = np.inf
        self.best_genome: list[float] = []
        self.history: list[float] = []
        self.t0 = time.time()
        with self.path.open("w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)

    def log(self, fitnesses: list[float], genomes: list[list[float]]) -> None:
        f = np.asarray(fitnesses)
        if f.min() < self.best_so_far:
            self.best_so_far = float(f.min())
            self.best_genome = list(genomes[int(f.argmin())])
        self.history.append(self.best_so_far)
        row = (self.generation, self.evaluations, f.min(), f.mean(),
               np.median(f), f.max(), f.std(), self.best_so_far,
               np.mean([g[-1] for g in genomes]), time.time() - self.t0)
        with self.path.open("a", newline="") as fh:
            csv.writer(fh).writerow(row)
        console.log(f"gen {self.generation:4d} | evals {self.evaluations:6d} | "
                    f"best {f.min():.4f} | mean {f.mean():.4f} | "
                    f"best-so-far {self.best_so_far:.4f}")
        self.generation += 1

    def stagnated(self, patience: int, tol: float = 1e-3) -> bool:
        if patience <= 0 or len(self.history) <= patience:
            return False
        return self.history[-patience - 1] - self.history[-1] < tol


@EAOperation
def parent_selection(population: Population, n_offspring: int) -> Population:
    parents = population.alive.evaluated.to_list()

    def tournament() -> Individual:
        idx = RNG.choice(len(parents), size=TOURNAMENT_K, replace=False)
        return min((parents[i] for i in idx), key=lambda ind: ind.fitness)

    for _ in range(n_offspring // 2):
        child = Individual()
        child.tags = {"p1": tournament().id, "p2": tournament().id}
        population.append(child)
    return population


@EAOperation
def crossover(population: Population) -> Population:
    by_id = {ind.id: ind for ind in population if ind.id is not None}
    for child in population.where(lambda ind: ind.requires_init).to_list():
        pa, pb = by_id[child.tags["p1"]], by_id[child.tags["p2"]]
        if RNG.random() < P_CROSSOVER:
            g1, g2 = Crossover.uniform(pa.genotype, pb.genotype)
        else:
            g1, g2 = pa.genotype, pb.genotype
        child.genotype = list(g1)
        sibling = Individual()
        sibling.genotype = list(g2)
        sibling.tags = dict(child.tags)
        population.append(sibling)
    return population


@EAOperation
def mutate(population: Population, variant: Algo) -> Population:
    for ind in population.where(lambda i: i.time_of_birth == -1 and not i.requires_init):
        genome = list(ind.genotype)
        weights, sigma = genome[:-1], float(genome[-1])
        if variant == "selfadapt":
            tau = 1.0 / np.sqrt(len(weights))
            sigma = max(sigma * float(np.exp(tau * RNG.normal())), SIGMA_MIN)
        weights = FloatMutator.gaussian(
            weights, std=sigma, mutation_probability=1.0,
            lower_bound=-W_MAX, upper_bound=W_MAX,
        )
        ind.genotype = [*weights, sigma]
        ind.requires_eval = True
    return population


@EAOperation
def evaluate(population: Population, logger: RunLogger) -> Population:
    todo = population.unevaluated.to_list()
    for ind, fit in zip(todo, evaluate_genomes([list(i.genotype) for i in todo]), strict=True):
        ind.fitness = fit
    logger.evaluations += len(todo)
    return population


@EAOperation
def survivor_selection(population: Population, mu: int) -> Population:
    for ind in population.alive.sort(sort="min")[mu:]:
        ind.alive = False
    return population


@EAOperation
def log_generation(population: Population, logger: RunLogger) -> Population:
    alive = population.alive.to_list()
    logger.log([i.fitness for i in alive], [list(i.genotype) for i in alive])
    return population


def run_ea(variant: Algo, seed: int, args: argparse.Namespace, run_dir: Path) -> RunLogger:
    seed_everything(seed)
    n_genes = genotype_length()
    logger = RunLogger(run_dir)

    config.target_population_size = args.mu
    initial = Population([Individual() for _ in range(args.mu)])
    for ind in initial:
        ind.genotype = random_genome(n_genes)
    initial = evaluate(logger=logger)(initial)
    log_generation(logger=logger)(initial)

    ops = [
        parent_selection(n_offspring=args.lam),
        crossover(),
        mutate(variant=variant),
        evaluate(logger=logger),
        survivor_selection(mu=args.mu),
        log_generation(logger=logger),
    ]
    ea = EA(initial, ops, num_steps=args.gens, is_maximisation=False, quiet=True,
            db_file_path=run_dir / "database.db", db_handling="delete")
    for _ in range(args.gens):
        ea.step()
        if logger.stagnated(args.patience):
            console.log(f"plateau: no improvement for {args.patience} generations")
            break
    return logger


def run_random_search(seed: int, args: argparse.Namespace, run_dir: Path) -> RunLogger:
    seed_everything(seed)
    n_genes = genotype_length()
    logger = RunLogger(run_dir)
    for gen in range(args.gens + 1):
        batch = [random_genome(n_genes) for _ in range(args.mu if gen == 0 else args.lam)]
        fits = evaluate_genomes(batch)
        logger.evaluations += len(batch)
        logger.log(fits, batch)
    return logger


def cmd_run(args: argparse.Namespace) -> None:
    for seed in args.seeds:
        run_dir = DATA / args.algo / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(
            {**vars(args), "seed": seed, "seeds": None, "func": None,
             "SIM_DURATION": SIM_DURATION, "SPAWN_POS": SPAWN_POS,
             "TARGET_POSITION": TARGET_POSITION, "HIDDEN_SIZE": HIDDEN_SIZE,
             "GAIT_FREQ": GAIT_FREQ, "SIGMA_INIT": SIGMA_INIT,
             "TOURNAMENT_K": TOURNAMENT_K, "P_CROSSOVER": P_CROSSOVER,
             "body": "john_set.gecko", "world": "SimpleFlatWorld"},
            indent=2, default=str))
        console.rule(f"{args.algo} | seed {seed}")
        if args.algo == "random":
            logger = run_random_search(seed, args, run_dir)
        else:
            logger = run_ea(args.algo, seed, args, run_dir)
        (run_dir / "best_genome.json").write_text(json.dumps(
            {"fitness": logger.best_so_far, "genome": logger.best_genome}))
        console.log(f"done: best fitness {logger.best_so_far:.4f} -> {run_dir}")


def load_runs(algo: str) -> list[dict[str, npt.NDArray[np.float64]]]:
    runs = []
    for path in sorted((DATA / algo).glob("seed_*/stats.csv")):
        with path.open() as f:
            rows = list(csv.DictReader(f))
        runs.append({k: np.array([float(r[k]) for r in rows]) for k in RunLogger.FIELDS})
    return runs


def pad(arrays: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    n = max(len(a) for a in arrays)
    return np.array([np.concatenate((a, np.full(n - len(a), a[-1]))) for a in arrays])


def cmd_plot(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    algos = [a for a in ("random", "fixed", "selfadapt") if load_runs(a)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    finals = {}
    for algo in algos:
        runs = load_runs(algo)
        best = pad([r["best_so_far"] for r in runs])
        evals = max((r["evaluations"] for r in runs), key=len)
        m, s = best.mean(0), best.std(0)
        (line,) = ax1.plot(evals, m, label=f"{algo} best (n={len(runs)})")
        ax1.fill_between(evals, m - s, m + s, alpha=0.2, color=line.get_color())
        if algo != "random":
            mean = pad([r["mean"] for r in runs])
            ax1.plot(evals, mean.mean(0), "--", color=line.get_color(),
                     label=f"{algo} pop. mean")
        finals[algo] = best[:, -1]
    ax1.set(xlabel="evaluations", ylabel="distance to target [m] (lower = better)",
            title="Best-so-far fitness, mean ± std over seeds")
    ax1.legend()
    ax2.boxplot(list(finals.values()), tick_labels=list(finals.keys()))
    ax2.set(ylabel="final best distance [m]", title="Final best fitness per run")
    fig.tight_layout()
    fig.savefig(DATA / "convergence.png", dpi=200)
    console.log(f"saved {DATA / 'convergence.png'}")

    for algo, f in finals.items():
        console.log(f"{algo:10s} final best: mean {f.mean():.4f}  std {f.std():.4f}  n={len(f)}")
    try:
        from scipy.stats import mannwhitneyu

        names = list(finals)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p = mannwhitneyu(finals[names[i]], finals[names[j]]).pvalue
                console.log(f"Mann-Whitney U {names[i]} vs {names[j]}: p = {p:.4g}")
    except ImportError:
        console.log("scipy not installed - skipping significance tests")

    if load_runs("selfadapt"):
        fig, ax = plt.subplots(figsize=(6, 4))
        sig = pad([r["mean_sigma"] for r in load_runs("selfadapt")])
        ax.plot(sig.mean(0))
        ax.fill_between(range(sig.shape[1]), sig.mean(0) - sig.std(0),
                        sig.mean(0) + sig.std(0), alpha=0.2)
        ax.set(xlabel="generation", ylabel="mean sigma in population",
               title="Self-adaptive step size")
        fig.tight_layout()
        fig.savefig(DATA / "sigma.png", dpi=200)
    if not args.no_show:
        plt.show()


def cmd_show(args: argparse.Namespace) -> None:
    saved = json.loads((DATA / args.algo / f"seed_{args.seed}" / "best_genome.json").read_text())
    fitness = evaluate_genome(saved["genome"], mode=args.view)
    console.log(f"stored fitness {saved['fitness']:.4f} | replayed fitness {fitness:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("run")
    p.add_argument("--algo", choices=["fixed", "selfadapt", "random"], required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--mu", type=int, default=30)
    p.add_argument("--lam", type=int, default=30)
    p.add_argument("--gens", type=int, default=100)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("plot")
    p.add_argument("--no-show", action="store_true")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("show")
    p.add_argument("--algo", choices=["fixed", "selfadapt", "random"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view", choices=["launcher", "video", "simple"], default="launcher")
    p.set_defaults(func=cmd_show)

    args = parser.parse_args()
    if getattr(args, "workers", 1) > 1:
        with mp.Pool(args.workers) as POOL:
            args.func(args)
    else:
        args.func(args)
