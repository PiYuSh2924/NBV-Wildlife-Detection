
import numpy as np
from math import isfinite


# All objectives are maximization objectives.
DEFAULT_OBJECTIVES = (
    "newly_visible_vertices",
    "visibility_fitness",
    "new_full_rank",
    "score_gain",
)

# Experimental search boundaries.

LOWER_BOUNDS = np.array([
    -25.0,       # x
    -25.0,       # y
    10.0,        # z
    -np.pi,      # yaw
    -np.pi / 2   # pitch
])

UPPER_BOUNDS = np.array([
    25.0,        # x
    25.0,        # y
    25.0,        # z
    np.pi,       # yaw
    0.0          # pitch
])


# Standard deviations used for Gaussian mutation.
MUTATION_SIGMA = np.array([
    3.0,     # x
    3.0,     # y
    2.0,     # z
    0.20,    # yaw
    0.15     # pitch
])


# --------------------------------------------------
# 1. PARETO DOMINANCE
# --------------------------------------------------

def dominates(candidate_a, candidate_b,
              objectives=DEFAULT_OBJECTIVES):

    values_a = [float(candidate_a[key]) for key in objectives]
    values_b = [float(candidate_b[key]) for key in objectives]

    if not all(isfinite(v) for v in values_a + values_b):
        raise ValueError("Objective values must be finite.")

    return (
        all(a >= b for a, b in zip(values_a, values_b))
        and any(a > b for a, b in zip(values_a, values_b))
    )


def pareto_front(candidates, objectives=DEFAULT_OBJECTIVES):

    if not candidates:
        return []

    selected = []

    for i, candidate in enumerate(candidates):

        for objective in objectives:
            if objective not in candidate:
                raise KeyError(
                    f"Missing objective: {objective}"
                )

            if not isfinite(float(candidate[objective])):
                raise ValueError(
                    f"Non-finite objective: {objective}"
                )

        is_dominated = any(
            dominates(other, candidate, objectives)
            for j, other in enumerate(candidates)
            if i != j
        )

        if not is_dominated:
            selected.append(candidate)

    return selected


def remove_duplicate_poses(candidates, decimals=8):
    """
    Remove candidates representing the same camera pose.

    Poses are compared using their five chromosome genes:
    x, y, z, yaw and pitch.
    """

    unique_candidates = []
    seen_poses = set()

    for candidate in candidates:

        genes = np.asarray(
            candidate["genes"],
            dtype=float
        )

        pose_key = tuple(
            np.round(genes, decimals=decimals)
        )

        if pose_key not in seen_poses:
            seen_poses.add(pose_key)
            unique_candidates.append(candidate)

    return unique_candidates



# --------------------------------------------------
# 2. INITIAL POPULATION
# --------------------------------------------------

def wrap_yaw(yaw):
    """Keep yaw inside [-pi, pi]."""

    return (yaw + np.pi) % (2.0 * np.pi) - np.pi


def constrain_chromosome(genes):

    genes = np.asarray(genes, dtype=float).copy()

    genes = np.clip(
        genes,
        LOWER_BOUNDS,
        UPPER_BOUNDS
    )

    genes[3] = wrap_yaw(genes[3])

    return genes


def create_random_chromosome(rng, target=None):

    if target is None:
        target = np.array([0.0, 0.0, 6.0])

    position = np.array([
    rng.uniform(-25.0, 25.0),
    rng.uniform(-25.0, 25.0),
    rng.uniform(10.0, 25.0)
    ])

    direction = target - position

    yaw = np.arctan2(
        direction[1],
        direction[0]
    )

    horizontal_distance = np.linalg.norm(
        direction[:2]
    )

    pitch = np.arctan2(
        direction[2],
        horizontal_distance
    )

    # Add orientation variation around the target direction.
    yaw += rng.normal(0.0, 0.35)
    pitch += rng.normal(0.0, 0.20)

    return constrain_chromosome([
        position[0],
        position[1],
        position[2],
        yaw,
        pitch
    ])


def create_initial_population(population_size, rng):

    return [
        create_random_chromosome(rng)
        for _ in range(population_size)
    ]


# --------------------------------------------------
# 3. NON-DOMINATED SORTING
# --------------------------------------------------

def non_dominated_sort(
    population,
    objectives=DEFAULT_OBJECTIVES
):
    """
    Divide the population into Pareto fronts.

    Front 0: non-dominated candidates.
    Front 1: candidates dominated only by front 0.
    Front 2: candidates dominated by earlier fronts.
    """

    domination_counts = [0] * len(population)
    dominated_indices = [[] for _ in population]

    first_front = []

    for p in range(len(population)):

        for q in range(len(population)):

            if p == q:
                continue

            if dominates(
                population[p],
                population[q],
                objectives
            ):
                dominated_indices[p].append(q)

            elif dominates(
                population[q],
                population[p],
                objectives
            ):
                domination_counts[p] += 1

        if domination_counts[p] == 0:
            first_front.append(p)

    fronts = [first_front]
    current_front = first_front

    while current_front:

        next_front = []

        for p in current_front:

            for q in dominated_indices[p]:

                domination_counts[q] -= 1

                if domination_counts[q] == 0:
                    next_front.append(q)

        if next_front:
            fronts.append(next_front)

        current_front = next_front

    return [
        [population[index] for index in front]
        for front in fronts
    ]


# --------------------------------------------------
# 4. CROWDING DISTANCE
# --------------------------------------------------


def calculate_crowding_distance(
    front,
    objectives=DEFAULT_OBJECTIVES
):
    """
    Estimate how isolated a candidate is in objective space.

    Higher crowding distance means a less crowded region.
    """

    distances = {
        id(candidate): 0.0
        for candidate in front
    }

    if len(front) <= 2:
        for candidate in front:
            distances[id(candidate)] = float("inf")

        return distances

    for objective in objectives:

        ordered = sorted(
            front,
            key=lambda item: item[objective]
        )

        minimum = float(ordered[0][objective])
        maximum = float(ordered[-1][objective])

        # Skip objectives with no variation.
        if maximum == minimum:
            continue

        # Preserve boundary candidates.
        distances[id(ordered[0])] = float("inf")
        distances[id(ordered[-1])] = float("inf")

        for i in range(1, len(ordered) - 1):

            previous_value = float(
                ordered[i - 1][objective]
            )

            next_value = float(
                ordered[i + 1][objective]
            )

            distances[id(ordered[i])] += (
                next_value - previous_value
            ) / (maximum - minimum)

    return distances



# --------------------------------------------------
# 5. TOURNAMENT SELECTION
# --------------------------------------------------

def assign_rank_and_crowding(
    population,
    objectives=DEFAULT_OBJECTIVES
):

    fronts = non_dominated_sort(
        population,
        objectives
    )

    for rank, front in enumerate(fronts):

        distances = calculate_crowding_distance(
            front,
            objectives
        )

        for candidate in front:

            candidate["_rank"] = rank

            candidate["_crowding"] = distances[
                id(candidate)
            ]

    return fronts


def tournament_selection(population, rng):

    indices = rng.choice(
        len(population),
        size=2,
        replace=False
    )

    first = population[indices[0]]
    second = population[indices[1]]

    if first["_rank"] < second["_rank"]:
        return first

    if second["_rank"] < first["_rank"]:
        return second

    if first["_crowding"] > second["_crowding"]:
        return first

    if second["_crowding"] > first["_crowding"]:
        return second

    return first if rng.random() < 0.5 else second


# --------------------------------------------------
# 6. TWO-POINT CROSSOVER
# --------------------------------------------------

def two_point_crossover(parent_a, parent_b, rng):

    child_a = np.asarray(parent_a, dtype=float).copy()
    child_b = np.asarray(parent_b, dtype=float).copy()

    # Five genes: choose two crossover points.
    points = sorted(
        rng.choice(
            np.arange(1, len(child_a)),
            size=2,
            replace=False
        )
    )

    start, end = points

    child_a[start:end], child_b[start:end] = (
        child_b[start:end].copy(),
        child_a[start:end].copy()
    )

    return (
        constrain_chromosome(child_a),
        constrain_chromosome(child_b)
    )


# --------------------------------------------------
# 7. GAUSSIAN MUTATION
# --------------------------------------------------

def gaussian_mutation(
    chromosome,
    rng,
    mutation_probability=0.20
):

    mutated = np.asarray(
        chromosome,
        dtype=float
    ).copy()

    for gene_index in range(len(mutated)):

        if rng.random() < mutation_probability:

            mutated[gene_index] += rng.normal(
                0.0,
                MUTATION_SIGMA[gene_index]
            )

    return constrain_chromosome(mutated)


# --------------------------------------------------
# 8. ELITIST SURVIVOR SELECTION
# --------------------------------------------------

def select_survivors(
    combined_population,
    population_size,
    objectives=DEFAULT_OBJECTIVES
):

    fronts = non_dominated_sort(
        combined_population,
        objectives
    )

    survivors = []

    for front in fronts:

        if len(survivors) + len(front) <= population_size:

            survivors.extend(front)

        else:

            distances = calculate_crowding_distance(
                front,
                objectives
            )

            ordered = sorted(
                front,
                key=lambda item: distances[id(item)],
                reverse=True
            )

            remaining = population_size - len(survivors)

            survivors.extend(ordered[:remaining])

            break

    return survivors


# --------------------------------------------------
# 9. GENETIC ALGORITHM
# --------------------------------------------------

def run_genetic_algorithm(
    evaluate_fn,
    population_size=12,
    generations=5,
    crossover_probability=0.90,
    mutation_probability=0.20,
    seed=42,
    objectives=DEFAULT_OBJECTIVES
):
    """
    Multi-objective genetic algorithm.

    evaluate_fn:
        Function accepting a five-value chromosome and returning
        a dictionary containing all objective values.

    Returns:
        Final population and Pareto-optimal candidates.
    """

    if population_size < 2:
        raise ValueError("Population size must be at least 2.")

    if generations < 1:
        raise ValueError("Generations must be at least 1.")

    rng = np.random.default_rng(seed)

    cache = {}

    def evaluate(chromosome):

        genes = constrain_chromosome(chromosome)

        # Avoid evaluating identical poses repeatedly.
        cache_key = tuple(
            np.round(genes, decimals=8)
        )

        if cache_key in cache:
            result = dict(cache[cache_key])
            result["genes"] = genes.copy()
            return result

        result = evaluate_fn(genes)

        for objective in objectives:

            if objective not in result:
                raise KeyError(
                    f"Evaluator did not return: {objective}"
                )

            if not isfinite(float(result[objective])):
                raise ValueError(
                    f"Invalid objective: {objective}"
                )

        result = dict(result)
        result["genes"] = genes.copy()

        cache[cache_key] = dict(result)

        return result

    # Initial generation.
    print("\nGenerating initial population...")

    population = [
        evaluate(genes)
        for genes in create_initial_population(
            population_size,
            rng
        )
    ]

    for generation in range(generations):

        print(
            f"\nGeneration {generation + 1}/{generations}"
        )

        assign_rank_and_crowding(
            population,
            objectives
        )

        offspring = []

        while len(offspring) < population_size:

            parent_a = tournament_selection(
                population,
                rng
            )

            parent_b = tournament_selection(
                population,
                rng
            )

            genes_a = parent_a["genes"]
            genes_b = parent_b["genes"]

            if rng.random() < crossover_probability:

                child_a, child_b = two_point_crossover(
                    genes_a,
                    genes_b,
                    rng
                )

            else:

                child_a = genes_a.copy()
                child_b = genes_b.copy()

            child_a = gaussian_mutation(
                child_a,
                rng,
                mutation_probability
            )

            child_b = gaussian_mutation(
                child_b,
                rng,
                mutation_probability
            )

            offspring.append(evaluate(child_a))

            if len(offspring) < population_size:
                offspring.append(evaluate(child_b))

        # Elitism: select from parents AND offspring.
        population = select_survivors(
            population + offspring,
            population_size,
            objectives
        )

        fronts = non_dominated_sort(
            population,
            objectives
        )

        print(
            "Pareto candidates in population:",
            len(fronts[0])
        )

        print(
            "Unique poses evaluated so far:",
            len(cache)
        )

    # Final Pareto front.
    final_front = pareto_front(
    population,
    objectives
    )

    final_front = remove_duplicate_poses(
        final_front
    )

    return {
        "population": population,
        "pareto_front": final_front,
        "evaluations": len(cache),
        "generations": generations
    }
