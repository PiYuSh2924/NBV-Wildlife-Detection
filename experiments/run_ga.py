
import numpy as np

from nbv_wildlife.environment import create_forest
from nbv_wildlife.viewpoints import CameraPose, create_initial_grid_views
from nbv_wildlife.visibility import VisibilityEngine

from nbv_wildlife.geometry import (
    focal_lengths_from_fov,
    compute_information_matrices,
    compute_projection_jacobian
)

from nbv_wildlife.nbv import run_genetic_algorithm


# --------------------------------------------------
# 1. GEOMETRY EVALUATION HELPERS
# --------------------------------------------------

def calculate_ranks(information):

    eigenvalues = np.linalg.eigvalsh(information)

    largest = np.maximum(
        eigenvalues[:, -1],
        0.0
    )

    tolerance = np.maximum(
        largest * 1e-10,
        1e-12
    )

    return np.sum(
        eigenvalues > tolerance[:, None],
        axis=1
    )


def regularized_logdet_score(
    information,
    regularization=1e-6
):

    regularized = (
        information +
        regularization * np.eye(3)
    )

    signs, log_determinants = np.linalg.slogdet(
        regularized
    )

    if np.any(signs <= 0):
        raise ValueError(
            "Non-positive determinant encountered."
        )

    return float(np.sum(log_determinants))


# --------------------------------------------------
# 2. MAIN EXPERIMENT
# --------------------------------------------------

def main():

    print("\n" + "=" * 75)
    print("GENETIC ALGORITHM FOR MULTI-OBJECTIVE NBV")
    print("=" * 75)

    # ----------------------------------------------
    # A. CREATE ENVIRONMENT
    # ----------------------------------------------

    print("\n[1/4] Creating forest...")

    forest = create_forest(
        width=30.0,
        depth=30.0,
        num_trees=45,
        seed=42
    )

    vertices = np.asarray(
        forest.combined_mesh.vertices
    )

    print("Total vertices:", len(vertices))

    # ----------------------------------------------
    # B. CREATE BASELINE
    # ----------------------------------------------

    print("\n[2/4] Calculating baseline information...")

    baseline_views = create_initial_grid_views(
        width=30.0,
        depth=30.0,
        grid_size=6,
        height=16.0
    )

    print("Baseline cameras:", len(baseline_views))

    engine = VisibilityEngine(
        forest.combined_mesh,
        horizontal_fov_deg=60.0,
        vertical_fov_deg=45.0
    )

    baseline_visibility = engine.compute_visibility_matrix(
        baseline_views
    )

    baseline_seen = np.any(
        baseline_visibility.astype(bool),
        axis=1
    )

    baseline_counts = engine.compute_visibility_counts(
        baseline_visibility
    )

    visibility_weights = engine.compute_visibility_weights(
        baseline_counts
    )

    fx, fy = focal_lengths_from_fov(
        image_width=640,
        image_height=480,
        horizontal_fov_deg=60.0,
        vertical_fov_deg=45.0
    )

    baseline_information = compute_information_matrices(
        vertices=vertices,
        camera_poses=baseline_views,
        visibility_matrix=baseline_visibility,
        camera_basis=engine.camera_basis,
        fx=fx,
        fy=fy
    )

    baseline_ranks = calculate_ranks(
        baseline_information
    )

    baseline_score = regularized_logdet_score(
        baseline_information
    )

    print(
        "Baseline full-rank vertices:",
        int(np.sum(baseline_ranks == 3))
    )

    print(
        "Baseline regularized geometry score:",
        baseline_score
    )

    # ----------------------------------------------
    # C. DEFINE CANDIDATE EVALUATOR
    # ----------------------------------------------

    print("\n[3/4] Preparing candidate evaluator...")

    evaluation_counter = 0

    def evaluate_candidate(chromosome):

        nonlocal evaluation_counter

        evaluation_counter += 1

        # Convert five genes into a camera pose.
        candidate = CameraPose(
            position=np.asarray(
                chromosome[:3],
                dtype=float
            ),
            yaw=float(chromosome[3]),
            pitch=float(chromosome[4]),
            roll=0.0
        )

        # Calculate candidate visibility.
        candidate_visibility = (
            engine.compute_visibility_vector(
                camera_position=candidate.position,
                yaw=candidate.yaw,
                pitch=candidate.pitch
            ).astype(bool)
        )

        # Previously unseen vertices.
        newly_visible = (
            candidate_visibility & ~baseline_seen
        )

        # Paper-inspired visibility objective.
        visibility_fitness = (
            engine.compute_visibility_fitness(
                candidate_visibility.astype(float),
                visibility_weights
            )
        )

        # Calculate candidate projection Jacobian.
        candidate_jacobian = compute_projection_jacobian(
            vertices=vertices,
            pose=candidate,
            camera_basis=engine.camera_basis,
            fx=fx,
            fy=fy,
            visible_mask=candidate_visibility
        )

        # Candidate information contribution.
        candidate_information = np.einsum(
            "nki,nkj->nij",
            candidate_jacobian,
            candidate_jacobian
        )

        # Combine candidate information with baseline.
        combined_information = (
            baseline_information + candidate_information
        )

        combined_ranks = calculate_ranks(
            combined_information
        )

        combined_score = regularized_logdet_score(
            combined_information
        )

        improved = (
            combined_ranks > baseline_ranks
        )

        newly_full_rank = (
            (baseline_ranks < 3) &
            (combined_ranks == 3)
        )

        result = {
            "newly_visible_vertices": int(
                np.sum(newly_visible)
            ),

            "visibility_fitness": float(
                visibility_fitness
            ),

            "new_full_rank": int(
                np.sum(newly_full_rank)
            ),

            "score_gain": float(
                combined_score - baseline_score
            ),

            # Additional diagnostic measurements.
            "visible_vertices": int(
                np.sum(candidate_visibility)
            ),

            "improved_vertices": int(
                np.sum(improved)
            ),

            "rank_gain": int(
                np.sum(combined_ranks - baseline_ranks)
            ),

            "full_rank_total": int(
                np.sum(combined_ranks == 3)
            ),

            "position": candidate.position.copy(),
            "yaw": candidate.yaw,
            "pitch": candidate.pitch
        }

        if evaluation_counter % 5 == 0:

            print(
                f"Evaluated {evaluation_counter} unique poses..."
            )

        return result

    # ----------------------------------------------
    # D. RUN GENETIC ALGORITHM
    # ----------------------------------------------

    print("\n[4/4] Running genetic algorithm...")

    ga_result = run_genetic_algorithm(
        evaluate_fn=evaluate_candidate,

        population_size=12,
        generations=5,

        crossover_probability=0.90,
        mutation_probability=0.20,

        seed=42
    )

    # ----------------------------------------------
    # E. REPORT RESULTS
    # ----------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL GA RESULTS")
    print("=" * 75)

    print(
        "Unique poses evaluated:",
        ga_result["evaluations"]
    )

    print(
        "Generations completed:",
        ga_result["generations"]
    )

    final_front = ga_result["pareto_front"]

    print(
        "Final Pareto-optimal candidates:",
        len(final_front)
    )

    for index, candidate in enumerate(final_front):

        genes = candidate["genes"]

        print("\n" + "-" * 60)
        print(f"GA Candidate {index + 1}")

        print(
            "Position:",
            np.round(genes[:3], 3)
        )

        print(
            "Yaw (degrees):",
            round(np.rad2deg(genes[3]), 3)
        )

        print(
            "Pitch (degrees):",
            round(np.rad2deg(genes[4]), 3)
        )

        print(
            "Newly visible vertices:",
            candidate["newly_visible_vertices"]
        )

        print(
            "Visibility fitness:",
            round(candidate["visibility_fitness"], 4)
        )

        print(
            "New full-rank vertices:",
            candidate["new_full_rank"]
        )

        print(
            "Geometry score gain:",
            round(candidate["score_gain"], 4)
        )

        print(
            "Total full-rank vertices:",
            candidate["full_rank_total"]
        )

    print("\n" + "=" * 75)
    print("GA EXPERIMENT COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
