
import numpy as np

from nbv_wildlife.environment import create_forest
from nbv_wildlife.viewpoints import CameraPose, create_initial_grid_views
from nbv_wildlife.visibility import VisibilityEngine
from nbv_wildlife.nbv import pareto_front

from nbv_wildlife.geometry import (
    focal_lengths_from_fov,
    compute_information_matrices,
    compute_projection_jacobian
)


def calculate_ranks(information):
    """Calculate the numerical rank of each 3x3 information matrix."""

    eigenvalues = np.linalg.eigvalsh(information)

    largest = np.maximum(eigenvalues[:, -1], 0.0)

    tolerance = np.maximum(
        largest * 1e-10,
        1e-12
    )

    return np.sum(
        eigenvalues > tolerance[:, None],
        axis=1
    )


def regularized_logdet_score(information, regularization=1e-6):
    """
    Regularized D-optimality comparison.

    This is a research adaptation, not the paper's
    literal product-of-determinants objective.
    """

    identity = np.eye(3)

    regularized = (
        information + regularization * identity
    )

    signs, log_determinants = np.linalg.slogdet(
        regularized
    )

    if np.any(signs <= 0):
        raise ValueError(
            "Non-positive determinant encountered."
        )

    return float(np.sum(log_determinants))


def create_candidate_views(radius=18.0, height=16.0):
    """
    Create eight oblique viewpoints around the forest.

    Each camera looks toward the central forest region.
    These are our experimental candidate poses.
    """

    candidates = []

    target = np.array([0.0, 0.0, 6.0])

    for angle_deg in range(0, 360, 45):

        angle = np.deg2rad(angle_deg)

        position = np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            height
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

        candidates.append(
            CameraPose(
                position=position,
                yaw=yaw,
                pitch=pitch,
                roll=0.0
            )
        )

    return candidates


def main():

    print("\n" + "=" * 65)
    print("NEXT-BEST-VIEW: GEOMETRY CANDIDATE EVALUATION")
    print("=" * 65)

    # 1. Create the forest.
    print("\n[1/5] Creating forest...")

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

    # 2. Create original viewpoints.
    print("\n[2/5] Creating baseline viewpoints...")

    baseline_views = create_initial_grid_views(
        width=30.0,
        depth=30.0,
        grid_size=6,
        height=16.0
    )

    print("Baseline cameras:", len(baseline_views))

    candidates = create_candidate_views()

    print("Candidate cameras:", len(candidates))

    # 3. Calculate baseline visibility and information.
    print("\n[3/5] Calculating baseline information...")

    engine = VisibilityEngine(
        forest.combined_mesh,
        horizontal_fov_deg=60.0,
        vertical_fov_deg=45.0
    )

    baseline_visibility = engine.compute_visibility_matrix(
        baseline_views
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

    # Identify vertices already observed by at least one baseline camera.
    baseline_seen = np.any(
        baseline_visibility.astype(bool),
        axis=1
    )

    # Calculate visibility weights using the paper's equation.
    baseline_counts = engine.compute_visibility_counts(
        baseline_visibility
    )

    visibility_weights = engine.compute_visibility_weights(
        baseline_counts
    )

    baseline_score = regularized_logdet_score(
        baseline_information
    )

    print("Baseline full-rank vertices:", np.sum(baseline_ranks == 3))
    print("Baseline regularized score:", baseline_score)

    # 4. Evaluate every candidate.
    print("\n[4/5] Evaluating candidate viewpoints...")

    results = []

    for index, candidate in enumerate(candidates):

        print(
            f"\nEvaluating candidate {index + 1}/{len(candidates)}"
        )

        candidate_visibility = engine.compute_visibility_vector(
            camera_position=candidate.position,
            yaw=candidate.yaw,
            pitch=candidate.pitch
        ).astype(bool)

                # Count vertices that this candidate reveals for the first time.
        newly_visible = (
            candidate_visibility & ~baseline_seen
        )

        newly_visible_count = int(
            np.sum(newly_visible)
        )

        # Calculate visibility-based fitness.
        visibility_fitness = engine.compute_visibility_fitness(
            candidate_visibility.astype(float),
            visibility_weights
        )

        candidate_jacobian = compute_projection_jacobian(
            vertices=vertices,
            pose=candidate,
            camera_basis=engine.camera_basis,
            fx=fx,
            fy=fy,
            visible_mask=candidate_visibility
        )

        candidate_information = np.einsum(
            "nki,nkj->nij",
            candidate_jacobian,
            candidate_jacobian
        )

        combined_information = (
            baseline_information + candidate_information
        )

        combined_ranks = calculate_ranks(
            combined_information
        )

        combined_score = regularized_logdet_score(
            combined_information
        )

        improved = combined_ranks > baseline_ranks

        newly_full_rank = (
            (baseline_ranks < 3) &
            (combined_ranks == 3)
        )

        result = {
            "candidate": index + 1,
            "position": candidate.position,
            "yaw": candidate.yaw,
            "pitch": candidate.pitch,

            "visible_vertices": int(np.sum(candidate_visibility)),
            "newly_visible_vertices": newly_visible_count,

            "visibility_fitness": visibility_fitness,

            "improved_vertices": int(np.sum(improved)),
            "rank_gain": int(np.sum(combined_ranks - baseline_ranks)),
            "new_full_rank": int(np.sum(newly_full_rank)),
            "full_rank_total": int(np.sum(combined_ranks == 3)),

            "score_gain": combined_score - baseline_score
        }

        results.append(result)

        # 5. Report candidate comparison.
    print("\n[5/5] Candidate comparison")

    print("\n" + "=" * 125)

    print(
        f"{'ID':<5}"
        f"{'Visible':<12}"
        f"{'New Visible':<14}"
        f"{'Jv':<15}"
        f"{'Improved':<12}"
        f"{'Rank Gain':<12}"
        f"{'New Rank-3':<14}"
        f"{'Total Rank-3':<15}"
        f"{'Score Gain':<15}"
    )

    print("=" * 125)

    for result in results:

        print(
            f"{result['candidate']:<5}"
            f"{result['visible_vertices']:<12}"
            f"{result['newly_visible_vertices']:<14}"
            f"{result['visibility_fitness']:<15.4f}"
            f"{result['improved_vertices']:<12}"
            f"{result['rank_gain']:<12}"
            f"{result['new_full_rank']:<14}"
            f"{result['full_rank_total']:<15}"
            f"{result['score_gain']:<15.4f}"
        )

    # 6. Identify Pareto-optimal candidates.
    print("\n" + "=" * 75)
    print("MULTI-OBJECTIVE NBV: PARETO-OPTIMAL CANDIDATES")
    print("=" * 75)

    pareto_candidates = pareto_front(results)

    for candidate in pareto_candidates:

        print(
            f"\nCandidate {candidate['candidate']}"
        )

        print(
            "Position:",
            candidate["position"]
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

    print("\n" + "=" * 75)
    print("PARETO SELECTION COMPLETED")
    print("=" * 75)

    print(
        "Number of evaluated candidates:",
        len(results)
    )

    print(
        "Number of Pareto-optimal candidates:",
        len(pareto_candidates)
    )

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()