
import numpy as np

from nbv_wildlife.environment import create_forest
from nbv_wildlife.viewpoints import create_initial_grid_views
from nbv_wildlife.visibility import VisibilityEngine

from nbv_wildlife.geometry import (
    focal_lengths_from_fov,
    compute_information_matrices,
    evaluate_geometry_fitness
)


def main():

    print("\n" + "=" * 60)
    print("GEOMETRY FITNESS BASELINE EXPERIMENT")
    print("=" * 60)

    # -------------------------------------------------
    # 1. Generate forest
    # -------------------------------------------------

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

    print("Forest vertices:", len(vertices))

    # -------------------------------------------------
    # 2. Generate initial camera viewpoints
    # -------------------------------------------------

    print("\n[2/5] Creating initial viewpoints...")

    viewpoints = create_initial_grid_views(
        width=30.0,
        depth=30.0,
        grid_size=6,
        height=16.0
    )

    print("Number of cameras:", len(viewpoints))

    # -------------------------------------------------
    # 3. Calculate visibility
    # -------------------------------------------------

    print("\n[3/5] Calculating visibility...")

    engine = VisibilityEngine(
        forest.combined_mesh,
        horizontal_fov_deg=60.0,
        vertical_fov_deg=45.0
    )

    visibility_matrix = engine.compute_visibility_matrix(
        viewpoints
    )

    print("\nVisibility matrix:", visibility_matrix.shape)

    # -------------------------------------------------
    # 4. Calculate camera intrinsics
    # -------------------------------------------------

    print("\n[4/5] Calculating camera intrinsics...")

    fx, fy = focal_lengths_from_fov(
        image_width=640,
        image_height=480,
        horizontal_fov_deg=60.0,
        vertical_fov_deg=45.0
    )

    print("Focal length fx:", round(fx, 3))
    print("Focal length fy:", round(fy, 3))

    # -------------------------------------------------
    # 5. Calculate geometric information
    # -------------------------------------------------

    print("\n[5/5] Calculating information matrices...")

    information = compute_information_matrices(
        vertices=vertices,
        camera_poses=viewpoints,
        visibility_matrix=visibility_matrix,
        camera_basis=engine.camera_basis,
        fx=fx,
        fy=fy
    )

    results = evaluate_geometry_fitness(
        information
    )

    # -------------------------------------------------
    # Report results
    # -------------------------------------------------

    ranks = results["vertex_ranks"]

    rank_counts = np.bincount(
        ranks,
        minlength=4
    )

    print("\n" + "=" * 60)
    print("GEOMETRY FITNESS RESULTS")
    print("=" * 60)

    print("Total vertices:", results["total_vertices"])

    print(
        "Rank 0 vertices:",
        rank_counts[0]
    )

    print(
        "Rank 1 vertices:",
        rank_counts[1]
    )

    print(
        "Rank 2 vertices:",
        rank_counts[2]
    )

    print(
        "Rank 3 vertices:",
        rank_counts[3]
    )

    print(
        "Fully constrained vertices:",
        results["full_rank_vertices"]
    )

    print(
        "Full-rank fraction:",
        round(results["full_rank_fraction"] * 100, 3),
        "%"
    )

    print(
        "Paper geometry fitness:",
        results["paper_fitness"]
    )

    full_rank_mask = ranks == 3

    if np.any(full_rank_mask):

        valid_logdet = results["log_determinants"][
            full_rank_mask
        ]

        print(
            "Mean log determinant (full-rank vertices):",
            np.mean(valid_logdet)
        )

        print(
            "Median log determinant (full-rank vertices):",
            np.median(valid_logdet)
        )

    else:
        print(
            "No full-rank vertices were found."
        )

    print("\nExperiment completed.")


if __name__ == "__main__":
    main()
