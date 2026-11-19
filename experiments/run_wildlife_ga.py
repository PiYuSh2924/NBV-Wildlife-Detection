"""
Wildlife-aware NBV experiment.

Compares:
  1) 36-camera baseline
  2) baseline + GA-optimized additional camera
  3) baseline + random additional camera (same evaluation budget)

Target boxes are simplified wildlife proxies. Metrics describe simulated
observability, not real detector accuracy.

Run from the repository root:
    python experiments/run_wildlife_ga.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from nbv_wildlife.environment import create_forest
from nbv_wildlife.viewpoints import CameraPose, create_initial_grid_views
from nbv_wildlife.visibility import VisibilityEngine
from nbv_wildlife.geometry import (
    focal_lengths_from_fov,
    compute_information_matrices,
    compute_projection_jacobian,
)
from nbv_wildlife.nbv import run_genetic_algorithm
from nbv_wildlife.targets import (
    generate_random_targets,
    compute_target_visibility_matrix,
    compute_target_visibility,
    summarize_target_visibility,
    evaluate_candidate_target_gain,
)


# -------------------- Experiment settings --------------------

FOREST_WIDTH = 30.0
FOREST_DEPTH = 30.0
TREE_COUNT = 45
FOREST_SEED = 42

BASELINE_GRID_SIZE = 6
BASELINE_HEIGHT = 16.0

TARGET_COUNT = 8
TARGET_SEED = 123
TARGET_SAMPLES_PER_AXIS = 5
OBSERVABLE_FRACTION_THRESHOLD = 0.25

GA_POPULATION = 12
GA_GENERATIONS = 5
GA_SEED = 42

# Same number of candidate evaluations as the GA's approximate
# evaluation budget; GA may cache duplicate chromosomes.
RANDOM_CONTROL_COUNT = GA_POPULATION * (GA_GENERATIONS + 1)
RANDOM_CONTROL_SEED = 2026

# Initial experimental weighting. These are hypotheses to test later
# through ablation/sensitivity analysis, not learned or validated weights.
WEIGHT_GEOMETRY = 0.35
WEIGHT_VISIBILITY = 0.25
WEIGHT_WILDLIFE = 0.40

# Fixed normalization references to keep objective scales comparable.
# The geometry information reference is intentionally explicit.
GEOMETRY_INFO_REFERENCE = 100000.0


def calculate_ranks(information):
    eigenvalues = np.linalg.eigvalsh(information)
    largest = np.maximum(eigenvalues[:, -1], 0.0)
    tolerance = np.maximum(largest * 1e-10, 1e-12)
    return np.sum(eigenvalues > tolerance[:, None], axis=1)


def regularized_logdet_score(information, regularization=1e-6):
    regularized = information + regularization * np.eye(3)
    signs, log_determinants = np.linalg.slogdet(regularized)
    if np.any(signs <= 0):
        raise ValueError("Non-positive determinant encountered.")
    return float(np.sum(log_determinants))


def weighted_score(
    newly_visible_vertices,
    new_full_rank,
    score_gain,
    visibility_fitness,
    total_vertices,
    visibility_weight_sum,
    newly_visible_target_samples,
    total_target_samples,
    new_targets_observable,
    target_count,
):
    """Return normalized components and weighted composite score."""

    # Geometry component: three normalized sub-metrics.
    geometry_visibility = (
        newly_visible_vertices / max(total_vertices, 1)
    )
    geometry_rank = new_full_rank / max(total_vertices, 1)
    geometry_information = (
        np.log1p(max(score_gain, 0.0))
        / np.log1p(GEOMETRY_INFO_REFERENCE)
    )
    geometry_information = float(np.clip(geometry_information, 0.0, 1.0))

    geometry_component = (
        0.50 * geometry_visibility
        + 0.25 * geometry_rank
        + 0.25 * geometry_information
    )

    visibility_component = (
        visibility_fitness / max(visibility_weight_sum, 1e-12)
    )
    visibility_component = float(
        np.clip(visibility_component, 0.0, 1.0)
    )

    target_sample_gain = (
        newly_visible_target_samples / max(total_target_samples, 1)
    )
    target_count_gain = (
        new_targets_observable / max(target_count, 1)
    )

    wildlife_component = (
        0.75 * target_sample_gain
        + 0.25 * target_count_gain
    )

    total = (
        WEIGHT_GEOMETRY * geometry_component
        + WEIGHT_VISIBILITY * visibility_component
        + WEIGHT_WILDLIFE * wildlife_component
    )

    return {
        "geometry_component": float(geometry_component),
        "geometry_visibility_component": float(geometry_visibility),
        "geometry_rank_component": float(geometry_rank),
        "geometry_information_component": float(geometry_information),
        "visibility_component": float(visibility_component),
        "target_sample_gain_component": float(target_sample_gain),
        "target_count_gain_component": float(target_count_gain),
        "wildlife_component": float(wildlife_component),
        "weighted_score": float(total),
    }


def main():
    print("\n" + "=" * 80)
    print("WILDLIFE-AWARE NBV EXPERIMENT")
    print("=" * 80)

    # 1) Environment
    print("\n[1/7] Creating forest...")
    forest = create_forest(
        width=FOREST_WIDTH,
        depth=FOREST_DEPTH,
        num_trees=TREE_COUNT,
        seed=FOREST_SEED,
    )
    vertices = np.asarray(forest.combined_mesh.vertices)
    print("Forest vertices:", len(vertices))

    # 2) Baseline views and visibility engine
    print("\n[2/7] Creating baseline cameras and visibility engine...")
    baseline_views = create_initial_grid_views(
        width=FOREST_WIDTH,
        depth=FOREST_DEPTH,
        grid_size=BASELINE_GRID_SIZE,
        height=BASELINE_HEIGHT,
    )
    print("Baseline cameras:", len(baseline_views))

    engine = VisibilityEngine(
        forest.combined_mesh,
        horizontal_fov_deg=60.0,
        vertical_fov_deg=45.0,
    )

    print("\nComputing forest baseline visibility...")
    baseline_visibility = engine.compute_visibility_matrix(
        baseline_views
    ).astype(bool)

    baseline_seen = np.any(baseline_visibility, axis=1)
    baseline_counts = engine.compute_visibility_counts(
        baseline_visibility.astype(np.uint8)
    )
    visibility_weights = engine.compute_visibility_weights(
        baseline_counts
    )

    fx, fy = focal_lengths_from_fov(
        image_width=640,
        image_height=480,
        horizontal_fov_deg=60.0,
        vertical_fov_deg=45.0,
    )

    print("\nComputing baseline geometry information...")
    baseline_information = compute_information_matrices(
        vertices=vertices,
        camera_poses=baseline_views,
        visibility_matrix=baseline_visibility,
        camera_basis=engine.camera_basis,
        fx=fx,
        fy=fy,
    )
    baseline_ranks = calculate_ranks(baseline_information)
    baseline_geometry_score = regularized_logdet_score(
        baseline_information
    )

    # 3) Targets and baseline target visibility
    print("\n[3/7] Generating simulated wildlife targets...")
    targets = generate_random_targets(
        width=FOREST_WIDTH,
        depth=FOREST_DEPTH,
        num_targets=TARGET_COUNT,
        seed=TARGET_SEED,
        samples_per_axis=TARGET_SAMPLES_PER_AXIS,
    )
    total_target_samples = sum(len(t.surface_points) for t in targets)
    print(f"Targets: {len(targets)}")
    print(f"Surface samples per target: {[len(t.surface_points) for t in targets]}")
    print("Total target surface samples:", total_target_samples)

    print("\nComputing target visibility from baseline cameras...")
    baseline_target_visibility = compute_target_visibility_matrix(
        engine=engine,
        targets=targets,
        camera_poses=baseline_views,
    ).astype(bool)

    baseline_target_summary = summarize_target_visibility(
        baseline_target_visibility,
        targets,
        observable_fraction_threshold=OBSERVABLE_FRACTION_THRESHOLD,
    )

    print("\nBaseline target observability:")
    print(
        "  Surface samples visible in baseline union:",
        baseline_target_summary["visible_surface_samples_union"],
        "/",
        baseline_target_summary["total_surface_samples"],
    )
    print(
        "  Union visible fraction:",
        f"{baseline_target_summary['overall_union_visible_fraction']:.4f}",
    )
    print(
        "  Targets with any visible surface:",
        baseline_target_summary["targets_with_any_visible_surface"],
        "/",
        TARGET_COUNT,
    )
    print(
        f"  Targets observable at threshold "
        f"({OBSERVABLE_FRACTION_THRESHOLD:.0%}):",
        baseline_target_summary["targets_observable_at_threshold"],
        "/",
        TARGET_COUNT,
    )

    # 4) Candidate evaluator
    print("\n[4/7] Preparing candidate evaluator...")
    evaluation_counter = 0

    def evaluate_candidate(chromosome):
        nonlocal evaluation_counter
        evaluation_counter += 1

        pose = CameraPose(
            position=np.asarray(chromosome[:3], dtype=float),
            yaw=float(chromosome[3]),
            pitch=float(chromosome[4]),
            roll=0.0,
        )

        # Forest vertex visibility.
        candidate_visibility = engine.compute_visibility_vector(
            camera_position=pose.position,
            yaw=pose.yaw,
            pitch=pose.pitch,
        ).astype(bool)

        newly_visible = candidate_visibility & ~baseline_seen

        visibility_fitness = engine.compute_visibility_fitness(
            candidate_visibility.astype(float),
            visibility_weights,
        )

        # Candidate geometric information.
        jacobian = compute_projection_jacobian(
            vertices=vertices,
            pose=pose,
            camera_basis=engine.camera_basis,
            fx=fx,
            fy=fy,
            visible_mask=candidate_visibility,
        )
        candidate_information = np.einsum(
            "nki,nkj->nij",
            jacobian,
            jacobian,
        )

        combined_information = baseline_information + candidate_information
        combined_ranks = calculate_ranks(combined_information)
        combined_geometry_score = regularized_logdet_score(
            combined_information
        )

        newly_full_rank = (
            (baseline_ranks < 3) & (combined_ranks == 3)
        )

        # Target visibility for this pose.
        candidate_target_visibility = compute_target_visibility(
            engine=engine,
            targets=targets,
            camera_pose=pose,
        )

        target_gain = evaluate_candidate_target_gain(
            baseline_visibility_matrix=baseline_target_visibility,
            candidate_visibility=candidate_target_visibility,
            targets=targets,
            observable_fraction_threshold=OBSERVABLE_FRACTION_THRESHOLD,
        )

        components = weighted_score(
            newly_visible_vertices=int(np.sum(newly_visible)),
            new_full_rank=int(np.sum(newly_full_rank)),
            score_gain=combined_geometry_score - baseline_geometry_score,
            visibility_fitness=float(visibility_fitness),
            total_vertices=len(vertices),
            visibility_weight_sum=float(np.sum(visibility_weights)),
            newly_visible_target_samples=target_gain[
                "newly_visible_target_samples"
            ],
            total_target_samples=total_target_samples,
            new_targets_observable=target_gain["new_targets_observable"],
            target_count=TARGET_COUNT,
        )

        result = {
            # Existing geometry objectives
            "newly_visible_vertices": int(np.sum(newly_visible)),
            "visibility_fitness": float(visibility_fitness),
            "new_full_rank": int(np.sum(newly_full_rank)),
            "score_gain": float(
                combined_geometry_score - baseline_geometry_score
            ),

            # Wildlife objectives
            "newly_visible_target_samples": int(
                target_gain["newly_visible_target_samples"]
            ),
            "target_visibility_gain": float(
                target_gain["target_visibility_gain"]
            ),
            "new_targets_observable": int(
                target_gain["new_targets_observable"]
            ),

            # Weighted composite
            **components,

            # Additional diagnostics
            "visible_vertices": int(np.sum(candidate_visibility)),
            "rank_gain": int(np.sum(combined_ranks - baseline_ranks)),
            "full_rank_total": int(np.sum(combined_ranks == 3)),
            "candidate_target_visible_samples": int(
                target_gain["candidate_visible_target_samples"]
            ),
            "candidate_target_visibility_fraction": float(
                target_gain["candidate_target_visibility_fraction"]
            ),
            "combined_target_union_visibility_fraction": float(
                target_gain["combined_union_visible_fraction"]
            ),
            "position": pose.position.copy(),
            "yaw": pose.yaw,
            "pitch": pose.pitch,
        }

        if evaluation_counter % 5 == 0:
            print(f"Evaluated {evaluation_counter} candidate poses...")

        return result

    # 5) Run GA
    print("\n[5/7] Running wildlife-aware genetic algorithm...")
    objectives = (
        "newly_visible_vertices",
        "visibility_fitness",
        "new_full_rank",
        "score_gain",
        "newly_visible_target_samples",
        "new_targets_observable",
        "weighted_score",
    )

    ga_result = run_genetic_algorithm(
        evaluate_fn=evaluate_candidate,
        population_size=GA_POPULATION,
        generations=GA_GENERATIONS,
        crossover_probability=0.90,
        mutation_probability=0.20,
        seed=GA_SEED,
        objectives=objectives,
    )

    pareto_candidates = ga_result["pareto_front"]
    if not pareto_candidates:
        raise RuntimeError("GA returned an empty Pareto front")

    best_ga = max(
        pareto_candidates,
        key=lambda c: c["weighted_score"],
    )

    # 6) Random control with a matching fixed evaluation budget.
    print("\n[6/7] Evaluating random-view control...")
    random_rng = np.random.default_rng(RANDOM_CONTROL_SEED)
    random_results = []

    # Uniform random control over the same feasible chromosome bounds:
    # x/y in [-25, 25], z in [10, 25], yaw in [-pi, pi],
    # pitch in [-pi/2, 0]. Unlike the GA initializer, this control
    # does not deliberately aim toward the forest center.
    from nbv_wildlife.nbv import LOWER_BOUNDS, UPPER_BOUNDS

    for index in range(RANDOM_CONTROL_COUNT):
        genes = random_rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS)
        result = evaluate_candidate(genes)
        result["control_index"] = index + 1
        result["genes"] = np.asarray(genes, dtype=float)
        random_results.append(result)

        if (index + 1) % 5 == 0:
            print(
                f"Random control evaluated "
                f"{index + 1}/{RANDOM_CONTROL_COUNT}"
            )

    best_random = max(
        random_results,
        key=lambda c: c["weighted_score"],
    )

    # 7) Report and save.
    print("\n[7/7] Reporting and saving results...")

    def print_result(label, result):
        print("\n" + "-" * 72)
        print(label)
        print("-" * 72)
        print("Position:", np.round(result["position"], 3))
        print("Yaw (deg):", round(np.rad2deg(result["yaw"]), 3))
        print("Pitch (deg):", round(np.rad2deg(result["pitch"]), 3))
        print("Weighted score:", round(result["weighted_score"], 6))
        print("Geometry component:", round(result["geometry_component"], 6))
        print("Visibility component:", round(result["visibility_component"], 6))
        print("Wildlife component:", round(result["wildlife_component"], 6))
        print("New forest vertices:", result["newly_visible_vertices"])
        print("Geometry score gain:", round(result["score_gain"], 4))
        print("New target surface samples:", result["newly_visible_target_samples"])
        print("New targets observable:", result["new_targets_observable"])
        print(
            "Candidate target surface visibility:",
            f"{result['candidate_target_visibility_fraction']:.4f}",
        )
        print(
            "Combined baseline + candidate target visibility:",
            f"{result['combined_target_union_visibility_fraction']:.4f}",
        )

    print_result("BEST GA CANDIDATE (weighted score among Pareto set)", best_ga)
    print_result("BEST RANDOM CONTROL (same approximate evaluation budget)", best_random)

    ga_score = float(best_ga["weighted_score"])
    random_score = float(best_random["weighted_score"])
    score_difference = ga_score - random_score
    relative_difference = (
        100.0 * score_difference / abs(random_score)
        if abs(random_score) > 1e-12 else float("nan")
    )

    print("\n" + "=" * 72)
    print("GA vs RANDOM CONTROL")
    print("=" * 72)
    print("GA weighted score:", ga_score)
    print("Random weighted score:", random_score)
    print("Absolute difference:", score_difference)
    print("Relative difference (%):", relative_difference)
    print(
        "\nInterpretation: this is a comparison of the experimental "
        "weighted observability/geometry proxy, not animal detector accuracy."
    )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    def json_safe(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        return value

    summary = {
        "experiment": "wildlife_aware_nbv",
        "seeds": {
            "forest": FOREST_SEED,
            "targets": TARGET_SEED,
            "ga": GA_SEED,
            "random_control": RANDOM_CONTROL_SEED,
        },
        "settings": {
            "forest_width": FOREST_WIDTH,
            "forest_depth": FOREST_DEPTH,
            "tree_count": TREE_COUNT,
            "baseline_cameras": len(baseline_views),
            "target_count": TARGET_COUNT,
            "target_samples_per_axis": TARGET_SAMPLES_PER_AXIS,
            "observable_fraction_threshold": OBSERVABLE_FRACTION_THRESHOLD,
            "weights": {
                "geometry": WEIGHT_GEOMETRY,
                "visibility": WEIGHT_VISIBILITY,
                "wildlife": WEIGHT_WILDLIFE,
            },
            "random_control_count": RANDOM_CONTROL_COUNT,
        },
        "baseline_target_summary": baseline_target_summary,
        "best_ga": json_safe(best_ga),
        "best_random": json_safe(best_random),
        "comparison": {
            "ga_weighted_score": ga_score,
            "random_weighted_score": random_score,
            "difference": score_difference,
            "relative_difference_percent": relative_difference,
        },
        "pareto_candidates": json_safe(pareto_candidates),
        "random_candidates": json_safe(random_results),
    }

    json_path = output_dir / "wildlife_ga_seed42.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = output_dir / "wildlife_ga_candidates_seed42.csv"
    fieldnames = [
        "source", "candidate_index", "weighted_score",
        "geometry_component", "visibility_component",
        "wildlife_component", "newly_visible_vertices",
        "visibility_fitness", "new_full_rank", "score_gain",
        "newly_visible_target_samples", "target_visibility_gain",
        "new_targets_observable", "candidate_target_visibility_fraction",
        "combined_target_union_visibility_fraction",
        "x", "y", "z", "yaw_deg", "pitch_deg",
    ]

    rows = []
    for source, candidates in (
        ("GA_PARETO", pareto_candidates),
        ("RANDOM_CONTROL", random_results),
    ):
        for index, c in enumerate(candidates, start=1):
            position = np.asarray(c["position"])
            rows.append({
                "source": source,
                "candidate_index": index,
                "weighted_score": c["weighted_score"],
                "geometry_component": c["geometry_component"],
                "visibility_component": c["visibility_component"],
                "wildlife_component": c["wildlife_component"],
                "newly_visible_vertices": c["newly_visible_vertices"],
                "visibility_fitness": c["visibility_fitness"],
                "new_full_rank": c["new_full_rank"],
                "score_gain": c["score_gain"],
                "newly_visible_target_samples": c["newly_visible_target_samples"],
                "target_visibility_gain": c["target_visibility_gain"],
                "new_targets_observable": c["new_targets_observable"],
                "candidate_target_visibility_fraction": c[
                    "candidate_target_visibility_fraction"
                ],
                "combined_target_union_visibility_fraction": c[
                    "combined_target_union_visibility_fraction"
                ],
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "yaw_deg": np.rad2deg(c["yaw"]),
                "pitch_deg": np.rad2deg(c["pitch"]),
            })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:")
    print(" ", json_path)
    print(" ", csv_path)
    print("\nExperiment completed.")


if __name__ == "__main__":
    main()
