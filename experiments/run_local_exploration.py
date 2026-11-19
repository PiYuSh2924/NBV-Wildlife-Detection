from __future__ import annotations

import argparse
import numpy as np

from nbv_wildlife.environment import create_forest
from nbv_wildlife.visibility import VisibilityEngine
from nbv_wildlife.viewpoints import CameraPose
from nbv_wildlife.exploration import ExplorationState, LocalAdaptiveNBVPlanner


def run_exploration(
    seed=42,
    steps=20,
    width=30.0,
    depth=30.0,
    num_trees=45,
    altitude=16.0,
    step_size=5.0,
):
    print("Creating procedural forest...")
    forest = create_forest(
        width=width,
        depth=depth,
        num_trees=num_trees,
        seed=seed,
    )
    engine = VisibilityEngine(forest.combined_mesh)

    # Begin near a forest corner. Keep the altitude fixed in this first study.
    start = np.array([-width / 2 + 1.0, -depth / 2 + 1.0, altitude])
    initial_pose = CameraPose(
        position=start,
        yaw=np.pi / 4,
        pitch=-np.pi / 3,
    )
    state = ExplorationState.create(start, len(forest.combined_mesh.vertices))

    print("Computing initial observation...")
    initial_visibility = engine.compute_visibility_vector(
        initial_pose.position, initial_pose.yaw, initial_pose.pitch
    ).astype(bool)
    state.update(initial_pose, initial_visibility, score=0.0)

    planner = LocalAdaptiveNBVPlanner(
        visibility_engine=engine,
        step_size=step_size,
        bounds=(-width / 2, width / 2, -depth / 2, depth / 2),
        connectivity=8,
        shortlist_size=8,
        max_expansion=3,
        novelty_weight=1.0,
        travel_cost_weight=0.10,
        rotation_cost_weight=0.03,
        revisit_penalty=0.25,
    )

    print("\n--- Local adaptive NBV exploration ---")
    print(f"Start position: {start.tolist()}")
    print(f"Initial observed vertices: {np.sum(state.observed_vertices)}")

    current_yaw = initial_pose.yaw
    for _ in range(steps):
        result = planner.step(state, current_yaw=current_yaw)
        if result is None:
            print("No candidate with additional FOV novelty; stopping.")
            break

        current_yaw = result["pose"].yaw
        pos = result["pose"].position
        print(
            f"Step {result['step']:02d}: pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) "
            f"gain={result['newly_observed']:4d} total={result['total_observed']:5d} "
            f"score={result['score']:.2f} travel={result['travel_distance']:.2f}"
        )

    total = len(state.observed_vertices)
    observed = int(np.sum(state.observed_vertices))
    print("\n--- Summary ---")
    print(f"Observations (including start): {len(state.history)}")
    print(f"Observed mesh vertices: {observed}/{total} ({observed / max(total, 1):.2%})")
    print(f"Visited positions: {len(state.visited_positions)}")
    return forest, engine, state


def main():
    parser = argparse.ArgumentParser(description="Run local adaptive NBV exploration simulation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=5.0)
    args = parser.parse_args()
    run_exploration(seed=args.seed, steps=args.steps, step_size=args.step_size)


if __name__ == "__main__":
    main()
