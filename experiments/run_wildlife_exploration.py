from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from nbv_wildlife.environment import create_forest
from nbv_wildlife.visibility import VisibilityEngine
from nbv_wildlife.viewpoints import CameraPose
from nbv_wildlife.targets import generate_random_targets, compute_target_visibility, summarize_target_visibility
from nbv_wildlife.wildlife_exploration import WildlifeExplorationState, WildlifeAwareLocalNBVPlanner


def run_wildlife_exploration(seed=42, target_seed=123, steps=20, width=30.0, depth=30.0,
                             num_trees=45, num_targets=8, altitude=16.0, step_size=5.0,
                             connectivity=8, geometry_weight=0.45, wildlife_weight=0.55,
                             threshold=0.25, output_dir="outputs/wildlife_exploration"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("Generating forest and wildlife proxy targets...")
    forest = create_forest(width=width, depth=depth, num_trees=num_trees, seed=seed)
    targets = generate_random_targets(width=width, depth=depth, num_targets=num_targets, seed=target_seed)
    engine = VisibilityEngine(forest.combined_mesh)
    start = np.array([-width/2+1.0, -depth/2+1.0, altitude], dtype=float)
    initial_pose = CameraPose(start, yaw=np.pi/4, pitch=-np.pi/3)
    state = WildlifeExplorationState.create(start, len(forest.combined_mesh.vertices),
                                           sum(len(t.surface_points) for t in targets))
    print("Casting initial camera rays...")
    gv0 = engine.compute_visibility_vector(initial_pose.position, initial_pose.yaw, initial_pose.pitch).astype(bool)
    tv0 = compute_target_visibility(engine, targets, initial_pose)
    initial_metrics = summarize_target_visibility(tv0, targets, threshold)
    state.update(initial_pose, gv0, tv0, 0.0, {
        "geometry_gain_normalized": 0.0, "target_gain_normalized": 0.0,
        "newly_observable_targets": initial_metrics["targets_observable_at_threshold"],
        "observable_targets_total": initial_metrics["targets_observable_at_threshold"],
        "target_union_fraction": initial_metrics["overall_visible_fraction"],
        "travel_distance": 0.0, "rotation_radians": 0.0,
        "revisited_position": False, "expansion_step": 0.0,
    })
    planner = WildlifeAwareLocalNBVPlanner(
        engine, targets, step_size=step_size,
        bounds=(-width/2,width/2,-depth/2,depth/2),
        connectivity=connectivity, geometry_weight=geometry_weight,
        wildlife_weight=wildlife_weight, observable_fraction_threshold=threshold,
    )
    yaw = initial_pose.yaw
    print(f"Start at {start.tolist()} | initial geometry={np.sum(state.observed_vertices)} "
          f"| wildlife samples={np.sum(state.observed_target_samples)}/{len(state.observed_target_samples)}")
    for _ in range(steps):
        result=planner.step(state,current_yaw=yaw)
        if result is None:
            print("No locally reachable candidate with additional proxy visibility; stopping.")
            break
        yaw=result["pose"].yaw
        p=result["pose"].position
        print(f"Step {result['step']:02d} | ({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}) "
              f"| +geom {result['new_vertices']:4d} | +target {result['new_target_samples']:3d} "
              f"| targets threshold +{result['metrics']['newly_observable_targets']} "
              f"| score {result['score']:.5f}")
    final_summary=summarize_target_visibility(state.observed_target_samples,targets,threshold)
    geom_total=len(state.observed_vertices)
    geom_observed=int(np.sum(state.observed_vertices))
    print("\n=== Wildlife-aware sequential NBV summary ===")
    print(f"Camera observations (including start): {len(state.history)}")
    print(f"Geometry coverage: {geom_observed}/{geom_total} ({geom_observed/max(geom_total,1):.2%})")
    print(f"Target surface observability: {final_summary['visible_surface_samples']}/{final_summary['total_surface_samples']} "
          f"({final_summary['overall_visible_fraction']:.2%})")
    print(f"Targets observable at threshold {threshold:.0%}: "
          f"{final_summary['targets_observable_at_threshold']}/{num_targets}")
    print(f"Visited positions: {len(state.visited_positions)}")
    # Save reproducible machine-readable results.
    serial_history=[]
    for h in state.history:
        d=dict(h)
        d["position"]=np.asarray(d["position"]).tolist()
        serial_history.append(d)
    payload={"experiment":"wildlife-aware local sequential NBV (proxy target observability)",
             "parameters":{"forest_seed":seed,"target_seed":target_seed,"steps_requested":steps,
                          "width":width,"depth":depth,"num_trees":num_trees,"num_targets":num_targets,
                          "altitude":altitude,"step_size":step_size,"connectivity":connectivity,
                          "geometry_weight":geometry_weight,"wildlife_weight":wildlife_weight,
                          "observable_fraction_threshold":threshold},
             "geometry":{"observed_vertices":geom_observed,"total_vertices":geom_total,
                        "coverage_fraction":geom_observed/max(geom_total,1)},
             "wildlife_proxy":final_summary,"history":serial_history,
             "targets":[{"target_id":t.target_id,"center":t.center.tolist(),
                         "dimensions":t.dimensions.tolist(),"samples":len(t.surface_points)} for t in targets]}
    json_path=out/"run_summary.json"
    json_path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    csv_path=out/"step_history.csv"
    if state.history:
        keys=[k for k in state.history[0] if k!="position"]
        with csv_path.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=keys+["x","y","z"])
            writer.writeheader()
            for h in state.history:
                row={k:h.get(k) for k in keys}
                row.update({"x":h["position"][0],"y":h["position"][1],"z":h["position"][2]})
                writer.writerow(row)
    print(f"Saved summary: {json_path}")
    print(f"Saved step log: {csv_path}")
    return forest,engine,targets,state,final_summary


def main():
    parser=argparse.ArgumentParser(description="Run sequential wildlife-aware local NBV simulation.")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--target-seed",type=int,default=123)
    parser.add_argument("--steps",type=int,default=20)
    parser.add_argument("--step-size",type=float,default=5.0)
    parser.add_argument("--targets",type=int,default=8)
    parser.add_argument("--connectivity",type=int,choices=(4,8),default=8)
    parser.add_argument("--geometry-weight",type=float,default=0.45)
    parser.add_argument("--wildlife-weight",type=float,default=0.55)
    parser.add_argument("--threshold",type=float,default=0.25)
    parser.add_argument("--output",default="outputs/wildlife_exploration")
    args=parser.parse_args()
    run_wildlife_exploration(seed=args.seed,target_seed=args.target_seed,steps=args.steps,
        step_size=args.step_size,num_targets=args.targets,connectivity=args.connectivity,
        geometry_weight=args.geometry_weight,wildlife_weight=args.wildlife_weight,
        threshold=args.threshold,output_dir=args.output)


if __name__=="__main__":
    main()
