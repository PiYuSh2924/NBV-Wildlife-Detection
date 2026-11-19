from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from nbv_wildlife.viewpoints import CameraPose, poses_at_neighbor_positions
from nbv_wildlife.targets import compute_target_visibility, summarize_target_visibility


@dataclass
class WildlifeExplorationState:
    current_position: np.ndarray
    observed_vertices: np.ndarray
    observed_target_samples: np.ndarray
    visited_positions: set = field(default_factory=set)
    history: list = field(default_factory=list)
    step_index: int = 0

    @classmethod
    def create(cls, start_position, num_vertices, num_target_samples):
        start = np.asarray(start_position, dtype=float)
        state = cls(start.copy(), np.zeros(num_vertices, dtype=bool),
                    np.zeros(num_target_samples, dtype=bool))
        state.visited_positions.add(cls.position_key(start))
        return state

    @staticmethod
    def position_key(position, decimals=2):
        return tuple(np.round(np.asarray(position, dtype=float), decimals))

    def update(self, pose, geometry_visibility, target_visibility, score, metrics):
        gv = np.asarray(geometry_visibility, dtype=bool)
        tv = np.asarray(target_visibility, dtype=bool)
        if gv.shape != self.observed_vertices.shape or tv.shape != self.observed_target_samples.shape:
            raise ValueError("Observation vector shape does not match exploration state.")
        new_g = gv & ~self.observed_vertices
        new_t = tv & ~self.observed_target_samples
        self.observed_vertices |= gv
        self.observed_target_samples |= tv
        self.current_position = np.asarray(pose.position, dtype=float).copy()
        self.visited_positions.add(self.position_key(self.current_position))
        row = {
            "step": self.step_index, "position": self.current_position.copy(),
            "yaw": float(pose.yaw), "pitch": float(pose.pitch),
            "new_vertices": int(np.sum(new_g)),
            "total_observed_vertices": int(np.sum(self.observed_vertices)),
            "new_target_samples": int(np.sum(new_t)),
            "total_observed_target_samples": int(np.sum(self.observed_target_samples)),
            "score": float(score), **metrics,
        }
        self.history.append(row)
        self.step_index += 1
        return row


class WildlifeAwareLocalNBVPlanner:
    """Sequential local NBV with geometry + wildlife-proxy gain and movement penalties."""

    def __init__(self, engine, targets, step_size=5.0, bounds=None, connectivity=8,
                 shortlist_size=10, max_expansion=3, geometry_weight=0.45,
                 wildlife_weight=0.55, travel_cost_weight=0.08,
                 rotation_cost_weight=0.02, revisit_penalty=0.15,
                 observable_fraction_threshold=0.25, min_gain=1,
                 yaw_offsets=(-np.pi/4, 0.0, np.pi/4),
                 pitch_choices=(-np.pi/2, -np.pi/3, -np.pi/4)):
        if geometry_weight < 0 or wildlife_weight < 0 or geometry_weight + wildlife_weight <= 0:
            raise ValueError("Gain weights must be nonnegative and not both zero.")
        self.engine, self.targets = engine, list(targets)
        self.step_size, self.bounds, self.connectivity = float(step_size), bounds, connectivity
        self.shortlist_size, self.max_expansion = int(shortlist_size), int(max_expansion)
        total = geometry_weight + wildlife_weight
        self.geometry_weight, self.wildlife_weight = geometry_weight/total, wildlife_weight/total
        self.travel_cost_weight, self.rotation_cost_weight = travel_cost_weight, rotation_cost_weight
        self.revisit_penalty, self.threshold, self.min_gain = revisit_penalty, observable_fraction_threshold, min_gain
        self.yaw_offsets, self.pitch_choices = tuple(yaw_offsets), tuple(pitch_choices)
        self.vertices = np.asarray(engine.mesh.vertices, dtype=np.float32)
        self.target_points = np.vstack([t.surface_points for t in self.targets]) if self.targets else np.zeros((0,3))
        self.total_target_samples = len(self.target_points)
        self.target_counts = np.asarray([len(t.surface_points) for t in self.targets], dtype=int)

    def _fov_mask(self, points, pose):
        camera = np.asarray(pose.position, dtype=np.float32)
        offsets = points - camera
        d = np.linalg.norm(offsets, axis=1)
        valid = d > 1e-8
        directions = np.zeros_like(offsets)
        directions[valid] = offsets[valid] / d[valid,None]
        right, up, forward = self.engine.camera_basis(pose.yaw, pose.pitch)
        x, y, z = directions @ right, directions @ up, directions @ forward
        return valid & (z > 0) & (np.arctan2(np.abs(x),z) <= self.engine.horizontal_fov/2) & (np.arctan2(np.abs(y),z) <= self.engine.vertical_fov/2)

    @staticmethod
    def _angle_diff(a,b):
        return abs((a-b+np.pi)%(2*np.pi)-np.pi)

    def _screen_gain(self, pose, state):
        geom = int(np.sum(self._fov_mask(self.vertices,pose) & ~state.observed_vertices))
        wildlife = int(np.sum(self._fov_mask(self.target_points,pose) & ~state.observed_target_samples)) if len(self.target_points) else 0
        return geom, wildlife

    def choose_next(self, state, current_yaw=0.0):
        best = None
        for expansion in range(self.max_expansion+1):
            step = self.step_size*(2**expansion)
            candidates = poses_at_neighbor_positions(state.current_position, step_size=step,
                bounds=self.bounds, connectivity=self.connectivity,
                yaw_offsets=self.yaw_offsets, pitches=self.pitch_choices)
            screened = []
            for pose in candidates:
                g,t = self._screen_gain(pose,state)
                screened.append((self.geometry_weight*g/max(len(self.vertices),1)
                                + self.wildlife_weight*t/max(self.total_target_samples,1), g,t,pose))
            screened.sort(key=lambda x:x[0],reverse=True)
            for _,sg,st,pose in screened[:self.shortlist_size]:
                if sg+st < self.min_gain: continue
                gv = self.engine.compute_visibility_vector(pose.position,pose.yaw,pose.pitch).astype(bool)
                tv = compute_target_visibility(self.engine,self.targets,pose)
                ng = int(np.sum(gv & ~state.observed_vertices))
                nt = int(np.sum(tv & ~state.observed_target_samples))
                geom_norm = ng/max(len(self.vertices),1)
                target_norm = nt/max(self.total_target_samples,1)
                distance = float(np.linalg.norm(pose.position-state.current_position))
                rotation = self._angle_diff(current_yaw,pose.yaw)
                revisit = state.position_key(pose.position) in state.visited_positions
                score = self.geometry_weight*geom_norm + self.wildlife_weight*target_norm
                score -= self.travel_cost_weight*(distance/max(self.step_size,1e-8))
                score -= self.rotation_cost_weight*(rotation/np.pi)
                score -= self.revisit_penalty*int(revisit)
                if ng+nt < self.min_gain: continue
                prior_summary = summarize_target_visibility(state.observed_target_samples,self.targets,self.threshold)
                updated = state.observed_target_samples | tv
                combined_summary = summarize_target_visibility(updated,self.targets,self.threshold)
                metrics = {
                    "geometry_gain_normalized": geom_norm,
                    "target_gain_normalized": target_norm,
                    "newly_observable_targets": combined_summary["targets_observable_at_threshold"]-prior_summary["targets_observable_at_threshold"],
                    "observable_targets_total": combined_summary["targets_observable_at_threshold"],
                    "target_union_fraction": combined_summary["overall_visible_fraction"],
                    "travel_distance": distance, "rotation_radians": rotation,
                    "revisited_position": bool(revisit), "expansion_step": step,
                }
                record = {"pose":pose,"geometry_visibility":gv,"target_visibility":tv,
                          "score":float(score),"new_vertices":ng,"new_target_samples":nt,
                          "metrics":metrics}
                if best is None or score > best["score"]: best=record
            if best is not None and (best["new_vertices"]+best["new_target_samples"] >= self.min_gain):
                return best
        return None

    def step(self,state,current_yaw=0.0):
        choice=self.choose_next(state,current_yaw)
        if choice is None:return None
        row=state.update(choice["pose"],choice["geometry_visibility"],choice["target_visibility"],
                         choice["score"],choice["metrics"])
        choice["step"]=row["step"]
        choice["total_observed_vertices"]=row["total_observed_vertices"]
        choice["total_observed_target_samples"]=row["total_observed_target_samples"]
        return choice
