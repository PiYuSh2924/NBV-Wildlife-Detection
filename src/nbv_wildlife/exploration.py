from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nbv_wildlife.viewpoints import CameraPose, poses_at_neighbor_positions


@dataclass
class ExplorationState:
    """Memory maintained by the online exploration process."""
    current_position: np.ndarray
    observed_vertices: np.ndarray
    visited_positions: set[tuple[float, float, float]] = field(default_factory=set)
    history: list[dict] = field(default_factory=list)
    step_index: int = 0

    @classmethod
    def create(cls, start_position, num_vertices):
        start = np.asarray(start_position, dtype=float)
        state = cls(
            current_position=start.copy(),
            observed_vertices=np.zeros(num_vertices, dtype=bool),
        )
        state.visited_positions.add(state.position_key(start))
        return state

    @staticmethod
    def position_key(position, decimals=2):
        return tuple(np.round(np.asarray(position, dtype=float), decimals=decimals))

    def update(self, pose: CameraPose, visibility: np.ndarray, score: float):
        visibility = np.asarray(visibility, dtype=bool)
        if visibility.shape != self.observed_vertices.shape:
            raise ValueError("Visibility vector size does not match state map.")
        new_vertices = visibility & ~self.observed_vertices
        newly_observed = int(np.sum(new_vertices))
        self.observed_vertices |= visibility
        self.current_position = np.asarray(pose.position, dtype=float).copy()
        self.visited_positions.add(self.position_key(self.current_position))
        self.history.append({
            "step": self.step_index,
            "position": self.current_position.copy(),
            "yaw": float(pose.yaw),
            "pitch": float(pose.pitch),
            "new_vertices": newly_observed,
            "total_observed": int(np.sum(self.observed_vertices)),
            "score": float(score),
        })
        self.step_index += 1
        return newly_observed


class LocalAdaptiveNBVPlanner:
    """
    Online local NBV planner:
      - generate reachable neighboring positions
      - cheap FOV-only screening
      - raycast a shortlist
      - score incremental visibility against travel/rotation costs
      - expand step size when local novelty is exhausted

    This initial implementation uses mesh-vertex novelty as a proxy. It does
    not represent animal detection probability or detector performance.
    """

    def __init__(
        self,
        visibility_engine,
        step_size=5.0,
        bounds=None,
        connectivity=8,
        shortlist_size=8,
        max_expansion=3,
        novelty_weight=1.0,
        travel_cost_weight=0.10,
        rotation_cost_weight=0.03,
        revisit_penalty=0.25,
        min_novelty=1,
        pitch_choices=(-np.pi / 2, -np.pi / 3, -np.pi / 4),
        yaw_offsets=(-np.pi / 4, 0.0, np.pi / 4),
    ):
        self.engine = visibility_engine
        self.step_size = float(step_size)
        self.bounds = bounds
        self.connectivity = connectivity
        self.shortlist_size = int(shortlist_size)
        self.max_expansion = int(max_expansion)
        self.novelty_weight = float(novelty_weight)
        self.travel_cost_weight = float(travel_cost_weight)
        self.rotation_cost_weight = float(rotation_cost_weight)
        self.revisit_penalty = float(revisit_penalty)
        self.min_novelty = int(min_novelty)
        self.pitch_choices = tuple(pitch_choices)
        self.yaw_offsets = tuple(yaw_offsets)
        self.vertices = np.asarray(self.engine.mesh.vertices, dtype=np.float32)

    def _fov_mask(self, pose):
        camera = np.asarray(pose.position, dtype=np.float32)
        offsets = self.vertices - camera
        distance = np.linalg.norm(offsets, axis=1)
        valid = distance > 1e-8
        directions = np.zeros_like(offsets)
        directions[valid] = offsets[valid] / distance[valid, None]
        right, up, forward = self.engine.camera_basis(pose.yaw, pose.pitch)
        x = directions @ right
        y = directions @ up
        z = directions @ forward
        return (
            valid & (z > 0)
            & (np.arctan2(np.abs(x), z) <= self.engine.horizontal_fov / 2)
            & (np.arctan2(np.abs(y), z) <= self.engine.vertical_fov / 2)
        )

    @staticmethod
    def _angle_difference(a, b):
        return abs((a - b + np.pi) % (2 * np.pi) - np.pi)

    def _pose_cost(self, current, candidate):
        distance = float(np.linalg.norm(candidate.position - current))
        return distance

    def _score(self, pose, visibility, state, current_yaw=0.0):
        newly_visible = visibility & ~state.observed_vertices
        gain = int(np.sum(newly_visible))
        travel = self._pose_cost(state.current_position, pose)
        rotation = self._angle_difference(current_yaw, pose.yaw)
        key = state.position_key(pose.position)
        revisit = key in state.visited_positions
        score = (
            self.novelty_weight * gain
            - self.travel_cost_weight * travel
            - self.rotation_cost_weight * rotation
            - (self.revisit_penalty if revisit else 0.0)
        )
        return score, gain, travel, rotation, revisit

    def choose_next(self, state: ExplorationState, current_yaw=0.0):
        """Select one next pose, expanding the search if exact gain is low."""
        if len(state.observed_vertices) != len(self.vertices):
            raise ValueError("Planner mesh and exploration-state map differ in size.")

        best_choice = None
        for expansion in range(self.max_expansion + 1):
            step = self.step_size * (2 ** expansion)
            candidates = poses_at_neighbor_positions(
                state.current_position,
                step_size=step,
                bounds=self.bounds,
                connectivity=self.connectivity,
                yaw_offsets=self.yaw_offsets,
                pitches=self.pitch_choices,
            )

            screened = []
            for pose in candidates:
                fov = self._fov_mask(pose)
                approximate_gain = int(np.sum(fov & ~state.observed_vertices))
                travel = self._pose_cost(state.current_position, pose)
                screened.append((approximate_gain, -travel, pose))

            screened.sort(key=lambda item: (item[0], item[1]), reverse=True)
            shortlist = screened[: self.shortlist_size]
            if not shortlist or shortlist[0][0] < self.min_novelty:
                continue

            best_score = -np.inf
            expansion_best = None
            for approximate_gain, _, pose in shortlist:
                visibility = self.engine.compute_visibility_vector(
                    pose.position, pose.yaw, pose.pitch
                ).astype(bool)
                score, gain, travel, rotation, revisit = self._score(
                    pose, visibility, state, current_yaw=current_yaw
                )
                record = {
                    "pose": pose,
                    "visibility": visibility,
                    "score": score,
                    "new_vertices": gain,
                    "travel_distance": travel,
                    "rotation_cost": rotation,
                    "revisited_position": revisit,
                    "expansion_step": step,
                    "screened_fov_gain": approximate_gain,
                }
                if score > best_score:
                    best_score = score
                    expansion_best = record

            # Expand only when the shortlisted detailed raycasts show that
            # the current neighborhood is occluded / already observed.
            if expansion_best is not None and expansion_best["new_vertices"] >= self.min_novelty:
                return expansion_best
            best_choice = expansion_best if best_choice is None else best_choice

        # If expansion found no useful visible gain, stop rather than move
        # to a viewpoint with zero newly visible geometry.
        return best_choice if best_choice and best_choice["new_vertices"] >= self.min_novelty else None

    def step(self, state: ExplorationState, current_yaw=0.0):
        """Plan, select, and update the simulated map for one observation."""
        choice = self.choose_next(state, current_yaw=current_yaw)
        if choice is None:
            return None
        newly_observed = state.update(
            choice["pose"], choice["visibility"], choice["score"]
        )
        choice["newly_observed"] = newly_observed
        choice["total_observed"] = int(np.sum(state.observed_vertices))
        choice["step"] = state.step_index - 1
        return choice
