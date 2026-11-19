from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import open3d as o3d


@dataclass
class WildlifeTarget:
    """Simplified wildlife proxy: axis-aligned box, surface samples only."""
    target_id: int
    center: np.ndarray
    dimensions: np.ndarray
    surface_points: np.ndarray

    @property
    def bounds_min(self):
        return self.center - self.dimensions / 2.0

    @property
    def bounds_max(self):
        return self.center + self.dimensions / 2.0

    def create_mesh(self, color=(0.85, 0.25, 0.12)):
        mesh = o3d.geometry.TriangleMesh.create_box(
            width=float(self.dimensions[0]),
            height=float(self.dimensions[1]),
            depth=float(self.dimensions[2])
        )
        mesh.translate(-self.dimensions / 2.0)
        mesh.translate(self.center)
        mesh.paint_uniform_color(list(color))
        mesh.compute_vertex_normals()
        return mesh


def sample_box_surface(center, dimensions, samples_per_axis=5):
    if samples_per_axis < 2:
        raise ValueError("samples_per_axis must be at least 2")
    center = np.asarray(center, dtype=float)
    dimensions = np.asarray(dimensions, dtype=float)
    if center.shape != (3,) or dimensions.shape != (3,):
        raise ValueError("center and dimensions must each have shape (3,)")
    if np.any(dimensions <= 0):
        raise ValueError("All target dimensions must be positive")
    axes = [np.linspace(-dimensions[i]/2, dimensions[i]/2, samples_per_axis) for i in range(3)]
    points = []
    for x in (-dimensions[0]/2, dimensions[0]/2):
        for y in axes[1]:
            for z in axes[2]:
                points.append([x, y, z])
    for y in (-dimensions[1]/2, dimensions[1]/2):
        for x in axes[0]:
            for z in axes[2]:
                points.append([x, y, z])
    for z in (-dimensions[2]/2, dimensions[2]/2):
        for x in axes[0]:
            for y in axes[1]:
                points.append([x, y, z])
    return np.unique(np.round(np.asarray(points), 10), axis=0) + center


def generate_random_targets(width=30.0, depth=30.0, num_targets=8, seed=123,
                           samples_per_axis=5, min_dimensions=(1.2, 0.5, 0.5),
                           max_dimensions=(2.2, 0.9, 1.2), edge_margin=1.0):
    if width <= 2*edge_margin or depth <= 2*edge_margin:
        raise ValueError("Forest dimensions are too small for edge margin")
    if num_targets < 1:
        raise ValueError("num_targets must be at least 1")
    rng = np.random.default_rng(seed)
    min_dimensions = np.asarray(min_dimensions, dtype=float)
    max_dimensions = np.asarray(max_dimensions, dtype=float)
    if np.any(min_dimensions <= 0) or np.any(max_dimensions < min_dimensions):
        raise ValueError("Target dimension bounds are invalid")
    targets = []
    for target_id in range(num_targets):
        dims = rng.uniform(min_dimensions, max_dimensions)
        x = rng.uniform(-width/2 + edge_margin, width/2 - edge_margin)
        y = rng.uniform(-depth/2 + edge_margin, depth/2 - edge_margin)
        center = np.array([x, y, dims[2]/2.0])
        targets.append(WildlifeTarget(target_id, center, dims,
                                     sample_box_surface(center, dims, samples_per_axis)))
    return targets


def all_target_points(targets: Iterable[WildlifeTarget]):
    targets = list(targets)
    return np.vstack([t.surface_points for t in targets]) if targets else np.zeros((0, 3))


def target_sample_counts(targets: Iterable[WildlifeTarget]):
    return np.asarray([len(t.surface_points) for t in targets], dtype=int)


def compute_target_visibility(engine, targets, camera_pose, chunk_size=65536, tolerance=1e-4):
    """Vectorized target-sample visibility. Forest mesh is the only occluder."""
    points = all_target_points(targets)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    camera = np.asarray(camera_pose.position, dtype=np.float32)
    pts = points.astype(np.float32)
    offsets = pts - camera
    distances = np.linalg.norm(offsets, axis=1)
    valid = distances > 1e-8
    directions = np.zeros_like(offsets)
    directions[valid] = offsets[valid] / distances[valid, None]
    right, up, forward = engine.camera_basis(camera_pose.yaw, camera_pose.pitch)
    xcam, ycam, zcam = directions @ right, directions @ up, directions @ forward
    in_fov = (valid & (zcam > 0)
              & (np.arctan2(np.abs(xcam), zcam) <= engine.horizontal_fov/2)
              & (np.arctan2(np.abs(ycam), zcam) <= engine.vertical_fov/2))
    ids = np.flatnonzero(in_fov)
    visible = np.zeros(len(points), dtype=bool)
    for begin in range(0, len(ids), chunk_size):
        idx = ids[begin:begin+chunk_size]
        rays = np.concatenate([np.repeat(camera[None, :], len(idx), axis=0), directions[idx]], axis=1)
        ray_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        hits = engine.scene.cast_rays(ray_tensor)["t_hit"].numpy()
        visible[idx[(~np.isfinite(hits)) | (hits >= distances[idx] - tolerance)]] = True
    return visible


def split_visibility_by_target(visibility, targets):
    visibility = np.asarray(visibility, dtype=bool)
    expected = sum(len(t.surface_points) for t in targets)
    if len(visibility) != expected:
        raise ValueError(f"Visibility length {len(visibility)} != target samples {expected}")
    offset, blocks = 0, []
    for t in targets:
        n = len(t.surface_points)
        blocks.append(visibility[offset:offset+n])
        offset += n
    return blocks


def summarize_target_visibility(visibility, targets, observable_fraction_threshold=0.25):
    if not 0 <= observable_fraction_threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    targets = list(targets)
    visibility = np.asarray(visibility, dtype=bool)
    blocks = split_visibility_by_target(visibility, targets)
    per_target = []
    observable = 0
    for target, block in zip(targets, blocks):
        fraction = float(np.mean(block)) if len(block) else 0.0
        is_observable = fraction >= observable_fraction_threshold
        observable += int(is_observable)
        per_target.append({
            "target_id": int(target.target_id),
            "sample_count": int(len(block)),
            "visible_samples": int(np.sum(block)),
            "visible_fraction": fraction,
            "observable_at_threshold": bool(is_observable),
        })
    return {
        "num_targets": len(targets),
        "total_surface_samples": int(len(visibility)),
        "visible_surface_samples": int(np.sum(visibility)),
        "overall_visible_fraction": float(np.mean(visibility)) if len(visibility) else 0.0,
        "targets_observable_at_threshold": int(observable),
        "threshold": float(observable_fraction_threshold),
        "per_target": per_target,
    }
