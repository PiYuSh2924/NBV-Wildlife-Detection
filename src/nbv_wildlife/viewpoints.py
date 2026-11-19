import numpy as np
from dataclasses import dataclass


@dataclass
class CameraPose:
    """A camera pose in world coordinates."""
    position: np.ndarray
    yaw: float
    pitch: float
    roll: float = 0.0


def create_initial_grid_views(width=30.0, depth=30.0, grid_size=6, height=16.0):
    """Create the legacy fixed-grid viewpoints for baseline comparisons."""
    x_values = np.linspace(-width / 2, width / 2, grid_size)
    y_values = np.linspace(-depth / 2, depth / 2, grid_size)
    viewpoints = []
    for x in x_values:
        for y in y_values:
            viewpoints.append(CameraPose(
                position=np.array([x, y, height], dtype=float),
                yaw=0.0,
                pitch=-np.pi / 2,
                roll=0.0,
            ))
    return viewpoints


def generate_neighbor_positions(
    current_position,
    step_size=5.0,
    bounds=None,
    connectivity=8,
    fixed_altitude=True,
):
    """
    Generate reachable planar neighbor positions around the drone.

    connectivity=4 uses cardinal neighbors; connectivity=8 also includes
    diagonals. Candidate locations are clipped by rejection to bounds.
    If bounds=(xmin, xmax, ymin, ymax), candidates outside are discarded.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    current = np.asarray(current_position, dtype=float)
    if current.shape != (3,):
        raise ValueError("current_position must be a 3D vector")

    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    candidates = []
    for dx, dy in offsets:
        candidate = current.copy()
        candidate[0] += dx * step_size
        candidate[1] += dy * step_size
        if fixed_altitude:
            candidate[2] = current[2]
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            if not (xmin <= candidate[0] <= xmax and ymin <= candidate[1] <= ymax):
                continue
        candidates.append(candidate)

    return candidates


def generate_orientations(
    movement_heading,
    yaw_offsets=(-np.pi / 4, 0.0, np.pi / 4),
    pitches=(-np.pi / 2, -np.pi / 3, -np.pi / 4),
):
    """Generate camera yaw/pitch choices independently of movement."""
    orientations = []
    for yaw_offset in yaw_offsets:
        yaw = (movement_heading + yaw_offset + np.pi) % (2 * np.pi) - np.pi
        for pitch in pitches:
            orientations.append((float(yaw), float(pitch)))
    return orientations


def poses_at_neighbor_positions(
    current_position,
    step_size=5.0,
    bounds=None,
    connectivity=8,
    yaw_offsets=(-np.pi / 4, 0.0, np.pi / 4),
    pitches=(-np.pi / 2, -np.pi / 3, -np.pi / 4),
):
    """Generate candidate CameraPose objects at adjacent positions."""
    positions = generate_neighbor_positions(
        current_position=current_position,
        step_size=step_size,
        bounds=bounds,
        connectivity=connectivity,
    )
    poses = []
    for position in positions:
        heading = np.arctan2(
            position[1] - current_position[1],
            position[0] - current_position[0],
        )
        for yaw, pitch in generate_orientations(
            heading, yaw_offsets=yaw_offsets, pitches=pitches
        ):
            poses.append(CameraPose(position=position.copy(), yaw=yaw, pitch=pitch))
    return poses
