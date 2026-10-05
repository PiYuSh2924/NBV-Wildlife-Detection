import numpy as np
from dataclasses import dataclass


@dataclass
class CameraPose:
    """
    Represents one UAV camera pose.

    position:
        [x, y, z]

    yaw:
        Rotation around vertical axis.

    pitch:
        Up/down orientation.

    roll:
        Camera roll.
    """

    position: np.ndarray
    yaw: float
    pitch: float
    roll: float = 0.0


def create_initial_grid_views(
    width=30.0,
    depth=30.0,
    grid_size=6,
    height=16.0
):
    """
    Create the 36 initial viewpoints.

    The paper describes:
        6 x 6 regular grid
        top-down orientation

    Exact initial camera altitude is not specified,
    so height=16 m is our reproduction assumption.
    """

    x_values = np.linspace(
        -width / 2,
        width / 2,
        grid_size
    )

    y_values = np.linspace(
        -depth / 2,
        depth / 2,
        grid_size
    )

    viewpoints = []

    for x in x_values:
        for y in y_values:

            pose = CameraPose(
                position=np.array(
                    [x, y, height],
                    dtype=float
                ),

                # Top-down camera
                yaw=0.0,

                # -90° means looking vertically downward
                pitch=-np.pi / 2,

                roll=0.0
            )

            viewpoints.append(pose)

    return viewpoints