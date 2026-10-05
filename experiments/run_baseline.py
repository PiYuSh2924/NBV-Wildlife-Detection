import open3d as o3d
import numpy as np

from nbv_wildlife.environment import (
    create_forest,
    create_ground
)

from nbv_wildlife.viewpoints import (
    create_initial_grid_views
)


def create_camera_marker(pose, size=0.5):
    """
    Create a small coordinate frame representing
    the camera/UAV viewpoint.
    """

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size,
        origin=pose.position
    )

    return frame


def create_view_direction(pose, length=2.0):
    """
    Draw a line showing the camera viewing direction.
    """

    start = pose.position

    # Top-down direction
    direction = np.array(
        [0.0, 0.0, -length]
    )

    end = start + direction

    points = o3d.utility.Vector3dVector(
        [start, end]
    )

    lines = o3d.utility.Vector2iVector(
        [[0, 1]]
    )

    line_set = o3d.geometry.LineSet(
        points=points,
        lines=lines
    )

    return line_set


def main():

    print("Creating forest...")

    forest = create_forest(
        width=30.0,
        depth=30.0,
        num_trees=45,
        seed=42
    )

    print("Creating 36 initial viewpoints...")

    viewpoints = create_initial_grid_views(
        width=30.0,
        depth=30.0,
        grid_size=6,
        height=16.0
    )

    print(f"Number of viewpoints: {len(viewpoints)}")

    # --------------------------------
    # Visualization objects
    # --------------------------------

    geometries = []

    # Forest
    geometries.append(
        forest.combined_mesh
    )

    # Ground
    geometries.append(
        create_ground(
            forest.width,
            forest.depth
        )
    )

    # Camera viewpoints
    for pose in viewpoints:

        geometries.append(
            create_camera_marker(
                pose
            )
        )

        geometries.append(
            create_view_direction(
                pose
            )
        )

    print()
    print("Opening Open3D visualization...")
    print("You should see:")
    print("  - 30m x 30m forest")
    print("  - randomized trees")
    print("  - 36 camera viewpoints")
    print("  - 6 x 6 grid")
    print("  - top-down viewing directions")

    o3d.visualization.draw_geometries(
        geometries,
        window_name="NBV Wildlife Detection - V0.1",
        width=1200,
        height=800
    )


if __name__ == "__main__":
    main()