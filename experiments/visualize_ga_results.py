import numpy as np
import open3d as o3d

from nbv_wildlife.environment import create_forest, create_ground
from nbv_wildlife.viewpoints import create_initial_grid_views


# ============================================================
# CONFIGURATION
# ============================================================

HORIZONTAL_FOV = 60.0
VERTICAL_FOV = 45.0

FRUSTUM_LENGTH = 8.0

BASELINE_MARKER_RADIUS = 0.20
GA_MARKER_RADIUS = 0.50


# ============================================================
# GA CANDIDATES
# Latest Pareto candidates from expanded +/- 25 m GA
# Angles are stored in DEGREES
# ============================================================

GA_CANDIDATES = [
    {
        "name": "Candidate 1",
        "position": [23.781, 13.057, 21.791],
        "yaw": -151.568,
        "pitch": -39.977,
        "color": [0.95, 0.12, 0.12],
    },
    {
        "name": "Candidate 2",
        "position": [17.116, -7.274, 24.823],
        "yaw": 147.493,
        "pitch": -56.315,
        "color": [0.10, 0.35, 0.95],
    },
    {
        "name": "Candidate 3",
        "position": [22.670, -7.274, 25.000],
        "yaw": 150.496,
        "pitch": -45.060,
        "color": [1.00, 0.62, 0.05],
    },
]


# ============================================================
# CAMERA ORIENTATION
# Independent of VisibilityEngine
# ============================================================

def calculate_camera_basis(yaw, pitch):
    """
    Calculate camera right, up, and forward vectors.

    Parameters
    ----------
    yaw : float
        Yaw in radians.

    pitch : float
        Pitch in radians.

    Convention
    ----------
    yaw = 0:
        Camera points along +X.

    yaw = 90 degrees:
        Camera points along +Y.

    pitch = 0:
        Camera points horizontally.

    pitch = -90 degrees:
        Camera points vertically downward.

    Returns
    -------
    right, up, forward : numpy arrays
    """

    forward = np.array([
        np.cos(pitch) * np.cos(yaw),
        np.cos(pitch) * np.sin(yaw),
        np.sin(pitch),
    ], dtype=float)

    right = np.array([
        -np.sin(yaw),
        np.cos(yaw),
        0.0,
    ], dtype=float)

    # Construct an orthogonal up vector.
    up = np.cross(forward, right)

    # Normalize vectors.
    forward = forward / np.linalg.norm(forward)
    right = right / np.linalg.norm(right)
    up = up / np.linalg.norm(up)

    return right, up, forward


# ============================================================
# CAMERA MARKER
# ============================================================

def create_camera_marker(position, color, radius=0.3):

    marker = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius,
        resolution=20,
    )

    marker.translate(
        np.asarray(position, dtype=float)
    )

    marker.paint_uniform_color(color)

    marker.compute_vertex_normals()

    return marker


# ============================================================
# CAMERA FRUSTUM
# ============================================================

def create_camera_frustum(
    position,
    yaw,
    pitch,
    color,
    length=FRUSTUM_LENGTH,
):

    position = np.asarray(
        position,
        dtype=float,
    )

    # Calculate camera orientation directly.
    right, up, forward = calculate_camera_basis(
        yaw,
        pitch,
    )

    # Calculate image-plane dimensions.
    half_width = (
        length *
        np.tan(np.deg2rad(HORIZONTAL_FOV / 2.0))
    )

    half_height = (
        length *
        np.tan(np.deg2rad(VERTICAL_FOV / 2.0))
    )

    # Center of the virtual image plane.
    image_center = position + length * forward

    # Four image-plane corners.
    corners = [
        image_center - half_width * right - half_height * up,
        image_center + half_width * right - half_height * up,
        image_center + half_width * right + half_height * up,
        image_center - half_width * right + half_height * up,
    ]

    # Point 0 = camera position
    # Points 1-4 = image plane corners
    points = [
        position,
        corners[0],
        corners[1],
        corners[2],
        corners[3],
    ]

    # Frustum edges.
    lines = [
        # Camera to image plane
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],

        # Image plane boundary
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
    ]

    frustum = o3d.geometry.LineSet()

    frustum.points = o3d.utility.Vector3dVector(
        np.asarray(points)
    )

    frustum.lines = o3d.utility.Vector2iVector(
        np.asarray(lines)
    )

    frustum.paint_uniform_color(color)

    return frustum


# ============================================================
# BUILD FOREST SCENE
# ============================================================

def build_scene():

    print("Generating forest environment...")

    forest = create_forest(
        width=30.0,
        depth=30.0,
        num_trees=45,
        seed=42,
    )

    geometries = []

    # --------------------------------------------------------
    # GROUND
    # --------------------------------------------------------

    print("Creating ground...")

    ground = create_ground(
        forest.width,
        forest.depth,
    )

    ground.paint_uniform_color([
        0.20,
        0.17,
        0.12,
    ])

    geometries.append(ground)

    # --------------------------------------------------------
    # FOREST
    # --------------------------------------------------------

    print("Adding individual trees...")

    for tree_mesh in forest.tree_meshes:

        tree_mesh.compute_vertex_normals()

        geometries.append(tree_mesh)

    # --------------------------------------------------------
    # BASELINE CAMERA POSITIONS
    # --------------------------------------------------------

    print("Generating baseline viewpoints...")

    baseline_views = create_initial_grid_views(
        width=30.0,
        depth=30.0,
        grid_size=6,
        height=16.0,
    )

    print(
        "Baseline viewpoints:",
        len(baseline_views),
    )

    for pose in baseline_views:

        marker = create_camera_marker(
            position=pose.position,
            color=[0.65, 0.68, 0.72],
            radius=BASELINE_MARKER_RADIUS,
        )

        geometries.append(marker)

    # --------------------------------------------------------
    # GA CAMERA POSITIONS
    # --------------------------------------------------------

    print("Adding GA viewpoints...")

    for candidate in GA_CANDIDATES:

        position = np.asarray(
            candidate["position"],
            dtype=float,
        )

        # Convert degrees to radians.
        yaw = np.deg2rad(
            candidate["yaw"]
        )

        pitch = np.deg2rad(
            candidate["pitch"]
        )

        color = candidate["color"]

        # Camera marker.
        marker = create_camera_marker(
            position=position,
            color=color,
            radius=GA_MARKER_RADIUS,
        )

        geometries.append(marker)

        # Camera viewing frustum.
        frustum = create_camera_frustum(
            position=position,
            yaw=yaw,
            pitch=pitch,
            color=color,
            length=FRUSTUM_LENGTH,
        )

        geometries.append(frustum)

        print(
            f"{candidate['name']}: "
            f"position={candidate['position']}, "
            f"yaw={candidate['yaw']:.3f}, "
            f"pitch={candidate['pitch']:.3f}"
        )

    # --------------------------------------------------------
    # COORDINATE FRAME
    # --------------------------------------------------------

    coordinate_frame = (
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=4.0,
            origin=[0.0, 0.0, 0.0],
        )
    )

    geometries.append(coordinate_frame)

    print("Scene geometry created.")

    return geometries


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_scene(geometries):

    print("Initializing Open3D visualizer...")

    visualizer = o3d.visualization.Visualizer()

    window_created = visualizer.create_window(
        window_name="NBV Wildlife Detection | GA Results",
        width=1600,
        height=1000,
        left=50,
        top=50,
        visible=True,
    )

    if not window_created:
        print("ERROR: Open3D failed to create the visualization window.")
        return

    try:

        # Add geometries and initialize the bounding box.
        for i, geometry in enumerate(geometries):

            visualizer.add_geometry(
                geometry,
                reset_bounding_box=(i == 0),
            )

        # Force an initial rendering update.
        visualizer.poll_events()
        visualizer.update_renderer()

        # Rendering settings.
        render_option = visualizer.get_render_option()

        render_option.background_color = np.array([
            0.12,
            0.14,
            0.17,
        ])

        render_option.mesh_show_back_face = True
        render_option.light_on = True
        render_option.line_width = 2.0

        # Initialize camera only after geometry/window initialization.
        view_control = visualizer.get_view_control()

        view_control.set_lookat([
            0.0,
            0.0,
            8.0,
        ])

        view_control.set_front([
            1.0,
            1.0,
            -0.65,
        ])

        view_control.set_up([
            0.0,
            0.0,
            1.0,
        ])

        view_control.set_zoom(0.65)

        # Refresh after camera settings.
        visualizer.poll_events()
        visualizer.update_renderer()

        print("Visualization window initialized successfully.")
        print("Keep the window open; close it to finish.")

        # Keep the window alive.
        visualizer.run()

    finally:

        visualizer.destroy_window()

        print("Visualization window closed.")

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NBV WILDLIFE DETECTION VISUALIZATION")
    print("=" * 60)

    geometries = build_scene()

    print("\nScene created successfully.")

    print("\nLegend:")
    print("Grey   = Baseline viewpoints")
    print("Red    = GA Candidate 1")
    print("Blue   = GA Candidate 2")
    print("Orange = GA Candidate 3")
    print("RGB    = Coordinate axes")

    print("\nOpening visualization window...")

    visualize_scene(geometries)

    print("\nVisualization completed.")


if __name__ == "__main__":
    main()