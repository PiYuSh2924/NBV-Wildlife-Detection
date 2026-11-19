
import numpy as np
import open3d as o3d

from dataclasses import dataclass


# ==================================================
# FOREST ENVIRONMENT
# ==================================================

@dataclass
class ForestEnvironment:
    width: float
    depth: float
    tree_meshes: list
    combined_mesh: o3d.geometry.TriangleMesh


# ==================================================
# HELPER: CREATE AN ELLIPSOID
# ==================================================

def create_ellipsoid(
    center,
    scale,
    resolution=16
):
    """
    Create a smooth ellipsoid using a scaled sphere.
    """

    mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=1.0,
        resolution=resolution
    )

    mesh.scale(
        1.0,
        center=[0, 0, 0]
    )

    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh.vertices) * np.asarray(scale)
    )

    mesh.translate(
        np.asarray(center, dtype=float)
    )

    mesh.compute_vertex_normals()

    return mesh


# ==================================================
# HELPER: CREATE ONE TREE
# ==================================================

def create_tree(
    x,
    y,
    trunk_height,
    crown_height,
    trunk_radius,
    crown_radius,
    crown_color=None
):
    """
    Create a stylized broadleaf tree.

    The trunk is tapered.
    The crown consists of overlapping rounded foliage clusters.

    These are procedural simulation assumptions, not
    an exact reconstruction of real tree species.
    """

    # --------------------------------------------------
    # A. TAPERED TRUNK
    # --------------------------------------------------

    trunk = o3d.geometry.TriangleMesh.create_cone(
        radius=trunk_radius * 1.25,
        height=trunk_height,
        resolution=16
    )

    trunk.translate([
        x,
        y,
        0.0
    ])

    trunk.paint_uniform_color([
        0.30,
        0.19,
        0.11
    ])

    trunk.compute_vertex_normals()

    # --------------------------------------------------
    # B. MAIN CROWN
    # --------------------------------------------------

    crown_center_z = (
        trunk_height + crown_height * 0.42
    )

    crown = create_ellipsoid(
        center=[
            x,
            y,
            crown_center_z
        ],
        scale=[
            crown_radius,
            crown_radius * 0.82,
            crown_height * 0.55
        ],
        resolution=20
    )

    # --------------------------------------------------
    # C. ADD ROUNDED FOLIAGE CLUSTERS
    # --------------------------------------------------

    # The overlapping clusters create a fuller,
    # less geometrically perfect canopy.

    cluster_positions = [
        (0.48, 0.05),
        (-0.45, 0.10),
        (0.05, 0.43),
        (-0.05, -0.42)
    ]

    for dx, dy in cluster_positions:

        cluster = create_ellipsoid(
            center=[
                x + dx * crown_radius,
                y + dy * crown_radius,
                crown_center_z + crown_height * 0.06
            ],
            scale=[
                crown_radius * 0.62,
                crown_radius * 0.58,
                crown_height * 0.37
            ],
            resolution=16
        )

        crown += cluster

    crown.compute_vertex_normals()

    # --------------------------------------------------
    # D. FOLIAGE COLOUR
    # --------------------------------------------------

    if crown_color is None:
        crown_color = [
            0.16,
            0.38,
            0.16
        ]

    crown.paint_uniform_color(crown_color)

    return trunk, crown


# ==================================================
# CREATE COMPLETE FOREST
# ==================================================

def create_forest(
    width=30.0,
    depth=30.0,
    num_trees=45,
    seed=42
):
    """
    Generate a randomized procedural broadleaf forest.

    Tree placement and dimensions are controlled by
    the random seed for reproducibility.
    """

    rng = np.random.default_rng(seed)

    tree_meshes = []

    # --------------------------------------------------
    # FOLIAGE COLOUR VARIATION
    # --------------------------------------------------

    foliage_palette = [
        [0.12, 0.32, 0.12],
        [0.16, 0.39, 0.15],
        [0.20, 0.43, 0.18],
        [0.25, 0.46, 0.19],
        [0.13, 0.35, 0.19],
        [0.28, 0.42, 0.16]
    ]

    for _ in range(num_trees):

        # ----------------------------------------------
        # A. RANDOM TREE POSITION
        # ----------------------------------------------

        x = rng.uniform(
            -width / 2 + 1,
            width / 2 - 1
        )

        y = rng.uniform(
            -depth / 2 + 1,
            depth / 2 - 1
        )

        # ----------------------------------------------
        # B. RANDOM TREE DIMENSIONS
        # ----------------------------------------------

        trunk_height = rng.uniform(
            5.0,
            9.0
        )

        crown_height = rng.uniform(
            4.0,
            8.0
        )

        trunk_radius = rng.uniform(
            0.15,
            0.35
        )

        crown_radius = rng.uniform(
            1.5,
            3.5
        )

        # ----------------------------------------------
        # C. SELECT FOLIAGE COLOUR
        # ----------------------------------------------

        color_index = rng.integers(
            0,
            len(foliage_palette)
        )

        crown_color = foliage_palette[
            color_index
        ]

        # ----------------------------------------------
        # D. CREATE TREE
        # ----------------------------------------------

        trunk, crown = create_tree(
            x=x,
            y=y,
            trunk_height=trunk_height,
            crown_height=crown_height,
            trunk_radius=trunk_radius,
            crown_radius=crown_radius,
            crown_color=crown_color
        )

        tree_meshes.append(trunk)
        tree_meshes.append(crown)

    # --------------------------------------------------
    # E. COMBINE TREE GEOMETRY
    # --------------------------------------------------

    combined_mesh = o3d.geometry.TriangleMesh()

    for mesh in tree_meshes:
        combined_mesh += mesh

    combined_mesh.compute_vertex_normals()

    return ForestEnvironment(
        width=width,
        depth=depth,
        tree_meshes=tree_meshes,
        combined_mesh=combined_mesh
    )


# ==================================================
# GROUND PLANE
# ==================================================

def create_ground(width, depth):
    """
    Create a forest-floor visualization surface.

    Ground is visual only and is NOT included in
    combined_mesh or the current raycasting engine.
    """

    ground = o3d.geometry.TriangleMesh.create_box(
        width=width,
        height=depth,
        depth=0.1
    )

    ground.translate([
        -width / 2,
        -depth / 2,
        -0.05
    ])

    ground.paint_uniform_color([
        0.20,
        0.17,
        0.12
    ])

    ground.compute_vertex_normals()

    return ground
