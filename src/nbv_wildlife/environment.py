import numpy as np
import open3d as o3d
from dataclasses import dataclass


@dataclass
class ForestEnvironment:
    width: float
    depth: float
    tree_meshes: list
    combined_mesh: o3d.geometry.TriangleMesh


def create_tree(
    x,
    y,
    trunk_height,
    crown_height,
    trunk_radius,
    crown_radius
):
    """
    Create one simple procedural tree.

    The paper describes randomized tree positions,
    shapes and densities, but does not specify the
    exact tree-generation procedure.

    Therefore, this is our reproduction assumption.
    """

    # -------------------------
    # Tree trunk
    # -------------------------
    trunk = o3d.geometry.TriangleMesh.create_cylinder(
        radius=trunk_radius,
        height=trunk_height
    )

    trunk.translate(
        [x, y, trunk_height / 2]
    )

    # -------------------------
    # Tree crown
    # -------------------------
    crown = o3d.geometry.TriangleMesh.create_cone(
        radius=crown_radius,
        height=crown_height
    )

    # The Open3D cone starts from its base.
    # Put that base slightly below the top of the trunk
    # to create a small overlap.
    crown.translate(
        [x, y, trunk_height - 0.5]
    )

    trunk.compute_vertex_normals()
    crown.compute_vertex_normals()

    return trunk, crown


def create_forest(
    width=30.0,
    depth=30.0,
    num_trees=45,
    seed=42
):
    """
    Generate a randomized forest environment.
    """

    rng = np.random.default_rng(seed)

    tree_meshes = []

    for _ in range(num_trees):

        # Random tree position
        x = rng.uniform(
            -width / 2 + 1,
            width / 2 - 1
        )

        y = rng.uniform(
            -depth / 2 + 1,
            depth / 2 - 1
        )

        # Random tree dimensions
        trunk_height = rng.uniform(5.0, 9.0)
        crown_height = rng.uniform(4.0, 8.0)

        trunk_radius = rng.uniform(0.15, 0.35)
        crown_radius = rng.uniform(1.5, 3.5)

        trunk, crown = create_tree(
            x,
            y,
            trunk_height,
            crown_height,
            trunk_radius,
            crown_radius
        )

        tree_meshes.append(trunk)
        tree_meshes.append(crown)

    # Combine all tree geometry
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


def create_ground(width, depth):
    """
    Create a ground plane for visualization.

    Ground is only for visualization at this stage.
    It will not be treated as an occluding surface
    in the later visibility calculations.
    """

    ground = o3d.geometry.TriangleMesh.create_box(
        width=width,
        height=depth,
        depth=0.1
    )

    ground.translate(
        [-width / 2, -depth / 2, -0.05]
    )

    ground.compute_vertex_normals()

    return ground