from __future__ import annotations

import numpy as np
import open3d as o3d

from nbv_wildlife.environment import create_forest
from nbv_wildlife.viewpoints import create_initial_grid_views


class VisibilityEngine:
    """
    Computes visibility from a camera using:

    1. Camera field-of-view constraints
    2. Ray-casting based occlusion testing
    """

    def __init__(
        self,
        mesh: o3d.geometry.TriangleMesh,
        horizontal_fov_deg: float = 60.0,
        vertical_fov_deg: float = 45.0,
    ):
        self.mesh = mesh

        self.horizontal_fov = np.deg2rad(horizontal_fov_deg)
        self.vertical_fov = np.deg2rad(vertical_fov_deg)

        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(
            mesh
        )

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(tensor_mesh)

    # =========================================================
    # CAMERA GEOMETRY
    # =========================================================

    def camera_basis(
        self,
        yaw: float,
        pitch: float,
    ):
        """
        Construct camera coordinate axes.

        Returns:
            right, up, forward
        """

        forward = np.array(
            [
                np.cos(pitch) * np.cos(yaw),
                np.cos(pitch) * np.sin(yaw),
                np.sin(pitch),
            ],
            dtype=float,
        )

        forward /= np.linalg.norm(forward)

        right = np.array(
            [
                np.sin(yaw),
                -np.cos(yaw),
                0.0,
            ],
            dtype=float,
        )

        right /= np.linalg.norm(right)

        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        return right, up, forward

    # =========================================================
    # FIELD OF VIEW
    # =========================================================

    def is_inside_fov(
        self,
        camera_position: np.ndarray,
        target_point: np.ndarray,
        yaw: float,
        pitch: float,
    ) -> bool:
        """
        Check whether a target point lies inside
        the camera field of view.
        """

        direction = target_point - camera_position

        distance = np.linalg.norm(direction)

        if distance == 0:
            return False

        direction = direction / distance

        right, up, forward = self.camera_basis(
            yaw,
            pitch,
        )

        x_camera = np.dot(
            direction,
            right,
        )

        y_camera = np.dot(
            direction,
            up,
        )

        z_camera = np.dot(
            direction,
            forward,
        )

        # Target must be in front of camera.
        if z_camera <= 0:
            return False

        horizontal_angle = np.arctan2(
            abs(x_camera),
            z_camera,
        )

        vertical_angle = np.arctan2(
            abs(y_camera),
            z_camera,
        )

        return (
            horizontal_angle <= self.horizontal_fov / 2
            and vertical_angle <= self.vertical_fov / 2
        )

    # =========================================================
    # OCCLUSION
    # =========================================================

    def is_occluded(
        self,
        camera_position: np.ndarray,
        target_point: np.ndarray,
        tolerance: float = 1e-4,
    ) -> bool:
        """
        Check whether another surface lies between
        the camera and the target point.
        """

        direction = target_point - camera_position

        distance = np.linalg.norm(direction)

        if distance == 0:
            return False

        direction = direction / distance

        ray = o3d.core.Tensor(
            [
                [
                    camera_position[0],
                    camera_position[1],
                    camera_position[2],
                    direction[0],
                    direction[1],
                    direction[2],
                ]
            ],
            dtype=o3d.core.Dtype.Float32,
        )

        result = self.scene.cast_rays(ray)

        t_hit = result["t_hit"].numpy()[0]

        if not np.isfinite(t_hit):
            return False

        return t_hit < distance - tolerance

    # =========================================================
    # COMPLETE VISIBILITY TEST
    # =========================================================

    def is_visible(
        self,
        camera_position: np.ndarray,
        target_point: np.ndarray,
        yaw: float,
        pitch: float,
    ) -> bool:
        """
        Visible = inside FOV AND not occluded.
        """

        if not self.is_inside_fov(
            camera_position,
            target_point,
            yaw,
            pitch,
        ):
            return False

        if self.is_occluded(
            camera_position,
            target_point,
        ):
            return False

        return True

    # =========================================================
    # SINGLE-CAMERA VISIBILITY VECTOR
    # =========================================================

    def compute_visibility_vector(
        self,
        camera_position: np.ndarray,
        yaw: float,
        pitch: float,
    ) -> np.ndarray:
        """
        Compute w_i for every mesh vertex.

        w_i = 1 -> visible
        w_i = 0 -> not visible
        """

        vertices = np.asarray(
            self.mesh.vertices
        )

        visibility = np.zeros(
            len(vertices),
            dtype=np.uint8,
        )

        for i, vertex in enumerate(vertices):

            if self.is_visible(
                camera_position,
                vertex,
                yaw,
                pitch,
            ):
                visibility[i] = 1

        return visibility

    # =========================================================
    # VISIBILITY MATRIX
    # =========================================================

    def compute_visibility_matrix(
        self,
        camera_poses,
    ) -> np.ndarray:
        """
        Compute visibility matrix M.

        Rows    = mesh vertices
        Columns = camera viewpoints

        M[i, k] = 1 if vertex i is visible
                  from camera k.
        """

        vertices = np.asarray(
            self.mesh.vertices
        )

        num_vertices = len(vertices)
        num_cameras = len(camera_poses)

        visibility_matrix = np.zeros(
            (num_vertices, num_cameras),
            dtype=np.uint8,
        )

        for k, camera in enumerate(camera_poses):

            print(
                f"Processing viewpoint "
                f"{k + 1}/{num_cameras}..."
            )

            visibility_matrix[:, k] = (
                self.compute_visibility_vector(
                    camera.position,
                    camera.yaw,
                    camera.pitch,
                )
            )

        return visibility_matrix

    # =========================================================
    # VISIBILITY COUNTS m_i
    # =========================================================

    def compute_visibility_counts(
        self,
        visibility_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute m_i.

        m_i = number of initial viewpoints
              from which vertex i is visible.
        """

        return np.sum(
            visibility_matrix,
            axis=1,
        )

    # =========================================================
    # VISIBILITY WEIGHTS alpha_i
    # =========================================================

    def compute_visibility_weights(
        self,
        visibility_counts: np.ndarray,
    ) -> np.ndarray:
        """
        Compute alpha_i from the paper:

            alpha_i =
            (1 - tanh(m_i - 3)) / 2

        Vertices observed fewer times receive
        greater weight.
        """

        visibility_counts = visibility_counts.astype(float)

        return (
            1.0
            - np.tanh(
                visibility_counts - 3.0
            )
        ) / 2.0

    # =========================================================
    # VISIBILITY FITNESS J_v
    # =========================================================

    def compute_visibility_fitness(
        self,
        candidate_visibility: np.ndarray,
        visibility_weights: np.ndarray,
    ) -> float:
        """
        Compute visibility fitness:

            J_v = sum(alpha_i * w_i)

        candidate_visibility:
            w_i for the candidate camera.

        visibility_weights:
            alpha_i derived from the initial viewpoints.
        """

        return float(
            np.sum(
                visibility_weights
                * candidate_visibility
            )
        )


# =============================================================
# TEST / DEMONSTRATION
# =============================================================

if __name__ == "__main__":

    print("Creating forest...")

    forest = create_forest(
        width=30.0,
        depth=30.0,
        num_trees=45,
        seed=42,
    )

    # ---------------------------------------------------------
    # VISIBILITY ENGINE
    # ---------------------------------------------------------

    engine = VisibilityEngine(
        forest.combined_mesh,
        horizontal_fov_deg=60.0,
        vertical_fov_deg=45.0,
    )

    # ---------------------------------------------------------
    # CONTROLLED TEST
    # ---------------------------------------------------------

    camera_position = np.array(
        [0.0, 0.0, 16.0]
    )

    yaw = 0.0
    pitch = -np.pi / 2

    print("\n===== CONTROLLED RAY TEST =====")

    test_points = {
        "center": np.array(
            [0.0, 0.0, 8.0]
        ),
        "near_center": np.array(
            [2.0, 0.0, 8.0]
        ),
    }

    for name, point in test_points.items():

        inside = engine.is_inside_fov(
            camera_position,
            point,
            yaw,
            pitch,
        )

        occluded = engine.is_occluded(
            camera_position,
            point,
        )

        visible = engine.is_visible(
            camera_position,
            point,
            yaw,
            pitch,
        )

        print(
            f"{name:15s} | "
            f"FOV: {inside} | "
            f"Occluded: {occluded} | "
            f"Visible: {visible}"
        )

    # ---------------------------------------------------------
    # SINGLE-CAMERA VISIBILITY
    # ---------------------------------------------------------

    print("\n===== VISIBILITY VECTOR TEST =====")

    visibility = engine.compute_visibility_vector(
        camera_position,
        yaw,
        pitch,
    )

    total_vertices = len(visibility)

    visible_vertices = int(
        np.sum(visibility)
    )

    print(
        f"Total vertices:     "
        f"{total_vertices}"
    )

    print(
        f"Visible vertices:   "
        f"{visible_vertices}"
    )

    print(
        f"Visibility ratio:   "
        f"{visible_vertices / total_vertices:.2%}"
    )

    # ---------------------------------------------------------
    # DIAGNOSTIC
    # ---------------------------------------------------------

    print("\n===== VISIBILITY DIAGNOSTIC =====")

    vertices = np.asarray(
        forest.combined_mesh.vertices
    )

    inside_fov_count = 0
    visible_count = 0

    for vertex in vertices:

        if engine.is_inside_fov(
            camera_position,
            vertex,
            yaw,
            pitch,
        ):

            inside_fov_count += 1

            if engine.is_visible(
                camera_position,
                vertex,
                yaw,
                pitch,
            ):

                visible_count += 1

    print(
        f"Vertices inside FOV: "
        f"{inside_fov_count}"
    )

    print(
        f"Vertices visible:    "
        f"{visible_count}"
    )

    if inside_fov_count > 0:

        print(
            f"Occlusion rate among "
            f"FOV vertices: "
            f"{1 - visible_count / inside_fov_count:.2%}"
        )

    # ---------------------------------------------------------
    # INITIAL 36 VIEWPOINTS
    # ---------------------------------------------------------

    print(
        "\n===== CREATING INITIAL "
        "36 VIEWPOINTS ====="
    )

    camera_poses = create_initial_grid_views(
        width=30.0,
        depth=30.0,
        grid_size=6,
    )

    print(
        f"Number of viewpoints: "
        f"{len(camera_poses)}"
    )

    # ---------------------------------------------------------
    # VISIBILITY MATRIX
    # ---------------------------------------------------------

    print(
        "\n===== 36-VIEW "
        "VISIBILITY MATRIX ====="
    )

    visibility_matrix = (
        engine.compute_visibility_matrix(
            camera_poses
        )
    )

    print(
        "\nMatrix shape:"
    )

    print(
        visibility_matrix.shape
    )

    # ---------------------------------------------------------
    # m_i
    # ---------------------------------------------------------

    print(
        "\n===== VISIBILITY COUNTS m_i ====="
    )

    visibility_counts = (
        engine.compute_visibility_counts(
            visibility_matrix
        )
    )

    print(
        f"Minimum m_i: "
        f"{np.min(visibility_counts)}"
    )

    print(
        f"Maximum m_i: "
        f"{np.max(visibility_counts)}"
    )

    print(
        f"Mean m_i: "
        f"{np.mean(visibility_counts):.2f}"
    )

    print(
        f"Vertices with m_i = 0: "
        f"{np.sum(visibility_counts == 0)}"
    )

    # ---------------------------------------------------------
    # alpha_i
    # ---------------------------------------------------------

    print(
        "\n===== VISIBILITY WEIGHTS alpha_i ====="
    )

    visibility_weights = (
        engine.compute_visibility_weights(
            visibility_counts
        )
    )

    print(
        f"Minimum alpha_i: "
        f"{np.min(visibility_weights):.4f}"
    )

    print(
        f"Maximum alpha_i: "
        f"{np.max(visibility_weights):.4f}"
    )

    print(
        f"Mean alpha_i: "
        f"{np.mean(visibility_weights):.4f}"
    )

    # Show the first few m_i / alpha_i pairs.
    print(
        "\nFirst 20 "
        "(m_i, alpha_i) pairs:"
    )

    for i in range(
        min(20, len(visibility_counts))
    ):

        print(
            f"Vertex {i:4d}: "
            f"m_i = {visibility_counts[i]:2d}, "
            f"alpha_i = "
            f"{visibility_weights[i]:.4f}"
        )

    # ---------------------------------------------------------
    # CANDIDATE VIEWPOINT
    # ---------------------------------------------------------

    print(
        "\n===== CANDIDATE VIEW FITNESS ====="
    )

    candidate_camera = camera_poses[0]

    candidate_visibility = (
        engine.compute_visibility_vector(
            candidate_camera.position,
            candidate_camera.yaw,
            candidate_camera.pitch,
        )
    )

    candidate_fitness = (
        engine.compute_visibility_fitness(
            candidate_visibility,
            visibility_weights,
        )
    )

    print(
        f"Candidate viewpoint: 1"
    )

    print(
        f"Visible vertices: "
        f"{np.sum(candidate_visibility)}"
    )

    print(
        f"J_v: "
        f"{candidate_fitness:.4f}"
    )