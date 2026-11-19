import numpy as np


def focal_lengths_from_fov(
    image_width=640,
    image_height=480,
    horizontal_fov_deg=60.0,
    vertical_fov_deg=45.0
):
    """
    Calculate pixel focal lengths from camera FOV.

    Image dimensions are reproduction assumptions.
    """

    hfov = np.deg2rad(horizontal_fov_deg)
    vfov = np.deg2rad(vertical_fov_deg)

    fx = image_width / (2.0 * np.tan(hfov / 2.0))
    fy = image_height / (2.0 * np.tan(vfov / 2.0))

    return fx, fy


def compute_projection_jacobian(
    vertices,
    pose,
    camera_basis,
    fx,
    fy,
    visible_mask=None
):
    """
    Calculate the 2x3 projection Jacobian for every
    3D vertex with respect to world coordinates.

    camera_basis is VisibilityEngine.camera_basis.
    """

    vertices = np.asarray(vertices, dtype=float)

    # Obtain the camera coordinate axes.
    right, up, forward = camera_basis(
        pose.yaw,
        pose.pitch
    )

    # World-to-camera rotation.
    R_cw = np.vstack((right, up, forward))

    # Transform world points into camera coordinates.
    camera_points = (
        R_cw @ (vertices - pose.position).T
    ).T

    x = camera_points[:, 0]
    y = camera_points[:, 1]
    z = camera_points[:, 2]

    # A point behind or at the camera cannot be projected.
    valid = z > 1e-9

    if visible_mask is not None:
        valid &= np.asarray(visible_mask, dtype=bool)

    number_of_vertices = len(vertices)

    # Local camera-coordinate Jacobians.
    J_local = np.zeros(
        (number_of_vertices, 2, 3),
        dtype=float
    )

    J_local[valid, 0, 0] = fx / z[valid]
    J_local[valid, 0, 2] = (
        -fx * x[valid] / z[valid]**2
    )

    J_local[valid, 1, 1] = fy / z[valid]
    J_local[valid, 1, 2] = (
        -fy * y[valid] / z[valid]**2
    )

    # Convert derivatives into world coordinates.
    A = np.einsum(
        "nij,jk->nik",
        J_local,
        R_cw
    )

    return A


def compute_information_matrices(
    vertices,
    camera_poses,
    visibility_matrix,
    camera_basis,
    fx,
    fy
):
    """
    Calculate one 3x3 information matrix per vertex.

    I_i = sum(A_ki.T @ A_ki)
    """

    vertices = np.asarray(vertices, dtype=float)
    visibility_matrix = np.asarray(
        visibility_matrix,
        dtype=bool
    )

    number_of_vertices = len(vertices)
    number_of_cameras = len(camera_poses)

    if visibility_matrix.shape != (
        number_of_vertices,
        number_of_cameras
    ):
        raise ValueError(
            "Visibility matrix dimensions do not "
            "match vertices and camera poses."
        )

    information = np.zeros(
        (number_of_vertices, 3, 3),
        dtype=float
    )

    for camera_index, pose in enumerate(camera_poses):

        print(
            f"Calculating geometry information: "
            f"camera {camera_index + 1}/{number_of_cameras}"
        )

        visible = visibility_matrix[:, camera_index]

        A = compute_projection_jacobian(
            vertices=vertices,
            pose=pose,
            camera_basis=camera_basis,
            fx=fx,
            fy=fy,
            visible_mask=visible
        )

        # Accumulate A.T @ A for every vertex.
        information += np.einsum(
            "nki,nkj->nij",
            A,
            A
        )

    return information


def evaluate_geometry_fitness(information):
    """
    Evaluate the paper's literal geometry objective:

        J_d = product(det(I_i))

    Singular matrices make the literal product zero.
    """

    information = np.asarray(information, dtype=float)

    eigenvalues = np.linalg.eigvalsh(information)

    largest_eigenvalue = np.maximum(
        eigenvalues[:, -1],
        0.0
    )

    tolerance = np.maximum(
        largest_eigenvalue * 1e-10,
        1e-12
    )

    ranks = np.sum(
        eigenvalues > tolerance[:, None],
        axis=1
    )

    signs, log_determinants = np.linalg.slogdet(
        information
    )

    full_rank = (ranks == 3) & (signs > 0)

    # Only full-rank matrices have a finite,
    # positive determinant.
    valid_log_determinants = log_determinants[full_rank]

    if np.all(full_rank) and len(full_rank) > 0:

        total_log_fitness = np.sum(
            valid_log_determinants
        )

        if total_log_fitness < np.log(np.finfo(float).max):
            paper_fitness = float(np.exp(total_log_fitness))
        else:
            paper_fitness = float("inf")

    else:
        # Literal product is zero if any vertex
        # has a singular information matrix.
        paper_fitness = 0.0

    return {
        "paper_fitness": paper_fitness,
        "full_rank_vertices": int(np.sum(full_rank)),
        "total_vertices": len(information),
        "full_rank_fraction": float(np.mean(full_rank)),
        "vertex_ranks": ranks,
        "log_determinants": log_determinants
    }
