
        marker = o3d.geometry.TriangleMesh.create_sphere(
            radius=CAMERA_MARKER_RADIUS,
            resolution=16,
        )

        marker.translate(
            np.asarray(pose.position, dtype=float)
        )

        marker.paint_uniform_color([
            0.70,
            0.72,
            0.76,
        ])

        marker.compute_vertex_normals()

        geometries.append(marker)

    # --------------------------------------------------------
    # Coordinate frame
    # --------------------------------------------------------

    coordinate_frame = (
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=4.0,
            origin=[0.0, 0.0, 0.0],
        )
    )

    geometries.append(coordinate_frame)

    # --------------------------------------------------------
    # Open3D visualizer
    # --------------------------------------------------------

    print("\nScene created.")
    print("Opening visualization window...")

    visualizer = o3d.visualization.Visualizer()

    visualizer.create_window(
        window_name="NBV Wildlife Detection - Baseline",
        width=1600,
        height=1000,
    )

    for geometry in geometries:
        visualizer.add_geometry(
            geometry,
            reset_bounding_box=False,
        )

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

    # Initial camera view.
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

    print("\nLegend:")
    print("Colored meshes = Forest trees")
    print("Grey spheres  = Baseline UAV viewpoints")
    print("RGB axes      = Coordinate frame")
    print("\nClose the visualization window to exit.")

    visualizer.run()

    visualizer.destroy_window()

    print("\nBaseline visualization closed.")


if __name__ == "__main__":
    main()
