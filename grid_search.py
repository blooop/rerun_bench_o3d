from itertools import product
from typing import Final

import open3d as o3d
import rerun as rr

pcl_path: Final = "poisson_pc_front.ply"

# Parameters
depth: Final = [7, 8, 9, 10]
width: Final = [0.0, 5.0, 10.0, 15.0]
scale: Final = [1.0, 1.1, 1.2, 1.3, 1.4]
linear_fit: Final = [False, True]

if __name__ == "__main__":
    rr.init("Open3D Poisson reconstruction -- grid search")
    rr.save("poisson_grid.rrd")

    # Load point cloud
    pcd = o3d.io.read_point_cloud(str("poisson_pc_front.ply"))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
    )

    rr.log("pcd", rr.Points3D(positions=pcd.points))

    # Grid search
    for d, w, s, lf in product(depth, width, scale, linear_fit):
        # Run you algorithm
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=d, width=w, scale=s, linear_fit=lf
        )  # Note: this example doesn't make sense, you can't change both depth and width at the same time

        # For each parameter, log it's value to timeline
        rr.set_time_sequence("depth", d)
        rr.set_time_seconds("width", w)
        rr.set_time_seconds("scale", s)
        rr.set_time_sequence("linear_fit", int(lf))
        rr.log(
            "mesh",
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                vertex_normals=mesh.vertex_normals,
                triangle_indices=mesh.triangles,
            ),
        )