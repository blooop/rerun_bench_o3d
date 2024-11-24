import rerun as rr
import open3d as o3d
import bencher as bch

rr.init("rerun_pointcloud", spawn=True)


class PoissonParams(bch.ParametrizedSweep):
    """A class to perform a ND grid search on the parameters of create_from_point_cloud_poisson"""

    # INPUT VARIABlES
    depth = bch.IntSweep(
        default=7,
        bounds=[3, 10],
        doc="Maximum depth of the tree that will be used for surface reconstruction.",
    )
    scale = bch.FloatSweep(
        default=1.1,
        bounds=[1.1, 1.4],
        doc="Specifies the ratio between the diameter of the cube used for reconstruction and the diameter of the samples’ bounding cube",
    )

    linear_fit = bch.BoolSweep(
        default=False,
        doc="If true, the reconstructor will use linear interpolation to estimate the positions of iso-vertices",
    )

    # RESULT VARIABLES
    rrd = bch.ResultContainer()

    def __init__(self, **params):
        super().__init__(**params)

        # Load and compute the normals once

        mesh = o3d.io.read_triangle_mesh(o3d.data.BunnyMesh().path)
        mesh.compute_vertex_normals()  # Ensure the normals are computed
# Convert mesh to point cloud
        point_cloud = mesh.sample_points_poisson_disk(
            number_of_points=10000
        )  

        # Estimate normals for the point cloud
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        point_cloud.orient_normals_consistent_tangent_plane(k=30)
        self.pcd = point_cloud


    def __call__(self, **kwargs) -> dict:
        """This function is called with the values of the ND sweep as kwargs

        Returns:
            dict: A dictionary of all the result variables
        """

        self.update_params_from_kwargs(**kwargs)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.pcd, depth=self.depth, width=0, scale=1.1, linear_fit=False
        )

        self.rrd = bch.capture_rerun_window(width=200, height=200)
        rr.log(
            "mesh",
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                vertex_normals=mesh.vertex_normals,
                triangle_indices=mesh.triangles,
            ),
        )

        return super().__call__()  # returns dict of all class parameters


if __name__ == "__main__":
    bch.run_flask_in_thread()  # hack to serve .rrd files

    run_cfg = bch.BenchRunCfg()
    # run_cfg.tag = "eagle"
    run_cfg.level = 3  # how finely to divide the search space.
    # Set to true if you want to store and reuse previously calculate values
    run_cfg.use_sample_cache = True 

    # allows cache to work across sweeps
    run_cfg.only_hash_tag = True

    #Set up grid search using the parameters defined in run_cfg
    bench = PoissonParams().to_bench(run_cfg)

    #redunce sample level otherwise there are too many rerun windows and they start becoming unstable
    run_cfg.level=3
    bench.plot_sweep(input_vars=["depth", "scale", "linear_fit"])


    # TURNED OFF FOR THE MOMENT AS TOO MANY RERUN WINDOWS CRASH CURRENTLY
    #  
    # run_cfg.level=4
    # bench.plot_sweep(input_vars=["depth"])
    # bench.plot_sweep(input_vars=["scale"])
    # bench.plot_sweep(input_vars=["linear_fit"])

    # #These are the values provided in the original example but it crashes when this many rerun windows are loaded at the same time
    # #
    # # sample using specific values
    # bench.plot_sweep(
    #     input_vars=[
    #         bch.p("depth", [7, 8, 9, 10]),
    #         bch.p("scale", [1.0, 1.1, 1.2, 1.3, 1.4]),
    #         "linear_fit",
    #     ],
    #     result_vars=["rrd"],
    # )
    bench.report.show()
