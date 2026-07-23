from xrspatial.gpu_rtx import has_rtx

from .common import get_xr_dataarray


class CreateTriangulation:
    # Times the gpu_rtx mesh build (data hash, triangulation kernel and
    # OptiX BVH build) that hillshade(shadows=True) and viewshed pay on
    # every call. Guards the fixes from issue #3691: the data hash must
    # not copy the full raster to the host, and the triangulation kernel
    # must run as a single launch.
    params = [100, 300, 1000, 3000]
    param_names = ("nx",)

    def setup(self, nx):
        if not has_rtx():
            raise NotImplementedError()
        import cupy
        from rtxpy import RTX

        self.RTX = RTX
        self.cupy = cupy
        ny = nx // 2
        self.xr = get_xr_dataarray((ny, nx), "rtxpy",
                                   different_each_call=True)

    def time_create_triangulation(self, nx):
        from xrspatial.gpu_rtx.mesh_utils import create_triangulation

        # A fresh RTX has no cached geometry, so each call rebuilds the
        # mesh, matching what the hillshade and viewshed entry points do.
        create_triangulation(self.xr, self.RTX())
        self.cupy.cuda.Device(0).synchronize()
