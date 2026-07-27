import os
import shutil
import tempfile

import numpy as np
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff

try:
    import cupy
    _has_cupy = True
except ImportError:
    _has_cupy = False

try:
    import dask.array as _da
    _has_dask = True
except ImportError:
    _has_dask = False


def _make_dataarray(ny, nx, backend):
    # Noisy float32 grid tagged with an EPSG CRS so the writer emits a
    # georeferenced file. Valid backends: "numpy", "dask", "cupy". A GPU
    # or dask backend that is unavailable raises NotImplementedError so
    # asv skips the parameter combination instead of failing.
    rng = np.random.default_rng(31607)
    z = rng.normal(0.0, 25.0, (ny, nx)).astype(np.float32)
    x = np.linspace(-180.0, 180.0, nx)
    y = np.linspace(90.0, -90.0, ny)

    if backend == "numpy":
        pass
    elif backend == "cupy":
        if not _has_cupy:
            raise NotImplementedError("CuPy not available")
        z = cupy.asarray(z)
    elif backend == "dask":
        if not _has_dask:
            raise NotImplementedError("dask not available")
        z = _da.from_array(z, chunks=(max(1, ny // 4), nx))
    else:
        raise RuntimeError(f"Unrecognised backend {backend}")

    da = xr.DataArray(z, coords=dict(y=y, x=x), dims=["y", "x"])
    da.attrs["crs"] = 4326
    return da


# -------------------------------------------------------------------------
# Write path (to_geotiff)
#
# numpy is the eager path; dask exercises the tile-row streaming writer;
# cupy exercises the GPU writer. All three are real, distinct code paths.
# -------------------------------------------------------------------------

class WriteGeoTIFF:
    params = ([512, 2048], ["numpy", "dask", "cupy"])
    param_names = ("nx", "backend")

    def setup(self, nx, backend):
        ny = nx // 2
        self.da = _make_dataarray(ny, nx, backend)
        self.dir = tempfile.mkdtemp(prefix="asv_geotiff_write_")
        self.path = os.path.join(self.dir, f"w_{nx}_{backend}.tif")
        self.gpu = backend == "cupy"

    def teardown(self, nx, backend):
        shutil.rmtree(getattr(self, "dir", ""), ignore_errors=True)

    def time_write_zstd(self, nx, backend):
        to_geotiff(self.da, self.path, compression="zstd", gpu=self.gpu)


# -------------------------------------------------------------------------
# COG write path (overview pyramid generation)
#
# cog=True materialises the array and builds an internal overview pyramid,
# a separate code path from the plain tiled writer above. dask is omitted
# because COG output materialises anyway.
# -------------------------------------------------------------------------

class WriteCOG:
    params = ([512, 2048], ["numpy", "cupy"])
    param_names = ("nx", "backend")

    def setup(self, nx, backend):
        ny = nx // 2
        self.da = _make_dataarray(ny, nx, backend)
        self.dir = tempfile.mkdtemp(prefix="asv_geotiff_cog_")
        self.path = os.path.join(self.dir, f"cog_{nx}_{backend}.tif")
        self.gpu = backend == "cupy"

    def teardown(self, nx, backend):
        shutil.rmtree(getattr(self, "dir", ""), ignore_errors=True)

    def time_write_cog(self, nx, backend):
        to_geotiff(self.da, self.path, cog=True,
                   overview_levels=[2, 4, 8], overview_resampling="mean",
                   compression="zstd", gpu=self.gpu)


# -------------------------------------------------------------------------
# COG write with symbology from a dask source (#3695)
#
# cog=True skips the streaming writer, so the color_ramp statistics used to
# be taken by re-running the source graph after the pixels were already
# materialised: the caller's whole pipeline executed twice. The fix folds
# the materialised buffer into the same StreamingStats accumulator the
# streaming path uses. The dask parameter is the one that regresses; numpy
# is kept as the no-graph control, and the no-ramp case below is the
# baseline the ramp variant should stay close to.
# -------------------------------------------------------------------------

class WriteCOGSymbology:
    params = ([512, 2048], ["numpy", "dask"])
    param_names = ("nx", "backend")

    def setup(self, nx, backend):
        ny = nx // 2
        self.da = _make_dataarray(ny, nx, backend)
        self.dir = tempfile.mkdtemp(prefix="asv_geotiff_cog_sym_")
        self.path = os.path.join(self.dir, f"cogsym_{nx}_{backend}.tif")

    def teardown(self, nx, backend):
        shutil.rmtree(getattr(self, "dir", ""), ignore_errors=True)

    def time_write_cog_color_ramp(self, nx, backend):
        to_geotiff(self.da, self.path, cog=True, color_ramp="viridis",
                   overview_levels=[2, 4, 8], overview_resampling="mean",
                   compression="zstd")

    def time_write_cog_no_ramp(self, nx, backend):
        to_geotiff(self.da, self.path, cog=True,
                   overview_levels=[2, 4, 8], overview_resampling="mean",
                   compression="zstd")


# -------------------------------------------------------------------------
# Eager read path (open_geotiff)
#
# The file is written once in setup; the benchmark measures decode +
# array assembly. gpu=True routes through the GPU decoder.
# -------------------------------------------------------------------------

class ReadGeoTIFF:
    params = ([512, 2048], ["numpy", "cupy"])
    param_names = ("nx", "backend")

    def setup(self, nx, backend):
        if backend == "cupy" and not _has_cupy:
            raise NotImplementedError("CuPy not available")
        ny = nx // 2
        self.dir = tempfile.mkdtemp(prefix="asv_geotiff_read_")
        self.path = os.path.join(self.dir, f"r_{nx}.tif")
        to_geotiff(_make_dataarray(ny, nx, "numpy"), self.path,
                   compression="zstd", tiled=True, tile_size=256)
        self.gpu = backend == "cupy"

    def teardown(self, nx, backend):
        shutil.rmtree(getattr(self, "dir", ""), ignore_errors=True)

    def time_read(self, nx, backend):
        open_geotiff(self.path, gpu=self.gpu)


# -------------------------------------------------------------------------
# Chunked (dask-backed) read path
#
# open_geotiff(chunks=) builds a lazy dask-backed DataArray; the benchmark
# materialises it to measure the chunked-read backend, not just graph build.
# -------------------------------------------------------------------------

class ReadGeoTIFFChunked:
    params = ([512, 2048],)
    param_names = ("nx",)

    def setup(self, nx):
        if not _has_dask:
            raise NotImplementedError("dask not available")
        ny = nx // 2
        self.dir = tempfile.mkdtemp(prefix="asv_geotiff_readchunk_")
        self.path = os.path.join(self.dir, f"rc_{nx}.tif")
        to_geotiff(_make_dataarray(ny, nx, "numpy"), self.path,
                   compression="zstd", tiled=True, tile_size=256)
        self.chunks = 256

    def teardown(self, nx):
        shutil.rmtree(getattr(self, "dir", ""), ignore_errors=True)

    def time_read_chunked(self, nx):
        open_geotiff(self.path, chunks=self.chunks).data.compute()
