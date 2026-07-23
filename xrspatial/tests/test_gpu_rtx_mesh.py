"""Tests for the gpu_rtx mesh-building guard (Issue #1378).

``create_triangulation()`` in ``xrspatial/gpu_rtx/mesh_utils.py`` divides
the raster's max width/height by ``cupy.amax(raster.data)`` to produce a
ratio-preserving z-scale.  An all-zero raster (or one whose max is NaN /
non-positive) made the divide produce ``inf`` / ``NaN`` and propagated
garbage geometry into OptiX.

These tests verify that the guard raises ``ValueError`` before any hash
or device-buffer work happens.  They only need cupy (for the cupy.amax
call inside the function); they do not need an actual RTX device because
the guard runs before ``optix.getHash()``.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from xrspatial.utils import has_cuda_and_cupy


pytestmark = pytest.mark.skipif(
    not has_cuda_and_cupy(),
    reason="cupy / CUDA not available",
)


def _cupy_raster(data, xs=None, ys=None):
    """Wrap a numpy array as a cupy-backed DataArray."""
    import cupy

    coords = {}
    if xs is not None:
        coords["x"] = xs
    if ys is not None:
        coords["y"] = ys
    return xr.DataArray(cupy.asarray(data), dims=["y", "x"], coords=coords)


def test_create_triangulation_rejects_all_zero_raster():
    """An all-zero raster has maxH=0.0 -> divide-by-zero -> ValueError."""
    from xrspatial.gpu_rtx.mesh_utils import create_triangulation

    raster = _cupy_raster(np.zeros((8, 8), dtype=np.float32))
    optix = MagicMock()  # never reached: guard runs first

    with pytest.raises(ValueError, match="no positive elevation variance"):
        create_triangulation(raster, optix)

    # The guard must short-circuit before touching optix.
    optix.getHash.assert_not_called()
    optix.build.assert_not_called()


def test_create_triangulation_rejects_all_nan_raster():
    """An all-NaN raster has maxH=nan -> non-finite -> ValueError."""
    from xrspatial.gpu_rtx.mesh_utils import create_triangulation

    raster = _cupy_raster(np.full((8, 8), np.nan, dtype=np.float32))
    optix = MagicMock()

    with pytest.raises(ValueError, match="no positive elevation variance"):
        create_triangulation(raster, optix)

    optix.getHash.assert_not_called()


def test_create_triangulation_rejects_non_positive_maxima():
    """A raster whose only non-NaN values are <= 0 must also be rejected.

    The mesh-scale formula assumes maxH > 0; otherwise the resulting
    scale either inverts the mesh (negative max) or is infinite (zero
    max).  Both are garbage-geometry conditions.
    """
    from xrspatial.gpu_rtx.mesh_utils import create_triangulation

    raster = _cupy_raster(np.full((8, 8), -3.0, dtype=np.float32))
    optix = MagicMock()

    with pytest.raises(ValueError, match="no positive elevation variance"):
        create_triangulation(raster, optix)


def test_create_triangulation_accepts_single_nonzero_pixel():
    """A raster with only one positive pixel must NOT raise.

    maxH is finite and > 0, so the guard passes and the function
    proceeds to mesh building.  We use a fake optix that returns a
    hash matching the data hash, so the build path is skipped and the
    function returns the computed scale without trying to allocate
    device buffers we don't care about for this test.
    """
    from xrspatial.gpu_rtx.mesh_utils import _data_hash, create_triangulation

    data = np.zeros((4, 4), dtype=np.float32)
    data[2, 2] = 5.0  # single non-zero -> maxH = 5.0
    raster = _cupy_raster(data)

    # Make optix.getHash() return the same hash the function will compute
    # for the data, so the `if optixhash != datahash:` block is skipped.
    expected_hash = _data_hash(raster.data)
    optix = MagicMock()
    optix.getHash.return_value = int(expected_hash)

    scale, ew_res, ns_res = create_triangulation(raster, optix)

    # max(H, W) / maxH = 4 / 5.0; no x/y coords -> unit resolution.
    assert scale == pytest.approx(4.0 / 5.0)
    assert ew_res == pytest.approx(1.0)
    assert ns_res == pytest.approx(1.0)
    optix.build.assert_not_called()  # hashes matched, no rebuild


def test_create_triangulation_error_message_includes_max_value():
    """The ValueError message should report the offending max value."""
    from xrspatial.gpu_rtx.mesh_utils import create_triangulation

    raster = _cupy_raster(np.zeros((4, 4), dtype=np.float32))

    with pytest.raises(ValueError) as excinfo:
        create_triangulation(raster, MagicMock())

    msg = str(excinfo.value)
    assert "max=" in msg
    assert "0.0" in msg


# ---------------------------------------------------------------------------
# Resolution-aware mesh geometry (issue #2861)
# ---------------------------------------------------------------------------


def test_cell_resolution_reads_anisotropic_coords():
    """_cell_resolution returns the real ew_res / ns_res from the coords."""
    from xrspatial.gpu_rtx.mesh_utils import _cell_resolution

    ny, nx = 6, 4
    xs = np.arange(nx, dtype=float) * 2.0     # ew_res = 2
    ys = np.arange(ny, dtype=float) * 5.0     # ns_res = 5
    raster = _cupy_raster(np.ones((ny, nx), dtype=np.float32), xs=xs, ys=ys)

    ew_res, ns_res = _cell_resolution(raster)
    assert ew_res == pytest.approx(2.0)
    assert ns_res == pytest.approx(5.0)


def test_cell_resolution_falls_back_to_unit_without_coords():
    """A coordinate-less raster keeps the old unit-spacing behaviour."""
    from xrspatial.gpu_rtx.mesh_utils import _cell_resolution

    raster = _cupy_raster(np.ones((5, 5), dtype=np.float32))
    ew_res, ns_res = _cell_resolution(raster)
    assert ew_res == pytest.approx(1.0)
    assert ns_res == pytest.approx(1.0)


def test_triangulate_places_vertices_at_real_resolution():
    """Mesh vertex x/y use the real cell resolution, not integer indices.

    This is the core of issue #2861: the GPU mesh was built on integer
    grid coordinates and ignored ew_res / ns_res.  The vertices must now
    sit at (col * ew_res, row * ns_res).
    """
    import cupy

    from xrspatial.gpu_rtx.mesh_utils import (_cell_resolution,
                                              _triangulate_terrain)

    ny, nx = 4, 3
    ew_res, ns_res = 2.0, 5.0
    xs = np.arange(nx, dtype=float) * ew_res
    ys = np.arange(ny, dtype=float) * ns_res
    data = np.zeros((ny, nx), dtype=np.float32)
    raster = _cupy_raster(data, xs=xs, ys=ys)

    ew, ns = _cell_resolution(raster)
    verts = cupy.empty(ny * nx * 3, np.float32)
    triangles = cupy.empty((ny - 1) * (nx - 1) * 2 * 3, np.int32)
    _triangulate_terrain(verts, triangles, raster, 1.0, ew, ns)
    cupy.cuda.Device(0).synchronize()

    v = verts.get().reshape(ny * nx, 3)
    for row in range(ny):
        for col in range(nx):
            idx = row * nx + col
            assert v[idx, 0] == pytest.approx(col * ew_res)
            assert v[idx, 1] == pytest.approx(row * ns_res)


def test_create_triangulation_returns_resolution():
    """create_triangulation reports the resolution it built the mesh with."""
    from xrspatial.gpu_rtx.mesh_utils import _data_hash, create_triangulation

    ny, nx = 4, 4
    xs = np.arange(nx, dtype=float) * 3.0
    ys = np.arange(ny, dtype=float) * 7.0
    data = np.zeros((ny, nx), dtype=np.float32)
    data[1, 1] = 5.0
    raster = _cupy_raster(data, xs=xs, ys=ys)

    # Skip the build path by matching the data hash.
    expected_hash = _data_hash(raster.data)
    optix = MagicMock()
    optix.getHash.return_value = int(expected_hash)

    scale, ew_res, ns_res = create_triangulation(raster, optix)
    assert ew_res == pytest.approx(3.0)
    assert ns_res == pytest.approx(7.0)
    # z-scale stays resolution-independent: max(H, W) / maxH = 4 / 5.0.
    assert scale == pytest.approx(4.0 / 5.0)


# ---------------------------------------------------------------------------
# Mesh-cache hash without a full device-to-host copy (issue #3691)
# ---------------------------------------------------------------------------


def test_data_hash_deterministic():
    """The same data always hashes to the same uint64."""
    import cupy

    from xrspatial.gpu_rtx.mesh_utils import _data_hash

    data = cupy.asarray(np.arange(64 * 48, dtype=np.float32).reshape(64, 48))
    h1 = _data_hash(data)
    h2 = _data_hash(cupy.array(data))  # independent copy
    assert h1 == h2
    assert isinstance(h1, np.uint64)


def test_data_hash_detects_corner_change():
    """Perturbing a corner value must change the hash.

    The asv benchmarks defeat mesh caching by changing ``z[-1, -1]``
    (benchmarks/benchmarks/common.py), so the hash has to stay
    sensitive to corner values.
    """
    import cupy

    from xrspatial.gpu_rtx.mesh_utils import _data_hash

    data = cupy.ones((64, 48), dtype=np.float32)
    h_before = _data_hash(data)
    data[-1, -1] = 2.0
    assert _data_hash(data) != h_before


def test_data_hash_detects_interior_change():
    """An interior-only change must alter the hash.

    The old ``hash(str(data.get()))`` hashed numpy's truncated repr, so
    two rasters differing only away from the edges collided.  The
    strided interior sample removes that blind spot for sampled cells.
    """
    import cupy

    from xrspatial.gpu_rtx.mesh_utils import _data_hash

    data = cupy.ones((64, 48), dtype=np.float32)
    h_before = _data_hash(data)
    data[32, 24] = 7.0  # far from every edge, on the stride lattice
    assert _data_hash(data) != h_before


def test_data_hash_distinguishes_dtypes():
    """Same bytes, different dtype -> different hash.

    All-zero float32 and int32 arrays are byte-identical, so this only
    passes if the dtype participates in the digest.
    """
    import cupy

    from xrspatial.gpu_rtx.mesh_utils import _data_hash

    a = cupy.zeros((16, 16), dtype=np.float32)
    b = cupy.zeros((16, 16), dtype=np.int32)
    assert _data_hash(a) != _data_hash(b)


def test_data_hash_stable_across_processes():
    """The digest must not depend on per-process hash salting.

    blake2b over shape/dtype/sample bytes is fully deterministic, so a
    fixed input maps to a fixed uint64.  The built-in ``hash()`` on
    bytes is salted via PYTHONHASHSEED and would fail this test on the
    next run with a different seed.
    """
    import cupy

    from xrspatial.gpu_rtx.mesh_utils import _data_hash

    data = cupy.asarray(
        np.arange(8 * 8, dtype=np.float32).reshape(8, 8))
    assert _data_hash(data) == np.uint64(7205926626332456187)


def test_data_hash_distinguishes_shapes():
    """Same values, different shape -> different hash."""
    import cupy

    from xrspatial.gpu_rtx.mesh_utils import _data_hash

    a = cupy.ones((32, 64), dtype=np.float32)
    b = cupy.ones((64, 32), dtype=np.float32)
    assert _data_hash(a) != _data_hash(b)


def test_triangulate_covers_grid_beyond_first_launch_batch():
    """A single launch must fill vertices for the whole grid.

    Guards the single-launch replacement of the old 100-block batching
    loop (issue #3691): 400 x 300 = 120,000 cells exceeds the 102,400
    threads the old loop dispatched per batch, so this fails if any part
    of the grid stops being covered.
    """
    import cupy

    from xrspatial.gpu_rtx.mesh_utils import _triangulate_terrain

    ny, nx = 400, 300
    ew_res, ns_res = 2.0, 3.0
    data = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx) + 1.0
    raster = xr.DataArray(cupy.asarray(data), dims=["y", "x"])

    verts = cupy.full(ny * nx * 3, np.nan, np.float32)
    triangles = cupy.full((ny - 1) * (nx - 1) * 2 * 3, -1, np.int32)
    _triangulate_terrain(verts, triangles, raster, 1.0, ew_res, ns_res)
    cupy.cuda.Device(0).synchronize()

    v = verts.get().reshape(ny * nx, 3)
    assert not np.isnan(v).any()  # every vertex written
    # Spot-check the last vertex, well past the old first-batch boundary.
    assert v[-1, 0] == pytest.approx((nx - 1) * ew_res)
    assert v[-1, 1] == pytest.approx((ny - 1) * ns_res)
    assert v[-1, 2] == pytest.approx(float(ny * nx))

    t = triangles.get()
    assert (t >= 0).all()  # every triangle index written
    assert t.max() == ny * nx - 1
