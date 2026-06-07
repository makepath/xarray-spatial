"""Public API consolidation contract (issue #2960).

The geotiff read/write surface is ``open_geotiff`` / ``to_geotiff``. The
four data backends (``_read_geotiff_dask``, ``_read_geotiff_gpu``,
``_read_vrt``, ``_write_geotiff_gpu``) are private; the dispatchers route
to them from the ``gpu=`` / ``chunks=`` / ``.vrt`` kwargs. The VRT-index
emitter ``_build_vrt`` is also private: ``to_geotiff``'s ``.vrt`` path
reaches it, and it has no DataArray to write (it indexes files that
already exist), so it is not part of the public surface (issue #2974).

These tests pin the new surface and confirm the dispatchers still reach
each backend.
"""
import numpy as np
import pytest

import xrspatial.geotiff as g
from xrspatial.geotiff import _build_vrt, open_geotiff, to_geotiff
from xrspatial.geotiff._backends.dask import _read_geotiff_dask
from xrspatial.geotiff._backends.gpu import _read_geotiff_gpu
from xrspatial.geotiff._backends.vrt import _read_vrt
from xrspatial.geotiff._geotags import GeoTransform
from xrspatial.geotiff._writer import write
from xrspatial.geotiff._writers.gpu import _write_geotiff_gpu

from .._helpers.markers import requires_gpu

_OLD_PUBLIC_NAMES = (
    "read_geotiff_dask",
    "read_geotiff_gpu",
    "read_vrt",
    "write_geotiff_gpu",
    "write_vrt",
)


def test_public_read_write_surface_is_consolidated():
    """The only lowercase ``__all__`` entries are the two public funcs."""
    fns = {name for name in g.__all__ if name[0].islower()}
    assert fns == {"open_geotiff", "to_geotiff"}


def test_build_vrt_is_not_public():
    """``_build_vrt`` is internal: not in ``__all__`` and no public alias."""
    assert "build_vrt" not in g.__all__
    assert "_build_vrt" not in g.__all__
    assert not hasattr(g, "build_vrt")
    # Still importable under its private name for internal callers / tests.
    assert callable(g._build_vrt)


@pytest.mark.parametrize("name", _OLD_PUBLIC_NAMES)
def test_old_backend_names_removed_from_public_namespace(name):
    """The five backend-named functions no longer exist on the package."""
    assert name not in g.__all__
    assert not hasattr(g, name)


def test_private_backends_remain_importable():
    """Made private, not deleted -- direct callers can still reach them."""
    for fn in (_read_geotiff_dask, _read_geotiff_gpu, _read_vrt,
               _write_geotiff_gpu):
        assert callable(fn)


def _two_tile_vrt(tmp_path):
    """Two side-by-side 4x4 tiles indexed by a VRT mosaic; returns the path."""
    left = np.arange(16, dtype=np.float32).reshape(4, 4)
    right = np.arange(16, 32, dtype=np.float32).reshape(4, 4)
    gt_left = GeoTransform(origin_x=0.0, origin_y=4.0,
                           pixel_width=1.0, pixel_height=-1.0)
    gt_right = GeoTransform(origin_x=4.0, origin_y=4.0,
                            pixel_width=1.0, pixel_height=-1.0)
    lpath = str(tmp_path / "left.tif")
    rpath = str(tmp_path / "right.tif")
    write(left, lpath, geo_transform=gt_left, compression="none", tiled=False)
    write(right, rpath, geo_transform=gt_right, compression="none", tiled=False)
    vrt_path = str(tmp_path / "mosaic.vrt")
    _build_vrt(vrt_path, [lpath, rpath])
    return vrt_path


def test_build_vrt_roundtrips_through_open_geotiff(tmp_path):
    """_build_vrt is the internal mosaic builder; open_geotiff reads it back."""
    vrt_path = _two_tile_vrt(tmp_path)
    mosaic = open_geotiff(vrt_path)
    assert mosaic.shape == (4, 8)


def test_open_geotiff_vrt_matches_direct_backend(tmp_path):
    """``.vrt`` source dispatches to the private VRT reader."""
    vrt_path = _two_tile_vrt(tmp_path)
    np.testing.assert_array_equal(
        open_geotiff(vrt_path).values, _read_vrt(vrt_path).values)


def test_open_geotiff_chunks_matches_direct_dask_backend(tmp_path):
    """``chunks=`` dispatches to the private dask reader."""
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    path = str(tmp_path / "src.tif")
    to_geotiff(arr, path, compression="deflate")
    via_dispatch = open_geotiff(path, chunks=4)
    direct = _read_geotiff_dask(path, chunks=4)
    assert via_dispatch.chunks is not None
    np.testing.assert_array_equal(
        via_dispatch.data.compute(), direct.data.compute())


@requires_gpu
def test_open_geotiff_gpu_matches_direct_backend(tmp_path):
    """``gpu=True`` dispatches to the private GPU reader."""
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    path = str(tmp_path / "src.tif")
    to_geotiff(arr, path, compression="deflate")
    via_dispatch = open_geotiff(path, gpu=True)
    direct = _read_geotiff_gpu(path)
    np.testing.assert_array_equal(
        via_dispatch.data.get(), direct.data.get())


@requires_gpu
def test_to_geotiff_gpu_matches_direct_backend(tmp_path):
    """``gpu=True`` on write dispatches to the private GPU writer."""
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    via_path = str(tmp_path / "via.tif")
    direct_path = str(tmp_path / "direct.tif")
    to_geotiff(arr, via_path, gpu=True, compression="deflate")
    _write_geotiff_gpu(arr, direct_path, compression="deflate")
    np.testing.assert_array_equal(
        open_geotiff(via_path).values, open_geotiff(direct_path).values)


def test_top_level_reexports_are_the_subpackage_functions():
    """``from xrspatial import open_geotiff, to_geotiff`` (issue #3005).

    The two public entry points are re-exported from the top-level
    package so they import the same way as every other public function.
    Both spellings must resolve to the same object.
    """
    import xrspatial

    assert xrspatial.open_geotiff is open_geotiff
    assert xrspatial.to_geotiff is to_geotiff
