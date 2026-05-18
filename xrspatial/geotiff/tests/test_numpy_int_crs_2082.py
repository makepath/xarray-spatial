"""Numpy integer CRS values must round-trip through the writers (#2082).

``_validate_crs_arg`` accepts ``numbers.Integral`` (incl. ``np.int32`` /
``np.int64`` / ``np.uint16``), but the eager and GPU writers used to
gate EPSG assignment on ``isinstance(crs, int)``. Numpy integers fail
that check, so the value passed validation, fell through the writer
branch, and the file landed on disk with no EPSG and no ``crs_wkt``
tag. Anything reading the file back got no CRS and silently dropped
into the no-georef code path.

These tests pin the round-trip at all three writer call sites that
used to do the gating: ``to_geotiff`` (eager), ``to_geotiff`` with a
``.vrt`` path (the deprecated ``_write_vrt_tiled`` branch), and
``write_geotiff_gpu`` (skipped without cuda).

There is an existing ``test_to_geotiff_accepts_numpy_int_epsg`` in
``test_crs_arg_validation_1971`` that only asserts ``buf.nbytes > 0``.
A file with no CRS still has bytes, so that test passed even with the
bug in place. The asserts below check ``geo_info.epsg`` (or the
file-level ``attrs['crs']``) instead, so a regression to the
silent-drop behaviour trips immediately.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff

pyproj = pytest.importorskip("pyproj")


def _square(dtype=np.float32):
    return xr.DataArray(
        np.zeros((4, 4), dtype=dtype),
        coords={'y': np.arange(4.0), 'x': np.arange(4.0)},
        dims=('y', 'x'),
    )


@pytest.mark.parametrize(
    "np_int",
    [np.int32(4326), np.int64(4326), np.uint16(4326)],
)
def test_to_geotiff_numpy_int_crs_roundtrips(tmp_path, np_int):
    path = str(tmp_path / f"tmp_2082_eager_{type(np_int).__name__}.tif")
    to_geotiff(_square(), path, crs=np_int)
    out = open_geotiff(path)
    assert out.attrs.get('crs') == 4326


@pytest.mark.parametrize(
    "np_int",
    [np.int32(3857), np.int64(3857)],
)
def test_to_geotiff_numpy_int_crs_roundtrips_bytesio(np_int):
    # The BytesIO branch is a separate code path from the path-based
    # writer for some metadata; pin both.
    buf = io.BytesIO()
    to_geotiff(_square(), buf, crs=np_int)
    buf.seek(0)
    out = open_geotiff(buf)
    assert out.attrs.get('crs') == 3857


def test_to_geotiff_vrt_tiled_numpy_int_crs_roundtrips(tmp_path):
    # ``to_geotiff(da, '*.vrt')`` dispatches to ``_write_vrt_tiled``,
    # which has its own crs resolution block (eager.py:871). Pin
    # numpy-int handling there separately. The writer emits per-tile
    # ``tile_NN_NN.tif`` files in a ``<stem>_tiles/`` directory next
    # to the .vrt; read one of those rather than the .vrt so the
    # check exercises the writer-side CRS resolution, not the
    # VRT reader's CRS handling.
    vrt_path = str(tmp_path / "tmp_2082_vrt_tiled.vrt")
    to_geotiff(_square(), vrt_path, crs=np.int64(4326))
    tile_path = str(tmp_path / "tmp_2082_vrt_tiled_tiles" / "tile_00_00.tif")
    out = open_geotiff(tile_path)
    assert out.attrs.get('crs') == 4326


def test_to_geotiff_python_int_crs_still_works(tmp_path):
    # Sanity: the existing plain-int path keeps working. If the
    # writer-side ``numbers.Integral`` widening broke this, every
    # existing caller that passes ``crs=4326`` would regress.
    path = str(tmp_path / "tmp_2082_plain_int.tif")
    to_geotiff(_square(), path, crs=4326)
    out = open_geotiff(path)
    assert out.attrs.get('crs') == 4326


def test_to_geotiff_numpy_int_crs_writes_real_int_to_attrs(tmp_path):
    # Coerce on the writer side so downstream consumers don't see a
    # numpy scalar in ``attrs['crs']``. Mostly a defensive check: if
    # the EPSG branch ever stores ``crs`` raw, ``attrs['crs']`` ends
    # up as ``np.int64(4326)`` instead of ``int(4326)``, which is a
    # serialisation hazard for downstream JSON / dict consumers.
    path = str(tmp_path / "tmp_2082_attrs_int_type.tif")
    to_geotiff(_square(), path, crs=np.int64(4326))
    out = open_geotiff(path)
    crs_val = out.attrs.get('crs')
    assert crs_val == 4326
    assert type(crs_val) is int


# --- GPU path -----------------------------------------------------------

cupy = pytest.importorskip("cupy", reason="GPU writer needs cupy")

try:
    import cupy  # noqa: F401  -- already imported via importorskip
    _GPU_AVAILABLE = True
except ImportError:
    _GPU_AVAILABLE = False


@pytest.mark.skipif(not _GPU_AVAILABLE, reason="cuda not available")
def test_write_geotiff_gpu_numpy_int_crs_roundtrips(tmp_path):
    from xrspatial.geotiff import write_geotiff_gpu

    arr = cupy.zeros((4, 4), dtype=cupy.float32)
    da = xr.DataArray(
        arr,
        coords={'y': np.arange(4.0), 'x': np.arange(4.0)},
        dims=('y', 'x'),
    )
    path = str(tmp_path / "tmp_2082_gpu.tif")
    write_geotiff_gpu(da, path, crs=np.int64(4326))
    out = open_geotiff(path)
    assert out.attrs.get('crs') == 4326
