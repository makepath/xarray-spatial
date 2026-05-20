"""Type validation for ``overview_level`` on the direct backend entry points.

Issue #2074 added the up-front ``_validate_overview_level_arg`` guard to
``open_geotiff`` so callers got a clean ``TypeError`` naming the offending
type (``bool`` / ``str`` / ``float``) instead of a leaked low-level error
from numeric comparison or list indexing. The direct backends
(``read_geotiff_dask``, ``read_geotiff_gpu``) reach the same selector but
only after source coercion, chunk validation, and (on the GPU path)
``on_gpu_failure`` resolution. A caller who passes both a bad
``overview_level`` and a bad source or bad ``chunks=`` got the unrelated
source / chunk error first, so the real defect in the call was hidden.

This module mirrors the issue #2074 tests against the two direct backends
and asserts ordering: the ``overview_level`` type check fires before
``_coerce_path``, ``_validate_chunks_arg``, or the ``on_gpu_failure``
alias handling. See issue #2160.
"""
from __future__ import annotations

import numpy as np
import pytest


tifffile = pytest.importorskip("tifffile")


def _write_cog_with_one_overview(path: str) -> np.ndarray:
    """Write a 64x64 single-band TIFF with one half-resolution overview."""
    rng = np.random.RandomState(0x2160)
    arr = rng.randint(0, 256, size=(64, 64), dtype=np.uint8)
    half = arr[::2, ::2]
    with tifffile.TiffWriter(path) as tw:
        tw.write(arr, tile=(32, 32), photometric="minisblack")
        tw.write(half, tile=(32, 32), photometric="minisblack",
                 subfiletype=1)
    return arr


@pytest.fixture
def cog_with_overview(tmp_path):
    path = str(tmp_path / "cog_overview_backend_2160.tif")
    arr = _write_cog_with_one_overview(path)
    return path, arr


# ---------------------------------------------------------------------------
# read_geotiff_dask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
def test_dask_overview_level_bool_raises_typeerror(cog_with_overview, value):
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = cog_with_overview
    with pytest.raises(TypeError, match="bool"):
        read_geotiff_dask(path, overview_level=value)


def test_dask_overview_level_str_raises_typeerror(cog_with_overview):
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = cog_with_overview
    with pytest.raises(TypeError, match="str"):
        read_geotiff_dask(path, overview_level="0")


def test_dask_overview_level_float_raises_typeerror(cog_with_overview):
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = cog_with_overview
    with pytest.raises(TypeError, match="float"):
        read_geotiff_dask(path, overview_level=1.0)


def test_dask_overview_level_zero_succeeds(cog_with_overview):
    from xrspatial.geotiff import read_geotiff_dask

    path, arr = cog_with_overview
    result = read_geotiff_dask(path, overview_level=0)
    assert result.shape == arr.shape


def test_dask_overview_level_one_succeeds(cog_with_overview):
    from xrspatial.geotiff import read_geotiff_dask

    path, arr = cog_with_overview
    result = read_geotiff_dask(path, overview_level=1)
    assert result.shape == (arr.shape[0] // 2, arr.shape[1] // 2)


def test_dask_overview_level_none_succeeds(cog_with_overview):
    from xrspatial.geotiff import read_geotiff_dask

    path, arr = cog_with_overview
    result = read_geotiff_dask(path, overview_level=None)
    assert result.shape == arr.shape


@pytest.mark.parametrize("value", [np.int64(0), np.int32(0)])
def test_dask_overview_level_numpy_int_zero_succeeds(cog_with_overview, value):
    """``np.int64`` / ``np.int32`` should be accepted like Python ints."""
    from xrspatial.geotiff import read_geotiff_dask

    path, arr = cog_with_overview
    result = read_geotiff_dask(path, overview_level=value)
    assert result.shape == arr.shape


@pytest.mark.parametrize("value", [np.int64(1), np.int32(1)])
def test_dask_overview_level_numpy_int_one_succeeds(cog_with_overview, value):
    from xrspatial.geotiff import read_geotiff_dask

    path, arr = cog_with_overview
    result = read_geotiff_dask(path, overview_level=value)
    assert result.shape == (arr.shape[0] // 2, arr.shape[1] // 2)


def test_dask_overview_level_typeerror_names_value(cog_with_overview):
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = cog_with_overview
    with pytest.raises(TypeError) as exc_info:
        read_geotiff_dask(path, overview_level="not-an-int")
    msg = str(exc_info.value)
    assert "str" in msg
    assert "not-an-int" in msg


# Ordering checks: the overview_level type error must fire before the
# unrelated source / chunk errors, matching open_geotiff's behaviour.


def test_dask_overview_level_check_runs_before_source_coercion():
    """Bad source + bad overview_level should report overview_level first."""
    from xrspatial.geotiff import read_geotiff_dask

    with pytest.raises(TypeError, match="bool"):
        read_geotiff_dask("/nonexistent/path-2160.tif", overview_level=True)


def test_dask_overview_level_check_runs_before_chunks_validation(
        cog_with_overview):
    """Bad chunks + bad overview_level should report overview_level first."""
    from xrspatial.geotiff import read_geotiff_dask

    path, _ = cog_with_overview
    with pytest.raises(TypeError, match="bool"):
        read_geotiff_dask(path, chunks=0, overview_level=True)


# ---------------------------------------------------------------------------
# read_geotiff_gpu
# ---------------------------------------------------------------------------
#
# These tests must not import cupy. The validator runs at the top of
# ``read_geotiff_gpu`` before the cupy import, so the bad-input cases
# raise ``TypeError`` on a CPU-only machine. The "succeeds" cases that
# actually need a GPU stay gated on cupy via ``importorskip``.


@pytest.mark.parametrize("value", [True, False])
def test_gpu_overview_level_bool_raises_typeerror_no_cupy(
        cog_with_overview, value):
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = cog_with_overview
    with pytest.raises(TypeError, match="bool"):
        read_geotiff_gpu(path, overview_level=value)


def test_gpu_overview_level_str_raises_typeerror_no_cupy(cog_with_overview):
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = cog_with_overview
    with pytest.raises(TypeError, match="str"):
        read_geotiff_gpu(path, overview_level="0")


def test_gpu_overview_level_float_raises_typeerror_no_cupy(cog_with_overview):
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = cog_with_overview
    with pytest.raises(TypeError, match="float"):
        read_geotiff_gpu(path, overview_level=1.0)


def test_gpu_overview_level_typeerror_names_value_no_cupy(cog_with_overview):
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = cog_with_overview
    with pytest.raises(TypeError) as exc_info:
        read_geotiff_gpu(path, overview_level="not-an-int")
    msg = str(exc_info.value)
    assert "str" in msg
    assert "not-an-int" in msg


def test_gpu_overview_level_check_runs_before_source_coercion():
    """Bad source + bad overview_level should report overview_level first."""
    from xrspatial.geotiff import read_geotiff_gpu

    with pytest.raises(TypeError, match="bool"):
        read_geotiff_gpu("/nonexistent/path-2160.tif", overview_level=True)


def test_gpu_overview_level_check_runs_before_chunks_validation(
        cog_with_overview):
    """Bad chunks + bad overview_level should report overview_level first."""
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = cog_with_overview
    with pytest.raises(TypeError, match="bool"):
        read_geotiff_gpu(path, chunks=0, overview_level=True)


def test_gpu_overview_level_check_runs_before_on_gpu_failure_validation(
        cog_with_overview):
    """Bad on_gpu_failure + bad overview_level reports overview_level first.

    The ``on_gpu_failure`` alias handling and the ``not in ('auto',
    'strict')`` check sit before the cupy import. Without the up-front
    overview validator, a caller who passes both a bad
    ``on_gpu_failure`` and a bad ``overview_level`` would get the
    ValueError from the policy check, masking the real type bug.
    """
    from xrspatial.geotiff import read_geotiff_gpu

    path, _ = cog_with_overview
    with pytest.raises(TypeError, match="bool"):
        read_geotiff_gpu(path, on_gpu_failure="bogus", overview_level=True)
