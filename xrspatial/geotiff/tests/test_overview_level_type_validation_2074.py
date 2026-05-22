"""Type validation for ``open_geotiff(overview_level=...)``.

The selector in ``_header.select_overview_ifd`` compares ``overview_level``
numerically and indexes a list with it. Without an upfront type check,
``True`` is silently coerced to ``1`` (because ``bool`` is a subclass of
``int`` in Python), so a caller passing a bool by mistake gets back the
first overview level instead of an error. Non-int types like ``str`` and
``float`` leak raw ``TypeError`` messages from the internal comparison
or indexing. See issue #2074.

This module asserts that:

* ``overview_level=True`` / ``False`` raise ``TypeError`` naming ``bool``.
* ``overview_level="0"`` raises ``TypeError`` naming ``str``.
* ``overview_level=1.0`` raises ``TypeError`` naming ``float``.
* ``overview_level=0``, ``1``, and ``None`` continue to work.
"""
from __future__ import annotations

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")


def _write_cog_with_one_overview(path: str) -> np.ndarray:
    """Write a 64x64 single-band TIFF with one half-resolution overview."""
    rng = np.random.RandomState(0x2074)
    arr = rng.randint(0, 256, size=(64, 64), dtype=np.uint8)
    half = arr[::2, ::2]
    with tifffile.TiffWriter(path) as tw:
        tw.write(arr, tile=(32, 32), photometric="minisblack")
        tw.write(half, tile=(32, 32), photometric="minisblack",
                 subfiletype=1)
    return arr


@pytest.fixture
def cog_with_overview(tmp_path):
    path = str(tmp_path / "cog_overview_level_type_2074.tif")
    arr = _write_cog_with_one_overview(path)
    return path, arr


@pytest.mark.parametrize("value", [True, False])
def test_overview_level_bool_raises_typeerror(cog_with_overview, value):
    from xrspatial.geotiff import open_geotiff

    path, _ = cog_with_overview
    with pytest.raises(TypeError, match="bool"):
        open_geotiff(path, overview_level=value)


def test_overview_level_str_raises_typeerror(cog_with_overview):
    from xrspatial.geotiff import open_geotiff

    path, _ = cog_with_overview
    with pytest.raises(TypeError, match="str"):
        open_geotiff(path, overview_level="0")


def test_overview_level_float_raises_typeerror(cog_with_overview):
    from xrspatial.geotiff import open_geotiff

    path, _ = cog_with_overview
    with pytest.raises(TypeError, match="float"):
        open_geotiff(path, overview_level=1.0)


def test_overview_level_zero_succeeds(cog_with_overview):
    from xrspatial.geotiff import open_geotiff

    path, arr = cog_with_overview
    result = open_geotiff(path, overview_level=0)
    assert result.shape == arr.shape


def test_overview_level_one_succeeds(cog_with_overview):
    from xrspatial.geotiff import open_geotiff

    path, arr = cog_with_overview
    result = open_geotiff(path, overview_level=1)
    # Half-resolution overview of a 64x64 source.
    assert result.shape == (arr.shape[0] // 2, arr.shape[1] // 2)


def test_overview_level_none_succeeds(cog_with_overview):
    from xrspatial.geotiff import open_geotiff

    path, arr = cog_with_overview
    result = open_geotiff(path, overview_level=None)
    assert result.shape == arr.shape


@pytest.mark.parametrize("value", [np.int64(0), np.int32(0)])
def test_overview_level_numpy_int_zero_succeeds(cog_with_overview, value):
    """``np.int64`` / ``np.int32`` should be accepted like Python ints."""
    from xrspatial.geotiff import open_geotiff

    path, arr = cog_with_overview
    result = open_geotiff(path, overview_level=value)
    assert result.shape == arr.shape


@pytest.mark.parametrize("value", [np.int64(1), np.int32(1)])
def test_overview_level_numpy_int_one_succeeds(cog_with_overview, value):
    """``np.int64`` / ``np.int32`` reach the overview level just like int."""
    from xrspatial.geotiff import open_geotiff

    path, arr = cog_with_overview
    result = open_geotiff(path, overview_level=value)
    assert result.shape == (arr.shape[0] // 2, arr.shape[1] // 2)


def test_overview_level_typeerror_names_value(cog_with_overview):
    """Error message should name the offending value, not just the type."""
    from xrspatial.geotiff import open_geotiff

    path, _ = cog_with_overview
    with pytest.raises(TypeError) as exc_info:
        open_geotiff(path, overview_level="not-an-int")
    msg = str(exc_info.value)
    assert "str" in msg
    assert "not-an-int" in msg
