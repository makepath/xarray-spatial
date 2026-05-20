"""Parameter coverage for ``compression_level=`` with ``compression='lz4'``.

The level-validation map in ``xrspatial.geotiff.__init__`` advertises a
``(0, 16)`` valid range for ``lz4``, but only the ``deflate`` ``(1, 9)``
and ``zstd`` ``(1, 22)`` ranges had direct round-trip + boundary-error
tests under ``test_compression_level.py``. ``lz4`` shares the same
range-validation call site (the dispatcher's eager numpy path, the dask
streaming path, and ``_write_vrt_tiled`` all share ``_LEVEL_RANGES``),
so a regression that drops ``lz4`` from the table -- or shifts the
range bounds -- would only surface against user code.

This module pins:

* Round-trip integrity at the boundary levels ``0`` and ``16``.
* Round-trip integrity at the documented default (``compression_level=None``)
  via the public ``to_geotiff`` API. The default uses the ``lz4_compress``
  signature default (``level=0``), so the no-arg path must still produce a
  decodable file.
* ValueError on out-of-range levels (``-1`` and ``17``) across both the
  eager (numpy) path and the dask streaming path.
* Tile-row segmentation for dask-streaming inputs: a low-level lz4 file
  and a high-level lz4 file built from the same input both decode to
  the original values bit-exact (lz4 is lossless across its level range).

Cat 4 MEDIUM: parameter coverage gap on numeric parameter with multiple
values where only the default was tested.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


_HAS_LZ4 = importlib.util.find_spec("lz4") is not None
_HAS_DASK = importlib.util.find_spec("dask") is not None

pytestmark = pytest.mark.skipif(not _HAS_LZ4, reason="lz4 package required")


def _make_da(seed: int = 0, shape: tuple = (64, 64)) -> xr.DataArray:
    """Return a small float32 DataArray with reproducible content."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(shape).astype(np.float32)
    return xr.DataArray(arr, dims=["y", "x"])


def _make_compressible(shape: tuple = (128, 128)) -> xr.DataArray:
    """Smooth gradient + small noise; high spatial coherence so level
    differences actually move the needle on compressed size."""
    rng = np.random.default_rng(42)
    y, x = np.mgrid[0: shape[0], 0: shape[1]]
    arr = ((y + x).astype(np.float32)
           + rng.standard_normal(shape).astype(np.float32) * 0.01)
    return xr.DataArray(arr, dims=["y", "x"])


# ---------------------------------------------------------------------------
# Round-trip integrity across the documented level range
# ---------------------------------------------------------------------------


class TestLZ4LevelRoundTrip:
    """Round-trips at the boundaries of the documented ``lz4`` range."""

    @pytest.mark.parametrize("level", [0, 1, 9, 16])
    def test_lz4_level_round_trip(self, level, tmp_path):
        """Every documented level produces a decodable file with exact
        pixel fidelity (lz4 is lossless)."""
        da = _make_da(seed=level)
        path = str(tmp_path / f"lz4_level_{level}.tif")
        to_geotiff(da, path, compression="lz4",
                   compression_level=level,
                   allow_experimental_codecs=True)
        result = open_geotiff(path)
        # lz4 is lossless: assert_array_equal, not assert_allclose.
        np.testing.assert_array_equal(result.values, da.values)

    def test_lz4_default_level_round_trip(self, tmp_path):
        """``compression_level=None`` falls through to ``lz4_compress``'s
        default (``level=0``). Pin the no-arg path so a future signature
        change is caught."""
        da = _make_da(seed=99)
        path = str(tmp_path / "lz4_default.tif")
        to_geotiff(da, path, compression="lz4",
                   allow_experimental_codecs=True)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, da.values)


# ---------------------------------------------------------------------------
# Higher level should not produce a larger file on compressible input
# ---------------------------------------------------------------------------


class TestLZ4LevelSizeEffect:
    """Higher ``compression_level`` yields the same or fewer bytes for
    a compressible input. lz4 supports level 0 (fast) through 16 (HC);
    levels above 0 invoke the high-compression mode."""

    def test_lz4_higher_level_not_larger(self, tmp_path):
        da = _make_compressible()
        path_lo = str(tmp_path / "lz4_lo.tif")
        path_hi = str(tmp_path / "lz4_hi.tif")
        to_geotiff(da, path_lo, compression="lz4", compression_level=0,
                   allow_experimental_codecs=True)
        to_geotiff(da, path_hi, compression="lz4", compression_level=16,
                   allow_experimental_codecs=True)
        size_lo = os.path.getsize(path_lo)
        size_hi = os.path.getsize(path_hi)
        # Allow equality: very small or already-compressed payloads can
        # land at the same byte count. The contract is "no worse".
        assert size_hi <= size_lo, (
            f"Expected level-16 file ({size_hi}) <= level-0 file ({size_lo})")


# ---------------------------------------------------------------------------
# Out-of-range level rejection (eager path)
# ---------------------------------------------------------------------------


class TestLZ4LevelOutOfRange:
    """The ``_LEVEL_RANGES`` table advertises ``lz4: (0, 16)``. Pin the
    rejection path so a future range change does not silently widen the
    accepted band."""

    @pytest.mark.parametrize("level", [-1, -10, 17, 100])
    def test_lz4_out_of_range_level_raises_eager(self, level, tmp_path):
        """Out-of-range level on the numpy/eager path raises with the
        same error message format as deflate/zstd."""
        da = _make_da()
        path = str(tmp_path / "lz4_bad.tif")
        with pytest.raises(ValueError, match="compression_level"):
            to_geotiff(da, path, compression="lz4",
                       compression_level=level,
                       allow_experimental_codecs=True)

    def test_lz4_out_of_range_message_includes_range(self, tmp_path):
        """Error message advertises the valid (0, 16) range so callers
        know the bound. Mirrors ``test_compression_level`` for zstd."""
        da = _make_da()
        path = str(tmp_path / "lz4_bad.tif")
        with pytest.raises(ValueError, match=r"lz4.*\(valid:\s*0-16\)"):
            to_geotiff(da, path, compression="lz4",
                       compression_level=999,
                       allow_experimental_codecs=True)


# ---------------------------------------------------------------------------
# Dask streaming path level handling
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_DASK, reason="dask package required")
class TestLZ4LevelDaskStreaming:
    """The dask streaming branch (``hasattr(raw, 'dask') and not cog``) has
    its own ``_LEVEL_RANGES`` check at a separate call site. Cover both
    accept and reject branches there."""

    def _make_dask_da(self, shape=(64, 64), chunks=(16, 16)):
        import dask.array as da_mod
        rng = np.random.default_rng(7)
        arr = rng.standard_normal(shape).astype(np.float32)
        return xr.DataArray(
            da_mod.from_array(arr, chunks=chunks),
            dims=["y", "x"],
        ), arr

    @pytest.mark.parametrize("level", [0, 1, 8, 16])
    def test_lz4_dask_streaming_level_round_trip(self, level, tmp_path):
        dask_da, np_arr = self._make_dask_da()
        path = str(tmp_path / f"lz4_dask_level_{level}.tif")
        to_geotiff(dask_da, path, compression="lz4",
                   compression_level=level, tile_size=16,
                   allow_experimental_codecs=True)
        result = open_geotiff(path)
        np.testing.assert_array_equal(result.values, np_arr)

    @pytest.mark.parametrize("level", [-1, 17, 50])
    def test_lz4_dask_streaming_out_of_range_raises(self, level, tmp_path):
        dask_da, _ = self._make_dask_da()
        path = str(tmp_path / "lz4_dask_bad.tif")
        with pytest.raises(ValueError, match="compression_level"):
            to_geotiff(dask_da, path, compression="lz4",
                       compression_level=level, tile_size=16,
                       allow_experimental_codecs=True)
