"""Tests for rechunk_no_shuffle."""

import numpy as np
import pytest
import xarray as xr

from xrspatial.utils import rechunk_no_shuffle

da = pytest.importorskip("dask.array")

_has_zarr = True
try:
    import zarr  # noqa: F401
except ImportError:
    _has_zarr = False

requires_zarr = pytest.mark.skipif(not _has_zarr, reason="zarr not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dask_raster(shape=(1024, 1024), chunks=256, dtype=np.float32):
    data = da.zeros(shape, chunks=chunks, dtype=dtype)
    dims = ['y', 'x'] if len(shape) == 2 else [f'd{i}' for i in range(len(shape))]
    return xr.DataArray(data, dims=dims)


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

def test_chunks_are_exact_multiples():
    """New chunks must be an integer multiple of original chunks."""
    raster = _make_dask_raster(shape=(2048, 2048), chunks=128)
    result = rechunk_no_shuffle(raster, target_mb=16)

    for orig, new in zip(raster.chunks, result.chunks):
        base = orig[0]
        # every chunk (except possibly the last) should be a multiple
        for c in new[:-1]:
            assert c % base == 0, f"chunk {c} is not a multiple of {base}"


def test_chunks_grow():
    """Output chunks should be larger than input when target is larger."""
    raster = _make_dask_raster(shape=(2048, 2048), chunks=64)
    result = rechunk_no_shuffle(raster, target_mb=16)
    assert result.chunks[0][0] > raster.chunks[0][0]


def test_already_large_returns_unchanged():
    """If chunks already meet or exceed target, return as-is."""
    raster = _make_dask_raster(shape=(512, 512), chunks=512)
    result = rechunk_no_shuffle(raster, target_mb=0.5)
    assert result.chunks == raster.chunks


def test_3d_input():
    """Works with 3-D arrays (e.g. stacked bands)."""
    raster = _make_dask_raster(shape=(4, 512, 512), chunks=(1, 128, 128))
    result = rechunk_no_shuffle(raster, target_mb=16)
    for orig, new in zip(raster.chunks, result.chunks):
        base = orig[0]
        for c in new[:-1]:
            assert c % base == 0


def test_preserves_values():
    """Rechunked array should contain identical data."""
    np.random.seed(1067)
    data = da.from_array(np.random.rand(256, 256).astype(np.float32), chunks=64)
    raster = xr.DataArray(data, dims=['y', 'x'])
    result = rechunk_no_shuffle(raster, target_mb=1)
    np.testing.assert_array_equal(raster.values, result.values)


def test_preserves_coords_and_attrs():
    """Coordinates and attributes must survive rechunking."""
    data = da.zeros((256, 256), chunks=64, dtype=np.float32)
    raster = xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={'y': np.arange(256), 'x': np.arange(256)},
        attrs={'crs': 'EPSG:4326'},
    )
    result = rechunk_no_shuffle(raster, target_mb=1)
    assert result.attrs == raster.attrs
    xr.testing.assert_equal(result.coords.to_dataset(), raster.coords.to_dataset())


# ---------------------------------------------------------------------------
# Non-dask passthrough
# ---------------------------------------------------------------------------

def test_numpy_passthrough():
    """Numpy-backed DataArray should be returned unchanged."""
    raster = xr.DataArray(np.zeros((100, 100)), dims=['y', 'x'])
    result = rechunk_no_shuffle(raster, target_mb=1)
    assert result is raster


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rejects_non_dataarray():
    with pytest.raises(TypeError, match="expected xr.DataArray"):
        rechunk_no_shuffle(np.zeros((10, 10)))


# ---------------------------------------------------------------------------
# Zarr re-open optimisation
# ---------------------------------------------------------------------------

@requires_zarr
def test_zarr_reopen_reduces_graph(tmp_path):
    """For a fresh zarr Dataset, rechunk should re-open with fewer tasks."""
    path = str(tmp_path / "rns_zarr_reopen.zarr")
    ds = xr.Dataset({"elev": xr.DataArray(
        np.random.rand(100, 100).astype(np.float64), dims=["y", "x"],
        coords={"y": np.arange(100), "x": np.arange(100)},
    )})
    ds.chunk({"y": 10, "x": 10}).to_zarr(path)

    ds_in = xr.open_zarr(path)
    tasks_before = len(ds_in["elev"].data.__dask_graph__())

    ds_out = rechunk_no_shuffle(ds_in, target_mb=1)
    tasks_after = len(ds_out["elev"].data.__dask_graph__())

    # Re-open should produce fewer tasks, not more
    assert tasks_after < tasks_before, (
        f"expected fewer tasks after rechunk, got {tasks_after} >= {tasks_before}"
    )
    # Values must match
    np.testing.assert_array_equal(ds_in["elev"].values, ds_out["elev"].values)


@requires_zarr
def test_zarr_reopen_skipped_after_sel(tmp_path):
    """After .sel(), the graph has >2 layers so re-open is skipped."""
    path = str(tmp_path / "rns_zarr_sel.zarr")
    ds = xr.Dataset({"elev": xr.DataArray(
        np.random.rand(100, 100).astype(np.float64), dims=["y", "x"],
        coords={"y": np.arange(100), "x": np.arange(100)},
    )})
    ds.chunk({"y": 10, "x": 10}).to_zarr(path)

    ds_sel = xr.open_zarr(path).sel(y=slice(10, 50))
    result = rechunk_no_shuffle(ds_sel, target_mb=1)

    # Should still rechunk (values match), just not via re-open
    np.testing.assert_array_equal(ds_sel["elev"].values, result["elev"].values)


def test_dataset_rechunk_fallback():
    """Dataset without zarr backing rechunks via the map() fallback."""
    ds = xr.Dataset({
        "elev": xr.DataArray(
            da.from_array(np.random.rand(100, 100).astype(np.float32),
                          chunks=(10, 10)),
            dims=["y", "x"],
        ),
        "slope": xr.DataArray(
            da.from_array(np.random.rand(100, 100).astype(np.float32),
                          chunks=(10, 10)),
            dims=["y", "x"],
        ),
    })
    result = rechunk_no_shuffle(ds, target_mb=1)
    assert isinstance(result, xr.Dataset)
    for name in ds.data_vars:
        xr.testing.assert_equal(ds[name], result[name])
        # Chunks should be at least as large as the originals.
        for orig, new in zip(ds[name].chunks, result[name].chunks):
            assert new[0] >= orig[0]


def test_rejects_nonpositive_target():
    raster = _make_dask_raster()
    with pytest.raises(ValueError, match="target_mb must be > 0"):
        rechunk_no_shuffle(raster, target_mb=0)
    with pytest.raises(ValueError, match="target_mb must be > 0"):
        rechunk_no_shuffle(raster, target_mb=-1)


# ---------------------------------------------------------------------------
# Accessor integration
# ---------------------------------------------------------------------------

def test_accessor():
    """The .xrs.rechunk_no_shuffle() accessor delegates correctly."""
    import xrspatial  # noqa: F401 — registers accessor
    raster = _make_dask_raster(shape=(1024, 1024), chunks=128)
    direct = rechunk_no_shuffle(raster, target_mb=16)
    via_accessor = raster.xrs.rechunk_no_shuffle(target_mb=16)
    assert direct.chunks == via_accessor.chunks
