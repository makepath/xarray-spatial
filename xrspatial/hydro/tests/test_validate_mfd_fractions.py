"""Tests for issue #2873: MFD public APIs validate fraction VALUES.

The public MFD functions document a fraction-grid contract: each cell's
8 bands are in [0, 1] and sum to either 1.0 (flow) or 0.0
(pit/flat/sink), with all-NaN bands at edges and nodata cells.  Before
this change they only checked the (8, H, W) shape and ran hydrology math
on whatever values they were given.

These tests pin the value validation: negative fractions, band sums that
are neither ~1.0 nor ~0.0, and partial-NaN band patterns all raise a
clear ValueError on the in-memory (numpy / cupy) backends.  Valid grids
from ``flow_direction_mfd`` still pass, and dask inputs skip the eager
value check so laziness is preserved.
"""

import numpy as np
import pytest
import xarray as xr

from xrspatial.hydro.flow_direction_mfd import flow_direction_mfd
from xrspatial.hydro.flow_accumulation_mfd import flow_accumulation_mfd
from xrspatial.hydro.flow_length_mfd import flow_length_mfd
from xrspatial.hydro.stream_order_mfd import stream_order_mfd
from xrspatial.hydro.stream_link_mfd import stream_link_mfd
from xrspatial.hydro.flow_path_mfd import flow_path_mfd
from xrspatial.hydro.hand_mfd import hand_mfd
from xrspatial.hydro.watershed_mfd import watershed_mfd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bowl(n=7):
    """A simple bowl whose center is the lowest cell."""
    y = np.arange(n, dtype=np.float64) - n // 2
    x = np.arange(n, dtype=np.float64) - n // 2
    yy, xx = np.meshgrid(y, x, indexing='ij')
    return xr.DataArray(yy ** 2 + xx ** 2, dims=['y', 'x'])


def _valid_mfd(n=7):
    """A valid MFD fraction grid produced by flow_direction_mfd."""
    return flow_direction_mfd(_bowl(n))


def _flow_accum(mfd):
    return flow_accumulation_mfd(mfd)


def _interior_flow_cell(mfd):
    """Return (r, c) of an interior cell whose bands sum to ~1.0.

    Used to inject corruption into a cell the validator will inspect.
    """
    vals = mfd.values
    sums = np.nansum(vals, axis=0)
    nan_count = np.isnan(vals).sum(axis=0)
    rows, cols = np.where((nan_count == 0) & (np.abs(sums - 1.0) <= 1e-6))
    assert len(rows) > 0, "test grid has no normal flow cell"
    return int(rows[0]), int(cols[0])


def _corrupt(mfd, mutate):
    """Copy *mfd* and apply *mutate(values)* in place, return DataArray."""
    bad = mfd.copy(deep=True)
    mutate(bad.values)
    return bad


# Each entry: (name, callable taking the (possibly corrupt) fraction grid)
def _callers(mfd):
    fa = _flow_accum(_valid_mfd())  # valid accumulation as a secondary arg
    sp = xr.DataArray(np.full(mfd.shape[1:], np.nan), dims=mfd.dims[1:])
    sp.values[0, 0] = 1.0
    pp = sp
    elev = _bowl()
    return [
        ("flow_accumulation_mfd", lambda g: flow_accumulation_mfd(g)),
        ("flow_length_mfd", lambda g: flow_length_mfd(g)),
        ("stream_order_mfd", lambda g: stream_order_mfd(g, fa, threshold=1)),
        ("stream_link_mfd", lambda g: stream_link_mfd(g, fa, threshold=1)),
        ("flow_path_mfd", lambda g: flow_path_mfd(g, sp)),
        ("hand_mfd", lambda g: hand_mfd(g, fa, elev, threshold=1)),
        ("watershed_mfd", lambda g: watershed_mfd(g, pp)),
    ]


# ---------------------------------------------------------------------------
# Valid input still passes
# ---------------------------------------------------------------------------

class TestValidInputPasses:
    def test_all_consumers_accept_valid_grid(self):
        mfd = _valid_mfd()
        for fname, call in _callers(mfd):
            # should not raise
            call(mfd)


# ---------------------------------------------------------------------------
# Negative fractions
# ---------------------------------------------------------------------------

class TestNegativeFractions:
    def test_each_consumer_rejects_negative(self):
        mfd = _valid_mfd()
        r, c = _interior_flow_cell(mfd)

        def mutate(v):
            v[0, r, c] = -0.5
            v[1, r, c] += 0.5  # keep the band sum at 1.0

        bad = _corrupt(mfd, mutate)
        for fname, call in _callers(mfd):
            with pytest.raises(ValueError, match="negative"):
                call(bad)


# ---------------------------------------------------------------------------
# Band sums outside {0, 1}
# ---------------------------------------------------------------------------

class TestBandSums:
    def test_each_consumer_rejects_sum_above_one(self):
        mfd = _valid_mfd()
        r, c = _interior_flow_cell(mfd)
        bad = _corrupt(mfd, lambda v: v.__setitem__((slice(None), r, c), 0.5))
        # 8 bands * 0.5 = 4.0
        for fname, call in _callers(mfd):
            with pytest.raises(ValueError, match="sum"):
                call(bad)

    def test_sink_cell_sum_zero_is_accepted(self):
        # The bowl center is a pit: all 8 bands are 0.0 (sum 0.0).  This
        # must remain valid.
        mfd = _valid_mfd()
        vals = mfd.values
        nan_count = np.isnan(vals).sum(axis=0)
        sums = np.nansum(vals, axis=0)
        has_sink = np.any((nan_count == 0) & (np.abs(sums) <= 1e-6))
        assert has_sink, "test grid has no pit/sink cell"
        flow_accumulation_mfd(mfd)  # should not raise


# ---------------------------------------------------------------------------
# Partial-NaN band pattern
# ---------------------------------------------------------------------------

class TestPartialNaN:
    def test_each_consumer_rejects_partial_nan(self):
        mfd = _valid_mfd()
        r, c = _interior_flow_cell(mfd)

        def mutate(v):
            # NaN one band, leave the rest finite -> partial NaN
            v[0, r, c] = np.nan

        bad = _corrupt(mfd, mutate)
        for fname, call in _callers(mfd):
            with pytest.raises(ValueError, match="partial-NaN"):
                call(bad)

    def test_all_nan_cell_is_accepted(self):
        # Edge cells from flow_direction_mfd are all-NaN; that is valid.
        mfd = _valid_mfd()
        assert np.isnan(mfd.values[:, 0, 0]).all()
        flow_accumulation_mfd(mfd)  # should not raise


# ---------------------------------------------------------------------------
# Dask skips eager value validation (laziness preserved)
# ---------------------------------------------------------------------------

class TestDaskSkipsValueCheck:
    def test_dask_invalid_values_not_validated_eagerly(self):
        dask = pytest.importorskip('dask.array')
        mfd = _valid_mfd()
        r, c = _interior_flow_cell(mfd)
        bad = _corrupt(mfd, lambda v: v.__setitem__((0, r, c), -5.0))
        dask_bad = xr.DataArray(
            dask.from_array(bad.values, chunks=(8, 5, 5)),
            dims=mfd.dims, coords=mfd.coords,
        )
        # Should not raise at validation time: dask value checks are
        # deferred (laziness preserved), not run eagerly here.
        out = flow_accumulation_mfd(dask_bad)
        assert isinstance(out.data, dask.Array)
