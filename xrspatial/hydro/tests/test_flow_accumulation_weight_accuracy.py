"""Accuracy checks for weighted flow accumulation (#3734).

These tests do not compare against hand-built grids.  They check physical
identities that any correct weighted accumulation must satisfy on a real
terrain surface:

1. Mass balance: every unit of weight entering the network leaves through
   exactly one outlet (a pit) or leaks off the valid grid where a cell
   drains into a NaN neighbour.  The leak term is computed from
   ``flow_dir`` alone, independently of the accumulation kernels.
2. Watershed cross-check (D8): the accumulation at a pour point equals
   the weight integrated over the watershed traced *downstream* to that
   point, which is a separate code path from the upstream BFS.
3. Linearity: accumulation is a linear operator in ``weight``.

The weight is an orographic "melt" field (zero below a snowline, growing
with elevation above it) rather than random noise, so the sign and scale
of any error show up in the identities.
"""

import numpy as np
import pytest
import xarray as xr

import xrspatial.accessor  # noqa: F401
from xrspatial.hydro import (
    basin,
    flow_accumulation,
    flow_direction_d8,
    flow_direction_dinf,
    flow_direction_mfd,
    watershed,
)
from xrspatial.hydro.flow_accumulation_d8 import _code_to_offset_py
from xrspatial.hydro.flow_accumulation_dinf import _angle_to_neighbors
from xrspatial.tests.general_checks import (
    cuda_and_cupy_available,
    dask_array_available,
)

_N = 80
_CHUNKS = (25, 25)  # ragged on an 80x80 grid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def dem():
    grid = xr.DataArray(np.zeros((_N, _N)), dims=['y', 'x'])
    return grid.xrs.generate_terrain(seed=3734)


@pytest.fixture(scope='module')
def melt(dem):
    """Orographic melt: 0 below the snowline, linear in elevation above."""
    e = dem.values
    lo, hi = np.nanmin(e), np.nanmax(e)
    snowline = lo + 0.4 * (hi - lo)
    rate = np.clip((e - snowline) / (hi - snowline), 0.0, None) * 12.0
    return xr.DataArray(rate, dims=dem.dims, coords=dem.coords, name='melt')


@pytest.fixture(scope='module')
def glacier(dem):
    """Indicator of glacierised surface: the top 30% of the elevation range."""
    e = dem.values
    lo, hi = np.nanmin(e), np.nanmax(e)
    return xr.DataArray((e > lo + 0.7 * (hi - lo)).astype(np.float64),
                        dims=dem.dims, coords=dem.coords, name='glacier')


@pytest.fixture(scope='module')
def routers(dem):
    return {
        'd8': flow_direction_d8(dem),
        'dinf': flow_direction_dinf(dem),
        'mfd': flow_direction_mfd(dem),
    }


def _to_backend(agg, backend):
    """Move a numpy-backed DataArray to *backend*."""
    if backend == 'numpy':
        return agg
    if backend == 'dask':
        return agg.chunk({'y': _CHUNKS[0], 'x': _CHUNKS[1]})
    import cupy
    out = agg.copy(data=cupy.asarray(agg.values))
    if backend == 'dask+cupy':
        import dask.array as da
        # Keep the 8 MFD bands in one chunk; tile only the spatial axes.
        chunks = ((agg.shape[0],) if agg.ndim == 3 else ()) + _CHUNKS
        out = agg.copy(data=da.from_array(cupy.asarray(agg.values),
                                          chunks=chunks))
    return out


def _to_numpy(agg):
    data = agg.data
    if hasattr(data, 'compute'):
        data = data.compute()
    if hasattr(data, 'get'):
        data = data.get()
    return np.asarray(data)


_BACKENDS = [
    'numpy',
    pytest.param('dask', marks=dask_array_available),
    pytest.param('cupy', marks=cuda_and_cupy_available),
    pytest.param('dask+cupy', marks=cuda_and_cupy_available),
]


# ---------------------------------------------------------------------------
# Leak computation (independent of the accumulation kernels)
# ---------------------------------------------------------------------------

def _valid_at(valid, r, c):
    h, w = valid.shape
    return 0 <= r < h and 0 <= c < w and valid[r, c]


def _outflow_loss_fraction(routing, fdir):
    """Per-cell fraction of outflow that drains into NaN or off the grid.

    Pits have no outflow and are handled separately as outlets.
    """
    if routing == 'mfd':
        frac = fdir  # (8, H, W)
        valid = ~np.isnan(frac[0])
        dy = [0, 1, 1, 1, 0, -1, -1, -1]
        dx = [1, 1, 0, -1, -1, -1, 0, 1]
    else:
        valid = ~np.isnan(fdir)
    h, w = valid.shape
    loss = np.zeros((h, w), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            if not valid[r, c]:
                continue
            if routing == 'd8':
                oy, ox = _code_to_offset_py(fdir[r, c])
                if (oy, ox) != (0, 0) and not _valid_at(valid, r + oy, c + ox):
                    loss[r, c] = 1.0
            elif routing == 'dinf':
                dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(fdir[r, c])
                if w1 > 0 and not _valid_at(valid, r + dy1, c + dx1):
                    loss[r, c] += w1
                if w2 > 0 and not _valid_at(valid, r + dy2, c + dx2):
                    loss[r, c] += w2
            else:
                for k in range(8):
                    f = frac[k, r, c]
                    if f > 0 and not _valid_at(valid, r + dy[k], c + dx[k]):
                        loss[r, c] += f
    return loss


def _pit_mask(routing, fdir):
    if routing == 'd8':
        return fdir == 0
    if routing == 'dinf':
        return fdir == -1.0
    valid = ~np.isnan(fdir[0])
    return valid & (np.nansum(fdir, axis=0) == 0)


# ---------------------------------------------------------------------------
# 1. Mass balance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('routing', ['d8', 'dinf', 'mfd'])
@pytest.mark.parametrize('backend', _BACKENDS)
def test_mass_balance(routers, melt, routing, backend):
    """sum(weight) == sum(accum at pits) + sum(accum * leak fraction)."""
    fdir = routers[routing]
    fd_np = fdir.values
    accum = _to_numpy(flow_accumulation(
        _to_backend(fdir, backend), weight=_to_backend(melt, backend),
        routing=routing))

    valid = ~np.isnan(fd_np[0] if routing == 'mfd' else fd_np)
    assert np.array_equal(np.isnan(accum), ~valid)

    total_in = float(np.sum(melt.values[valid]))
    pits = _pit_mask(routing, fd_np)
    loss = _outflow_loss_fraction(routing, fd_np)
    total_out = float(np.sum(accum[pits])) + float(
        np.nansum(accum * loss))
    assert total_in > 0
    np.testing.assert_allclose(total_out, total_in, rtol=1e-9)


def test_mass_balance_per_basin(routers, melt):
    """D8: weight summed over each basin equals the accumulation at its outlet."""
    fdir = routers['d8']
    accum = flow_accumulation(fdir, weight=melt).values
    basins = basin(fdir).values
    ids = np.unique(basins[~np.isnan(basins)])
    assert ids.size > 10
    for bid in ids:
        members = basins == bid
        # The outlet is the one cell in the basin that everything drains
        # to, so it carries the basin's total.
        outlet_total = float(np.max(accum[members]))
        np.testing.assert_allclose(
            outlet_total, float(np.sum(melt.values[members])), rtol=1e-9)


# ---------------------------------------------------------------------------
# 2. Watershed cross-check
# ---------------------------------------------------------------------------

def _pour_points(fdir, accum, n_channel=8, n_random=8):
    """Row/col pairs: the highest-accumulation cells plus random valid cells."""
    valid = ~np.isnan(fdir)
    flat = np.where(valid, accum, -np.inf).ravel()
    channel = np.argsort(flat)[::-1][:n_channel]
    rng = np.random.default_rng(3734)
    pool = np.flatnonzero(valid.ravel())
    random = rng.choice(pool, size=n_random, replace=False)
    return [np.unravel_index(i, fdir.shape) for i in
            np.concatenate([channel, random])]


@pytest.mark.parametrize('weight_name', ['melt', 'glacier'])
def test_watershed_cross_check(routers, melt, glacier, weight_name):
    """accum[p] == sum(weight over watershed(p)) for many pour points."""
    fdir = routers['d8']
    weight = {'melt': melt, 'glacier': glacier}[weight_name]
    accum = flow_accumulation(fdir, weight=weight).values
    count = flow_accumulation(fdir).values

    for r, c in _pour_points(fdir.values, count):
        pp = xr.full_like(fdir, np.nan)
        pp.values[r, c] = 1.0
        ws = watershed(fdir, pp).values
        inside = ws == 1.0
        assert inside[r, c]
        np.testing.assert_allclose(
            accum[r, c], float(np.sum(weight.values[inside])), rtol=1e-9)
        # The unweighted count is the same identity with weight == 1.
        assert count[r, c] == inside.sum()


def test_glacier_melt_is_zero_without_upstream_glacier(routers, glacier):
    """Cells with no glacier anywhere upstream accumulate exactly zero."""
    fdir = routers['d8']
    accum = flow_accumulation(fdir, weight=glacier).values
    basins = basin(fdir).values
    for bid in np.unique(basins[~np.isnan(basins)]):
        members = basins == bid
        if glacier.values[members].sum() == 0:
            assert np.all(accum[members] == 0.0)
        else:
            assert np.max(accum[members]) == glacier.values[members].sum()


# ---------------------------------------------------------------------------
# 3. Linearity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('routing', ['d8', 'dinf', 'mfd'])
def test_linearity(routers, melt, glacier, routing):
    """accum(a*w1 + b*w2) == a*accum(w1) + b*accum(w2)."""
    fdir = routers[routing]
    a, b = 2.5, -0.75
    combined = a * melt + b * glacier
    lhs = flow_accumulation(fdir, weight=combined, routing=routing).values
    rhs = (a * flow_accumulation(fdir, weight=melt, routing=routing).values
           + b * flow_accumulation(fdir, weight=glacier, routing=routing).values)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize('routing', ['d8', 'dinf', 'mfd'])
def test_monotone_along_flow_paths(routers, melt, routing):
    """With non-negative weight, accumulation never decreases downstream.

    For D8 this holds cell to cell. For D-inf and MFD each downstream
    neighbour receives only a fraction, so check that the total handed
    downstream (accum * fraction) is bounded by the receiver's value.
    """
    fdir = routers[routing]
    fd_np = fdir.values
    accum = flow_accumulation(fdir, weight=melt, routing=routing).values
    valid = ~np.isnan(fd_np[0] if routing == 'mfd' else fd_np)
    h, w = valid.shape
    checked = 0
    for r in range(h):
        for c in range(w):
            if not valid[r, c]:
                continue
            if routing == 'd8':
                targets = [(_code_to_offset_py(fd_np[r, c]), 1.0)]
            elif routing == 'dinf':
                dy1, dx1, w1, dy2, dx2, w2 = _angle_to_neighbors(fd_np[r, c])
                targets = [((dy1, dx1), w1), ((dy2, dx2), w2)]
            else:
                dy = [0, 1, 1, 1, 0, -1, -1, -1]
                dx = [1, 1, 0, -1, -1, -1, 0, 1]
                targets = [((dy[k], dx[k]), fd_np[k, r, c]) for k in range(8)]
            for (oy, ox), f in targets:
                if f <= 0 or (oy, ox) == (0, 0):
                    continue
                if _valid_at(valid, r + oy, c + ox):
                    assert accum[r + oy, c + ox] >= accum[r, c] * f - 1e-9
                    checked += 1
    assert checked > 0
