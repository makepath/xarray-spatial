# Hypsometric Integral Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `hypsometric_integral(zones, values)` function to `xrspatial/zonal.py` that computes HI = (mean - min) / (max - min) per zone and paints the result back into a raster.

**Architecture:** A new public function in `zonal.py` with four backend implementations (numpy, cupy, dask+numpy, dask+cupy) following the existing `ArrayTypeFunctionMapping` dispatch pattern. The numpy path uses `np.unique` + boolean masking per zone (simpler than `_sort_and_stride` since we only need min/mean/max; `_sort_and_stride` exists for the generic multi-stat case in `stats()`). The cupy path transfers to host and reuses the numpy path (same pattern as `_stats_dask_cupy`; spec's "device-side scatter/gather" is aspirational, host transfer is fine for this use case). The dask path uses module-level `@delayed` functions for per-block aggregation (min/max/sum/count), reduces to global per-zone HI, then `map_blocks` to paint back (preserving chunk structure). Exposed via the `.xrs` accessor and the top-level `xrspatial` namespace.

**Tech Stack:** numpy, cupy (optional), dask (optional), xarray, pytest

**Spec:** `docs/superpowers/specs/2026-03-24-hypsometric-integral-design.md`

---

### Task 1: Write failing tests for numpy backend

**Files:**
- Create: `xrspatial/tests/test_hypsometric_integral.py`

Every test function needs `@pytest.mark.parametrize("backend", ...)` — there is no global `backend` fixture. This matches the pattern in `test_zonal.py`.

- [ ] **Step 1: Create test file with hand-crafted test cases**

```python
# xrspatial/tests/test_hypsometric_integral.py
try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from .general_checks import create_test_raster


# --- fixtures ---------------------------------------------------------------

@pytest.fixture
def hi_zones(backend):
    """Two zones (1, 2) plus nodata (0).

    Zone 1: 5 cells — (0,1), (0,2), (0,3), (1,1), (1,2)
    Zone 2: 6 cells — (1,3), (2,1), (2,2), (2,3), (3,1), (3,2)
    Nodata: 4 cells — column 0 and (3,3)
    """
    data = np.array([
        [0, 1, 1, 1],
        [0, 1, 1, 2],
        [0, 2, 2, 2],
        [0, 2, 2, 0],
    ], dtype=np.float64)
    return create_test_raster(data, backend, dims=['y', 'x'],
                              attrs={'res': (1.0, 1.0)}, chunks=(2, 2))


@pytest.fixture
def hi_values(backend):
    """Elevation values.

    Zone 1 cells: 10, 20, 30, 40, 50
      min=10, max=50, mean=30, HI=(30-10)/(50-10) = 0.5

    Zone 2 cells: 100, 60, 70, 80, 90, 95
      min=60, max=100, mean=82.5, HI=(82.5-60)/(100-60) = 0.5625
    """
    data = np.array([
        [999., 10., 20., 30.],
        [999., 40., 50., 100.],
        [999., 60., 70., 80.],
        [999., 90., 95., 999.],
    ], dtype=np.float64)
    return create_test_raster(data, backend, dims=['y', 'x'],
                              attrs={'res': (1.0, 1.0)}, chunks=(2, 2))


# --- basic -------------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_basic(backend, hi_zones, hi_values):
    from xrspatial.zonal import hypsometric_integral

    result = hypsometric_integral(hi_zones, hi_values)

    assert isinstance(result, xr.DataArray)
    assert result.shape == hi_values.shape
    assert result.dims == hi_values.dims
    assert result.name == 'hypsometric_integral'

    out = result.values if not (da and isinstance(result.data, da.Array)) else result.compute().values

    # zone 0 (nodata) cells should be NaN
    nodata_mask = np.array([
        [True, False, False, False],
        [True, False, False, False],
        [True, False, False, False],
        [True, False, False, True],
    ])
    assert np.all(np.isnan(out[nodata_mask]))

    # zone 1: HI = 0.5
    z1_mask = np.array([
        [False, True, True, True],
        [False, True, True, False],
        [False, False, False, False],
        [False, False, False, False],
    ])
    np.testing.assert_allclose(out[z1_mask], 0.5, rtol=1e-10)

    # zone 2: HI = 0.5625
    z2_mask = np.array([
        [False, False, False, False],
        [False, False, False, True],
        [False, True, True, True],
        [False, True, True, False],
    ])
    np.testing.assert_allclose(out[z2_mask], 0.5625, rtol=1e-10)


# --- edge cases --------------------------------------------------------------

@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_flat_zone(backend):
    """A zone with all identical values has range=0, so HI should be NaN."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[1, 1], [1, 1]], dtype=np.float64), backend,
        chunks=(2, 2))
    values = create_test_raster(
        np.array([[5.0, 5.0], [5.0, 5.0]]), backend,
        chunks=(2, 2))

    result = hypsometric_integral(zones, values, nodata=0)
    out = result.values if not (da and isinstance(result.data, da.Array)) else result.compute().values
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_nan_in_values(backend):
    """NaN elevation cells should be excluded from per-zone stats."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[1, 1], [1, 1]], dtype=np.float64), backend,
        chunks=(2, 2))
    values = create_test_raster(
        np.array([[10.0, np.nan], [20.0, 30.0]]), backend,
        chunks=(2, 2))

    result = hypsometric_integral(zones, values, nodata=0)
    out = result.values if not (da and isinstance(result.data, da.Array)) else result.compute().values

    # zone 1 finite values: 10, 20, 30 -> HI = (20-10)/(30-10) = 0.5
    # the NaN cell should remain NaN in output
    assert np.isnan(out[0, 1])
    np.testing.assert_allclose(out[0, 0], 0.5, rtol=1e-10)
    np.testing.assert_allclose(out[1, 0], 0.5, rtol=1e-10)
    np.testing.assert_allclose(out[1, 1], 0.5, rtol=1e-10)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_single_cell_zone(backend):
    """A zone with a single cell has range=0, so HI=NaN."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[1, 2]], dtype=np.float64), backend,
        chunks=(1, 2))
    values = create_test_raster(
        np.array([[10.0, 20.0]]), backend,
        chunks=(1, 2))

    result = hypsometric_integral(zones, values, nodata=0)
    out = result.values if not (da and isinstance(result.data, da.Array)) else result.compute().values
    # single cell -> range=0 -> NaN
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_all_nan_zone(backend):
    """A zone whose elevation cells are all NaN should produce NaN."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[1, 1], [2, 2]], dtype=np.float64), backend,
        chunks=(2, 2))
    values = create_test_raster(
        np.array([[np.nan, np.nan], [10.0, 20.0]]), backend,
        chunks=(2, 2))

    result = hypsometric_integral(zones, values, nodata=0)
    out = result.values if not (da and isinstance(result.data, da.Array)) else result.compute().values

    # zone 1: all NaN -> NaN
    assert np.all(np.isnan(out[0, :]))
    # zone 2: HI = (15-10)/(20-10) = 0.5
    np.testing.assert_allclose(out[1, :], 0.5, rtol=1e-10)


@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_nodata_none(backend):
    """When nodata=None, all zone IDs are included (even 0)."""
    from xrspatial.zonal import hypsometric_integral

    zones = create_test_raster(
        np.array([[0, 0], [1, 1]], dtype=np.float64), backend,
        chunks=(2, 2))
    values = create_test_raster(
        np.array([[10.0, 20.0], [30.0, 40.0]]), backend,
        chunks=(2, 2))

    result = hypsometric_integral(zones, values, nodata=None)
    out = result.values if not (da and isinstance(result.data, da.Array)) else result.compute().values

    # zone 0: HI = (15-10)/(20-10) = 0.5
    np.testing.assert_allclose(out[0, :], 0.5, rtol=1e-10)
    # zone 1: HI = (35-30)/(40-30) = 0.5
    np.testing.assert_allclose(out[1, :], 0.5, rtol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest xrspatial/tests/test_hypsometric_integral.py -v -x -k "numpy" --no-header 2>&1 | head -30`
Expected: FAIL — `ImportError: cannot import name 'hypsometric_integral' from 'xrspatial.zonal'`

- [ ] **Step 3: Commit**

```bash
git add xrspatial/tests/test_hypsometric_integral.py
git commit -m "Add failing tests for hypsometric_integral"
```

---

### Task 2: Implement numpy backend and public function

**Files:**
- Modify: `xrspatial/zonal.py` (add `_hi_numpy` and `hypsometric_integral`)

Note: `zonal.py` does not import `functools.partial`. The dispatcher uses lambdas to bind arguments, following the existing pattern in `stats()`.

- [ ] **Step 1: Add numpy backend function to zonal.py**

Add in a new section before the existing `_apply_numpy` helper function (search for `def _apply_numpy`):

```python
# ---------------------------------------------------------------------------
# Hypsometric integral
# ---------------------------------------------------------------------------

def _hi_numpy(zones_data, values_data, nodata):
    """Numpy backend for hypsometric integral."""
    unique_zones = np.unique(zones_data[np.isfinite(zones_data)])
    if nodata is not None:
        unique_zones = unique_zones[unique_zones != nodata]

    out = np.full(values_data.shape, np.nan, dtype=np.float64)

    for z in unique_zones:
        mask = (zones_data == z) & np.isfinite(values_data)
        if not np.any(mask):
            continue
        vals = values_data[mask]
        mn, mx = vals.min(), vals.max()
        if mx == mn:
            continue  # flat zone -> NaN
        hi = (vals.mean() - mn) / (mx - mn)
        out[mask] = hi
    return out
```

- [ ] **Step 2: Add public `hypsometric_integral` function**

Add immediately after `_hi_numpy`:

```python
def hypsometric_integral(
    zones,
    values,
    nodata=0,
    column=None,
    rasterize_kw=None,
    name='hypsometric_integral',
):
    """Hypsometric integral (HI) per zone, painted back to a raster.

    HI measures geomorphic maturity: ``(mean - min) / (max - min)``
    computed over elevations within each zone.  Values range from 0 to 1.

    Parameters
    ----------
    zones : xr.DataArray, GeoDataFrame, or list of (geometry, value) pairs
        Zone definitions.  Integer zone IDs.  GeoDataFrame and list-of-pairs
        inputs are rasterized using *values* as the template grid.
    values : xr.DataArray
        2D elevation raster (float), same shape as *zones*.
    nodata : int or None, default 0
        Zone ID that means "no zone".  Excluded from computation; those
        cells get NaN in the output.  Set to ``None`` to include all IDs.
    column : str, optional
        Column in a GeoDataFrame containing zone IDs.
    rasterize_kw : dict, optional
        Extra keyword arguments for ``rasterize()`` when *zones* is vector.
    name : str, default ``'hypsometric_integral'``
        Name for the output DataArray.

    Returns
    -------
    xr.DataArray
        Float64 raster, same shape/dims/coords as *values*.  Each cell
        holds the HI of its zone.  NaN for nodata zones, non-finite
        elevation cells, and flat zones (elevation range = 0).
    """
    zones = _maybe_rasterize_zones(zones, values, column=column,
                                   rasterize_kw=rasterize_kw)
    validate_arrays(zones, values)

    _nodata = nodata  # capture for closures

    mapper = ArrayTypeFunctionMapping(
        numpy_func=lambda z, v: _hi_numpy(z, v, _nodata),
        cupy_func=not_implemented_func,
        dask_func=not_implemented_func,
        dask_cupy_func=not_implemented_func,
    )

    out = mapper(zones)(zones.data, values.data)

    return xr.DataArray(
        out,
        name=name,
        dims=values.dims,
        coords=values.coords,
        attrs=values.attrs,
    )
```

- [ ] **Step 3: Run numpy tests to verify they pass**

Run: `pytest xrspatial/tests/test_hypsometric_integral.py -v -k "numpy and not dask and not cupy" --no-header 2>&1 | tail -20`
Expected: all `numpy` backend tests PASS

- [ ] **Step 4: Commit**

```bash
git add xrspatial/zonal.py
git commit -m "Add hypsometric_integral with numpy backend"
```

---

### Task 3: Implement cupy backend

**Files:**
- Modify: `xrspatial/zonal.py` (add `_hi_cupy`, update dispatcher)

- [ ] **Step 1: Add cupy backend function**

Add after `_hi_numpy`:

```python
def _hi_cupy(zones_data, values_data, nodata):
    """CuPy backend for hypsometric integral — transfer to host, compute, return."""
    import cupy as cp
    result_np = _hi_numpy(cp.asnumpy(zones_data), cp.asnumpy(values_data), nodata)
    return cp.asarray(result_np)
```

- [ ] **Step 2: Update dispatcher in `hypsometric_integral`**

Change `cupy_func=not_implemented_func` to:

```python
cupy_func=lambda z, v: _hi_cupy(z, v, _nodata),
```

- [ ] **Step 3: Run tests (numpy still passes)**

Run: `pytest xrspatial/tests/test_hypsometric_integral.py -v -k "numpy and not dask and not cupy" --no-header 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add xrspatial/zonal.py
git commit -m "Add cupy backend for hypsometric_integral"
```

---

### Task 4: Implement dask+numpy backend

**Files:**
- Modify: `xrspatial/zonal.py` (add `_hi_dask_numpy`, update dispatcher)

The dask path: (1) compute per-zone min/max/sum/count as delayed tasks across blocks, (2) reduce to global per-zone HI lookup, (3) `map_blocks` to paint HI back per chunk (preserving chunk structure).

- [ ] **Step 1: Add module-level delayed helpers and dask+numpy backend function**

Add three functions after `_hi_cupy`. The `@delayed` helpers are defined at module level to match the existing pattern in `zonal.py` (see `_single_stats_func` at line ~280).

```python
@delayed
def _hi_block_stats(z_block, v_block, uzones):
    """Per-chunk: return (n_zones, 4) array of [min, max, sum, count]."""
    result = np.full((len(uzones), 4), np.nan, dtype=np.float64)
    result[:, 3] = 0  # count starts at 0
    for i, z in enumerate(uzones):
        mask = (z_block == z) & np.isfinite(v_block)
        if not np.any(mask):
            continue
        vals = v_block[mask]
        result[i, 0] = vals.min()
        result[i, 1] = vals.max()
        result[i, 2] = vals.sum()
        result[i, 3] = len(vals)
    return result


@delayed
def _hi_reduce(partials_list, uzones):
    """Reduce per-block stats to global per-zone HI lookup dict."""
    stacked = np.stack(partials_list)  # (n_blocks, n_zones, 4)
    g_min = np.nanmin(stacked[:, :, 0], axis=0)
    g_max = np.nanmax(stacked[:, :, 1], axis=0)
    g_sum = np.nansum(stacked[:, :, 2], axis=0)
    g_count = np.nansum(stacked[:, :, 3], axis=0)

    hi_lookup = {}
    for i, z in enumerate(uzones):
        if g_count[i] == 0 or g_max[i] == g_min[i]:
            hi_lookup[z] = np.nan
        else:
            mean = g_sum[i] / g_count[i]
            hi_lookup[z] = (mean - g_min[i]) / (g_max[i] - g_min[i])
    return hi_lookup


def _hi_dask_numpy(zones_data, values_data, nodata):
    """Dask+numpy backend for hypsometric integral."""
    # Step 1: find all unique zones across all chunks
    unique_zones = _unique_finite_zones(zones_data)
    if nodata is not None:
        unique_zones = unique_zones[unique_zones != nodata]

    if len(unique_zones) == 0:
        return da.full(values_data.shape, np.nan, dtype=np.float64,
                       chunks=values_data.chunks)

    # Step 2: per-block aggregation -> global reduce
    zones_blocks = zones_data.to_delayed().ravel()
    values_blocks = values_data.to_delayed().ravel()

    partials = [
        _hi_block_stats(zb, vb, unique_zones)
        for zb, vb in zip(zones_blocks, values_blocks)
    ]

    # Compute the HI lookup eagerly so map_blocks can use it as a parameter.
    hi_lookup = dask.compute(_hi_reduce(partials, unique_zones))[0]

    # Step 3: paint back using map_blocks (preserves chunk structure)
    def _paint(zones_chunk, values_chunk, hi_map):
        out = np.full(zones_chunk.shape, np.nan, dtype=np.float64)
        for z, hi_val in hi_map.items():
            mask = (zones_chunk == z) & np.isfinite(values_chunk)
            out[mask] = hi_val
        return out

    return da.map_blocks(
        _paint, zones_data, values_data, hi_map=hi_lookup,
        dtype=np.float64, meta=np.array(()),
    )
```

Note: `delayed` and `dask` are already imported at module level in `zonal.py`.

- [ ] **Step 2: Update dispatcher**

Change `dask_func=not_implemented_func` to:

```python
dask_func=lambda z, v: _hi_dask_numpy(z, v, _nodata),
```

- [ ] **Step 3: Run all tests including dask**

Run: `pytest xrspatial/tests/test_hypsometric_integral.py -v -k "numpy or dask" --no-header 2>&1 | tail -20`
Expected: all numpy and dask+numpy tests PASS

- [ ] **Step 4: Commit**

```bash
git add xrspatial/zonal.py
git commit -m "Add dask+numpy backend for hypsometric_integral"
```

---

### Task 5: Implement dask+cupy backend

**Files:**
- Modify: `xrspatial/zonal.py` (add `_hi_dask_cupy`, update dispatcher)

Follows the same pattern as `_stats_dask_cupy`: convert dask+cupy chunks to numpy, delegate to the dask+numpy path.

- [ ] **Step 1: Add dask+cupy backend function**

Add after `_hi_dask_numpy`:

```python
def _hi_dask_cupy(zones_data, values_data, nodata):
    """Dask+cupy backend: convert chunks to numpy, delegate."""
    zones_cpu = zones_data.map_blocks(
        lambda x: x.get(), dtype=zones_data.dtype, meta=np.array(()),
    )
    values_cpu = values_data.map_blocks(
        lambda x: x.get(), dtype=values_data.dtype, meta=np.array(()),
    )
    return _hi_dask_numpy(zones_cpu, values_cpu, nodata)
```

- [ ] **Step 2: Update dispatcher**

Change `dask_cupy_func=not_implemented_func` to:

```python
dask_cupy_func=lambda z, v: _hi_dask_cupy(z, v, _nodata),
```

- [ ] **Step 3: Run tests (numpy and dask still pass)**

Run: `pytest xrspatial/tests/test_hypsometric_integral.py -v -k "numpy and not cupy" --no-header 2>&1 | tail -10`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add xrspatial/zonal.py
git commit -m "Add dask+cupy backend for hypsometric_integral"
```

---

### Task 6: Wire up public exports and accessor

**Files:**
- Modify: `xrspatial/__init__.py` (add export)
- Modify: `xrspatial/accessor.py` (add accessor method)
- Modify: `xrspatial/tests/test_hypsometric_integral.py` (add accessor test)

- [ ] **Step 1: Write failing test for accessor**

Add to the end of `xrspatial/tests/test_hypsometric_integral.py`:

```python
@pytest.mark.parametrize("backend", ['numpy', 'dask+numpy', 'cupy', 'dask+cupy'])
def test_hypsometric_integral_accessor(backend, hi_zones, hi_values):
    """Verify the .xrs accessor method works."""
    result = hi_values.xrs.zonal_hypsometric_integral(hi_zones)
    assert isinstance(result, xr.DataArray)
    assert result.shape == hi_values.shape

    out = result.values if not (da and isinstance(result.data, da.Array)) else result.compute().values
    z1_mask = np.array([
        [False, True, True, True],
        [False, True, True, False],
        [False, False, False, False],
        [False, False, False, False],
    ])
    np.testing.assert_allclose(out[z1_mask], 0.5, rtol=1e-10)


def test_hypsometric_integral_list_of_pairs_zones():
    """Vector zones via list of (geometry, value) pairs."""
    from shapely.geometry import box
    from xrspatial.zonal import hypsometric_integral

    pytest.importorskip("shapely")
    pytest.importorskip("rasterio")

    values_data = np.array([
        [10., 20., 30.],
        [40., 50., 60.],
        [70., 80., 90.],
    ], dtype=np.float64)
    values = xr.DataArray(values_data, dims=['y', 'x'])
    values['y'] = [2.0, 1.0, 0.0]
    values['x'] = [0.0, 1.0, 2.0]
    values.attrs['res'] = (1.0, 1.0)

    # Zone 1 covers left half, zone 2 covers right half
    zones_pairs = [
        (box(-0.5, -0.5, 1.5, 2.5), 1),
        (box(1.5, -0.5, 2.5, 2.5), 2),
    ]

    result = hypsometric_integral(zones_pairs, values, nodata=0)
    assert isinstance(result, xr.DataArray)
    assert result.shape == values.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest xrspatial/tests/test_hypsometric_integral.py::test_hypsometric_integral_accessor -v -k "numpy and not dask and not cupy" --no-header 2>&1 | tail -10`
Expected: FAIL — `AttributeError`

- [ ] **Step 3: Add accessor method to `xrspatial/accessor.py`**

Add after the `zonal_crosstab` method (search for `def zonal_crosstab`):

```python
    def zonal_hypsometric_integral(self, zones, **kwargs):
        from .zonal import hypsometric_integral
        return hypsometric_integral(zones, self._obj, **kwargs)
```

- [ ] **Step 4: Add top-level export to `xrspatial/__init__.py`**

Add after the existing zonal imports (search for `zonal_stats`):

```python
from xrspatial.zonal import hypsometric_integral  # noqa
```

- [ ] **Step 5: Run all tests**

Run: `pytest xrspatial/tests/test_hypsometric_integral.py -v -k "numpy and not dask and not cupy" --no-header 2>&1 | tail -20`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add xrspatial/__init__.py xrspatial/accessor.py xrspatial/tests/test_hypsometric_integral.py
git commit -m "Wire up hypsometric_integral accessor and public export"
```

---

### Task 7: Final validation

**Files:** (none — verification only)

- [ ] **Step 1: Run the full zonal test suite to check for regressions**

Run: `pytest xrspatial/tests/test_zonal.py -v -k "numpy and not dask and not cupy" --no-header -q 2>&1 | tail -10`
Expected: existing zonal tests still PASS

- [ ] **Step 2: Run the hypsometric integral tests one final time**

Run: `pytest xrspatial/tests/test_hypsometric_integral.py -v --no-header 2>&1`
Expected: all PASS (cupy/dask+cupy tests skip if no GPU)

- [ ] **Step 3: Verify import works from top level**

Run: `python -c "from xrspatial import hypsometric_integral; print(hypsometric_integral.__doc__[:60])"`
Expected: prints first 60 chars of docstring without error
