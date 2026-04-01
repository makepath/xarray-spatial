# Multi-Observer Viewshed & Line-of-Sight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `cumulative_viewshed`, `visibility_frequency`, and `line_of_sight` functions to a new `xrspatial/visibility.py` module, with tests, docs, notebook, and README updates.

**Architecture:** New `visibility.py` composes over existing `viewshed()` for multi-observer work and implements Bresenham-based transect extraction for line-of-sight. No changes to `viewshed.py` internals.

**Tech Stack:** numpy, xarray, dask.delayed (for lazy multi-observer), Bresenham's line algorithm, Fresnel zone math.

**Spec:** `docs/superpowers/specs/2026-04-01-multi-observer-viewshed-design.md`

---

### Task 1: Create worktree and scaffold module

**Files:**
- Create: `xrspatial/visibility.py`

- [ ] **Step 1: Create worktree**

```bash
git worktree add .claude/worktrees/issue-1145 -b issue-1145
```

- [ ] **Step 2: Create empty visibility module with docstring**

Create `xrspatial/visibility.py` in the worktree:

```python
"""
Multi-observer viewshed and line-of-sight profile tools.

Functions
---------
cumulative_viewshed
    Count how many observers can see each cell.
visibility_frequency
    Fraction of observers that can see each cell.
line_of_sight
    Elevation profile and visibility along a straight line between two points.
"""
```

- [ ] **Step 3: Commit scaffold**

```bash
git add xrspatial/visibility.py
git commit -m "Add empty visibility module scaffold (#1145)"
```

---

### Task 2: Implement and test `_bresenham_line`

**Files:**
- Modify: `xrspatial/visibility.py`
- Create: `xrspatial/tests/test_visibility.py`

- [ ] **Step 1: Write the failing test**

Create `xrspatial/tests/test_visibility.py`:

```python
import numpy as np
import pytest

from xrspatial.visibility import _bresenham_line


class TestBresenhamLine:
    def test_horizontal(self):
        cells = _bresenham_line(0, 0, 0, 4)
        assert cells == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

    def test_vertical(self):
        cells = _bresenham_line(0, 0, 4, 0)
        assert cells == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]

    def test_diagonal(self):
        cells = _bresenham_line(0, 0, 3, 3)
        assert cells == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_single_cell(self):
        cells = _bresenham_line(2, 3, 2, 3)
        assert cells == [(2, 3)]

    def test_steep_negative(self):
        cells = _bresenham_line(4, 2, 0, 0)
        # Must start at (4, 2) and end at (0, 0)
        assert cells[0] == (4, 2)
        assert cells[-1] == (0, 0)
        assert len(cells) == 5

    def test_includes_endpoints(self):
        cells = _bresenham_line(1, 1, 5, 8)
        assert cells[0] == (1, 1)
        assert cells[-1] == (5, 8)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest xrspatial/tests/test_visibility.py::TestBresenhamLine -v
```

Expected: FAIL with `ImportError` (function doesn't exist yet).

- [ ] **Step 3: Implement `_bresenham_line`**

Add to `xrspatial/visibility.py`:

```python
def _bresenham_line(r0, c0, r1, c1):
    """Return list of (row, col) cells along the line from (r0,c0) to (r1,c1).

    Uses Bresenham's line algorithm. Both endpoints are included.
    """
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest xrspatial/tests/test_visibility.py::TestBresenhamLine -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add xrspatial/visibility.py xrspatial/tests/test_visibility.py
git commit -m "Add Bresenham line algorithm for LOS transects (#1145)"
```

---

### Task 3: Implement and test `_extract_transect`

**Files:**
- Modify: `xrspatial/visibility.py`
- Modify: `xrspatial/tests/test_visibility.py`

- [ ] **Step 1: Write the failing test**

Append to `xrspatial/tests/test_visibility.py`:

```python
import xarray as xr
from xrspatial.visibility import _extract_transect


class TestExtractTransect:
    def _make_raster(self, data):
        h, w = data.shape
        return xr.DataArray(
            data,
            dims=['y', 'x'],
            coords={'y': np.arange(h, dtype=float),
                    'x': np.arange(w, dtype=float)},
        )

    def test_numpy_diagonal(self):
        data = np.arange(25, dtype=float).reshape(5, 5)
        raster = self._make_raster(data)
        cells = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        elev, xs, ys = _extract_transect(raster, cells)
        np.testing.assert_array_equal(elev, [0, 6, 12, 18, 24])
        np.testing.assert_array_equal(xs, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(ys, [0, 1, 2, 3, 4])

    def test_dask_matches_numpy(self):
        import dask.array as da
        data = np.arange(25, dtype=float).reshape(5, 5)
        raster_np = self._make_raster(data)
        raster_dask = raster_np.copy()
        raster_dask.data = da.from_array(data, chunks=(3, 3))
        cells = [(0, 0), (2, 3), (4, 4)]
        elev_np, _, _ = _extract_transect(raster_np, cells)
        elev_da, _, _ = _extract_transect(raster_dask, cells)
        np.testing.assert_array_equal(elev_np, elev_da)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest xrspatial/tests/test_visibility.py::TestExtractTransect -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `_extract_transect`**

Add to `xrspatial/visibility.py`:

```python
import numpy as np
import xarray

from .utils import has_cuda_and_cupy, has_dask_array, is_cupy_array


def _extract_transect(raster, cells):
    """Extract elevation, x-coords, and y-coords for a list of (row, col) cells.

    For dask or cupy-backed rasters the values are pulled to numpy.
    Returns (elevations, x_coords, y_coords) as 1-D numpy arrays.
    """
    rows = np.array([r for r, c in cells])
    cols = np.array([c for r, c in cells])

    x_coords = raster.coords['x'].values[cols]
    y_coords = raster.coords['y'].values[rows]

    data = raster.data
    if has_dask_array():
        import dask.array as da
        if isinstance(data, da.Array):
            data = data.compute()
    if has_cuda_and_cupy() and is_cupy_array(data):
        data = data.get()

    elevations = data[rows, cols].astype(np.float64)
    return elevations, x_coords, y_coords
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest xrspatial/tests/test_visibility.py::TestExtractTransect -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add xrspatial/visibility.py xrspatial/tests/test_visibility.py
git commit -m "Add transect extraction helper for LOS profiles (#1145)"
```

---

### Task 4: Implement and test `line_of_sight`

**Files:**
- Modify: `xrspatial/visibility.py`
- Modify: `xrspatial/tests/test_visibility.py`

- [ ] **Step 1: Write the failing tests**

Append to `xrspatial/tests/test_visibility.py`:

```python
from xrspatial.visibility import line_of_sight


def _make_raster(data):
    """Module-level helper for creating test rasters."""
    h, w = data.shape
    return xr.DataArray(
        data,
        dims=['y', 'x'],
        coords={'y': np.arange(h, dtype=float),
                'x': np.arange(w, dtype=float)},
    )


class TestLineOfSight:
    def test_flat_terrain_all_visible(self):
        data = np.zeros((5, 10), dtype=float)
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=2, x1=9, y1=2,
                               observer_elev=10, target_elev=10)
        assert isinstance(result, xr.Dataset)
        assert 'visible' in result
        assert 'elevation' in result
        assert 'los_height' in result
        assert 'distance' in result
        assert result['visible'].all()

    def test_obstruction_blocks_view(self):
        data = np.zeros((1, 10), dtype=float)
        data[0, 5] = 100  # tall wall in the middle
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=0, x1=9, y1=0,
                               observer_elev=1, target_elev=0)
        vis = result['visible'].values
        # observer cell is visible
        assert vis[0]
        # cells before the wall are visible
        assert all(vis[:6])
        # at least some cells after the wall are blocked
        assert not all(vis[6:])

    def test_observer_equals_target(self):
        data = np.ones((5, 5), dtype=float)
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=2, y0=2, x1=2, y1=2)
        assert len(result['sample']) == 1
        assert result['visible'].values[0]

    def test_elevation_offsets(self):
        data = np.zeros((1, 5), dtype=float)
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=0, x1=4, y1=0,
                               observer_elev=10, target_elev=20)
        los = result['los_height'].values
        # LOS starts at 10, ends at 20
        assert abs(los[0] - 10.0) < 1e-10
        assert abs(los[-1] - 20.0) < 1e-10

    def test_distance_monotonic(self):
        data = np.zeros((5, 10), dtype=float)
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=0, x1=9, y1=4)
        d = result['distance'].values
        assert all(d[i] <= d[i + 1] for i in range(len(d) - 1))

    def test_fresnel_zone(self):
        data = np.zeros((1, 11), dtype=float)
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=0, x1=10, y1=0,
                               observer_elev=50, target_elev=50,
                               frequency_mhz=900)
        assert 'fresnel_radius' in result
        assert 'fresnel_clear' in result
        # midpoint has largest Fresnel radius
        fr = result['fresnel_radius'].values
        mid = len(fr) // 2
        assert fr[mid] >= fr[1]
        assert fr[mid] >= fr[-2]
        # with 50m clearance and flat terrain, Fresnel should be clear
        assert result['fresnel_clear'].all()

    def test_no_fresnel_by_default(self):
        data = np.zeros((5, 5), dtype=float)
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=0, x1=4, y1=4)
        assert 'fresnel_radius' not in result
        assert 'fresnel_clear' not in result

    def test_xy_coords_in_output(self):
        data = np.zeros((5, 10), dtype=float)
        raster = _make_raster(data)
        result = line_of_sight(raster, x0=0, y0=2, x1=9, y1=2)
        # first point should match observer
        assert abs(result['x'].values[0] - 0.0) < 1e-10
        assert abs(result['y'].values[0] - 2.0) < 1e-10
        # last point should match target
        assert abs(result['x'].values[-1] - 9.0) < 1e-10
        assert abs(result['y'].values[-1] - 2.0) < 1e-10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest xrspatial/tests/test_visibility.py::TestLineOfSight -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `line_of_sight`**

Add to `xrspatial/visibility.py`:

```python
from .utils import _validate_raster

SPEED_OF_LIGHT = 299_792_458.0  # m/s


def _fresnel_radius_1(d1, d2, freq_hz):
    """First Fresnel zone radius at a point d1 from transmitter, d2 from receiver."""
    D = d1 + d2
    if D == 0 or freq_hz == 0:
        return 0.0
    wavelength = SPEED_OF_LIGHT / freq_hz
    return np.sqrt(wavelength * d1 * d2 / D)


def line_of_sight(
    raster: xarray.DataArray,
    x0: float, y0: float,
    x1: float, y1: float,
    observer_elev: float = 0,
    target_elev: float = 0,
    frequency_mhz: float = None,
) -> xarray.Dataset:
    """Compute elevation profile and visibility along a straight line.

    Parameters
    ----------
    raster : xarray.DataArray
        Elevation raster.
    x0, y0 : float
        Observer location in data-space coordinates.
    x1, y1 : float
        Target location in data-space coordinates.
    observer_elev : float
        Height above terrain at the observer.
    target_elev : float
        Height above terrain at the target.
    frequency_mhz : float, optional
        Radio frequency in MHz. When set, first Fresnel zone clearance
        is computed at each sample point.

    Returns
    -------
    xarray.Dataset
        Dataset with dimension ``sample`` containing variables
        ``distance``, ``elevation``, ``los_height``, ``visible``,
        ``x``, ``y``, and optionally ``fresnel_radius`` and
        ``fresnel_clear``.
    """
    _validate_raster(raster, func_name='line_of_sight', name='raster')

    x_coords = raster.coords['x'].values
    y_coords = raster.coords['y'].values

    # snap to nearest grid cell
    c0 = int(np.argmin(np.abs(x_coords - x0)))
    r0 = int(np.argmin(np.abs(y_coords - y0)))
    c1 = int(np.argmin(np.abs(x_coords - x1)))
    r1 = int(np.argmin(np.abs(y_coords - y1)))

    cells = _bresenham_line(r0, c0, r1, c1)
    elevations, xs, ys = _extract_transect(raster, cells)

    n = len(cells)

    # compute cumulative distance along the transect
    distance = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        distance[i] = distance[i - 1] + np.sqrt(dx * dx + dy * dy)

    total_dist = distance[-1] if n > 1 else 0.0

    # LOS height: linear interpolation from observer to target
    obs_h = elevations[0] + observer_elev
    tgt_h = elevations[-1] + target_elev if n > 1 else obs_h
    if total_dist > 0:
        los_height = obs_h + (tgt_h - obs_h) * (distance / total_dist)
    else:
        los_height = np.array([obs_h])

    # visibility: track max elevation angle from observer
    visible = np.ones(n, dtype=bool)
    max_angle = -np.inf
    for i in range(1, n):
        if distance[i] == 0:
            continue
        angle = (elevations[i] - obs_h) / distance[i]
        if angle >= max_angle:
            max_angle = angle
        else:
            visible[i] = False

    data_vars = {
        'distance': ('sample', distance),
        'elevation': ('sample', elevations),
        'los_height': ('sample', los_height),
        'visible': ('sample', visible),
        'x': ('sample', xs),
        'y': ('sample', ys),
    }

    if frequency_mhz is not None:
        freq_hz = frequency_mhz * 1e6
        fresnel = np.zeros(n, dtype=np.float64)
        fresnel_clear = np.ones(n, dtype=bool)
        for i in range(n):
            d1 = distance[i]
            d2 = total_dist - d1
            fresnel[i] = _fresnel_radius_1(d1, d2, freq_hz)
            clearance = los_height[i] - elevations[i]
            if clearance < fresnel[i]:
                fresnel_clear[i] = False
        data_vars['fresnel_radius'] = ('sample', fresnel)
        data_vars['fresnel_clear'] = ('sample', fresnel_clear)

    return xarray.Dataset(data_vars)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest xrspatial/tests/test_visibility.py::TestLineOfSight -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add xrspatial/visibility.py xrspatial/tests/test_visibility.py
git commit -m "Add line_of_sight with Fresnel zone support (#1145)"
```

---

### Task 5: Implement and test `cumulative_viewshed`

**Files:**
- Modify: `xrspatial/visibility.py`
- Modify: `xrspatial/tests/test_visibility.py`

- [ ] **Step 1: Write the failing tests**

Append to `xrspatial/tests/test_visibility.py`:

```python
import dask.array as da
from xrspatial.visibility import cumulative_viewshed


class TestCumulativeViewshed:
    def test_flat_terrain_all_visible(self):
        """On flat terrain with elevated observers, every cell is visible."""
        data = np.zeros((10, 10), dtype=float)
        raster = _make_raster(data)
        observers = [
            {'x': 2.0, 'y': 2.0, 'observer_elev': 10},
            {'x': 7.0, 'y': 7.0, 'observer_elev': 10},
        ]
        result = cumulative_viewshed(raster, observers)
        assert result.dtype == np.int32
        # every cell should be seen by both observers
        assert (result.values == 2).all()

    def test_single_observer_matches_viewshed(self):
        """Single-observer cumulative should match binary viewshed."""
        from xrspatial import viewshed
        from xrspatial.viewshed import INVISIBLE
        data = np.random.RandomState(42).rand(15, 15).astype(float) * 100
        raster = _make_raster(data)
        obs = {'x': 7.0, 'y': 7.0, 'observer_elev': 50}
        result = cumulative_viewshed(raster, [obs])
        vs = viewshed(raster, x=7.0, y=7.0, observer_elev=50)
        expected = (vs.values != INVISIBLE).astype(np.int32)
        np.testing.assert_array_equal(result.values, expected)

    def test_wall_blocks_one_side(self):
        """A tall wall blocks visibility from the other side."""
        data = np.zeros((1, 11), dtype=float)
        data[0, 5] = 1000  # tall wall
        raster = _make_raster(data)
        obs_left = {'x': 0.0, 'y': 0.0, 'observer_elev': 1}
        obs_right = {'x': 10.0, 'y': 0.0, 'observer_elev': 1}
        result = cumulative_viewshed(raster, [obs_left, obs_right])
        # the wall cell itself is visible to both
        assert result.values[0, 5] == 2
        # cells immediately next to wall on each side visible to both
        # cells far from wall visible to only one observer
        assert result.values[0, 0] >= 1
        assert result.values[0, 10] >= 1

    def test_per_observer_max_distance(self):
        """Per-observer max_distance limits the analysis radius."""
        data = np.zeros((20, 20), dtype=float)
        raster = _make_raster(data)
        obs = {'x': 10.0, 'y': 10.0, 'observer_elev': 10, 'max_distance': 3}
        result = cumulative_viewshed(raster, [obs])
        # corners should be 0 (beyond max_distance)
        assert result.values[0, 0] == 0
        assert result.values[19, 19] == 0
        # center should be 1
        assert result.values[10, 10] == 1

    def test_empty_observers_raises(self):
        data = np.zeros((5, 5), dtype=float)
        raster = _make_raster(data)
        with pytest.raises(ValueError):
            cumulative_viewshed(raster, [])

    def test_dask_matches_numpy(self):
        """Dask backend should produce the same result as numpy."""
        data = np.random.RandomState(99).rand(15, 15).astype(float) * 50
        raster_np = _make_raster(data)
        raster_dask = raster_np.copy()
        raster_dask.data = da.from_array(data, chunks=(8, 8))
        observers = [
            {'x': 3.0, 'y': 3.0, 'observer_elev': 30},
            {'x': 12.0, 'y': 12.0, 'observer_elev': 30},
        ]
        result_np = cumulative_viewshed(raster_np, observers)
        result_dask = cumulative_viewshed(raster_dask, observers)
        np.testing.assert_array_equal(result_np.values, result_dask.values)

    def test_preserves_coords_and_dims(self):
        data = np.zeros((5, 5), dtype=float)
        raster = _make_raster(data)
        raster.attrs['crs'] = 'EPSG:4326'
        observers = [{'x': 2.0, 'y': 2.0, 'observer_elev': 10}]
        result = cumulative_viewshed(raster, observers)
        assert result.dims == raster.dims
        np.testing.assert_array_equal(result.coords['x'].values,
                                      raster.coords['x'].values)
        np.testing.assert_array_equal(result.coords['y'].values,
                                      raster.coords['y'].values)
        assert result.attrs.get('crs') == 'EPSG:4326'
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest xrspatial/tests/test_visibility.py::TestCumulativeViewshed -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `cumulative_viewshed`**

Add to `xrspatial/visibility.py`:

```python
from .viewshed import viewshed, INVISIBLE


def cumulative_viewshed(
    raster: xarray.DataArray,
    observers: list,
    target_elev: float = 0,
    max_distance: float = None,
) -> xarray.DataArray:
    """Count how many observers can see each cell.

    Parameters
    ----------
    raster : xarray.DataArray
        Elevation raster (numpy, cupy, or dask-backed).
    observers : list of dict
        Each dict must have ``x`` and ``y`` keys (data-space coords).
        Optional keys: ``observer_elev`` (default 0), ``target_elev``
        (overrides function-level default), ``max_distance`` (per-observer
        analysis radius).
    target_elev : float
        Default target elevation for observers that don't specify one.
    max_distance : float, optional
        Default maximum analysis radius.

    Returns
    -------
    xarray.DataArray
        Integer raster (int32) with the count of observers that have
        line-of-sight to each cell.
    """
    _validate_raster(raster, func_name='cumulative_viewshed', name='raster')
    if not observers:
        raise ValueError("observers list must not be empty")

    count = np.zeros(raster.shape, dtype=np.int32)

    for obs in observers:
        ox = obs['x']
        oy = obs['y']
        oe = obs.get('observer_elev', 0)
        te = obs.get('target_elev', target_elev)
        md = obs.get('max_distance', max_distance)

        vs = viewshed(raster, x=ox, y=oy, observer_elev=oe,
                      target_elev=te, max_distance=md)

        mask = vs.values if isinstance(vs.data, np.ndarray) else vs.values
        count += (mask != INVISIBLE).astype(np.int32)

    return xarray.DataArray(count, coords=raster.coords,
                            dims=raster.dims, attrs=raster.attrs)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest xrspatial/tests/test_visibility.py::TestCumulativeViewshed -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add xrspatial/visibility.py xrspatial/tests/test_visibility.py
git commit -m "Add cumulative_viewshed for multi-observer analysis (#1145)"
```

---

### Task 6: Implement and test `visibility_frequency`

**Files:**
- Modify: `xrspatial/visibility.py`
- Modify: `xrspatial/tests/test_visibility.py`

- [ ] **Step 1: Write the failing tests**

Append to `xrspatial/tests/test_visibility.py`:

```python
from xrspatial.visibility import visibility_frequency


class TestVisibilityFrequency:
    def test_flat_terrain_all_ones(self):
        data = np.zeros((10, 10), dtype=float)
        raster = _make_raster(data)
        observers = [
            {'x': 2.0, 'y': 2.0, 'observer_elev': 10},
            {'x': 7.0, 'y': 7.0, 'observer_elev': 10},
        ]
        result = visibility_frequency(raster, observers)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result.values, 1.0)

    def test_equals_cumulative_divided_by_n(self):
        data = np.random.RandomState(7).rand(15, 15).astype(float) * 100
        raster = _make_raster(data)
        observers = [
            {'x': 3.0, 'y': 3.0, 'observer_elev': 50},
            {'x': 10.0, 'y': 10.0, 'observer_elev': 50},
            {'x': 7.0, 'y': 2.0, 'observer_elev': 50},
        ]
        freq = visibility_frequency(raster, observers)
        cum = cumulative_viewshed(raster, observers)
        expected = cum.values.astype(np.float64) / 3.0
        np.testing.assert_allclose(freq.values, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest xrspatial/tests/test_visibility.py::TestVisibilityFrequency -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `visibility_frequency`**

Add to `xrspatial/visibility.py`:

```python
def visibility_frequency(
    raster: xarray.DataArray,
    observers: list,
    target_elev: float = 0,
    max_distance: float = None,
) -> xarray.DataArray:
    """Fraction of observers that can see each cell.

    Parameters are the same as :func:`cumulative_viewshed`.

    Returns
    -------
    xarray.DataArray
        Float64 raster with values in [0, 1].
    """
    cum = cumulative_viewshed(raster, observers, target_elev, max_distance)
    freq = cum.astype(np.float64) / len(observers)
    return freq
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest xrspatial/tests/test_visibility.py::TestVisibilityFrequency -v
```

Expected: all 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add xrspatial/visibility.py xrspatial/tests/test_visibility.py
git commit -m "Add visibility_frequency wrapper (#1145)"
```

---

### Task 7: Integration — exports, accessor, docs, README

**Files:**
- Modify: `xrspatial/__init__.py:121` (add imports after viewshed import)
- Modify: `xrspatial/accessor.py:213` (add accessor methods after viewshed)
- Modify: `docs/source/reference/surface.rst:82` (add section after viewshed)
- Modify: `README.md:318` (add rows after viewshed row)

- [ ] **Step 1: Add imports to `__init__.py`**

After line 121 (`from xrspatial.viewshed import viewshed  # noqa`), add:

```python
from xrspatial.visibility import cumulative_viewshed  # noqa
from xrspatial.visibility import line_of_sight  # noqa
from xrspatial.visibility import visibility_frequency  # noqa
```

- [ ] **Step 2: Add accessor methods to `accessor.py`**

After the `viewshed` method (line 213), add:

```python
    def cumulative_viewshed(self, observers, **kwargs):
        from .visibility import cumulative_viewshed
        return cumulative_viewshed(self._obj, observers, **kwargs)

    def visibility_frequency(self, observers, **kwargs):
        from .visibility import visibility_frequency
        return visibility_frequency(self._obj, observers, **kwargs)

    def line_of_sight(self, x0, y0, x1, y1, **kwargs):
        from .visibility import line_of_sight
        return line_of_sight(self._obj, x0, y0, x1, y1, **kwargs)
```

- [ ] **Step 3: Update docs reference**

In `docs/source/reference/surface.rst`, after the Viewshed section (line 82), add:

```rst

Cumulative Viewshed
===================
.. autosummary::
    :toctree: _autosummary

    xrspatial.visibility.cumulative_viewshed

Visibility Frequency
====================
.. autosummary::
    :toctree: _autosummary

    xrspatial.visibility.visibility_frequency

Line of Sight
=============
.. autosummary::
    :toctree: _autosummary

    xrspatial.visibility.line_of_sight
```

- [ ] **Step 4: Update README feature matrix**

After the Viewshed row (line 318), add:

```markdown
| [Cumulative Viewshed](xrspatial/visibility.py) | Counts how many observers can see each cell | Custom | ✅️ | 🔄 | 🔄 | 🔄 |
| [Visibility Frequency](xrspatial/visibility.py) | Fraction of observers with line-of-sight to each cell | Custom | ✅️ | 🔄 | 🔄 | 🔄 |
| [Line of Sight](xrspatial/visibility.py) | Elevation profile and visibility along a point-to-point transect | Custom | ✅️ | 🔄 | 🔄 | 🔄 |
```

(These inherit viewshed's backend support. CuPy/Dask marked as 🔄 because cumulative_viewshed falls back through `viewshed()` and LOS always runs on CPU.)

- [ ] **Step 5: Run full test suite to verify nothing is broken**

```bash
pytest xrspatial/tests/test_visibility.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add xrspatial/__init__.py xrspatial/accessor.py docs/source/reference/surface.rst README.md
git commit -m "Add visibility module to exports, accessor, docs, and README (#1145)"
```

---

### Task 8: User guide notebook

**Files:**
- Create: `examples/user_guide/37_Visibility_Analysis.ipynb`

- [ ] **Step 1: Create the notebook**

Create `examples/user_guide/37_Visibility_Analysis.ipynb` with cells:

**Cell 1 (markdown):**
```markdown
# Visibility Analysis

This notebook demonstrates multi-observer viewshed analysis and point-to-point
line-of-sight profiling.

**Functions covered:**
- `cumulative_viewshed` -- count of observers with line-of-sight to each cell
- `visibility_frequency` -- normalized version (fraction of observers)
- `line_of_sight` -- elevation profile, visibility, and optional Fresnel zone clearance
```

**Cell 2 (code):**
```python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from xrspatial.terrain import generate_terrain
from xrspatial.hillshade import hillshade
from xrspatial.visibility import (
    cumulative_viewshed,
    visibility_frequency,
    line_of_sight,
)
```

**Cell 3 (markdown):**
```markdown
## Generate synthetic terrain
```

**Cell 4 (code):**
```python
terrain = generate_terrain(width=200, height=200, seed=42)
hs = hillshade(terrain)

fig, ax = plt.subplots(figsize=(8, 8))
hs.plot.imshow(ax=ax, cmap='gray', add_colorbar=False)
terrain.plot.imshow(ax=ax, alpha=0.4, cmap='terrain', add_colorbar=True)
ax.set_title('Synthetic terrain')
plt.tight_layout()
```

**Cell 5 (markdown):**
```markdown
## Cumulative viewshed

Place three observers on the terrain and count how many can see each cell.
```

**Cell 6 (code):**
```python
xs = terrain.coords['x'].values
ys = terrain.coords['y'].values

observers = [
    {'x': float(xs[40]),  'y': float(ys[40]),  'observer_elev': 20},
    {'x': float(xs[160]), 'y': float(ys[80]),  'observer_elev': 20},
    {'x': float(xs[100]), 'y': float(ys[160]), 'observer_elev': 20},
]

cum = cumulative_viewshed(terrain, observers)

fig, ax = plt.subplots(figsize=(8, 8))
hs.plot.imshow(ax=ax, cmap='gray', add_colorbar=False)
cum.plot.imshow(ax=ax, alpha=0.6, cmap='YlOrRd',
                add_colorbar=True, vmin=0, vmax=3)
for obs in observers:
    ax.plot(obs['x'], obs['y'], 'k^', markersize=10)
ax.set_title('Cumulative viewshed (observer count)')
plt.tight_layout()
```

**Cell 7 (markdown):**
```markdown
## Visibility frequency

Same data, but normalized to [0, 1].
```

**Cell 8 (code):**
```python
freq = visibility_frequency(terrain, observers)

fig, ax = plt.subplots(figsize=(8, 8))
hs.plot.imshow(ax=ax, cmap='gray', add_colorbar=False)
freq.plot.imshow(ax=ax, alpha=0.6, cmap='RdYlGn',
                 add_colorbar=True, vmin=0, vmax=1)
for obs in observers:
    ax.plot(obs['x'], obs['y'], 'k^', markersize=10)
ax.set_title('Visibility frequency')
plt.tight_layout()
```

**Cell 9 (markdown):**
```markdown
## Line of sight

Draw a transect between two points and check visibility along it.
```

**Cell 10 (code):**
```python
ox, oy = float(xs[20]), float(ys[100])
tx, ty = float(xs[180]), float(ys[100])

los = line_of_sight(terrain, x0=ox, y0=oy, x1=tx, y1=ty,
                    observer_elev=15, target_elev=5)

fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(los['distance'].values, 0, los['elevation'].values,
                color='sienna', alpha=0.4, label='Terrain')
ax.plot(los['distance'].values, los['los_height'].values,
        'r--', label='LOS ray')

vis = los['visible'].values
d = los['distance'].values
ax.scatter(d[vis], los['elevation'].values[vis],
           c='green', s=8, label='Visible', zorder=3)
ax.scatter(d[~vis], los['elevation'].values[~vis],
           c='red', s=8, label='Blocked', zorder=3)

ax.set_xlabel('Distance')
ax.set_ylabel('Elevation')
ax.set_title('Line-of-sight profile')
ax.legend()
plt.tight_layout()
```

**Cell 11 (markdown):**
```markdown
## Fresnel zone clearance

For radio link planning, check whether the first Fresnel zone is clear at 900 MHz.
```

**Cell 12 (code):**
```python
los_f = line_of_sight(terrain, x0=ox, y0=oy, x1=tx, y1=ty,
                      observer_elev=50, target_elev=50,
                      frequency_mhz=900)

fig, ax = plt.subplots(figsize=(12, 4))
d = los_f['distance'].values
ax.fill_between(d, 0, los_f['elevation'].values,
                color='sienna', alpha=0.4, label='Terrain')
ax.plot(d, los_f['los_height'].values, 'r--', label='LOS ray')

# Fresnel zone envelope
lh = los_f['los_height'].values
fr = los_f['fresnel_radius'].values
ax.fill_between(d, lh - fr, lh + fr, alpha=0.15, color='blue',
                label='1st Fresnel zone')

fc = los_f['fresnel_clear'].values
ax.scatter(d[fc], los_f['elevation'].values[fc],
           c='green', s=8, label='Fresnel clear', zorder=3)
ax.scatter(d[~fc], los_f['elevation'].values[~fc],
           c='red', s=8, label='Fresnel blocked', zorder=3)

ax.set_xlabel('Distance')
ax.set_ylabel('Elevation')
ax.set_title('Fresnel zone clearance (900 MHz)')
ax.legend()
plt.tight_layout()
```

- [ ] **Step 2: Commit**

```bash
git add examples/user_guide/37_Visibility_Analysis.ipynb
git commit -m "Add visibility analysis user guide notebook (#1145)"
```

---

### Task 9: Final validation

- [ ] **Step 1: Run full test suite**

```bash
pytest xrspatial/tests/test_visibility.py -v
```

Expected: all tests PASS.

- [ ] **Step 2: Verify imports work**

```bash
python -c "from xrspatial import cumulative_viewshed, visibility_frequency, line_of_sight; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Verify accessor works**

```bash
python -c "
import numpy as np, xarray as xr
r = xr.DataArray(np.zeros((5,5)), dims=['y','x'],
                 coords={'y': np.arange(5.0), 'x': np.arange(5.0)})
r.xrs.cumulative_viewshed([{'x':2.0,'y':2.0,'observer_elev':10}])
print('accessor OK')
"
```

Expected: `accessor OK`.
