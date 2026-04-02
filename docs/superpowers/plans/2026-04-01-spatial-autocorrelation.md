# Spatial Autocorrelation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Global Moran's I and Local Moran's I (LISA) with queen/rook contiguity weights, supporting all four backends.

**Architecture:** New `xrspatial/autocorrelation.py` module following the existing `ArrayTypeFunctionMapping` dispatch pattern. Global Moran's I reuses the existing convolution infrastructure for spatial lag computation. LISA uses a fused `@ngjit` kernel for numpy (lag + permutation in one pass) and separate `map_overlap` calls for dask. All four backends: numpy, cupy, dask+numpy, dask+cupy.

**Tech Stack:** NumPy, Numba (`@ngjit`), CuPy, Dask (`map_overlap`, `map_blocks`), xarray. CuPy backends fall back to CPU for the branching-heavy permutation step (same pattern as `emerging_hotspots.py`).

**Spec:** `docs/superpowers/specs/2026-04-01-spatial-autocorrelation-design.md`

---

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `xrspatial/autocorrelation.py` | Create | Public API (`morans_i`, `lisa`), backend dispatch, all backend implementations |
| `xrspatial/tests/test_autocorrelation.py` | Create | Tests for both functions across all backends and edge cases |
| `xrspatial/__init__.py` | Modify (line ~2) | Add `morans_i` and `lisa` exports |
| `docs/source/reference/autocorrelation.rst` | Create | Sphinx API docs |
| `docs/source/reference/index.rst` | Modify (line 10) | Add `autocorrelation` to toctree |
| `README.md` | Modify (line ~567) | Add Spatial Statistics section to feature matrix |
| `examples/user_guide/48_Spatial_Autocorrelation.ipynb` | Create | User guide notebook |

---

### Task 1: Create Worktree and Scaffold Module

**Files:**
- Create: `xrspatial/autocorrelation.py`
- Create: `xrspatial/tests/test_autocorrelation.py`

- [ ] **Step 1: Create worktree**

```bash
git worktree add .claude/worktrees/issue-1135 -b issue-1135
```

- [ ] **Step 2: Create module scaffold**

Create `xrspatial/autocorrelation.py` in the worktree:

```python
"""Spatial autocorrelation statistics.

Global and local measures of spatial autocorrelation for raster data,
using queen or rook contiguity weights derived from the grid structure.
"""

import math
from functools import partial

import numpy as np
import xarray as xr
from numba import jit, prange

from xrspatial.convolution import (
    _convolve_2d_numpy,
    _convolve_2d_numpy_boundary,
)
from xrspatial.utils import (
    ArrayTypeFunctionMapping,
    _boundary_to_dask,
    _validate_boundary,
    _validate_raster,
    cuda_args,
    has_cuda_and_cupy,
    has_dask_array,
    is_cupy_array,
    is_dask_cupy,
    ngjit,
)

# Contiguity kernels (center pixel excluded)
try:
    import cupy
    from xrspatial.convolution import _convolve_2d_cupy
except ImportError:
    cupy = None

# Contiguity kernels (center pixel excluded)
_QUEEN_KERNEL = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.float32)

_ROOK_KERNEL = np.array([[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]], dtype=np.float32)

VALID_CONTIGUITY = ('queen', 'rook')


def _contiguity_kernel(contiguity):
    """Return the 3x3 weight kernel for the given contiguity type."""
    if contiguity == 'queen':
        return _QUEEN_KERNEL.copy()
    elif contiguity == 'rook':
        return _ROOK_KERNEL.copy()
    else:
        raise ValueError(
            f"Invalid contiguity '{contiguity}'. "
            f"Expected one of {VALID_CONTIGUITY}"
        )


def _not_implemented(*args, **kwargs):
    raise NotImplementedError("Backend not yet implemented")


def morans_i(raster, contiguity='queen', boundary='nan'):
    """Global Moran's I statistic for spatial autocorrelation.

    Parameters
    ----------
    raster : xr.DataArray
        2D raster of numeric values.
    contiguity : str, default 'queen'
        Contiguity type: 'queen' (8 neighbors) or 'rook' (4 neighbors).
    boundary : str, default 'nan'
        How to handle edges: 'nan', 'nearest', 'reflect', or 'wrap'.

    Returns
    -------
    xr.DataArray
        Scalar (0-dimensional) DataArray with the I statistic as its value.
        Attrs include expected_I, variance_I, z_score, p_value, N, S0,
        and contiguity.
    """
    _validate_raster(raster, func_name='morans_i', name='raster', ndim=2)
    _validate_boundary(boundary)
    kernel = _contiguity_kernel(contiguity)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=partial(_morans_i_numpy, kernel=kernel, boundary=boundary),
        cupy_func=partial(_morans_i_cupy, kernel=kernel, boundary=boundary),
        dask_func=partial(_morans_i_dask_numpy, kernel=kernel, boundary=boundary),
        dask_cupy_func=partial(_morans_i_dask_cupy, kernel=kernel, boundary=boundary),
    )
    return mapper(raster)(raster)


def lisa(raster, contiguity='queen', n_permutations=999, boundary='nan'):
    """Local Indicators of Spatial Association (Local Moran's I).

    Parameters
    ----------
    raster : xr.DataArray
        2D raster of numeric values.
    contiguity : str, default 'queen'
        Contiguity type: 'queen' (8 neighbors) or 'rook' (4 neighbors).
    n_permutations : int, default 999
        Number of random permutations for pseudo p-value computation.
    boundary : str, default 'nan'
        How to handle edges: 'nan', 'nearest', 'reflect', or 'wrap'.

    Returns
    -------
    xr.Dataset
        Dataset with variables:
        - lisa_values (y, x) float32: local I_i per pixel
        - p_values (y, x) float32: pseudo p-values from permutation
        - cluster (y, x) int8: 0=NS, 1=HH, 2=LL, 3=HL, 4=LH
    """
    _validate_raster(raster, func_name='lisa', name='raster', ndim=2)
    _validate_boundary(boundary)
    kernel = _contiguity_kernel(contiguity)

    mapper = ArrayTypeFunctionMapping(
        numpy_func=partial(_lisa_numpy, kernel=kernel,
                           n_permutations=n_permutations, boundary=boundary),
        cupy_func=partial(_lisa_cupy, kernel=kernel,
                          n_permutations=n_permutations, boundary=boundary),
        dask_func=partial(_lisa_dask_numpy, kernel=kernel,
                          n_permutations=n_permutations, boundary=boundary),
        dask_cupy_func=partial(_lisa_dask_cupy, kernel=kernel,
                               n_permutations=n_permutations, boundary=boundary),
    )
    result = mapper(raster)(raster)

    dims_2d = raster.dims[-2:]
    coords_2d = {k: v for k, v in raster.coords.items()
                 if k in dims_2d or set(v.dims).issubset(set(dims_2d))}

    lisa_vals, p_vals, cluster_vals = result
    return xr.Dataset(
        {
            'lisa_values': xr.DataArray(lisa_vals, dims=dims_2d, coords=coords_2d),
            'p_values': xr.DataArray(p_vals, dims=dims_2d, coords=coords_2d),
            'cluster': xr.DataArray(cluster_vals, dims=dims_2d, coords=coords_2d),
        },
        attrs={
            'n_permutations': n_permutations,
            'contiguity': contiguity,
        },
    )


# --- Backend stubs (filled in by subsequent tasks) ---

_morans_i_numpy = _not_implemented
_morans_i_cupy = _not_implemented
_morans_i_dask_numpy = _not_implemented
_morans_i_dask_cupy = _not_implemented
_lisa_numpy = _not_implemented
_lisa_cupy = _not_implemented
_lisa_dask_numpy = _not_implemented
_lisa_dask_cupy = _not_implemented
```

- [ ] **Step 3: Create test file scaffold**

Create `xrspatial/tests/test_autocorrelation.py` in the worktree:

```python
"""Tests for xrspatial.autocorrelation (Moran's I, LISA)."""

import numpy as np
import pytest
import xarray as xr

from xrspatial.autocorrelation import morans_i, lisa, _contiguity_kernel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raster(data, dims=('y', 'x')):
    """Wrap a numpy array as an xr.DataArray."""
    return xr.DataArray(data.astype(np.float32), dims=dims)


def _checkerboard(n=4):
    """Return an n x n checkerboard of 0s and 1s (float32)."""
    arr = np.indices((n, n)).sum(axis=0) % 2
    return arr.astype(np.float32)


def _gradient(n=4):
    """Return an n x n array where each row is a constant (0..n-1)."""
    return np.tile(np.arange(n, dtype=np.float32).reshape(-1, 1), (1, n))
```

- [ ] **Step 4: Commit scaffold**

```bash
cd .claude/worktrees/issue-1135
git add xrspatial/autocorrelation.py xrspatial/tests/test_autocorrelation.py
git commit -m "Scaffold autocorrelation module and tests (#1135)"
```

---

### Task 2: Global Moran's I -- NumPy Backend (TDD)

**Files:**
- Modify: `xrspatial/tests/test_autocorrelation.py`
- Modify: `xrspatial/autocorrelation.py`

- [ ] **Step 1: Write failing tests**

Append to `xrspatial/tests/test_autocorrelation.py`:

```python
# ---------------------------------------------------------------------------
# Global Moran's I -- NumPy
# ---------------------------------------------------------------------------

class TestMoransINumpy:
    """Global Moran's I on numpy-backed DataArrays."""

    def test_checkerboard_negative(self):
        """Checkerboard: perfectly alternating -> strong negative I."""
        raster = _make_raster(_checkerboard(6))
        result = morans_i(raster, contiguity='queen')
        assert result.shape == ()
        I = float(result)
        assert I < -0.5, f"Checkerboard should have negative I, got {I}"

    def test_gradient_positive(self):
        """Row gradient: spatially smooth -> strong positive I."""
        raster = _make_raster(_gradient(6))
        result = morans_i(raster, contiguity='queen')
        I = float(result)
        assert I > 0.3, f"Gradient should have positive I, got {I}"

    def test_rook_vs_queen_differ(self):
        """Queen and rook should produce different I values."""
        raster = _make_raster(_checkerboard(6))
        I_queen = float(morans_i(raster, contiguity='queen'))
        I_rook = float(morans_i(raster, contiguity='rook'))
        assert I_queen != I_rook

    def test_attrs_present(self):
        """Result attrs should contain analytical inference fields."""
        raster = _make_raster(_gradient(6))
        result = morans_i(raster, contiguity='queen')
        for key in ('expected_I', 'variance_I', 'z_score', 'p_value', 'N', 'S0'):
            assert key in result.attrs, f"Missing attr: {key}"
        assert result.attrs['N'] == 36
        assert result.attrs['contiguity'] == 'queen'

    def test_constant_raster_nan(self):
        """Constant raster (zero variance) -> NaN."""
        data = np.ones((4, 4), dtype=np.float32)
        result = morans_i(_make_raster(data))
        assert np.isnan(float(result))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd .claude/worktrees/issue-1135
python -m pytest xrspatial/tests/test_autocorrelation.py::TestMoransINumpy -v 2>&1 | head -30
```

Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement numpy backend**

In `xrspatial/autocorrelation.py`, replace the `_morans_i_numpy = _not_implemented` stub with:

```python
def _morans_i_numpy(raster, kernel, boundary='nan'):
    """Global Moran's I -- numpy backend."""
    data = raster.values.astype(np.float32)
    mask = ~np.isnan(data)
    N = int(np.sum(mask))

    if N < 2:
        return _scalar_result(np.nan, contiguity='unknown')

    mean = np.nanmean(data)
    z = np.where(mask, data - mean, np.nan)
    var = np.nansum(z[mask] ** 2) / N

    if var == 0.0:
        return _scalar_result(np.nan, contiguity='unknown')

    # Spatial lag via existing convolution
    lag = _convolve_2d_numpy_boundary(z, kernel, boundary=boundary)

    # S0: total weight count (sum of neighbor counts for valid pixels)
    mask_f = mask.astype(np.float32)
    n_neighbors = _convolve_2d_numpy_boundary(mask_f, kernel, boundary=boundary)
    S0 = float(np.nansum(n_neighbors[mask]))

    # Moran's I
    numerator = float(np.nansum(z * lag))
    denominator = float(np.nansum(z[mask] ** 2))
    I = (N / S0) * numerator / denominator

    # Analytical inference (normality assumption, Cliff & Ord 1981)
    S1 = 2.0 * S0  # symmetric binary weights: S1 = 2*S0
    S2 = 4.0 * float(np.nansum(n_neighbors[mask] ** 2))
    expected_I = -1.0 / (N - 1)
    var_I = (
        (N ** 2 * S1 - N * S2 + 3 * S0 ** 2)
        / (S0 ** 2 * (N ** 2 - 1))
    ) - expected_I ** 2
    var_I = max(var_I, 0.0)

    z_score = (I - expected_I) / math.sqrt(var_I) if var_I > 0 else np.nan
    p_value = float(2.0 * (1.0 - _norm_cdf(abs(z_score)))) if not np.isnan(z_score) else np.nan

    return xr.DataArray(
        np.float64(I),
        attrs={
            'expected_I': expected_I,
            'variance_I': var_I,
            'z_score': z_score,
            'p_value': p_value,
            'N': N,
            'S0': S0,
            'contiguity': 'queen' if kernel.sum() == 8 else 'rook',
        },
    )
```

Also add these helper functions above the backend stubs:

```python
def _norm_cdf(x):
    """Standard normal CDF via the error function (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _scalar_result(value, contiguity='unknown'):
    """Return a scalar DataArray with NaN attrs."""
    return xr.DataArray(
        np.float64(value),
        attrs={
            'expected_I': np.nan,
            'variance_I': np.nan,
            'z_score': np.nan,
            'p_value': np.nan,
            'N': 0,
            'S0': 0.0,
            'contiguity': contiguity,
        },
    )
```

Update the `morans_i` public function's mapper to pass `contiguity` string through. In the numpy backend, infer contiguity from kernel sum (8=queen, 4=rook). This is already handled in the code above.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd .claude/worktrees/issue-1135
python -m pytest xrspatial/tests/test_autocorrelation.py::TestMoransINumpy -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add xrspatial/autocorrelation.py xrspatial/tests/test_autocorrelation.py
git commit -m "Add Global Moran's I numpy backend (#1135)"
```

---

### Task 3: LISA -- NumPy Backend (TDD)

**Files:**
- Modify: `xrspatial/tests/test_autocorrelation.py`
- Modify: `xrspatial/autocorrelation.py`

- [ ] **Step 1: Write failing tests**

Append to `xrspatial/tests/test_autocorrelation.py`:

```python
# ---------------------------------------------------------------------------
# LISA -- NumPy
# ---------------------------------------------------------------------------

class TestLisaNumpy:
    """Local Moran's I (LISA) on numpy-backed DataArrays."""

    def test_returns_dataset(self):
        """LISA returns a Dataset with expected variables."""
        raster = _make_raster(_gradient(6))
        ds = lisa(raster, n_permutations=99)
        assert isinstance(ds, xr.Dataset)
        assert 'lisa_values' in ds
        assert 'p_values' in ds
        assert 'cluster' in ds
        assert ds['lisa_values'].shape == (6, 6)
        assert ds['p_values'].dtype == np.float32
        assert ds['cluster'].dtype == np.int8

    def test_checkerboard_negative_lisa(self):
        """Checkerboard: all local I_i should be negative."""
        raster = _make_raster(_checkerboard(6))
        ds = lisa(raster, n_permutations=99)
        vals = ds['lisa_values'].values
        # Interior pixels (not on boundary) should be negative
        interior = vals[1:-1, 1:-1]
        valid = interior[~np.isnan(interior)]
        assert np.all(valid < 0), "Checkerboard interior LISA should be negative"

    def test_checkerboard_clusters_hl_lh(self):
        """Checkerboard: significant clusters should be HL or LH."""
        raster = _make_raster(_checkerboard(8))
        ds = lisa(raster, n_permutations=199)
        cluster = ds['cluster'].values
        sig = cluster[cluster != 0]
        # All significant pixels should be HL (3) or LH (4)
        assert np.all((sig == 3) | (sig == 4)), f"Unexpected clusters: {np.unique(sig)}"

    def test_gradient_positive_lisa(self):
        """Gradient: interior pixels should have positive local I_i."""
        raster = _make_raster(_gradient(8))
        ds = lisa(raster, n_permutations=99)
        vals = ds['lisa_values'].values
        interior = vals[2:-2, 2:-2]
        valid = interior[~np.isnan(interior)]
        assert np.all(valid > 0), "Gradient interior LISA should be positive"

    def test_pvalues_in_range(self):
        """p-values should be in [0, 1] for all valid pixels."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 10)).astype(np.float32)
        ds = lisa(_make_raster(data), n_permutations=99)
        p = ds['p_values'].values
        valid = p[~np.isnan(p)]
        assert np.all(valid >= 0.0) and np.all(valid <= 1.0)

    def test_cluster_codes_valid(self):
        """Cluster codes should be in {0, 1, 2, 3, 4}."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 10)).astype(np.float32)
        ds = lisa(_make_raster(data), n_permutations=99)
        codes = ds['cluster'].values
        assert set(np.unique(codes)).issubset({0, 1, 2, 3, 4})

    def test_nan_propagation(self):
        """NaN input pixels produce NaN in output."""
        data = np.ones((6, 6), dtype=np.float32)
        data[0, 0] = np.nan
        data[2, 3] = 5.0  # break constant to avoid zero-var
        ds = lisa(_make_raster(data), n_permutations=99)
        assert np.isnan(ds['lisa_values'].values[0, 0])
        assert np.isnan(ds['p_values'].values[0, 0])
        assert ds['cluster'].values[0, 0] == 0

    def test_constant_raster_nan(self):
        """Constant raster (zero variance) -> NaN LISA values."""
        data = np.full((4, 4), 5.0, dtype=np.float32)
        ds = lisa(_make_raster(data), n_permutations=99)
        assert np.all(np.isnan(ds['lisa_values'].values))

    def test_attrs(self):
        """Dataset attrs carry metadata."""
        ds = lisa(_make_raster(_gradient(4)), n_permutations=99)
        assert ds.attrs['n_permutations'] == 99
        assert ds.attrs['contiguity'] == 'queen'
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd .claude/worktrees/issue-1135
python -m pytest xrspatial/tests/test_autocorrelation.py::TestLisaNumpy -v 2>&1 | head -30
```

Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Implement the @ngjit fused LISA kernel**

In `xrspatial/autocorrelation.py`, add the Numba JIT function (place it after `_scalar_result` and before the backend stubs):

```python
@ngjit
def _lisa_fused_ngjit(z, kernel, inv_var, n_perms, seed,
                      out_lisa, out_pval, out_cluster):
    """Fused LISA computation: lag + permutation + cluster in one pass.

    Parameters
    ----------
    z : float32 2D array (data - mean, NaN where input is NaN)
    kernel : float32 3x3 contiguity kernel
    inv_var : float (1 / variance)
    n_perms : int
    seed : int64
    out_lisa, out_pval : float32 2D arrays (output)
    out_cluster : int8 2D array (output)
    """
    rows, cols = z.shape
    kr, kc = kernel.shape
    hr, hc = kr // 2, kc // 2
    max_neighbors = kr * kc  # 9 for 3x3, but center is 0

    for i in range(rows):
        for j in range(cols):
            zi = z[i, j]
            # NaN pixel
            if zi != zi:
                out_lisa[i, j] = np.nan
                out_pval[i, j] = np.nan
                out_cluster[i, j] = 0
                continue

            # Extract valid neighbors
            nbr_z = np.empty(max_neighbors, dtype=np.float32)
            nbr_w = np.empty(max_neighbors, dtype=np.float32)
            n_nbr = 0
            for di in range(kr):
                for dj in range(kc):
                    w = kernel[di, dj]
                    if w == 0.0:
                        continue
                    ni = i + di - hr
                    nj = j + dj - hc
                    if 0 <= ni < rows and 0 <= nj < cols:
                        val = z[ni, nj]
                        if val == val:  # not NaN
                            nbr_z[n_nbr] = val
                            nbr_w[n_nbr] = w
                            n_nbr += 1

            if n_nbr == 0:
                out_lisa[i, j] = np.nan
                out_pval[i, j] = np.nan
                out_cluster[i, j] = 0
                continue

            # Observed spatial lag and LISA value
            lag = 0.0
            for k in range(n_nbr):
                lag += nbr_w[k] * nbr_z[k]
            I_obs = zi * inv_var * lag
            out_lisa[i, j] = I_obs

            # Permutation test (Fisher-Yates shuffle)
            abs_I = abs(I_obs)
            count = 0
            rng = np.int64(seed) + np.int64(i * cols + j)

            for p in range(n_perms):
                # Shuffle neighbor z values in place
                for k in range(n_nbr - 1, 0, -1):
                    rng = rng * np.int64(6364136223846793005) + np.int64(1442695040888963407)
                    idx = int((rng >> 33) & np.int64(0x7fffffff)) % (k + 1)
                    tmp = nbr_z[k]
                    nbr_z[k] = nbr_z[idx]
                    nbr_z[idx] = tmp

                perm_lag = 0.0
                for k in range(n_nbr):
                    perm_lag += nbr_w[k] * nbr_z[k]
                I_perm = zi * inv_var * perm_lag
                if abs(I_perm) >= abs_I:
                    count += 1

            out_pval[i, j] = np.float32(count + 1) / np.float32(n_perms + 1)

            # Cluster classification (p <= 0.05)
            if out_pval[i, j] > 0.05:
                out_cluster[i, j] = 0  # not significant
            elif zi > 0.0 and lag > 0.0:
                out_cluster[i, j] = 1  # HH
            elif zi < 0.0 and lag < 0.0:
                out_cluster[i, j] = 2  # LL
            elif zi > 0.0 and lag < 0.0:
                out_cluster[i, j] = 3  # HL
            else:
                out_cluster[i, j] = 4  # LH
```

- [ ] **Step 4: Implement _lisa_numpy backend**

Replace the `_lisa_numpy = _not_implemented` stub:

```python
def _lisa_numpy(raster, kernel, n_permutations=999, boundary='nan'):
    """LISA -- numpy backend."""
    data = raster.values.astype(np.float32)
    mask = ~np.isnan(data)
    N = int(np.sum(mask))

    if N < 2:
        nans = np.full(data.shape, np.nan, dtype=np.float32)
        zeros = np.zeros(data.shape, dtype=np.int8)
        return nans, nans.copy(), zeros

    mean = np.nanmean(data)
    z = np.where(mask, data - mean, np.nan).astype(np.float32)
    var = np.nansum(z[mask] ** 2) / N

    if var == 0.0:
        nans = np.full(data.shape, np.nan, dtype=np.float32)
        zeros = np.zeros(data.shape, dtype=np.int8)
        return nans, nans.copy(), zeros

    inv_var = np.float32(1.0 / var)
    out_lisa = np.empty(data.shape, dtype=np.float32)
    out_pval = np.empty(data.shape, dtype=np.float32)
    out_cluster = np.empty(data.shape, dtype=np.int8)

    _lisa_fused_ngjit(z, kernel, inv_var, n_permutations, 42,
                      out_lisa, out_pval, out_cluster)
    return out_lisa, out_pval, out_cluster
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd .claude/worktrees/issue-1135
python -m pytest xrspatial/tests/test_autocorrelation.py::TestLisaNumpy -v
```

Expected: all 9 tests PASS

- [ ] **Step 6: Commit**

```bash
git add xrspatial/autocorrelation.py xrspatial/tests/test_autocorrelation.py
git commit -m "Add LISA numpy backend with permutation testing (#1135)"
```

---

### Task 4: Global Moran's I and LISA -- CuPy Backends

**Files:**
- Modify: `xrspatial/tests/test_autocorrelation.py`
- Modify: `xrspatial/autocorrelation.py`

- [ ] **Step 1: Write cupy tests**

Append to `xrspatial/tests/test_autocorrelation.py`:

```python
# ---------------------------------------------------------------------------
# CuPy backends
# ---------------------------------------------------------------------------

@pytest.fixture
def cupy_available():
    return pytest.importorskip("cupy")


class TestMoransICuPy:
    def test_matches_numpy(self, cupy_available):
        cp = cupy_available
        data = _gradient(6)
        I_np = float(morans_i(_make_raster(data)))
        I_cp = float(morans_i(_make_raster(cp.asarray(data))))
        np.testing.assert_allclose(I_cp, I_np, rtol=1e-5)


class TestLisaCuPy:
    def test_matches_numpy(self, cupy_available):
        cp = cupy_available
        data = _gradient(8)
        ds_np = lisa(_make_raster(data), n_permutations=99)
        ds_cp = lisa(_make_raster(cp.asarray(data)), n_permutations=99)
        np.testing.assert_allclose(
            ds_cp['lisa_values'].values,
            ds_np['lisa_values'].values,
            rtol=1e-4, atol=1e-6,
        )
```

- [ ] **Step 2: Implement _morans_i_cupy**

Replace the `_morans_i_cupy = _not_implemented` stub (cupy import is already in the scaffold):

```python
def _morans_i_cupy(raster, kernel, boundary='nan'):
    """Global Moran's I -- cupy backend.

    Transfers to CPU for the analytical inference step since it's
    just scalar arithmetic.
    """
    data = raster.data.astype(cupy.float32)
    mask = ~cupy.isnan(data)
    N = int(cupy.sum(mask))

    if N < 2:
        return _scalar_result(np.nan)

    mean = float(cupy.nanmean(data))
    z = cupy.where(mask, data - mean, cupy.nan).astype(cupy.float32)
    var = float(cupy.nansum(z[mask] ** 2)) / N

    if var == 0.0:
        return _scalar_result(np.nan)

    # Spatial lag via GPU convolution
    lag = _convolve_2d_cupy(z, kernel, boundary=boundary)

    # S0 from neighbor count
    mask_f = mask.astype(cupy.float32)
    n_neighbors = _convolve_2d_cupy(mask_f, kernel, boundary=boundary)
    S0 = float(cupy.nansum(n_neighbors[mask]))

    numerator = float(cupy.nansum(z * lag))
    denominator = float(cupy.nansum(z[mask] ** 2))
    I = (N / S0) * numerator / denominator

    # Analytical inference (same scalar math as numpy)
    S1 = 2.0 * S0
    S2 = 4.0 * float(cupy.nansum(n_neighbors[mask] ** 2))
    expected_I = -1.0 / (N - 1)
    var_I = (
        (N ** 2 * S1 - N * S2 + 3 * S0 ** 2)
        / (S0 ** 2 * (N ** 2 - 1))
    ) - expected_I ** 2
    var_I = max(var_I, 0.0)
    z_score = (I - expected_I) / math.sqrt(var_I) if var_I > 0 else np.nan
    p_value = float(2.0 * (1.0 - _norm_cdf(abs(z_score)))) if not np.isnan(z_score) else np.nan

    return xr.DataArray(
        np.float64(I),
        attrs={
            'expected_I': expected_I, 'variance_I': var_I,
            'z_score': z_score, 'p_value': p_value,
            'N': N, 'S0': S0,
            'contiguity': 'queen' if kernel.sum() == 8 else 'rook',
        },
    )
```

- [ ] **Step 3: Implement _lisa_cupy**

Replace `_lisa_cupy = _not_implemented`:

```python
def _lisa_cupy(raster, kernel, n_permutations=999, boundary='nan'):
    """LISA -- cupy backend.

    Falls back to CPU for the permutation step (branching-heavy Fisher-Yates
    shuffle is faster on CPU, same pattern as emerging_hotspots.py).
    """
    data = raster.data.astype(cupy.float32)
    mask = ~cupy.isnan(data)
    N = int(cupy.sum(mask))

    if N < 2:
        shape = data.shape
        nans = cupy.full(shape, cupy.nan, dtype=cupy.float32)
        zeros = cupy.zeros(shape, dtype=cupy.int8)
        return cupy.asnumpy(nans), cupy.asnumpy(nans), cupy.asnumpy(zeros)

    mean = float(cupy.nanmean(data))
    z_gpu = cupy.where(mask, data - mean, cupy.nan).astype(cupy.float32)
    var = float(cupy.nansum(z_gpu[mask] ** 2)) / N

    if var == 0.0:
        shape = data.shape
        nans = np.full(shape, np.nan, dtype=np.float32)
        zeros = np.zeros(shape, dtype=np.int8)
        return nans, nans.copy(), zeros

    # Transfer to CPU for branching-heavy permutation
    z_cpu = cupy.asnumpy(z_gpu)
    inv_var = np.float32(1.0 / var)
    out_lisa = np.empty(z_cpu.shape, dtype=np.float32)
    out_pval = np.empty(z_cpu.shape, dtype=np.float32)
    out_cluster = np.empty(z_cpu.shape, dtype=np.int8)

    _lisa_fused_ngjit(z_cpu, kernel, inv_var, n_permutations, 42,
                      out_lisa, out_pval, out_cluster)

    return cupy.asarray(out_lisa), cupy.asarray(out_pval), cupy.asarray(out_cluster)
```

- [ ] **Step 4: Run cupy tests**

```bash
cd .claude/worktrees/issue-1135
python -m pytest xrspatial/tests/test_autocorrelation.py::TestMoransICuPy -v
python -m pytest xrspatial/tests/test_autocorrelation.py::TestLisaCuPy -v
```

Expected: PASS (or SKIP if no GPU)

- [ ] **Step 5: Commit**

```bash
git add xrspatial/autocorrelation.py xrspatial/tests/test_autocorrelation.py
git commit -m "Add Moran's I and LISA cupy backends (#1135)"
```

---

### Task 5: Global Moran's I and LISA -- Dask Backends

**Files:**
- Modify: `xrspatial/tests/test_autocorrelation.py`
- Modify: `xrspatial/autocorrelation.py`

- [ ] **Step 1: Write dask tests**

Append to `xrspatial/tests/test_autocorrelation.py`:

```python
# ---------------------------------------------------------------------------
# Dask backends
# ---------------------------------------------------------------------------

@pytest.fixture
def dask_available():
    return pytest.importorskip("dask.array")


def _make_dask_raster(data, chunks=(4, 4)):
    import dask.array as da
    darr = da.from_array(data.astype(np.float32), chunks=chunks)
    return xr.DataArray(darr, dims=('y', 'x'))


class TestMoransIDask:
    def test_matches_numpy(self, dask_available):
        data = _gradient(8)
        I_np = float(morans_i(_make_raster(data)))
        I_dask = float(morans_i(_make_dask_raster(data)))
        np.testing.assert_allclose(I_dask, I_np, rtol=1e-5)

    def test_checkerboard(self, dask_available):
        data = _checkerboard(8)
        result = morans_i(_make_dask_raster(data))
        assert float(result) < -0.5


class TestLisaDask:
    def test_matches_numpy(self, dask_available):
        data = _gradient(8)
        ds_np = lisa(_make_raster(data), n_permutations=99)
        ds_dask = lisa(_make_dask_raster(data, chunks=(8, 8)), n_permutations=99)
        np.testing.assert_allclose(
            ds_dask['lisa_values'].values,
            ds_np['lisa_values'].values,
            rtol=1e-4, atol=1e-6,
        )

    def test_chunked_pvalues_in_range(self, dask_available):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((16, 16)).astype(np.float32)
        ds = lisa(_make_dask_raster(data, chunks=(8, 8)), n_permutations=99)
        p = ds['p_values'].values
        valid = p[~np.isnan(p)]
        assert np.all(valid >= 0.0) and np.all(valid <= 1.0)
```

- [ ] **Step 2: Implement dask chunk functions**

In `xrspatial/autocorrelation.py`, add after the `_lisa_fused_ngjit` function:

```python
def _lag_chunk_numpy(chunk, kernel):
    """Compute spatial lag for one dask chunk (called by map_overlap)."""
    return _convolve_2d_numpy(chunk.astype(np.float32), kernel)


def _perm_chunk_numpy(chunk_z, kernel, inv_var, n_perms, seed):
    """Compute LISA p-values for one dask chunk (called by map_overlap)."""
    rows, cols = chunk_z.shape
    out_pval = np.empty((rows, cols), dtype=np.float32)
    out_pval[:] = np.nan

    kr, kc = kernel.shape
    hr, hc = kr // 2, kc // 2
    max_nbr = kr * kc

    for i in range(rows):
        for j in range(cols):
            zi = chunk_z[i, j]
            if zi != zi:
                continue

            nbr_z = np.empty(max_nbr, dtype=np.float32)
            nbr_w = np.empty(max_nbr, dtype=np.float32)
            n_nbr = 0
            for di in range(kr):
                for dj in range(kc):
                    w = kernel[di, dj]
                    if w == 0.0:
                        continue
                    ni = i + di - hr
                    nj = j + dj - hc
                    if 0 <= ni < rows and 0 <= nj < cols:
                        val = chunk_z[ni, nj]
                        if val == val:
                            nbr_z[n_nbr] = val
                            nbr_w[n_nbr] = w
                            n_nbr += 1

            if n_nbr == 0:
                continue

            lag = 0.0
            for k in range(n_nbr):
                lag += nbr_w[k] * nbr_z[k]
            I_obs = zi * inv_var * lag
            abs_I = abs(I_obs)
            count = 0
            rng = np.int64(seed) + np.int64(i * cols + j)

            for p in range(n_perms):
                for k in range(n_nbr - 1, 0, -1):
                    rng = rng * np.int64(6364136223846793005) + np.int64(1442695040888963407)
                    idx = int((rng >> 33) & np.int64(0x7fffffff)) % (k + 1)
                    tmp = nbr_z[k]
                    nbr_z[k] = nbr_z[idx]
                    nbr_z[idx] = tmp

                perm_lag = 0.0
                for k in range(n_nbr):
                    perm_lag += nbr_w[k] * nbr_z[k]
                I_perm = zi * inv_var * perm_lag
                if abs(I_perm) >= abs_I:
                    count += 1

            out_pval[i, j] = np.float32(count + 1) / np.float32(n_perms + 1)

    return out_pval


_perm_chunk_numpy_jit = ngjit(_perm_chunk_numpy)
```

Wait -- `_perm_chunk_numpy` can't be JIT'd directly because it's the chunk function called by `map_overlap`. The JIT version would need the same inner loop extracted. Let me restructure: keep `_perm_chunk_numpy` as a plain Python wrapper that calls a JIT inner function.

Actually, the simpler approach: reuse `_lisa_fused_ngjit` inside the chunk function, since it already does exactly what we need. The chunk function wraps it:

```python
def _lisa_chunk_numpy(chunk_z, kernel, inv_var, n_perms, seed):
    """Fused LISA chunk: lag + pval + cluster for map_overlap."""
    out_lisa = np.empty(chunk_z.shape, dtype=np.float32)
    out_pval = np.empty(chunk_z.shape, dtype=np.float32)
    out_cluster = np.empty(chunk_z.shape, dtype=np.int8)
    _lisa_fused_ngjit(chunk_z.astype(np.float32), kernel, inv_var,
                      n_perms, seed, out_lisa, out_pval, out_cluster)
    return out_lisa  # or out_pval or out_cluster depending on which pass
```

But we need three separate outputs. So we run `_lisa_fused_ngjit` once inside a wrapper that stores all three, and we call `map_overlap` three times each returning a different band. This triples the computation but avoids the multi-output problem.

Better approach: run the fused kernel once in a wrapper, cache the results, return the requested band. But caching across `map_overlap` calls is fragile.

Best practical approach for dask: call the fused kernel in one `map_overlap`, encode all three outputs into a single 2D array by interleaving rows:

No -- too hacky. Let me use the cleanest approach: three `map_overlap` calls sharing the same chunk function structure, each returning one output. The chunk function computes everything but only returns one band.

Even better: just live with computing 3x. For depth=1, the overlap is tiny and the dominant cost is the permutation loop anyway. Each `map_overlap` call does the same permutation work. The total cost is 3x, but for a plan, clarity beats optimization. We can fuse later if profiling shows it matters.

Actually, the best practical approach given the existing pattern in the codebase is:

1. `map_overlap` → spatial lag (one call)
2. `lisa_values = z * lag / var` (element-wise)
3. `map_overlap` → p_values (one call, does its own permutation)
4. `map_blocks` → cluster (element-wise from z, lag, p)

This is only 2 `map_overlap` calls, not 3.

Let me restructure the chunk functions to be cleaner.

Alright, I'll restructure the dask approach in the plan. Let me update the chunk functions section.

- [ ] **Step 3: Implement _morans_i_dask_numpy**

Replace `_morans_i_dask_numpy = _not_implemented`:

```python
def _morans_i_dask_numpy(raster, kernel, boundary='nan'):
    """Global Moran's I -- dask+numpy backend."""
    import dask.array as da

    data = raster.data.astype(np.float32)
    mask = ~da.isnan(data)

    # Eagerly compute scalars
    mean, N = da.compute(da.nanmean(data), da.sum(mask))
    N = int(N)
    if N < 2:
        return _scalar_result(np.nan)

    z = da.where(mask, data - mean, np.nan)
    var_total = da.nansum(z ** 2) / N

    # Spatial lag via map_overlap
    _lag = partial(_lag_chunk_numpy, kernel=kernel)
    lag = z.map_overlap(_lag, depth=(1, 1),
                        boundary=_boundary_to_dask(boundary),
                        meta=np.array((), dtype=np.float32))

    # S0 from mask convolution
    mask_f = mask.astype(np.float32)
    n_neighbors = mask_f.map_overlap(_lag, depth=(1, 1),
                                     boundary=_boundary_to_dask(boundary),
                                     meta=np.array((), dtype=np.float32))
    S0 = da.nansum(da.where(mask, n_neighbors, 0.0))

    numerator = da.nansum(z * lag)
    denominator = da.nansum(z ** 2)

    # Compute all dask scalars at once
    I_num, I_den, S0_val, var_val, S2_inner = da.compute(
        numerator, denominator, S0,
        var_total,
        da.nansum(da.where(mask, n_neighbors ** 2, 0.0)),
    )
    S0_val = float(S0_val)

    if float(var_val) == 0.0:
        return _scalar_result(np.nan)

    I = (N / S0_val) * float(I_num) / float(I_den)

    S1 = 2.0 * S0_val
    S2 = 4.0 * float(S2_inner)
    expected_I = -1.0 / (N - 1)
    var_I = (
        (N ** 2 * S1 - N * S2 + 3 * S0_val ** 2)
        / (S0_val ** 2 * (N ** 2 - 1))
    ) - expected_I ** 2
    var_I = max(var_I, 0.0)
    z_score = (I - expected_I) / math.sqrt(var_I) if var_I > 0 else np.nan
    p_value = float(2.0 * (1.0 - _norm_cdf(abs(z_score)))) if not np.isnan(z_score) else np.nan

    return xr.DataArray(
        np.float64(I),
        attrs={
            'expected_I': expected_I, 'variance_I': var_I,
            'z_score': z_score, 'p_value': p_value,
            'N': N, 'S0': S0_val,
            'contiguity': 'queen' if kernel.sum() == 8 else 'rook',
        },
    )
```

- [ ] **Step 4: Implement dask LISA chunk functions and _lisa_dask_numpy**

Add chunk functions in `xrspatial/autocorrelation.py`:

```python
def _lag_chunk_numpy(chunk, kernel):
    """Spatial lag for one dask chunk. Called by map_overlap."""
    return _convolve_2d_numpy(chunk.astype(np.float32), kernel)


@ngjit
def _perm_pvalue_ngjit(z, kernel, inv_var, n_perms, seed, out_pval):
    """Compute permutation p-values only (LISA value computed separately)."""
    rows, cols = z.shape
    kr, kc = kernel.shape
    hr, hc = kr // 2, kc // 2
    max_nbr = kr * kc

    for i in range(rows):
        for j in range(cols):
            zi = z[i, j]
            if zi != zi:
                out_pval[i, j] = np.nan
                continue

            nbr_z = np.empty(max_nbr, dtype=np.float32)
            nbr_w = np.empty(max_nbr, dtype=np.float32)
            n_nbr = 0
            for di in range(kr):
                for dj in range(kc):
                    w = kernel[di, dj]
                    if w == 0.0:
                        continue
                    ni = i + di - hr
                    nj = j + dj - hc
                    if 0 <= ni < rows and 0 <= nj < cols:
                        val = z[ni, nj]
                        if val == val:
                            nbr_z[n_nbr] = val
                            nbr_w[n_nbr] = w
                            n_nbr += 1

            if n_nbr == 0:
                out_pval[i, j] = np.nan
                continue

            # Observed lag and LISA
            lag = 0.0
            for k in range(n_nbr):
                lag += nbr_w[k] * nbr_z[k]
            I_obs = zi * inv_var * lag
            abs_I = abs(I_obs)

            count = 0
            rng = np.int64(seed) + np.int64(i * cols + j)

            for p in range(n_perms):
                for k in range(n_nbr - 1, 0, -1):
                    rng = rng * np.int64(6364136223846793005) + np.int64(1442695040888963407)
                    idx = int((rng >> 33) & np.int64(0x7fffffff)) % (k + 1)
                    tmp = nbr_z[k]
                    nbr_z[k] = nbr_z[idx]
                    nbr_z[idx] = tmp

                perm_lag = 0.0
                for k in range(n_nbr):
                    perm_lag += nbr_w[k] * nbr_z[k]
                if abs(zi * inv_var * perm_lag) >= abs_I:
                    count += 1

            out_pval[i, j] = np.float32(count + 1) / np.float32(n_perms + 1)


def _perm_chunk_wrapper(chunk_z, kernel, inv_var, n_perms, seed):
    """Wrapper for map_overlap: compute p-values for one chunk."""
    chunk_z = chunk_z.astype(np.float32)
    out = np.empty(chunk_z.shape, dtype=np.float32)
    _perm_pvalue_ngjit(chunk_z, kernel, inv_var, n_perms, seed, out)
    return out
```

Replace `_lisa_dask_numpy = _not_implemented`:

```python
def _lisa_dask_numpy(raster, kernel, n_permutations=999, boundary='nan'):
    """LISA -- dask+numpy backend."""
    import dask.array as da

    data = raster.data.astype(np.float32)
    mask = ~da.isnan(data)

    mean, var_sum, N = da.compute(
        da.nanmean(data),
        da.nansum((data - da.nanmean(data)) ** 2),
        da.sum(mask),
    )
    N = int(N)
    if N < 2:
        nans = np.full(data.shape, np.nan, dtype=np.float32)
        zeros = np.zeros(data.shape, dtype=np.int8)
        return nans, nans.copy(), zeros

    mean = float(mean)
    var = float(var_sum) / N
    if var == 0.0:
        nans = np.full(data.shape, np.nan, dtype=np.float32)
        zeros = np.zeros(data.shape, dtype=np.int8)
        return nans, nans.copy(), zeros

    inv_var = np.float32(1.0 / var)
    z = da.where(mask, data - mean, np.nan).astype(np.float32)
    bnd = _boundary_to_dask(boundary)

    # 1. Spatial lag via map_overlap
    lag = z.map_overlap(
        partial(_lag_chunk_numpy, kernel=kernel),
        depth=(1, 1), boundary=bnd,
        meta=np.array((), dtype=np.float32),
    )

    # 2. LISA values (element-wise, lazy)
    lisa_vals = z * lag * inv_var

    # 3. p-values via map_overlap (permutation)
    p_vals = z.map_overlap(
        partial(_perm_chunk_wrapper, kernel=kernel,
                inv_var=inv_var, n_perms=n_permutations, seed=42),
        depth=(1, 1), boundary=bnd,
        meta=np.array((), dtype=np.float32),
    )

    # 4. Cluster classification (element-wise via map_blocks)
    def _classify_block(z_blk, lag_blk, p_blk):
        out = np.zeros(z_blk.shape, dtype=np.int8)
        sig = p_blk <= 0.05
        out[sig & (z_blk > 0) & (lag_blk > 0)] = 1  # HH
        out[sig & (z_blk < 0) & (lag_blk < 0)] = 2  # LL
        out[sig & (z_blk > 0) & (lag_blk < 0)] = 3  # HL
        out[sig & (z_blk < 0) & (lag_blk > 0)] = 4  # LH
        nan_mask = np.isnan(z_blk)
        out[nan_mask] = 0
        return out

    cluster = da.map_blocks(
        _classify_block, z, lag, p_vals,
        dtype=np.int8, meta=np.array((), dtype=np.int8),
    )

    # Compute all lazy arrays
    lisa_vals, p_vals, cluster = da.compute(lisa_vals, p_vals, cluster)

    return (
        lisa_vals.astype(np.float32),
        p_vals.astype(np.float32),
        cluster.astype(np.int8),
    )
```

- [ ] **Step 5: Implement _morans_i_dask_cupy and _lisa_dask_cupy**

Replace both dask+cupy stubs. These fall back to the dask+numpy implementations since the permutation step is CPU-bound anyway:

```python
def _morans_i_dask_cupy(raster, kernel, boundary='nan'):
    """Global Moran's I -- dask+cupy backend.

    Falls back to dask+numpy. The convolution is too small (3x3 kernel)
    to benefit from GPU, and the reduction is scalar.
    """
    import dask.array as da
    data_np = raster.data.map_blocks(
        lambda b: cupy.asnumpy(b), dtype=np.float32,
        meta=np.array((), dtype=np.float32),
    )
    raster_np = xr.DataArray(data_np, dims=raster.dims, coords=raster.coords)
    return _morans_i_dask_numpy(raster_np, kernel=kernel, boundary=boundary)


def _lisa_dask_cupy(raster, kernel, n_permutations=999, boundary='nan'):
    """LISA -- dask+cupy backend.

    Falls back to dask+numpy. Permutation is branching-heavy and runs
    faster on CPU (same rationale as emerging_hotspots.py).
    """
    import dask.array as da
    data_np = raster.data.map_blocks(
        lambda b: cupy.asnumpy(b), dtype=np.float32,
        meta=np.array((), dtype=np.float32),
    )
    raster_np = xr.DataArray(data_np, dims=raster.dims, coords=raster.coords)
    return _lisa_dask_numpy(raster_np, kernel=kernel,
                            n_permutations=n_permutations, boundary=boundary)
```

- [ ] **Step 6: Run all dask tests**

```bash
cd .claude/worktrees/issue-1135
python -m pytest xrspatial/tests/test_autocorrelation.py::TestMoransIDask -v
python -m pytest xrspatial/tests/test_autocorrelation.py::TestLisaDask -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add xrspatial/autocorrelation.py xrspatial/tests/test_autocorrelation.py
git commit -m "Add dask backends for Moran's I and LISA (#1135)"
```

---

### Task 6: Edge Cases and Cross-Backend Tests

**Files:**
- Modify: `xrspatial/tests/test_autocorrelation.py`

- [ ] **Step 1: Add edge case and contiguity kernel tests**

Append to `xrspatial/tests/test_autocorrelation.py`:

```python
# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_cell(self):
        data = np.array([[5.0]], dtype=np.float32)
        result = morans_i(_make_raster(data))
        assert np.isnan(float(result))

    def test_all_nan(self):
        data = np.full((4, 4), np.nan, dtype=np.float32)
        result = morans_i(_make_raster(data))
        assert np.isnan(float(result))

    def test_single_non_nan(self):
        data = np.full((4, 4), np.nan, dtype=np.float32)
        data[2, 2] = 1.0
        result = morans_i(_make_raster(data))
        assert np.isnan(float(result))

    def test_nan_corners(self):
        data = _gradient(6)
        data[0, 0] = np.nan
        data[0, -1] = np.nan
        data[-1, 0] = np.nan
        data[-1, -1] = np.nan
        result = morans_i(_make_raster(data))
        assert not np.isnan(float(result))

    def test_lisa_single_cell(self):
        data = np.array([[5.0]], dtype=np.float32)
        ds = lisa(_make_raster(data), n_permutations=9)
        assert np.all(np.isnan(ds['lisa_values'].values))

    def test_lisa_all_nan(self):
        data = np.full((4, 4), np.nan, dtype=np.float32)
        ds = lisa(_make_raster(data), n_permutations=9)
        assert np.all(np.isnan(ds['lisa_values'].values))

    def test_contiguity_kernel_invalid(self):
        with pytest.raises(ValueError, match="Invalid contiguity"):
            _contiguity_kernel('bishop')

    def test_contiguity_kernel_queen(self):
        k = _contiguity_kernel('queen')
        assert k.shape == (3, 3)
        assert k[1, 1] == 0.0
        assert k.sum() == 8.0

    def test_contiguity_kernel_rook(self):
        k = _contiguity_kernel('rook')
        assert k.shape == (3, 3)
        assert k[1, 1] == 0.0
        assert k.sum() == 4.0
        assert k[0, 0] == 0.0  # corners are zero

    def test_rook_checkerboard_perfect_negative(self):
        """4-connected rook on checkerboard: every neighbor is opposite."""
        raster = _make_raster(_checkerboard(8))
        I = float(morans_i(raster, contiguity='rook'))
        assert I < -0.9, f"Rook checkerboard should be near -1, got {I}"
```

- [ ] **Step 2: Run edge case tests**

```bash
cd .claude/worktrees/issue-1135
python -m pytest xrspatial/tests/test_autocorrelation.py::TestEdgeCases -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add xrspatial/tests/test_autocorrelation.py
git commit -m "Add edge case and contiguity tests (#1135)"
```

---

### Task 7: Register Exports

**Files:**
- Modify: `xrspatial/__init__.py` (line 2 area)

- [ ] **Step 1: Add imports to __init__.py**

Add after line 1 (`from xrspatial.aspect import aspect  # noqa`):

```python
from xrspatial.autocorrelation import lisa  # noqa
from xrspatial.autocorrelation import morans_i  # noqa
```

- [ ] **Step 2: Verify imports work**

```bash
cd .claude/worktrees/issue-1135
python -c "from xrspatial import morans_i, lisa; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run full test suite for autocorrelation**

```bash
cd .claude/worktrees/issue-1135
python -m pytest xrspatial/tests/test_autocorrelation.py -v
```

Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add xrspatial/__init__.py
git commit -m "Export morans_i and lisa from xrspatial (#1135)"
```

---

### Task 8: Documentation

**Files:**
- Create: `docs/source/reference/autocorrelation.rst`
- Modify: `docs/source/reference/index.rst` (line 10)
- Modify: `README.md` (line ~567)

- [ ] **Step 1: Create autocorrelation.rst**

Create `docs/source/reference/autocorrelation.rst`:

```rst
.. _reference.autocorrelation:

***********************
Spatial Autocorrelation
***********************

.. caution::

   LISA uses ``dask.array.map_overlap`` with depth 1.  Each chunk dimension
   must be **at least 3 cells** (larger than the contiguity kernel radius).

.. note::

   Permutation-based p-values use a fixed internal seed for reproducibility
   within a single call.  Results are deterministic for the same input and
   ``n_permutations`` value.

Global Moran's I
================
.. autosummary::
   :toctree: _autosummary

   xrspatial.autocorrelation.morans_i

Local Moran's I (LISA)
======================
.. autosummary::
   :toctree: _autosummary

   xrspatial.autocorrelation.lisa
```

- [ ] **Step 2: Add to docs toctree**

In `docs/source/reference/index.rst`, add `autocorrelation` after line 9 (after the `:maxdepth: 2` line, before `classification`):

```rst
   autocorrelation
   classification
```

- [ ] **Step 3: Add Spatial Statistics section to README**

In `README.md`, insert after line 567 (`-----------` after Dasymetric section) and before `#### Usage`:

```markdown

### **Spatial Statistics**

| Name | Description | Source | NumPy xr.DataArray | Dask xr.DataArray | CuPy GPU xr.DataArray | Dask GPU xr.DataArray |
|:----------:|:------------|:------:|:----------------------:|:--------------------:|:-------------------:|:------:|
| [Moran's I](xrspatial/autocorrelation.py) | Global spatial autocorrelation with analytical inference | Cliff & Ord 1981 | ✅️ | ✅️ | ✅️ | ✅️ |
| [LISA](xrspatial/autocorrelation.py) | Local Indicators of Spatial Association with permutation p-values | Anselin 1995 | ✅️ | ✅️ | ✅️ | ✅️ |

-----------
```

- [ ] **Step 4: Commit**

```bash
git add docs/source/reference/autocorrelation.rst docs/source/reference/index.rst README.md
git commit -m "Add spatial autocorrelation docs and README entry (#1135)"
```

---

### Task 9: User Guide Notebook

**Files:**
- Create: `examples/user_guide/48_Spatial_Autocorrelation.ipynb`

- [ ] **Step 1: Create notebook**

Create `examples/user_guide/48_Spatial_Autocorrelation.ipynb` with cells:

**Cell 1 (markdown):**
```markdown
# Spatial Autocorrelation: Moran's I and LISA

This notebook demonstrates how to measure spatial autocorrelation in raster data using `morans_i` (global) and `lisa` (local).

**Spatial autocorrelation** measures whether nearby pixels tend to have similar values (positive autocorrelation) or dissimilar values (negative autocorrelation). It answers the question: "Is the spatial pattern in this raster clustered, dispersed, or random?"

- **Global Moran's I** produces a single statistic for the entire raster.
- **LISA (Local Indicators of Spatial Association)** produces a per-pixel statistic, identifying where clusters and outliers are.
```

**Cell 2 (code):**
```python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from xrspatial import morans_i, lisa
from xrspatial.terrain import generate_terrain
```

**Cell 3 (markdown):**
```markdown
## Generate sample data

We'll create a synthetic elevation surface with spatial structure, plus a random noise surface for comparison.
```

**Cell 4 (code):**
```python
# Spatially structured surface (elevation)
terrain = generate_terrain(canvas_width=200, canvas_height=200)

# Random noise (no spatial structure)
rng = np.random.default_rng(42)
noise = xr.DataArray(
    rng.standard_normal((200, 200)).astype(np.float32),
    dims=('y', 'x'),
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
terrain.plot(ax=axes[0], cmap='terrain')
axes[0].set_title('Elevation (spatially structured)')
noise.plot(ax=axes[1], cmap='RdBu_r')
axes[1].set_title('Random noise')
plt.tight_layout()
plt.show()
```

**Cell 5 (markdown):**
```markdown
## Global Moran's I

A single statistic summarizing the degree of spatial autocorrelation:
- **I > E[I]**: positive autocorrelation (clustering)
- **I < E[I]**: negative autocorrelation (dispersion)
- **I ≈ E[I]**: random spatial pattern
```

**Cell 6 (code):**
```python
for name, raster in [('Elevation', terrain), ('Noise', noise)]:
    result = morans_i(raster, contiguity='queen')
    I = float(result)
    print(f"{name}:")
    print(f"  Moran's I = {I:.4f}")
    print(f"  Expected   = {result.attrs['expected_I']:.4f}")
    print(f"  z-score    = {result.attrs['z_score']:.2f}")
    print(f"  p-value    = {result.attrs['p_value']:.2e}")
    print()
```

**Cell 7 (markdown):**
```markdown
## Queen vs Rook contiguity

Queen contiguity uses all 8 neighbors (including diagonals). Rook contiguity uses only 4 (up/down/left/right).
```

**Cell 8 (code):**
```python
I_queen = float(morans_i(terrain, contiguity='queen'))
I_rook = float(morans_i(terrain, contiguity='rook'))
print(f"Queen I = {I_queen:.4f}")
print(f"Rook  I = {I_rook:.4f}")
```

**Cell 9 (markdown):**
```markdown
## LISA: Local spatial autocorrelation

LISA identifies **where** clustering occurs. Each pixel gets:
- A local I value (positive = similar neighbors, negative = dissimilar)
- A p-value from permutation testing
- A cluster classification: HH (hot spot), LL (cold spot), HL/LH (spatial outliers)
```

**Cell 10 (code):**
```python
ds = lisa(terrain, contiguity='queen', n_permutations=999)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ds['lisa_values'].plot(ax=axes[0], cmap='RdBu_r', robust=True)
axes[0].set_title('Local Moran\'s I')

ds['p_values'].plot(ax=axes[1], cmap='YlOrRd_r', vmin=0, vmax=0.1)
axes[1].set_title('p-values')

# Cluster map with custom colors
cluster_cmap = ListedColormap(['lightgrey', 'red', 'blue', 'pink', 'lightblue'])
ds['cluster'].plot(ax=axes[2], cmap=cluster_cmap, vmin=0, vmax=4,
                   add_colorbar=False)
axes[2].set_title('Clusters: grey=NS, red=HH, blue=LL, pink=HL, cyan=LH')

plt.tight_layout()
plt.show()
```

**Cell 11 (markdown):**
```markdown
## Comparison: structured vs random

On random noise, LISA should find few significant clusters (mostly not-significant grey).
```

**Cell 12 (code):**
```python
ds_noise = lisa(noise, n_permutations=999)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cluster_cmap = ListedColormap(['lightgrey', 'red', 'blue', 'pink', 'lightblue'])

ds['cluster'].plot(ax=axes[0], cmap=cluster_cmap, vmin=0, vmax=4, add_colorbar=False)
axes[0].set_title('Elevation clusters')

ds_noise['cluster'].plot(ax=axes[1], cmap=cluster_cmap, vmin=0, vmax=4, add_colorbar=False)
axes[1].set_title('Noise clusters (mostly NS)')

plt.tight_layout()
plt.show()

sig_terrain = float((ds['cluster'].values != 0).mean() * 100)
sig_noise = float((ds_noise['cluster'].values != 0).mean() * 100)
print(f"Significant pixels: elevation={sig_terrain:.1f}%, noise={sig_noise:.1f}%")
```

- [ ] **Step 2: Verify notebook runs**

```bash
cd .claude/worktrees/issue-1135
python -m jupyter nbconvert --to notebook --execute examples/user_guide/48_Spatial_Autocorrelation.ipynb --output /dev/null 2>&1 | tail -5
```

If jupyter is not available, verify the key cells run as a script:

```bash
cd .claude/worktrees/issue-1135
python -c "
from xrspatial import morans_i, lisa
from xrspatial.terrain import generate_terrain
terrain = generate_terrain(canvas_width=50, canvas_height=50)
print('Global I:', float(morans_i(terrain)))
ds = lisa(terrain, n_permutations=99)
print('LISA vars:', list(ds.data_vars))
print('Clusters:', set(ds['cluster'].values.flat))
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add examples/user_guide/48_Spatial_Autocorrelation.ipynb
git commit -m "Add spatial autocorrelation user guide notebook (#1135)"
```

---

### Task 10: Final Verification

- [ ] **Step 1: Run the full autocorrelation test suite**

```bash
cd .claude/worktrees/issue-1135
python -m pytest xrspatial/tests/test_autocorrelation.py -v --tb=short
```

Expected: all tests PASS

- [ ] **Step 2: Run a smoke test of the full xrspatial test suite**

```bash
cd .claude/worktrees/issue-1135
python -m pytest xrspatial/tests/ -x -q --tb=line 2>&1 | tail -10
```

Verify no regressions in existing tests.

- [ ] **Step 3: Review git log**

```bash
cd .claude/worktrees/issue-1135
git log --oneline master..issue-1135
```

Expected: 7-8 commits, each referencing #1135.
