# Spatial Autocorrelation: Moran's I and LISA

**Issue:** #1135 (partial -- Global Moran's I, Local Moran's I, queen/rook contiguity)  
**Date:** 2026-04-01

## Scope

This spec covers the first increment of #1135:

- Global Moran's I with analytical inference
- Local Moran's I (LISA) with permutation-based pseudo p-values
- Queen and rook contiguity weights (3x3 kernel)

Geary's C, join count statistics, and distance-band weights are deferred to follow-up issues.

## Public API

Two functions in `xrspatial/autocorrelation.py`:

### `morans_i`

```python
def morans_i(
    raster: xr.DataArray,
    contiguity: str = 'queen',
    boundary: str = 'nan',
) -> xr.DataArray:
```

Returns a scalar (0-dimensional) DataArray. The `.item()` value is the I statistic.
Attrs carry analytical inference results:

| Attr | Type | Description |
|------|------|-------------|
| `expected_I` | float | -1/(N-1) |
| `variance_I` | float | Cliff & Ord analytical variance |
| `z_score` | float | (I - E[I]) / sqrt(Var[I]) |
| `p_value` | float | Two-sided, from normal approximation |
| `N` | int | Count of non-NaN pixels |
| `S0` | float | Sum of all weights |
| `contiguity` | str | 'queen' or 'rook' |

### `lisa`

```python
def lisa(
    raster: xr.DataArray,
    contiguity: str = 'queen',
    n_permutations: int = 999,
    boundary: str = 'nan',
) -> xr.Dataset:
```

Returns a Dataset with three DataVariables:

| Variable | Dims | Dtype | Description |
|----------|------|-------|-------------|
| `lisa_values` | (y, x) | float32 | Local I_i per pixel |
| `p_values` | (y, x) | float32 | Pseudo p-value from permutation |
| `cluster` | (y, x) | int8 | 0=NS, 1=HH, 2=LL, 3=HL, 4=LH |

Dataset attrs: `n_permutations`, `contiguity`, `global_morans_I`.

Cluster codes use significance threshold p <= 0.05. Pixels with p > 0.05 get code 0 regardless of their quadrant.

## Mathematics

### Global Moran's I

```
z = x - mean(x)
lag = convolve(z, W)           # W = queen or rook kernel
I = (N / S0) * sum(z * lag) / sum(z^2)
```

S0 (total weight sum) accounts for border effects and NaN gaps by convolving a non-NaN mask with the weight kernel and summing the result.

Analytical expected value and variance follow Cliff & Ord (1981):

```
E[I] = -1 / (N - 1)
Var[I] uses S0, S1, S2, N, and the kurtosis of z
```

Where S1 = (1/2) * sum_ij (w_ij + w_ji)^2 and S2 = sum_i (sum_j w_ij + sum_j w_ji)^2. For symmetric binary weights (queen/rook), S1 = 2 * S0 and S2 simplifies.

### Local Moran's I (LISA)

```
I_i = (z_i / var(x)) * sum_j(w_ij * z_j)
```

### Permutation pseudo p-value

For each pixel i:
1. Extract the neighbor values (up to 8 for queen, 4 for rook).
2. Shuffle the neighbor values n_permutations times (Fisher-Yates).
3. Recompute I_i with each shuffled set.
4. p_value = (count(|I_perm| >= |I_obs|) + 1) / (n_permutations + 1)

The +1 correction includes the observed value as one permutation (Davison & Hinkley, 1997).

### Cluster classification

| Code | Label | Condition (when p <= 0.05) |
|------|-------|---------------------------|
| 0 | NS | not significant |
| 1 | HH | z_i > 0 and lag_i > 0 |
| 2 | LL | z_i < 0 and lag_i < 0 |
| 3 | HL | z_i > 0 and lag_i < 0 |
| 4 | LH | z_i < 0 and lag_i > 0 |

### NaN handling

- NaN input pixels produce NaN in lisa_values and p_values, 0 in cluster.
- NaN neighbors are excluded from lag sums (their weight drops to zero).
- Constant rasters (zero variance) produce NaN for all statistics.

## Contiguity kernels

Queen (8 neighbors):
```
[[1, 1, 1],
 [1, 0, 1],
 [1, 1, 1]]
```

Rook (4 neighbors):
```
[[0, 1, 0],
 [1, 0, 1],
 [0, 1, 0]]
```

## Internal Architecture

### Backend dispatch

Both public functions validate input via `_validate_raster`, build the contiguity kernel, then dispatch through `ArrayTypeFunctionMapping`:

```
morans_i / lisa
  -> _validate_raster(raster, ndim=2)
  -> kernel = _contiguity_kernel(contiguity)
  -> ArrayTypeFunctionMapping(numpy, cupy, dask, dask_cupy)(raster)(...)
```

### Backend implementations

**numpy:** Compute mean/var eagerly. Spatial lag via `_convolve_2d_numpy` (imported from convolution module) or a local implementation. LISA permutation via `@ngjit` loop that iterates pixels, extracts 8 neighbors, runs Fisher-Yates shuffle n_permutations times.

**cupy:** Same structure but GPU arrays. Spatial lag via CuPy convolution. Permutation via `@cuda.jit` kernel where each thread handles one pixel. RNG uses `numba.cuda.random.xoroshiro128p_uniform_float32` for Fisher-Yates shuffle.

**dask+numpy:** Eagerly compute global mean, var, N with `da.compute()`. Pass as scalars to chunk function via `partial()`. Single `map_overlap(depth=1, boundary=...)` call with fused chunk function that computes lag + I_i + permutation + cluster. Returns `(3, H, W)` float32 array (lisa_values, p_values, cluster). Unpacked into Dataset after.

**dask+cupy:** Same structure as dask+numpy but chunk function dispatches to CUDA kernel internally.

### Fused LISA chunk function

```python
def _lisa_chunk_numpy(chunk, kernel, global_mean, global_var, n_permutations, seed):
    """Single-pass: lag + LISA + permutation + cluster for one chunk."""
    rows, cols = chunk.shape
    z = chunk - global_mean
    out = np.empty((3, rows, cols), dtype=np.float32)
    # For each pixel: read neighbors, compute lag, permute, classify
    _lisa_fused_ngjit(z, kernel, global_var, n_permutations, seed, out)
    return out
```

The `@ngjit` inner function handles the pixel loop with neighbor extraction, lag computation, Fisher-Yates permutation, and cluster assignment in a single pass.

### Global Moran's I backend flow

```
numpy:
  z = data - mean
  lag = convolve(z, kernel)          # reuse focal convolution
  mask = ~isnan(data)
  S0 = sum(convolve(mask, kernel))   # total weight count
  N = sum(mask)
  I = (N / S0) * nansum(z * lag) / nansum(z^2)
  # analytical variance via S0, S1, S2, N

dask:
  mean, var, N = da.compute(da.nanmean(data), da.nanvar(data), da.sum(~da.isnan(data)))
  z = data - mean
  lag = map_overlap(_convolve_chunk, depth=1, ...)
  S0 = da.sum(map_overlap(_convolve_mask_chunk, depth=1, ...))
  I = (N / S0) * da.nansum(z * lag) / da.nansum(z**2)
  I = I.compute()
```

## map_overlap specifics

- **depth:** 1 in both dimensions (3x3 kernel, half-size = 1)
- **boundary:** passed through `_boundary_to_dask(boundary)`, default `np.nan`
- **meta:** `np.array((), dtype=np.float32)` for numpy, `cupy.array((), dtype=cupy.float32)` for cupy
- **Fused output:** LISA chunk function receives a 2D chunk and returns a `(3, H, W)` array. To make `map_overlap` accept this shape change, pass `new_axis=0` so dask knows the output gains a leading dimension. After the `map_overlap` call, slice `result[0]`, `result[1]`, `result[2]` to get the three output bands. If `new_axis` with `map_overlap` proves brittle at runtime, fall back to three separate `map_overlap` calls (one per output band) at the cost of 3x neighbor reads.

## Seed handling

Permutation tests need reproducible results for cross-backend testing.

- numpy/dask+numpy: `np.random.SeedSequence(seed)` spawns per-pixel child sequences. In practice, the `@ngjit` function uses a simple LCG seeded with `seed + pixel_linear_index` for Fisher-Yates shuffles over 8 elements. Good enough for 8-element shuffles.
- cupy/dask+cupy: `xoroshiro128p` states initialized with `create_xoroshiro128p_states(n_threads, seed)`.
- Public API does not expose seed. Internal backends accept it for testing. Default seed is 0 for determinism within a single call (users wanting different random draws can call twice -- permutation p-values are not sensitive to seed choice for n >= 999).

## Testing

File: `xrspatial/tests/test_autocorrelation.py`

### Known-value tests

| Input | Expected I | Rationale |
|-------|-----------|-----------|
| 4x4 checkerboard | I < -0.8 | Strong negative autocorrelation |
| 4x4 row gradient | I > 0.5 | Positive autocorrelation |
| 8x8 random (seeded) | -0.3 < I < 0.3 | Near zero |
| Constant value | NaN | Zero variance |

### LISA tests

- Checkerboard: all I_i negative, all clusters HL or LH, all p-values < 0.05
- Gradient: center pixels positive (HH or LL), p-values < 0.05
- NaN corners: NaN in output at those positions, valid elsewhere

### Edge cases

- Single-cell raster: returns NaN
- All-NaN raster: returns NaN
- Raster with one non-NaN pixel: returns NaN

### Cross-backend parity

- `assert_numpy_equals_dask_numpy` for both functions
- `assert_numpy_equals_cupy` (skip if no GPU)
- Fixed seed ensures identical permutation sequences

### Contiguity

- Queen vs rook produce different I values on same input
- Rook on 4x4 checkerboard: I = -1.0 (perfect negative, all 4 neighbors opposite)

## Documentation

- Add API entry in `docs/source/reference/` (new section or extend focal tools)
- Add row to README feature matrix under a new "Spatial Statistics" category

## Files changed

| File | Change |
|------|--------|
| `xrspatial/autocorrelation.py` | New module |
| `xrspatial/__init__.py` | Export `morans_i`, `lisa` |
| `xrspatial/tests/test_autocorrelation.py` | New test file |
| `docs/source/reference/autocorrelation.rst` | New API docs |
| `README.md` | Feature matrix row |
| `examples/user_guide/` | New notebook |
