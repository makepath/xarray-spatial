# GeoTIFF performance and memory controls implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `dtype` to `open_geotiff`, `compression_level` to `to_geotiff`, and VRT tiled output when `to_geotiff` is given a `.vrt` path. Issue #1083.

**Architecture:** Three independent features threaded into the existing geotiff module. `dtype` intercepts each read path after tile/strip decode. `compression_level` passes through `to_geotiff` → `write()` → `_write_tiled`/`_write_stripped` → `compress()`. VRT output adds a new code path in `to_geotiff` that slices the input into per-chunk GeoTIFFs and calls `write_vrt()`.

**Tech Stack:** numpy, xarray, dask (optional), numba, cupy (optional). All existing dependencies.

---

## File map

| File | Role | Changes |
|------|------|---------|
| `xrspatial/geotiff/__init__.py` | Public API | Add `dtype` param to `open_geotiff`, `read_geotiff_dask`, `read_geotiff_gpu`, `_delayed_read_window`. Add `compression_level` param to `to_geotiff`, `write_geotiff_gpu`. Add VRT output path in `to_geotiff`. Add `_validate_dtype_cast()` helper. |
| `xrspatial/geotiff/_writer.py` | Tile/strip compression, file assembly | Thread `compression_level` through `write()`, `_write_tiled()`, `_write_stripped()`, `_prepare_tile()`. |
| `xrspatial/geotiff/_compression.py` | Codec dispatch | No changes needed -- `compress()` already accepts `level`. |
| `xrspatial/geotiff/tests/test_dtype_read.py` | New test file | Tests for `dtype` on eager, dask, validation. |
| `xrspatial/geotiff/tests/test_compression_level.py` | New test file | Tests for `compression_level` round-trips. |
| `xrspatial/geotiff/tests/test_vrt_write.py` | New test file | Tests for `.vrt` output path, dask streaming, numpy slicing, edge cases. |

---

### Task 1: `compression_level` plumbing through the writer

The simplest of the three features. Thread the level integer from the public API down to the `compress()` call.

**Files:**
- Modify: `xrspatial/geotiff/_writer.py:298-403` (`_write_stripped`, `_prepare_tile`, `_write_tiled`, `write`)
- Modify: `xrspatial/geotiff/__init__.py:342-519` (`to_geotiff`, `write_geotiff_gpu`)
- Test: `xrspatial/geotiff/tests/test_compression_level.py` (create)

- [ ] **Step 1: Write the failing test**

Create `xrspatial/geotiff/tests/test_compression_level.py`:

```python
"""Tests for compression_level parameter on to_geotiff."""
import numpy as np
import os
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


@pytest.fixture
def sample_float32(tmp_path):
    """100x100 float32 raster with coords and CRS."""
    arr = np.random.default_rng(42).random((100, 100), dtype=np.float32)
    y = np.linspace(40.0, 41.0, 100)
    x = np.linspace(-105.0, -104.0, 100)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 4326})
    return da


class TestCompressionLevel:
    """Round-trip tests: write with level, read back, verify data matches."""

    def test_zstd_level_1_round_trip(self, sample_float32, tmp_path):
        path = str(tmp_path / 'test_1083_zstd_l1.tif')
        to_geotiff(sample_float32, path, compression='zstd',
                   compression_level=1)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values,
                                             sample_float32.values, decimal=6)

    def test_zstd_level_22_round_trip(self, sample_float32, tmp_path):
        path = str(tmp_path / 'test_1083_zstd_l22.tif')
        to_geotiff(sample_float32, path, compression='zstd',
                   compression_level=22)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values,
                                             sample_float32.values, decimal=6)

    def test_deflate_level_1_round_trip(self, sample_float32, tmp_path):
        path = str(tmp_path / 'test_1083_deflate_l1.tif')
        to_geotiff(sample_float32, path, compression='deflate',
                   compression_level=1)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values,
                                             sample_float32.values, decimal=6)

    def test_deflate_level_9_round_trip(self, sample_float32, tmp_path):
        path = str(tmp_path / 'test_1083_deflate_l9.tif')
        to_geotiff(sample_float32, path, compression='deflate',
                   compression_level=9)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values,
                                             sample_float32.values, decimal=6)

    def test_higher_level_produces_smaller_file(self, sample_float32, tmp_path):
        path_l1 = str(tmp_path / 'test_1083_small_l1.tif')
        path_l22 = str(tmp_path / 'test_1083_small_l22.tif')
        to_geotiff(sample_float32, path_l1, compression='zstd',
                   compression_level=1)
        to_geotiff(sample_float32, path_l22, compression='zstd',
                   compression_level=22)
        assert os.path.getsize(path_l22) <= os.path.getsize(path_l1)

    def test_level_none_uses_default(self, sample_float32, tmp_path):
        path = str(tmp_path / 'test_1083_default.tif')
        to_geotiff(sample_float32, path, compression='zstd',
                   compression_level=None)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values,
                                             sample_float32.values, decimal=6)

    def test_level_ignored_for_lzw(self, sample_float32, tmp_path):
        """LZW has no level support; setting one should not error."""
        path = str(tmp_path / 'test_1083_lzw_level.tif')
        to_geotiff(sample_float32, path, compression='lzw',
                   compression_level=5)
        result = open_geotiff(path)
        np.testing.assert_array_almost_equal(result.values,
                                             sample_float32.values, decimal=6)

    def test_invalid_level_raises(self, sample_float32, tmp_path):
        path = str(tmp_path / 'test_1083_bad_level.tif')
        with pytest.raises(ValueError, match='compression_level'):
            to_geotiff(sample_float32, path, compression='zstd',
                       compression_level=99)

    def test_invalid_deflate_level_raises(self, sample_float32, tmp_path):
        path = str(tmp_path / 'test_1083_bad_deflate.tif')
        with pytest.raises(ValueError, match='compression_level'):
            to_geotiff(sample_float32, path, compression='deflate',
                       compression_level=10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/issue-1083 && python -m pytest xrspatial/geotiff/tests/test_compression_level.py -v --no-header -x 2>&1 | head -30`
Expected: FAIL -- `to_geotiff()` got an unexpected keyword argument `compression_level`.

- [ ] **Step 3: Add `compression_level` validation to `to_geotiff`**

In `xrspatial/geotiff/__init__.py`, change the `to_geotiff` signature and add validation before the write call. Add `compression_level: int | None = None` parameter after `compression`. Add this validation block before the `write()` call (before line 499):

```python
    # Validate compression_level
    _LEVEL_RANGES = {
        'deflate': (1, 9), 'zstd': (1, 22), 'lz4': (0, 16),
    }
    if compression_level is not None:
        level_range = _LEVEL_RANGES.get(compression)
        if level_range is not None:
            lo, hi = level_range
            if not (lo <= compression_level <= hi):
                raise ValueError(
                    f"compression_level={compression_level} out of range "
                    f"for {compression} (valid: {lo}-{hi})")
```

Pass `compression_level=compression_level` to the `write()` call at line 499.

- [ ] **Step 4: Thread `compression_level` through `write()` → `_write_tiled` → `_prepare_tile`**

In `xrspatial/geotiff/_writer.py`:

1. Add `compression_level: int | None = None` parameter to `write()` (after `predictor`).
2. Pass `compression_level=compression_level` to `_write_tiled()` and `_write_stripped()` calls inside `write()`.
3. Add `compression_level: int | None = None` parameter to `_write_tiled()` and `_write_stripped()`.
4. Add `compression_level: int | None = None` parameter to `_prepare_tile()`.
5. In `_prepare_tile()`, change `return compress(tile_data, compression)` to `return compress(tile_data, compression, level=compression_level)` when `compression_level is not None`, else `return compress(tile_data, compression)`. Simplest: `return compress(tile_data, compression) if compression_level is None else compress(tile_data, compression, level=compression_level)`.
6. In `_write_stripped()`, do the same for the `compress(strip_data, compression)` call at the sequential path.
7. Pass `compression_level` through all `_prepare_tile` call sites in `_write_tiled`.

The `compress()` function in `_compression.py` already accepts `level` as a keyword argument with default 6, so we just need to pass it when non-None.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd .claude/worktrees/issue-1083 && python -m pytest xrspatial/geotiff/tests/test_compression_level.py -v --no-header 2>&1 | tail -20`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
cd .claude/worktrees/issue-1083
git add xrspatial/geotiff/__init__.py xrspatial/geotiff/_writer.py xrspatial/geotiff/tests/test_compression_level.py
git commit -m "Add compression_level parameter to to_geotiff (#1083)"
```

---

### Task 2: `dtype` parameter on `open_geotiff` (eager and dask paths)

**Files:**
- Modify: `xrspatial/geotiff/__init__.py:151-636` (`open_geotiff`, `read_geotiff_dask`, `_delayed_read_window`, `read_geotiff_gpu`)
- Test: `xrspatial/geotiff/tests/test_dtype_read.py` (create)

- [ ] **Step 1: Write the failing test**

Create `xrspatial/geotiff/tests/test_dtype_read.py`:

```python
"""Tests for dtype parameter on open_geotiff."""
import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


@pytest.fixture
def float64_tif(tmp_path):
    """Write a float64 GeoTIFF for dtype cast tests."""
    arr = np.random.default_rng(99).random((80, 80)).astype(np.float64)
    y = np.linspace(40.0, 41.0, 80)
    x = np.linspace(-105.0, -104.0, 80)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 4326})
    path = str(tmp_path / 'test_1083_f64.tif')
    to_geotiff(da, path, compression='none')
    return path, arr


@pytest.fixture
def uint16_tif(tmp_path):
    """Write a uint16 GeoTIFF for dtype cast tests."""
    arr = np.random.default_rng(77).integers(0, 10000, (60, 60),
                                             dtype=np.uint16)
    y = np.linspace(40.0, 41.0, 60)
    x = np.linspace(-105.0, -104.0, 60)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 4326})
    path = str(tmp_path / 'test_1083_u16.tif')
    to_geotiff(da, path, compression='none')
    return path, arr


class TestDtypeEager:
    """dtype parameter on eager (numpy) reads."""

    def test_float64_to_float32(self, float64_tif):
        path, orig = float64_tif
        result = open_geotiff(path, dtype='float32')
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(
            result.values, orig.astype(np.float32), decimal=6)

    def test_float64_to_float16(self, float64_tif):
        path, orig = float64_tif
        result = open_geotiff(path, dtype=np.float16)
        assert result.dtype == np.float16

    def test_uint16_to_int32(self, uint16_tif):
        path, orig = uint16_tif
        result = open_geotiff(path, dtype='int32')
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result.values, orig.astype(np.int32))

    def test_uint16_to_uint8(self, uint16_tif):
        """Narrowing int cast is allowed (user asked for it)."""
        path, _ = uint16_tif
        result = open_geotiff(path, dtype='uint8')
        assert result.dtype == np.uint8

    def test_float_to_int_raises(self, float64_tif):
        path, _ = float64_tif
        with pytest.raises(ValueError, match='float.*int'):
            open_geotiff(path, dtype='int32')

    def test_dtype_none_preserves_native(self, float64_tif):
        path, _ = float64_tif
        result = open_geotiff(path, dtype=None)
        assert result.dtype == np.float64


class TestDtypeDask:
    """dtype parameter on dask reads."""

    def test_float64_to_float32_dask(self, float64_tif):
        path, orig = float64_tif
        result = open_geotiff(path, dtype='float32', chunks=40)
        assert result.dtype == np.float32
        computed = result.values
        np.testing.assert_array_almost_equal(
            computed, orig.astype(np.float32), decimal=6)

    def test_chunks_are_target_dtype(self, float64_tif):
        path, _ = float64_tif
        result = open_geotiff(path, dtype='float32', chunks=40)
        # Each chunk should be float32, not float64
        assert result.data.dtype == np.float32

    def test_float_to_int_raises_dask(self, float64_tif):
        path, _ = float64_tif
        with pytest.raises(ValueError, match='float.*int'):
            open_geotiff(path, dtype='int32', chunks=40)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd .claude/worktrees/issue-1083 && python -m pytest xrspatial/geotiff/tests/test_dtype_read.py -v --no-header -x 2>&1 | head -20`
Expected: FAIL -- `open_geotiff()` got an unexpected keyword argument `dtype`.

- [ ] **Step 3: Add `_validate_dtype_cast` helper and `dtype` to `open_geotiff`**

In `xrspatial/geotiff/__init__.py`, add a helper function after the `_geo_to_coords` function (around line 58):

```python
def _validate_dtype_cast(source_dtype, target_dtype):
    """Validate that casting source_dtype to target_dtype is allowed.

    Raises ValueError for float-to-int casts (lossy in a way users
    often don't intend).  All other casts are permitted -- the user
    asked for them explicitly.
    """
    src = np.dtype(source_dtype)
    tgt = np.dtype(target_dtype)
    if src.kind == 'f' and tgt.kind in ('u', 'i'):
        raise ValueError(
            f"Cannot cast float ({src}) to int ({tgt}). "
            f"This loses fractional data and is usually unintentional. "
            f"Cast explicitly after reading if you really want this.")
```

Then modify `open_geotiff` signature to add `dtype=None` after `source`. In the eager path (after `arr, geo_info = read_to_array(...)` at line 204), add:

```python
    if dtype is not None:
        target = np.dtype(dtype)
        _validate_dtype_cast(arr.dtype, target)
        arr = arr.astype(target)
```

Pass `dtype=dtype` through to `read_geotiff_dask()` and `read_geotiff_gpu()` calls.

- [ ] **Step 4: Add `dtype` to `read_geotiff_dask` and `_delayed_read_window`**

In `read_geotiff_dask`:
1. Add `dtype` parameter to signature.
2. Before building dask blocks, validate: `if dtype is not None: target = np.dtype(dtype); _validate_dtype_cast(file_dtype, target)` where `file_dtype` is the dtype from the metadata read.
3. If dtype is set, use `target` instead of `dtype` (the file dtype) for `da.from_delayed(..., dtype=target)`.
4. Pass `dtype` to `_delayed_read_window`.

In `_delayed_read_window`:
1. Add `target_dtype=None` parameter.
2. Inside the `_read()` closure, after the nodata masking, add: `if target_dtype is not None: arr = arr.astype(target_dtype)`.

- [ ] **Step 5: Add `dtype` to `read_geotiff_gpu`**

In `read_geotiff_gpu`:
1. Add `dtype` parameter to signature.
2. After the final `arr_gpu` is built (before building the DataArray), add: `if dtype is not None: target = np.dtype(dtype); _validate_dtype_cast(np.dtype(str(arr_gpu.dtype)), target); arr_gpu = arr_gpu.astype(target)`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd .claude/worktrees/issue-1083 && python -m pytest xrspatial/geotiff/tests/test_dtype_read.py -v --no-header 2>&1 | tail -20`
Expected: All PASS.

- [ ] **Step 7: Run existing tests to check for regressions**

Run: `cd .claude/worktrees/issue-1083 && python -m pytest xrspatial/geotiff/tests/ -v --no-header -x -q 2>&1 | tail -20`
Expected: All PASS.

- [ ] **Step 8: Commit**

```bash
cd .claude/worktrees/issue-1083
git add xrspatial/geotiff/__init__.py xrspatial/geotiff/tests/test_dtype_read.py
git commit -m "Add dtype parameter to open_geotiff (#1083)"
```

---

### Task 3: VRT tiled output from `to_geotiff`

**Files:**
- Modify: `xrspatial/geotiff/__init__.py:342-519` (`to_geotiff`)
- Test: `xrspatial/geotiff/tests/test_vrt_write.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `xrspatial/geotiff/tests/test_vrt_write.py`:

```python
"""Tests for VRT tiled output from to_geotiff."""
import numpy as np
import os
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


@pytest.fixture
def sample_raster():
    """200x200 float32 raster with coords and CRS."""
    arr = np.random.default_rng(55).random((200, 200), dtype=np.float32)
    y = np.linspace(41.0, 40.0, 200)  # north-to-south
    x = np.linspace(-106.0, -105.0, 200)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': y, 'x': x},
                      attrs={'crs': 4326, 'nodata': -9999.0})
    return da


class TestVrtOutputNumpy:
    """VRT output from numpy-backed DataArrays."""

    def test_creates_vrt_and_tiles_dir(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'out_1083.vrt')
        to_geotiff(sample_raster, vrt_path)
        assert os.path.exists(vrt_path)
        tiles_dir = str(tmp_path / 'out_1083_tiles')
        assert os.path.isdir(tiles_dir)
        tile_files = os.listdir(tiles_dir)
        assert len(tile_files) > 0
        assert all(f.endswith('.tif') for f in tile_files)

    def test_round_trip_numpy(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'rt_1083.vrt')
        to_geotiff(sample_raster, vrt_path)
        result = open_geotiff(vrt_path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_tile_naming_convention(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'named_1083.vrt')
        to_geotiff(sample_raster, vrt_path, tile_size=100)
        tiles_dir = str(tmp_path / 'named_1083_tiles')
        files = sorted(os.listdir(tiles_dir))
        # 200x200 with tile_size=100 -> 2x2 grid
        assert files == [
            'tile_00_00.tif', 'tile_00_01.tif',
            'tile_01_00.tif', 'tile_01_01.tif',
        ]

    def test_relative_paths_in_vrt(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'rel_1083.vrt')
        to_geotiff(sample_raster, vrt_path)
        with open(vrt_path) as f:
            content = f.read()
        # Paths should be relative (no leading /)
        assert 'rel_1083_tiles/' in content
        assert str(tmp_path) not in content

    def test_compression_level_passed_to_tiles(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'cl_1083.vrt')
        to_geotiff(sample_raster, vrt_path, compression='zstd',
                   compression_level=1)
        result = open_geotiff(vrt_path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)


class TestVrtOutputDask:
    """VRT output from dask-backed DataArrays."""

    def test_dask_round_trip(self, sample_raster, tmp_path):
        dask_da = sample_raster.chunk({'y': 100, 'x': 100})
        vrt_path = str(tmp_path / 'dask_1083.vrt')
        to_geotiff(dask_da, vrt_path)
        result = open_geotiff(vrt_path)
        np.testing.assert_array_almost_equal(
            result.values, sample_raster.values, decimal=5)

    def test_dask_one_tile_per_chunk(self, sample_raster, tmp_path):
        dask_da = sample_raster.chunk({'y': 100, 'x': 100})
        vrt_path = str(tmp_path / 'chunks_1083.vrt')
        to_geotiff(dask_da, vrt_path)
        tiles_dir = str(tmp_path / 'chunks_1083_tiles')
        # 200x200 chunked 100x100 -> 2x2 = 4 tiles
        assert len(os.listdir(tiles_dir)) == 4


class TestVrtEdgeCases:
    """Edge cases and validation."""

    def test_cog_with_vrt_raises(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'cog_1083.vrt')
        with pytest.raises(ValueError, match='cog.*vrt|vrt.*cog'):
            to_geotiff(sample_raster, vrt_path, cog=True)

    def test_overview_levels_with_vrt_raises(self, sample_raster, tmp_path):
        vrt_path = str(tmp_path / 'ovr_1083.vrt')
        with pytest.raises(ValueError, match='overview.*vrt|vrt.*overview'):
            to_geotiff(sample_raster, vrt_path, overview_levels=[2, 4])

    def test_nonempty_tiles_dir_raises(self, sample_raster, tmp_path):
        tiles_dir = tmp_path / 'exist_1083_tiles'
        tiles_dir.mkdir()
        (tiles_dir / 'dummy.tif').write_text('x')
        vrt_path = str(tmp_path / 'exist_1083.vrt')
        with pytest.raises(FileExistsError):
            to_geotiff(sample_raster, vrt_path)

    def test_empty_tiles_dir_ok(self, sample_raster, tmp_path):
        tiles_dir = tmp_path / 'empty_1083_tiles'
        tiles_dir.mkdir()
        vrt_path = str(tmp_path / 'empty_1083.vrt')
        to_geotiff(sample_raster, vrt_path)
        assert os.path.exists(vrt_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd .claude/worktrees/issue-1083 && python -m pytest xrspatial/geotiff/tests/test_vrt_write.py -v --no-header -x 2>&1 | head -20`
Expected: FAIL -- VRT path not yet handled.

- [ ] **Step 3: Implement VRT output path in `to_geotiff`**

In `xrspatial/geotiff/__init__.py`, add the VRT detection and dispatch at the top of `to_geotiff`, right after the docstring and before the GPU dispatch:

```python
    # VRT tiled output
    if path.lower().endswith('.vrt'):
        if cog:
            raise ValueError(
                "cog=True is not compatible with VRT output. "
                "VRT writes tiled GeoTIFFs, not a single COG.")
        if overview_levels is not None:
            raise ValueError(
                "overview_levels is not compatible with VRT output. "
                "VRT tiles do not include overviews.")
        _write_vrt_tiled(data, path,
                         crs=crs, nodata=nodata,
                         compression=compression,
                         compression_level=compression_level,
                         tile_size=tile_size,
                         predictor=predictor,
                         bigtiff=bigtiff,
                         gpu=gpu)
        return
```

Then add the `_write_vrt_tiled` function (new function in `__init__.py`):

```python
def _write_vrt_tiled(data, vrt_path: str, *,
                     crs=None, nodata=None,
                     compression='zstd', compression_level=None,
                     tile_size=256, predictor=False,
                     bigtiff=None, gpu=None):
    """Write a DataArray as a directory of tiled GeoTIFFs with a VRT index.

    For dask inputs, each chunk is computed and written independently
    so the full array never materialises in RAM.
    """
    import os
    import math
    from ._vrt import write_vrt as _write_vrt_fn

    stem = os.path.splitext(os.path.basename(vrt_path))[0]
    tiles_dir = os.path.join(os.path.dirname(vrt_path) or '.', f'{stem}_tiles')

    # Validate tiles directory
    if os.path.isdir(tiles_dir) and os.listdir(tiles_dir):
        raise FileExistsError(
            f"Tiles directory already exists and is not empty: {tiles_dir}")
    os.makedirs(tiles_dir, exist_ok=True)

    # Resolve metadata from the DataArray
    epsg = None
    wkt = None
    nodata_val = nodata
    geo_transform = None

    if isinstance(data, xr.DataArray):
        geo_transform = _coords_to_transform(data)
        if crs is None:
            crs_attr = data.attrs.get('crs')
            if isinstance(crs_attr, str):
                epsg = _wkt_to_epsg(crs_attr)
                if epsg is None:
                    wkt = crs_attr
            elif crs_attr is not None:
                epsg = int(crs_attr)
            if epsg is None:
                wkt_attr = data.attrs.get('crs_wkt')
                if isinstance(wkt_attr, str):
                    epsg = _wkt_to_epsg(wkt_attr)
                    if epsg is None:
                        wkt = wkt_attr
        elif isinstance(crs, int):
            epsg = crs
        elif isinstance(crs, str):
            epsg = _wkt_to_epsg(crs)
            if epsg is None:
                wkt = crs
        if nodata_val is None:
            nodata_val = data.attrs.get('nodata')

    raw = data.data if isinstance(data, xr.DataArray) else data
    is_dask = hasattr(raw, 'dask')
    is_cupy = hasattr(raw, 'device') or hasattr(raw, 'get')

    if is_dask:
        # Dask path: one tile per chunk
        import dask
        chunks_y = raw.chunks[0]
        chunks_x = raw.chunks[1]
        n_rows = len(chunks_y)
        n_cols = len(chunks_x)
    else:
        # Numpy/CuPy path: slice by tile_size
        if is_cupy:
            arr = raw
        else:
            arr = np.asarray(raw)
        h, w = arr.shape[:2]
        n_rows = math.ceil(h / tile_size)
        n_cols = math.ceil(w / tile_size)

    pad_width = len(str(max(n_rows, n_cols) - 1))
    tile_paths = []

    if is_dask:
        delayed_writes = []
        row_offset = 0
        for ri, ch_y in enumerate(chunks_y):
            col_offset = 0
            for ci, ch_x in enumerate(chunks_x):
                tile_name = f'tile_{ri:0{pad_width}d}_{ci:0{pad_width}d}.tif'
                tile_path = os.path.join(tiles_dir, tile_name)
                tile_paths.append(tile_path)

                # Extract the chunk as a dask array
                chunk_slice = raw[
                    row_offset:row_offset + ch_y,
                    col_offset:col_offset + ch_x,
                ]

                # Build per-tile geo_transform
                tile_gt = None
                if geo_transform is not None:
                    t = geo_transform
                    tile_gt = GeoTransform(
                        origin_x=t.origin_x + col_offset * t.pixel_width,
                        origin_y=t.origin_y + row_offset * t.pixel_height,
                        pixel_width=t.pixel_width,
                        pixel_height=t.pixel_height,
                    )

                delayed_writes.append(
                    dask.delayed(_write_single_tile)(
                        chunk_slice, tile_path, tile_gt, epsg, wkt,
                        nodata_val, compression, compression_level,
                        tile_size, predictor, bigtiff))

                col_offset += ch_x
            row_offset += ch_y

        dask.compute(*delayed_writes)

    else:
        # Numpy/CuPy: slice and write sequentially
        h, w = arr.shape[:2]
        for ri in range(n_rows):
            for ci in range(n_cols):
                r0 = ri * tile_size
                c0 = ci * tile_size
                r1 = min(r0 + tile_size, h)
                c1 = min(c0 + tile_size, w)

                tile_name = f'tile_{ri:0{pad_width}d}_{ci:0{pad_width}d}.tif'
                tile_path = os.path.join(tiles_dir, tile_name)
                tile_paths.append(tile_path)

                tile_data = arr[r0:r1, c0:c1]

                tile_gt = None
                if geo_transform is not None:
                    t = geo_transform
                    tile_gt = GeoTransform(
                        origin_x=t.origin_x + c0 * t.pixel_width,
                        origin_y=t.origin_y + r0 * t.pixel_height,
                        pixel_width=t.pixel_width,
                        pixel_height=t.pixel_height,
                    )

                _write_single_tile(
                    tile_data, tile_path, tile_gt, epsg, wkt,
                    nodata_val, compression, compression_level,
                    tile_size, predictor, bigtiff)

    # Generate VRT index with relative paths
    write_vrt(vrt_path, tile_paths, relative=True,
              nodata=nodata_val)


def _write_single_tile(chunk_data, path, geo_transform, epsg, wkt,
                       nodata, compression, compression_level,
                       tile_size, predictor, bigtiff):
    """Write a single tile GeoTIFF. Used by _write_vrt_tiled."""
    if hasattr(chunk_data, 'compute'):
        chunk_data = chunk_data.compute()
    if hasattr(chunk_data, 'get'):
        chunk_data = chunk_data.get()  # CuPy -> numpy

    arr = np.asarray(chunk_data)

    # Auto-promote unsupported dtypes
    if arr.dtype == np.float16:
        arr = arr.astype(np.float32)
    elif arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)

    # Restore NaN to nodata sentinel
    if nodata is not None and arr.dtype.kind == 'f' and not np.isnan(nodata):
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            arr = arr.copy()
            arr[nan_mask] = arr.dtype.type(nodata)

    write(arr, path,
          geo_transform=geo_transform,
          crs_epsg=epsg,
          crs_wkt=wkt if epsg is None else None,
          nodata=nodata,
          compression=compression,
          tiled=True,
          tile_size=tile_size,
          predictor=predictor,
          compression_level=compression_level,
          bigtiff=bigtiff)
```

Note: The import of `GeoTransform` is already at the top of `__init__.py` (line 19). The import of `write_vrt` should come from `._vrt`. Adjust the import inside `_write_vrt_tiled` to: `from ._vrt import write_vrt as _write_vrt_fn` and call `_write_vrt_fn(...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd .claude/worktrees/issue-1083 && python -m pytest xrspatial/geotiff/tests/test_vrt_write.py -v --no-header 2>&1 | tail -30`
Expected: All PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `cd .claude/worktrees/issue-1083 && python -m pytest xrspatial/geotiff/tests/ -v --no-header -q 2>&1 | tail -20`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
cd .claude/worktrees/issue-1083
git add xrspatial/geotiff/__init__.py xrspatial/geotiff/tests/test_vrt_write.py
git commit -m "Add VRT tiled output from to_geotiff (#1083)"
```

---

### Task 4: Update documentation and README

**Files:**
- Modify: `docs/source/reference/io.rst` (or equivalent -- check for existing geotiff docs)
- Modify: `README.md`

- [ ] **Step 1: Update API docs**

Check if `docs/source/reference/` has an entry for `open_geotiff`/`to_geotiff`. If so, no code change needed since the docstrings will auto-generate. If there's a manually maintained parameter list, add `dtype`, `compression_level`, and the `.vrt` extension behaviour.

- [ ] **Step 2: Update README usage examples**

In `README.md`, find the GeoTIFF I/O section (around line 140-201 based on the exploration). Add these examples to the existing list:

```python
open_geotiff('dem.tif', dtype='float32')              # half memory
open_geotiff('dem.tif', dtype='float32', chunks=512)   # dask + half memory
to_geotiff(data, 'out.tif', compression_level=1)       # fast scratch write
to_geotiff(data, 'out.tif', compression_level=22)      # max compression
to_geotiff(dask_da, 'mosaic.vrt')                      # stream dask to VRT
```

- [ ] **Step 3: Commit**

```bash
cd .claude/worktrees/issue-1083
git add README.md docs/
git commit -m "Update docs for dtype, compression_level, VRT output (#1083)"
```

---

### Task 5: User guide notebook

**Files:**
- Create: `examples/user_guide/46_GeoTIFF_Performance.ipynb`

- [ ] **Step 1: Create the notebook**

Create `examples/user_guide/46_GeoTIFF_Performance.ipynb` with these cells:

1. **Markdown: title** -- "GeoTIFF Performance Controls: dtype, compression_level, and VRT output"
2. **Code: imports** -- `import numpy as np, xarray as xr, os, tempfile` and `from xrspatial.geotiff import open_geotiff, to_geotiff`
3. **Markdown: dtype section** -- explain what `dtype` does and when to use it
4. **Code: create a float64 raster, write it, read back with dtype='float32'** -- show the memory savings (arr.nbytes before and after)
5. **Code: dask dtype** -- same with `chunks=256`, show `.dtype` on the result
6. **Markdown: compression_level section** -- explain the speed/size tradeoff
7. **Code: write same raster at level=1 and level=22** -- compare file sizes and write times with `%%time`
8. **Markdown: VRT output section** -- explain the streaming write and directory layout
9. **Code: create a larger raster, chunk it, write to .vrt** -- show the output directory listing
10. **Code: read the VRT back** -- round-trip verification
11. **Markdown: summary** -- one-paragraph recap

- [ ] **Step 2: Run the notebook to verify it executes**

Run: `cd .claude/worktrees/issue-1083 && jupyter nbconvert --to notebook --execute examples/user_guide/46_GeoTIFF_Performance.ipynb --output /dev/null 2>&1 | tail -5`
Expected: No errors.

- [ ] **Step 3: Commit**

```bash
cd .claude/worktrees/issue-1083
git add examples/user_guide/46_GeoTIFF_Performance.ipynb
git commit -m "Add user guide notebook for geotiff performance controls (#1083)"
```
