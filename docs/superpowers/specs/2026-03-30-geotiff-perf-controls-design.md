# GeoTIFF Performance and Memory Controls

Adds three parameters to `open_geotiff` and `to_geotiff` that let callers
control memory usage, compression speed, and large-raster write strategy.
All three are opt-in; default behaviour is unchanged.

## 1. `dtype` parameter on `open_geotiff`

### API

```python
open_geotiff(source, *, dtype=None, ...)
```

`dtype` accepts any numpy dtype string or object (`np.float32`, `'float32'`,
etc.). `None` preserves the file's native dtype (current behaviour).

### Read paths

| Path | Behaviour |
|------|-----------|
| Eager (numpy) | Output array allocated at target dtype. Each decoded tile/strip cast before copy-in. Peak overhead: one tile at native dtype. |
| Dask | Each delayed chunk function casts after decode. Output chunks are target dtype. Same per-tile overhead. |
| GPU (CuPy) | Cast on device after decode. |
| Dask + CuPy | Combination of dask and GPU paths. |

### Numba LZW fast path

The LZW decoder is a numba JIT function that emits values one at a time into a
byte buffer. A variant will decode each value and cast inline to the target
dtype so the per-tile buffer is never allocated at native dtype. Other codecs
(deflate, zstd) return byte buffers from C libraries where per-value
interception isn't possible, so those fall back to the tile-level cast.

### Validation

- Narrowing float casts (float64 to float32): allowed.
- Narrowing int casts (int64 to int16): allowed (user asked for it explicitly).
- Widening casts (float32 to float64, uint8 to int32): allowed.
- Float to int: `ValueError` (lossy in a way users often don't intend).
- Unsupported casts (e.g. complex128 to uint8): `ValueError`.

## 2. `compression_level` parameter on `to_geotiff`

### API

```python
to_geotiff(data, path, *, compression='zstd', compression_level=None, ...)
```

`compression_level` is `int | None`. `None` uses the codec's existing default.

### Ranges

| Codec | Range | Default | Direction |
|-------|-------|---------|-----------|
| deflate | 1 -- 9 | 6 | 1 = fastest, 9 = smallest |
| zstd | 1 -- 22 | 3 | 1 = fastest, 22 = smallest |
| lz4 | 0 -- 16 | 0 | 0 = fastest |
| lzw | n/a | n/a | No level support; ignored silently |
| jpeg | n/a | n/a | Quality is a separate axis; ignored |
| packbits | n/a | n/a | Ignored |
| none | n/a | n/a | Ignored |

### Plumbing

`to_geotiff` passes `compression_level` to `write()`, which passes it to
`compress()`. The internal `compress()` already accepts a `level` argument; we
just thread it through the two intermediate call sites that currently hardcode
it.

### Validation

- Out-of-range level for a codec that supports levels: `ValueError`.
- Level set for a codec without level support: silently ignored.

### GPU path

`write_geotiff_gpu` also accepts and forwards the level to nvCOMP batch
compression, which supports levels for zstd and deflate.

## 3. VRT output from `to_geotiff` via `.vrt` extension

### Trigger

When `path` ends in `.vrt`, `to_geotiff` writes a tiled VRT instead of a
monolithic TIFF. No new parameter needed.

### Output layout

```
output.vrt
output_tiles/
  tile_0000_0000.tif   # row_col, zero-padded
  tile_0000_0001.tif
  ...
```

Directory name derived from the VRT stem (`foo.vrt` -> `foo_tiles/`).
Zero-padding width scales to the grid dimensions.

### Behaviour per input type

| Input | Tiling strategy | Memory profile |
|-------|----------------|----------------|
| Dask DataArray | One tile per dask chunk. Each task computes its chunk and writes one `.tif`. | One chunk in RAM at a time (scheduler controlled). |
| Dask + CuPy | Same, GPU compress per tile. | One chunk in GPU memory at a time. |
| Numpy / ndarray | Slice into `tile_size`-sized pieces, write each. | Source array already in RAM; tile slices are views (no duplication). |
| CuPy | Same as numpy but GPU compress. | Source on GPU; tiles are views. |

### Per-tile properties

- Same `compression`, `compression_level`, `predictor`, `nodata`, `crs` as the
  parent call.
- `tiled=True` with the caller's `tile_size` (internal TIFF tiling within each
  chunk-file).
- GeoTransform adjusted to each tile's spatial position (row/col offset from
  the full raster origin).
- No COG overviews on individual tiles.

### VRT generation

After all tiles are written, call `write_vrt()` with relative paths. The VRT
XML references each tile by its spatial extent and band mapping.

### Edge cases and validation

- `cog=True` with a `.vrt` path: `ValueError` (mutually exclusive).
- Tiles directory exists and is non-empty: `FileExistsError` to prevent silent
  overwrites.
- Tiles directory doesn't exist: created automatically.
- `overview_levels` with `.vrt` path: `ValueError` (overviews don't apply).

### Dask scheduling

For dask inputs, all delayed tile-write tasks are submitted to
`dask.compute()` at once. The scheduler manages parallelism and memory. Each
task is: compute chunk, compress, write tile file. No coordination between
tasks.

## Out of scope

- Streaming write of a monolithic `.tif` from dask input (tracked as a separate
  issue). Users who need a single file from a large dask array can write to VRT
  and convert externally, or ensure sufficient RAM.
- JPEG quality parameter (separate concern from compression level).
- Automatic chunk-size recommendation based on available memory.
