"""Benchmark xrspatial.geotiff vs rioxarray for read/write performance and consistency."""
from __future__ import annotations

import os
import tempfile
import time

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timer(fn, warmup=1, runs=5):
    """Time a callable, returning (median_seconds, result_from_last_call)."""
    for _ in range(warmup):
        result = fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], result


def _fmt_ms(seconds):
    return f"{seconds * 1000:.1f} ms"


# ---------------------------------------------------------------------------
# Consistency check
# ---------------------------------------------------------------------------

def check_consistency(path):
    """Compare pixel values and geo metadata between the two readers."""
    import rioxarray  # noqa: F401

    from xrspatial.geotiff import open_geotiff

    rio_da = xr.open_dataarray(path, engine='rasterio')
    rio_arr = rio_da.squeeze('band').values.astype(np.float64)

    our_da = open_geotiff(path)
    our_arr = our_da.values.astype(np.float64)

    # Shape
    assert rio_arr.shape == our_arr.shape, (
        f"Shape mismatch: rioxarray {rio_arr.shape} vs ours {our_arr.shape}")

    # Pixel values (count NaN agreement as exact match)
    rio_nan = np.isnan(rio_arr)
    our_nan = np.isnan(our_arr)
    both_nan = rio_nan & our_nan
    valid = ~(rio_nan | our_nan)
    diff = np.zeros_like(rio_arr)
    diff[valid] = np.abs(rio_arr[valid] - our_arr[valid])
    max_diff = float(diff[valid].max()) if valid.any() else 0.0
    mean_diff = float(diff[valid].mean()) if valid.any() else 0.0
    # Exact = same value on valid pixels + both NaN on NaN pixels
    exact_count = int(np.sum(diff[valid] == 0)) + int(both_nan.sum())
    pct_exact = exact_count / diff.size * 100

    # CRS
    rio_epsg = rio_da.rio.crs.to_epsg() if rio_da.rio.crs else None
    our_epsg = our_da.attrs.get('crs')

    # Coordinate comparison
    rio_y = rio_da.coords['y'].values
    rio_x = rio_da.coords['x'].values
    our_y = our_da.coords['y'].values
    our_x = our_da.coords['x'].values

    y_max_diff = float(np.max(np.abs(rio_y - our_y))) if len(rio_y) == len(our_y) else float('inf')
    x_max_diff = float(np.max(np.abs(rio_x - our_x))) if len(rio_x) == len(our_x) else float('inf')

    return {
        'shape': rio_arr.shape,
        'dtype_rio': str(rio_da.dtype),
        'dtype_ours': str(our_da.dtype),
        'max_pixel_diff': max_diff,
        'mean_pixel_diff': mean_diff,
        'pct_exact_match': pct_exact,
        'epsg_rio': rio_epsg,
        'epsg_ours': our_epsg,
        'epsg_match': rio_epsg == our_epsg,
        'y_max_diff': y_max_diff,
        'x_max_diff': x_max_diff,
    }


# ---------------------------------------------------------------------------
# Read benchmark
# ---------------------------------------------------------------------------

def bench_read(path, runs=10):
    """Benchmark read performance."""
    import rioxarray  # noqa: F401

    from xrspatial.geotiff import open_geotiff

    def rio_read():
        da = xr.open_dataarray(path, engine='rasterio')
        _ = da.values  # force load
        da.close()
        return da

    def our_read():
        return open_geotiff(path)

    rio_time, _ = _timer(rio_read, warmup=2, runs=runs)
    our_time, _ = _timer(our_read, warmup=2, runs=runs)

    return rio_time, our_time


# ---------------------------------------------------------------------------
# Write benchmark
# ---------------------------------------------------------------------------

def bench_write(shape=(512, 512), compression='deflate', runs=5):
    """Benchmark write performance."""
    import rioxarray  # noqa: F401

    from xrspatial.geotiff import to_geotiff

    rng = np.random.RandomState(42)
    arr = rng.rand(*shape).astype(np.float32)

    y = np.linspace(45.0, 44.0, shape[0])
    x = np.linspace(-120.0, -119.0, shape[1])
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x})
    da = da.rio.write_crs(4326)
    da = da.rio.write_nodata(np.nan)

    da_ours = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x},
                           attrs={'crs': 4326})

    tmpdir = tempfile.mkdtemp()

    comp_map = {'deflate': 'deflate', 'lzw': 'lzw', 'none': None}
    rio_comp = comp_map.get(compression, compression)

    def rio_write():
        p = os.path.join(tmpdir, 'rio_out.tif')
        if rio_comp:
            da.rio.to_raster(p, compress=rio_comp.upper())
        else:
            da.rio.to_raster(p)
        return os.path.getsize(p)

    def our_write():
        p = os.path.join(tmpdir, 'our_out.tif')
        to_geotiff(da_ours, p, compression=compression, tiled=False)
        return os.path.getsize(p)

    rio_time, rio_size = _timer(rio_write, warmup=1, runs=runs)
    our_time, our_size = _timer(our_write, warmup=1, runs=runs)

    return rio_time, our_time, rio_size, our_size


# ---------------------------------------------------------------------------
# Write + read-back consistency
# ---------------------------------------------------------------------------

def bench_round_trip(shape=(256, 256), compression='deflate'):
    """Write with our module, read back with rioxarray, and vice versa."""
    import rioxarray  # noqa: F401

    from xrspatial.geotiff import open_geotiff, to_geotiff

    rng = np.random.RandomState(99)
    arr = rng.rand(*shape).astype(np.float32)
    y = np.linspace(45.0, 44.0, shape[0])
    x = np.linspace(-120.0, -119.0, shape[1])

    tmpdir = tempfile.mkdtemp()

    # Ours write -> rioxarray read
    our_path = os.path.join(tmpdir, 'ours.tif')
    da_ours = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x},
                           attrs={'crs': 4326})
    to_geotiff(da_ours, our_path, compression=compression, tiled=False)

    rio_da = xr.open_dataarray(our_path, engine='rasterio')
    rio_arr = rio_da.squeeze('band').values if 'band' in rio_da.dims else rio_da.values
    rio_da.close()

    diff1 = float(np.nanmax(np.abs(arr - rio_arr)))

    # rioxarray write -> ours read
    rio_path = os.path.join(tmpdir, 'rio.tif')
    da_rio = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x})
    da_rio = da_rio.rio.write_crs(4326)
    comp_map = {'deflate': 'DEFLATE', 'lzw': 'LZW', 'none': None}
    rio_comp = comp_map.get(compression)
    if rio_comp:
        da_rio.rio.to_raster(rio_path, compress=rio_comp)
    else:
        da_rio.rio.to_raster(rio_path)

    our_da = open_geotiff(rio_path)
    our_arr = our_da.values

    diff2 = float(np.nanmax(np.abs(arr - our_arr)))

    return diff1, diff2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    landsat_dir = 'docs/source/user_guide/data'
    bands = [
        'LC80030172015001LGN00_B2.tiff',
        'LC80030172015001LGN00_B3.tiff',
        'LC80030172015001LGN00_B4.tiff',
        'LC80030172015001LGN00_B5.tiff',
    ]

    print("=" * 72)
    print("xrspatial.geotiff vs rioxarray -- Benchmark & Consistency")
    print("=" * 72)

    # --- Consistency on real Landsat files ---
    print("\n--- Pixel & Metadata Consistency (Landsat 8 bands) ---\n")
    for band_file in bands:
        path = os.path.join(landsat_dir, band_file)
        if not os.path.exists(path):
            print(f"  {band_file}: SKIPPED (not found)")
            continue
        c = check_consistency(path)
        name = band_file.split('_')[1].replace('.tiff', '')
        print(f"  {name}:  shape={c['shape']}  dtype rio={c['dtype_rio']} ours={c['dtype_ours']}")
        print(f"         pixels: max_diff={c['max_pixel_diff']:.6f}  "
              f"mean_diff={c['mean_pixel_diff']:.6f}  exact={c['pct_exact_match']:.1f}%")
        print(f"         EPSG: rio={c['epsg_rio']} ours={c['epsg_ours']} match={c['epsg_match']}")
        print(f"         coords: y_max_diff={c['y_max_diff']:.6f}  x_max_diff={c['x_max_diff']:.6f}")  # noqa: E501

    # --- Read performance ---
    print("\n--- Read Performance (median of 10 runs) ---\n")
    print(f"  {'File':<8} {'rioxarray':>12} {'xrspatial':>12} {'ratio':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*8}")
    for band_file in bands:
        path = os.path.join(landsat_dir, band_file)
        if not os.path.exists(path):
            continue
        rio_t, our_t = bench_read(path, runs=10)
        name = band_file.split('_')[1].replace('.tiff', '')
        ratio = our_t / rio_t if rio_t > 0 else float('inf')
        print(f"  {name:<8} {_fmt_ms(rio_t):>12} {_fmt_ms(our_t):>12} {ratio:>7.2f}x")

    # --- Write performance ---
    print("\n--- Write Performance (512x512 float32, median of 5 runs) ---\n")
    print(f"  {'Compression':<12} {'rioxarray':>12} {'xrspatial':>12} {'ratio':>8} {'size rio':>10} {'size ours':>10}")  # noqa: E501
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*8} {'-'*10} {'-'*10}")
    for comp in ['none', 'deflate', 'lzw']:
        rio_t, our_t, rio_sz, our_sz = bench_write((512, 512), comp, runs=5)
        ratio = our_t / rio_t if rio_t > 0 else float('inf')
        print(f"  {comp:<12} {_fmt_ms(rio_t):>12} {_fmt_ms(our_t):>12} {ratio:>7.2f}x "
              f"{rio_sz:>9,} {our_sz:>9,}")

    # --- Write performance (larger) ---
    print("\n--- Write Performance (2048x2048 float32, median of 3 runs) ---\n")
    print(f"  {'Compression':<12} {'rioxarray':>12} {'xrspatial':>12} {'ratio':>8} {'size rio':>10} {'size ours':>10}")  # noqa: E501
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*8} {'-'*10} {'-'*10}")
    for comp in ['none', 'deflate']:
        rio_t, our_t, rio_sz, our_sz = bench_write((2048, 2048), comp, runs=3)
        ratio = our_t / rio_t if rio_t > 0 else float('inf')
        print(f"  {comp:<12} {_fmt_ms(rio_t):>12} {_fmt_ms(our_t):>12} {ratio:>7.2f}x "
              f"{rio_sz:>9,} {our_sz:>9,}")

    # --- Cross-library round-trip ---
    print("\n--- Cross-Library Round-Trip Consistency ---\n")
    for comp in ['none', 'deflate']:
        d1, d2 = bench_round_trip((256, 256), comp)
        print(f"  {comp}: ours->rioxarray max_diff={d1:.8f}  rioxarray->ours max_diff={d2:.8f}")

    # --- Real-world files from rtxpy ---
    rtxpy_dir = '../rtxpy/examples'
    rtxpy_files = [
        ('render_demo_terrain.tif', 'uncompressed strip'),
        ('Copernicus_DSM_COG_10_N40_00_W075_00_DEM.tif', 'deflate+fpred COG'),
        ('Copernicus_DSM_COG_10_S23_00_W044_00_DEM.tif', 'deflate+fpred COG'),
        ('USGS_1_n43w122.tif', 'LZW+fpred COG'),
        ('USGS_1_n39w106.tif', 'LZW+fpred COG'),
        ('USGS_one_meter_x65y454_NY_LongIsland_Z18_2014.tif', 'LZW tiled COG'),
    ]

    print("\n--- Real-World Files: Consistency & Read Performance ---\n")
    print(f"  {'File':<52} {'Format':<20} {'Shape':>12} {'Exact%':>7} {'rio':>9} {'ours':>9} {'ratio':>7}")  # noqa: E501
    print(f"  {'-'*52} {'-'*20} {'-'*12} {'-'*7} {'-'*9} {'-'*9} {'-'*7}")

    for fname, desc in rtxpy_files:
        path = os.path.join(rtxpy_dir, fname)
        if not os.path.exists(path):
            continue

        # Consistency
        c = check_consistency(path)

        # Performance (fewer runs for large files)
        fsize = os.path.getsize(path)
        runs = 3 if fsize > 50_000_000 else 5
        rio_t, our_t = bench_read(path, runs=runs)
        ratio = our_t / rio_t if rio_t > 0 else float('inf')

        shape_str = f"{c['shape'][0]}x{c['shape'][1]}"
        short_name = fname[:50]
        print(f"  {short_name:<52} {desc:<20} {shape_str:>12} {c['pct_exact_match']:>6.1f}% "
              f"{_fmt_ms(rio_t):>9} {_fmt_ms(our_t):>9} {ratio:>6.2f}x")

    print()


if __name__ == '__main__':
    main()
