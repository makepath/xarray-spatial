#!/usr/bin/env python
"""
Benchmark xrspatial.reproject vs rioxarray.reproject
====================================================

Compares performance and pixel-level consistency across raster sizes,
CRS pairs, and resampling methods.

Usage
-----
    python -m xrspatial.tests.bench_reproject_vs_rioxarray
"""

import time
import sys

import numpy as np
import xarray as xr

from xrspatial.reproject import reproject as xrs_reproject

try:
    import rioxarray  # noqa: F401
    HAS_RIOXARRAY = True
except ImportError:
    HAS_RIOXARRAY = False

try:
    from pyproj import CRS
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


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


def _make_raster(h, w, crs='EPSG:4326', x_range=(-10, 10), y_range=(-10, 10),
                 nodata=np.nan):
    """Create a test DataArray with geographic coordinates and CRS metadata."""
    y = np.linspace(y_range[1], y_range[0], h)
    x = np.linspace(x_range[0], x_range[1], w)
    xx, yy = np.meshgrid(x, y)
    data = (xx + yy).astype(np.float64)
    return xr.DataArray(
        data, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        name='gradient',
        attrs={'crs': crs, 'nodata': nodata},
    )


def _make_rio_raster(da, crs_str='EPSG:4326'):
    """Convert an xrspatial-style DataArray to rioxarray-compatible form."""
    da_rio = da.copy()
    res_y = float(da.y[1] - da.y[0])  # negative for north-up
    res_x = float(da.x[1] - da.x[0])
    left = float(da.x[0]) - res_x / 2
    top = float(da.y[0]) - res_y / 2   # y descending, so y[0] is top
    from rasterio.transform import from_origin
    transform = from_origin(left, top, res_x, abs(res_y))
    da_rio.rio.write_crs(crs_str, inplace=True)
    da_rio.rio.write_transform(transform, inplace=True)
    da_rio.rio.write_nodata(np.nan, inplace=True)
    return da_rio


RESAMPLING_MAP_RIO = {
    'nearest': 0,   # rasterio.enums.Resampling.nearest
    'bilinear': 1,  # rasterio.enums.Resampling.bilinear
    'cubic': 2,     # rasterio.enums.Resampling.cubic
}


def _fmt_time(seconds):
    if seconds < 1:
        return f'{seconds * 1000:.1f}ms'
    return f'{seconds:.2f}s'


def _fmt_shape(shape):
    return f'{shape[0]}x{shape[1]}'


# CRS-specific coordinate ranges (square aspect ratio in source units)
CRS_RANGES = {
    'EPSG:4326': {'x_range': (-10, 10), 'y_range': (40, 60)},
    'EPSG:32633': {'x_range': (300000, 700000), 'y_range': (5200000, 5600000)},
}


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------

SIZES = [
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
]

CRS_PAIRS = [
    ('EPSG:4326', 'EPSG:32633'),   # WGS84 -> UTM zone 33N
    ('EPSG:32633', 'EPSG:4326'),   # UTM -> WGS84
    ('EPSG:4326', 'EPSG:3857'),    # WGS84 -> Web Mercator
]

RESAMPLINGS = ['nearest', 'bilinear', 'cubic']


def run_performance(sizes=None, crs_pairs=None, resamplings=None):
    """Run performance benchmarks (approx, exact, and rioxarray)."""
    sizes = sizes or SIZES
    crs_pairs = crs_pairs or CRS_PAIRS
    resamplings = resamplings or ['bilinear']

    print()
    print('=' * 90)
    print('PERFORMANCE BENCHMARK: xrspatial (approx / exact) vs rioxarray')
    print('=' * 90)

    for src_crs, dst_crs in crs_pairs:
        ranges = CRS_RANGES[src_crs]

        print(f'\n### {src_crs} -> {dst_crs}')
        print()
        print(f'| {"Size":>12} | {"Resampling":>10} '
              f'| {"xrs approx":>12} | {"xrs exact":>12} '
              f'| {"rioxarray":>12} | {"approx/rio":>10} | {"exact/rio":>10} |')
        print(f'|{"-"*14}|{"-"*12}'
              f'|{"-"*14}|{"-"*14}'
              f'|{"-"*14}|{"-"*12}|{"-"*12}|')

        for h, w in sizes:
            da = _make_raster(h, w, crs=src_crs, **ranges)
            da_rio = _make_rio_raster(da, src_crs)

            for resampling in resamplings:
                # xrspatial approx (default, precision=16)
                approx_time, _ = _timer(
                    lambda: xrs_reproject(da, dst_crs,
                                         resampling=resampling,
                                         transform_precision=16),
                    warmup=2, runs=5,
                )

                # xrspatial exact (precision=0)
                exact_time, _ = _timer(
                    lambda: xrs_reproject(da, dst_crs,
                                         resampling=resampling,
                                         transform_precision=0),
                    warmup=2, runs=5,
                )

                # rioxarray
                rio_resamp = RESAMPLING_MAP_RIO[resampling]
                rio_time, _ = _timer(
                    lambda: da_rio.rio.reproject(dst_crs,
                                                resampling=rio_resamp),
                    warmup=2, runs=5,
                )

                approx_ratio = rio_time / approx_time if approx_time > 0 else float('inf')
                exact_ratio = rio_time / exact_time if exact_time > 0 else float('inf')

                print(f'| {_fmt_shape((h, w)):>12} | {resampling:>10} '
                      f'| {_fmt_time(approx_time):>12} '
                      f'| {_fmt_time(exact_time):>12} '
                      f'| {_fmt_time(rio_time):>12} '
                      f'| {approx_ratio:>9.2f}x '
                      f'| {exact_ratio:>9.2f}x |')


def run_consistency(sizes=None, crs_pairs=None, resamplings=None):
    """Run pixel-level consistency checks.

    Forces both libraries to produce the same output grid by running
    rioxarray first, then passing its resolution and bounds to xrspatial.
    """
    sizes = sizes or [(256, 256), (512, 512), (1024, 1024)]
    crs_pairs = crs_pairs or CRS_PAIRS
    resamplings = resamplings or RESAMPLINGS

    print()
    print('=' * 80)
    print('CONSISTENCY CHECK: xrspatial vs rioxarray (same output grid)')
    print('=' * 80)
    print()
    print(f'| {"Size":>12} | {"CRS":>24} | {"Resampling":>10} '
          f'| {"Out shape":>11} | {"RMSE":>10} | {"MaxErr":>10} '
          f'| {"R²":>8} | {"NaN agree":>9} |')
    print(f'|{"-"*14}|{"-"*26}|{"-"*12}'
          f'|{"-"*13}|{"-"*12}|{"-"*12}'
          f'|{"-"*10}|{"-"*11}|')

    for src_crs, dst_crs in crs_pairs:
        ranges = CRS_RANGES[src_crs]

        for h, w in sizes:
            da = _make_raster(h, w, crs=src_crs, **ranges)
            da_rio = _make_rio_raster(da, src_crs)

            for resampling in resamplings:
                # Run rioxarray first to get the reference output grid
                rio_resamp = RESAMPLING_MAP_RIO[resampling]
                rio_result = da_rio.rio.reproject(dst_crs,
                                                 resampling=rio_resamp)
                rio_vals = rio_result.values

                # Extract rioxarray's output grid parameters
                rio_transform = rio_result.rio.transform()
                rio_res_x = rio_transform.a
                rio_res_y = abs(rio_transform.e)
                rio_h, rio_w = rio_vals.shape
                rio_left = rio_transform.c
                rio_top = rio_transform.f
                rio_bounds = (
                    rio_left,                        # left
                    rio_top - rio_res_y * rio_h,     # bottom
                    rio_left + rio_res_x * rio_w,    # right
                    rio_top,                         # top
                )

                # Run xrspatial with the same grid
                xrs_result = xrs_reproject(
                    da, dst_crs,
                    resampling=resampling,
                    resolution=(rio_res_y, rio_res_x),
                    bounds=rio_bounds,
                )
                xrs_vals = xrs_result.values

                shape_ok = xrs_vals.shape == rio_vals.shape
                if not shape_ok:
                    # Crop to common area
                    common_h = min(xrs_vals.shape[0], rio_vals.shape[0])
                    common_w = min(xrs_vals.shape[1], rio_vals.shape[1])
                    xrs_vals = xrs_vals[:common_h, :common_w]
                    rio_vals = rio_vals[:common_h, :common_w]

                # Compare where both have valid data
                xrs_nan = np.isnan(xrs_vals)
                rio_nan = np.isnan(rio_vals)
                both_valid = ~xrs_nan & ~rio_nan
                nan_agree = np.mean(xrs_nan == rio_nan) * 100

                if both_valid.sum() > 0:
                    diff = xrs_vals[both_valid] - rio_vals[both_valid]
                    rmse = np.sqrt(np.mean(diff ** 2))
                    max_err = np.max(np.abs(diff))
                    ss_res = np.sum(diff ** 2)
                    ss_tot = np.sum(
                        (rio_vals[both_valid]
                         - np.mean(rio_vals[both_valid])) ** 2
                    )
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
                    rmse_str = f'{rmse:.6f}'
                    max_str = f'{max_err:.6f}'
                    r2_str = f'{r2:.6f}'
                else:
                    rmse_str = 'N/A'
                    max_str = 'N/A'
                    r2_str = 'N/A'

                out_shape = _fmt_shape(xrs_vals.shape)
                if not shape_ok:
                    out_shape += '*'
                crs_label = f'{src_crs}->{dst_crs}'

                print(f'| {_fmt_shape((h, w)):>12} | {crs_label:>24} '
                      f'| {resampling:>10} '
                      f'| {out_shape:>11} | {rmse_str:>10} '
                      f'| {max_str:>10} | {r2_str:>8} '
                      f'| {nan_agree:>8.1f}% |')


REAL_WORLD_FILES = [
    {
        'path': '~/rtxpy/examples/render_demo_terrain.tif',
        'target_crs': 'EPSG:32618',
        'label': 'render_demo 187x253 NAD83->UTM18',
    },
    {
        'path': '~/rtxpy/examples/USGS_1_n43w123.tif',
        'target_crs': 'EPSG:32610',
        'label': 'USGS 1as Oregon 3612x3612 NAD83->UTM10',
    },
    {
        'path': '~/rtxpy/examples/USGS_1_n39w106.tif',
        'target_crs': 'EPSG:32613',
        'label': 'USGS 1as Colorado 3612x3612 NAD83->UTM13',
    },
    {
        'path': '~/rtxpy/examples/Copernicus_DSM_COG_10_N40_00_W075_00_DEM.tif',
        'target_crs': 'EPSG:32618',
        'label': 'Copernicus DEM 3600x3600 WGS84->UTM18',
    },
    {
        'path': '~/rtxpy/examples/USGS_one_meter_x66y454_NY_LongIsland_Z18_2014.tif',
        'target_crs': 'EPSG:4326',
        'label': 'USGS 1m LongIsland 10012x10012 UTM18->WGS84',
    },
]


def _load_for_both(path):
    """Load a GeoTIFF for both xrspatial and rioxarray."""
    import os
    path = os.path.expanduser(path)

    from xrspatial.geotiff import open_geotiff
    da_xrs = open_geotiff(path)

    da_rio = rioxarray.open_rasterio(path).squeeze(drop=True)
    return da_xrs, da_rio


def run_real_world(files=None, resamplings=None):
    """Benchmark and compare on real-world GeoTIFF files."""
    import os
    files = files or REAL_WORLD_FILES
    resamplings = resamplings or ['bilinear']

    # Filter to files that exist
    files = [f for f in files if os.path.exists(os.path.expanduser(f['path']))]
    if not files:
        print('\nNo real-world files found, skipping.')
        return

    print()
    print('=' * 130)
    print('REAL-WORLD FILES: performance and consistency (approx vs exact vs rioxarray)')
    print('=' * 130)
    print()
    print(f'| {"File":>48} '
          f'| {"xrs approx":>11} | {"xrs exact":>11} | {"rioxarray":>11} '
          f'| {"ap/rio":>6} | {"ex/rio":>6} '
          f'| {"RMSE(approx)":>12} | {"RMSE(exact)":>12} '
          f'| {"MaxE(approx)":>12} | {"MaxE(exact)":>12} |')
    print(f'|{"-"*50}'
          f'|{"-"*13}|{"-"*13}|{"-"*13}'
          f'|{"-"*8}|{"-"*8}'
          f'|{"-"*14}|{"-"*14}'
          f'|{"-"*14}|{"-"*14}|')

    for entry in files:
        da_xrs, da_rio = _load_for_both(entry['path'])
        dst_crs = entry['target_crs']
        label = entry['label']

        for resampling in resamplings:
            rio_resamp = RESAMPLING_MAP_RIO[resampling]

            # Performance: xrspatial approx
            approx_time, _ = _timer(
                lambda: xrs_reproject(da_xrs, dst_crs, resampling=resampling,
                                     transform_precision=16),
                warmup=2, runs=5,
            )

            # Performance: xrspatial exact
            exact_time, _ = _timer(
                lambda: xrs_reproject(da_xrs, dst_crs, resampling=resampling,
                                     transform_precision=0),
                warmup=2, runs=5,
            )

            # Performance: rioxarray
            rio_time, rio_result = _timer(
                lambda: da_rio.rio.reproject(dst_crs, resampling=rio_resamp),
                warmup=2, runs=5,
            )

            approx_ratio = rio_time / approx_time if approx_time > 0 else float('inf')
            exact_ratio = rio_time / exact_time if exact_time > 0 else float('inf')

            # Consistency: force same grid, test both modes
            rio_vals = rio_result.values
            rio_transform = rio_result.rio.transform()
            rio_res_x = rio_transform.a
            rio_res_y = abs(rio_transform.e)
            rio_h, rio_w = rio_vals.shape
            rio_left = rio_transform.c
            rio_top = rio_transform.f
            rio_bounds = (
                rio_left,
                rio_top - rio_res_y * rio_h,
                rio_left + rio_res_x * rio_w,
                rio_top,
            )

            nodata = da_xrs.attrs.get('nodata', None)
            stats = {}
            for mode_name, precision in [('approx', 16), ('exact', 0)]:
                xrs_matched = xrs_reproject(
                    da_xrs, dst_crs,
                    resampling=resampling,
                    resolution=(rio_res_y, rio_res_x),
                    bounds=rio_bounds,
                    transform_precision=precision,
                )
                xrs_vals = xrs_matched.values
                rv = rio_vals

                if xrs_vals.shape != rv.shape:
                    ch = min(xrs_vals.shape[0], rv.shape[0])
                    cw = min(xrs_vals.shape[1], rv.shape[1])
                    xrs_vals = xrs_vals[:ch, :cw]
                    rv = rv[:ch, :cw]

                xf = xrs_vals.astype(np.float64)
                rf = rv.astype(np.float64)

                if nodata is not None and not np.isnan(nodata):
                    both_valid = (xf != nodata) & (rf != nodata)
                else:
                    both_valid = np.isfinite(xf) & np.isfinite(rf)

                if both_valid.sum() > 0:
                    diff = xf[both_valid] - rf[both_valid]
                    rmse = np.sqrt(np.mean(diff ** 2))
                    max_err = np.max(np.abs(diff))
                else:
                    rmse = max_err = float('nan')
                stats[mode_name] = (rmse, max_err)

            print(f'| {label:>48} '
                  f'| {_fmt_time(approx_time):>11} '
                  f'| {_fmt_time(exact_time):>11} '
                  f'| {_fmt_time(rio_time):>11} '
                  f'| {approx_ratio:>5.2f}x '
                  f'| {exact_ratio:>5.2f}x '
                  f'| {stats["approx"][0]:>12.6f} '
                  f'| {stats["exact"][0]:>12.6f} '
                  f'| {stats["approx"][1]:>12.6f} '
                  f'| {stats["exact"][1]:>12.6f} |')


def run_merge(sizes=None):
    """Benchmark xrspatial.merge vs rioxarray.merge_arrays.

    Creates 4 overlapping rasters in a 2x2 grid arrangement and merges
    them into a single mosaic with each library.
    """
    from rioxarray.merge import merge_arrays as rio_merge_arrays

    from xrspatial.reproject import merge as xrs_merge

    sizes = sizes or [(512, 512), (1024, 1024), (2048, 2048)]

    print()
    print('=' * 100)
    print('MERGE BENCHMARK: xrspatial.merge vs rioxarray.merge_arrays (4 overlapping tiles)')
    print('=' * 100)
    print()
    print(f'| {"Tile size":>12} '
          f'| {"xrs merge":>11} | {"rio merge":>11} '
          f'| {"xrs/rio":>7} '
          f'| {"RMSE":>10} | {"MaxErr":>10} '
          f'| {"Valid px":>10} | {"NaN agree":>9} |')
    print(f'|{"-" * 14}'
          f'|{"-" * 13}|{"-" * 13}'
          f'|{"-" * 9}'
          f'|{"-" * 12}|{"-" * 12}'
          f'|{"-" * 12}|{"-" * 11}|')

    for h, w in sizes:
        # Build 4 overlapping tiles in a 2x2 grid.
        # Each tile spans 10 degrees; overlap is 2 degrees on each shared edge.
        # Total coverage: 18 x 18 degrees (from -9 to 9 lon, 41 to 59 lat).
        tile_specs = [
            # (x_range, y_range) -- 2-degree overlap between neighbours
            ((-9, 1),  (49, 59)),   # top-left
            ((-1, 9),  (49, 59)),   # top-right
            ((-9, 1),  (41, 51)),   # bottom-left
            ((-1, 9),  (41, 51)),   # bottom-right
        ]

        tiles_xrs = []
        tiles_rio = []
        for x_range, y_range in tile_specs:
            da = _make_raster(h, w, crs='EPSG:4326',
                              x_range=x_range, y_range=y_range)
            tiles_xrs.append(da)
            tiles_rio.append(_make_rio_raster(da, 'EPSG:4326'))

        # Benchmark xrspatial merge
        xrs_time, xrs_result = _timer(
            lambda: xrs_merge(tiles_xrs),
            warmup=1, runs=3,
        )

        # Benchmark rioxarray merge
        rio_time, rio_result = _timer(
            lambda: rio_merge_arrays(tiles_rio),
            warmup=1, runs=3,
        )

        xrs_vals = xrs_result.values
        rio_vals = rio_result.values

        # Crop to common shape if they differ
        common_h = min(xrs_vals.shape[0], rio_vals.shape[0])
        common_w = min(xrs_vals.shape[1], rio_vals.shape[1])
        xrs_vals = xrs_vals[:common_h, :common_w]
        rio_vals = rio_vals[:common_h, :common_w]

        # Compare where both have valid data
        xrs_nan = np.isnan(xrs_vals)
        rio_nan = np.isnan(rio_vals)
        both_valid = ~xrs_nan & ~rio_nan
        n_valid = int(both_valid.sum())
        nan_agree = np.mean(xrs_nan == rio_nan) * 100

        if n_valid > 0:
            diff = xrs_vals[both_valid] - rio_vals[both_valid]
            rmse = np.sqrt(np.mean(diff ** 2))
            max_err = np.max(np.abs(diff))
            rmse_str = f'{rmse:.6f}'
            max_str = f'{max_err:.6f}'
        else:
            rmse_str = 'N/A'
            max_str = 'N/A'

        ratio = xrs_time / rio_time if rio_time > 0 else float('inf')

        print(f'| {_fmt_shape((h, w)):>12} '
              f'| {_fmt_time(xrs_time):>11} '
              f'| {_fmt_time(rio_time):>11} '
              f'| {ratio:>6.2f}x '
              f'| {rmse_str:>10} | {max_str:>10} '
              f'| {n_valid:>10} | {nan_agree:>8.1f}% |')


def main():
    if not HAS_PYPROJ:
        print('ERROR: pyproj is required for reprojection benchmarks')
        sys.exit(1)
    if not HAS_RIOXARRAY:
        print('ERROR: rioxarray is required for comparison benchmarks')
        print('  pip install rioxarray')
        sys.exit(1)

    print(f'NumPy {np.__version__}')
    try:
        import numba
        print(f'Numba {numba.__version__}')
    except ImportError:
        pass
    try:
        import rasterio
        print(f'Rasterio {rasterio.__version__}')
    except ImportError:
        pass

    run_consistency()
    run_performance()
    run_real_world()
    run_merge()


if __name__ == '__main__':
    main()
