"""
Benchmark: xrspatial vs rasterio polygonize (raster-to-vector).

Compares wall-clock time and output consistency across a range of raster
sizes and region patterns.  Both libraries polygonize the same synthetic
rasters.

Usage:
    python benchmarks/benchmark_polygonize.py

Outputs:
    - Console: timing tables, consistency checks
    - benchmarks/POLYGONIZE_BENCHMARKS.md: formatted markdown report
"""

import os
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import xarray as xr
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shapely_shape

from xrspatial import polygonize as xrs_polygonize


# ---------------------------------------------------------------------------
# Raster generators
# ---------------------------------------------------------------------------

def make_few_large(ny, nx, rng):
    """Few large contiguous regions (low unique-value count)."""
    n_vals = max(3, min(8, nx // 50))
    base = rng.integers(0, n_vals, size=(ny // 8 + 1, nx // 8 + 1),
                        dtype=np.int32)
    # Upscale with nearest-neighbour to get big blobs.
    rows = np.repeat(np.arange(base.shape[0]), 8)[:ny]
    cols = np.repeat(np.arange(base.shape[1]), 8)[:nx]
    return base[np.ix_(rows, cols)].copy()


def make_many_small(ny, nx, rng):
    """Many small regions (high unique-value count)."""
    n_vals = max(10, nx // 5)
    return rng.integers(0, n_vals, size=(ny, nx), dtype=np.int32)


def make_checkerboard(ny, nx, rng):
    """Alternating checkerboard pattern (worst case for polygon count)."""
    row = np.arange(nx, dtype=np.int32)
    col = np.arange(ny, dtype=np.int32)
    return ((row[np.newaxis, :] + col[:, np.newaxis]) % 2).astype(np.int32)


def make_rings(ny, nx, rng):
    """Concentric rings from the center (regions with holes)."""
    cy, cx = ny / 2, nx / 2
    yy, xx = np.mgrid[:ny, :nx]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring_width = max(5, min(nx, ny) // 15)
    return (dist / ring_width).astype(np.int32)


def make_masked_blobs(ny, nx, rng):
    """Large regions with ~15% masked-out pixels."""
    n_vals = max(3, min(8, nx // 50))
    base = rng.integers(0, n_vals, size=(ny // 6 + 1, nx // 6 + 1),
                        dtype=np.int32)
    rows = np.repeat(np.arange(base.shape[0]), 6)[:ny]
    cols = np.repeat(np.arange(base.shape[1]), 6)[:nx]
    raster = base[np.ix_(rows, cols)].copy()
    return raster


def make_mask_for(ny, nx, rng):
    """Return a boolean mask with ~15% False pixels."""
    return (rng.uniform(size=(ny, nx)) > 0.15)


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------

def run_xrspatial(raster_da, mask_da):
    """Run xrspatial polygonize, return list of (value, shapely_polygon)."""
    column, polygon_points = xrs_polygonize(
        raster_da, mask=mask_da, return_type="numpy")
    # Convert to shapely for consistency comparison.
    from shapely.geometry import Polygon
    polys = []
    for val, rings in zip(column, polygon_points):
        try:
            p = Polygon(rings[0], rings[1:])
            polys.append((val, p))
        except Exception:
            pass
    return polys


def run_xrspatial_cupy(raster_da, mask_da):
    """Run xrspatial polygonize with cupy input."""
    import cupy
    data_cu = cupy.asarray(raster_da.values)
    raster_cu = xr.DataArray(data_cu)
    mask_cu = None
    if mask_da is not None:
        mask_cu = xr.DataArray(cupy.asarray(mask_da.values))
    column, polygon_points = xrs_polygonize(
        raster_cu, mask=mask_cu, return_type="numpy")
    cupy.cuda.Device().synchronize()
    from shapely.geometry import Polygon
    polys = []
    for val, rings in zip(column, polygon_points):
        try:
            p = Polygon(rings[0], rings[1:])
            polys.append((val, p))
        except Exception:
            pass
    return polys


def run_rasterio(raster_np, mask_np):
    """Run rasterio.features.shapes, return list of (value, shapely_polygon)."""
    results = []
    for geojson_dict, value in rio_shapes(raster_np, mask=mask_np):
        try:
            geom = shapely_shape(geojson_dict)
            results.append((float(value), geom))
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def bench(fn, *args, warmup=1, repeats=5):
    """Return (result, mean_seconds, std_seconds)."""
    for _ in range(warmup):
        fn(*args)
    result = None
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    arr = np.array(times)
    return result, arr.mean(), arr.std()


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------

def check_consistency(results_dict):
    """Compare polygon outputs pairwise by count and total area."""
    stats = {}
    for label, polys in results_dict.items():
        total_area = sum(p.area for _, p in polys if p.is_valid)
        stats[label] = {"count": len(polys), "area": total_area}

    labels = list(stats.keys())
    checks = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a_lbl, b_lbl = labels[i], labels[j]
            a, b = stats[a_lbl], stats[b_lbl]

            count_ratio = (min(a["count"], b["count"]) /
                           max(a["count"], b["count"])
                           if max(a["count"], b["count"]) > 0 else 1.0)
            area_diff = abs(a["area"] - b["area"])
            max_area = max(a["area"], b["area"])
            area_ratio = (1.0 - area_diff / max_area) if max_area > 0 else 1.0

            checks.append({
                "a": a_lbl, "b": b_lbl,
                "count_a": a["count"], "count_b": b["count"],
                "count_ratio": count_ratio,
                "area_a": a["area"], "area_b": b["area"],
                "area_ratio": area_ratio,
                "ok": area_ratio > 0.95,
            })
    return checks


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GEN_LABELS = {
    "few_large":    "Few large regions",
    "many_small":   "Many small regions",
    "checkerboard": "Checkerboard",
    "rings":        "Concentric rings (holes)",
    "masked_blobs": "Large regions + 15% mask",
}

GENERATORS = {
    "few_large":    {"fn": make_few_large,    "masked": False},
    "many_small":   {"fn": make_many_small,   "masked": False},
    "checkerboard": {"fn": make_checkerboard, "masked": False},
    "rings":        {"fn": make_rings,        "masked": False},
    "masked_blobs": {"fn": make_masked_blobs, "masked": True},
}

GEN_ORDER = ["few_large", "many_small", "checkerboard", "rings",
             "masked_blobs"]


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt_ms(seconds):
    ms = seconds * 1000
    if ms >= 100:
        return f"{ms:.0f} ms"
    if ms >= 10:
        return f"{ms:.1f} ms"
    return f"{ms:.2f} ms"


def generate_markdown(all_results, slower_cases, has_cupy):
    lines = []
    w = lines.append

    w("# Polygonize benchmarks")
    w("")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    w(f"Generated: {now}")
    w("")
    w("Compares xrspatial polygonize (numpy and cupy backends) against "
      "rasterio.features.shapes (GDAL) across 5 region patterns and "
      "4 raster sizes. The **fastest** time in each row is bold.")
    w("")

    for gen_key in GEN_ORDER:
        if gen_key not in all_results:
            continue
        label = GEN_LABELS[gen_key]
        entries = all_results[gen_key]

        w(f"## {label}")
        w("")

        # Timing table
        w("### Timings")
        w("")
        w("| size | xrs-numpy | xrs-cupy | rasterio |")
        w("|---:|---:|---:|---:|")

        for e in entries:
            size = f"{e['nx']}x{e['ny']}"
            times = {"xrs-numpy": e["xrs_mean"]}
            if e["cupy_mean"] is not None:
                times["xrs-cupy"] = e["cupy_mean"]
            times["rasterio"] = e["rio_mean"]
            fastest = min(times, key=times.get)

            def _bold_if(key):
                val = times.get(key)
                if val is None:
                    return "--"
                s = _fmt_ms(val)
                return f"**{s}**" if key == fastest else s

            xnp = _bold_if("xrs-numpy")
            xcu = _bold_if("xrs-cupy")
            rio = _bold_if("rasterio")
            w(f"| {size} | {xnp} | {xcu} | {rio} |")

        w("")

        # Ratio table
        w("### Relative to xrs-numpy")
        w("")
        w("Values below 1.0 mean the competitor is faster.")
        w("")
        w("| size | xrs-cupy | rasterio |")
        w("|---:|---:|---:|")

        for e in entries:
            size = f"{e['nx']}x{e['ny']}"
            xrs = e["xrs_mean"]
            cu_r = (f"{e['cupy_mean']/xrs:.2f}x"
                    if e["cupy_mean"] is not None and xrs > 0 else "--")
            rio_r = f"{e['rio_mean']/xrs:.2f}x" if xrs > 0 else "--"
            w(f"| {size} | {cu_r} | {rio_r} |")

        w("")

        # Consistency
        w("### Consistency")
        w("")
        pair_data = defaultdict(lambda: {"area_ratios": [], "count_ratios": []})
        for e in entries:
            for c in e["consistency"]:
                key = f"{c['a']} vs {c['b']}"
                pair_data[key]["area_ratios"].append(c["area_ratio"])
                pair_data[key]["count_ratios"].append(c["count_ratio"])

        w("| pair | area agreement min | area agreement max | count ratio range |")
        w("|:-----|---:|---:|---:|")
        for pair, d in pair_data.items():
            amin = min(d["area_ratios"])
            amax = max(d["area_ratios"])
            cmin = min(d["count_ratios"])
            cmax = max(d["count_ratios"])
            w(f"| {pair} | {amin:.4f} | {amax:.4f} | {cmin:.2f} - {cmax:.2f} |")

        w("")

    # Top slower cases
    if slower_cases:
        slower_cases.sort(key=lambda x: -x[1])
        top = slower_cases[:10]

        w("## Where xrs-numpy is slower")
        w("")
        w("Configurations where rasterio beats xrs-numpy.")
        w("")
        w("| # | pattern | size | faster lib | xrs-numpy | other | xrs slower by |")
        w("|--:|:--------|-----:|:-----------|----------:|------:|--------------:|")
        for i, (comp, ratio, gen, nx, ny, xrs_ms, comp_ms) in enumerate(
                top, 1):
            size = f"{nx}x{ny}"
            w(f"| {i} | {gen} | {size} | {comp} "
              f"| {xrs_ms:.1f} ms | {comp_ms:.1f} ms | {ratio:.1f}x |")
        w("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)

    resolutions = [(100, 50), (500, 250), (1000, 500), (2000, 1000)]

    try:
        import cupy  # noqa: F401
        has_cupy = True
    except ImportError:
        has_cupy = False

    all_results = defaultdict(list)
    slower_cases = []

    header = (
        f"{'pattern':<18} {'size':>10}"
        f"  {'xrs-numpy (s)':>14}"
        f"  {'xrs-cupy (s)':>14}"
        f"  {'rasterio (s)':>14}"
        f"  {'rio/xrs':>8}"
        f"  {'cupy/xrs':>8}"
    )
    print(header)
    print("-" * len(header))

    for gen_name in GEN_ORDER:
        gen_cfg = GENERATORS[gen_name]
        gen_fn = gen_cfg["fn"]
        use_mask = gen_cfg["masked"]

        for nx, ny in resolutions:
            raster_np = gen_fn(ny, nx, rng)
            mask_np = make_mask_for(ny, nx, rng) if use_mask else None

            raster_da = xr.DataArray(raster_np)
            mask_da = xr.DataArray(mask_np) if mask_np is not None else None

            # xrspatial numpy
            xrs_res, xrs_mean, xrs_std = bench(
                run_xrspatial, raster_da, mask_da)

            # rasterio
            rio_mask = mask_np.astype(np.uint8) if mask_np is not None else None
            rio_res, rio_mean, rio_std = bench(
                run_rasterio, raster_np, rio_mask)

            # xrspatial cupy
            cupy_mean = cupy_std = None
            cupy_res = None
            if has_cupy:
                cupy_res, cupy_mean, cupy_std = bench(
                    run_xrspatial_cupy, raster_da, mask_da)
                cupy_ratio = (cupy_mean / xrs_mean
                              if xrs_mean > 0 else float("inf"))
                cupy_col = f"  {cupy_mean:>8.4f}+-{cupy_std:.4f}"
                cupy_rat = f"  {cupy_ratio:>7.2f}x"
            else:
                cupy_col = f"  {'n/a':>14}"
                cupy_rat = f"  {'n/a':>8}"

            rio_ratio = rio_mean / xrs_mean if xrs_mean > 0 else float("inf")

            if rio_ratio < 1.0:
                slower_cases.append((
                    "rasterio", 1.0 / rio_ratio, gen_name,
                    nx, ny, xrs_mean * 1000, rio_mean * 1000))

            # Console output
            size_str = f"{nx}x{ny}"
            print(
                f"{gen_name:<18} {size_str:>10}"
                f"  {xrs_mean:>8.4f}+-{xrs_std:.4f}"
                f"{cupy_col}"
                f"  {rio_mean:>8.4f}+-{rio_std:.4f}"
                f"  {rio_ratio:>7.2f}x"
                f"{cupy_rat}"
            )

            # Consistency
            consistency_inputs = {"xrs-numpy": xrs_res}
            consistency_inputs["rasterio"] = rio_res
            if cupy_res is not None:
                consistency_inputs["xrs-cupy"] = cupy_res
            checks = check_consistency(consistency_inputs)

            for c in checks:
                status = "OK" if c["ok"] else "WARN"
                print(f"    {c['a']} vs {c['b']}: "
                      f"count={c['count_a']}/{c['count_b']}  "
                      f"area_agree={c['area_ratio']:.4f}  "
                      f"[{status}]")

            all_results[gen_name].append({
                "nx": nx, "ny": ny,
                "xrs_mean": xrs_mean, "xrs_std": xrs_std,
                "cupy_mean": cupy_mean, "cupy_std": cupy_std,
                "rio_mean": rio_mean, "rio_std": rio_std,
                "consistency": checks,
            })

    # Console: slower cases
    if slower_cases:
        slower_cases.sort(key=lambda x: -x[1])
        top = slower_cases[:10]
        print()
        print("=" * 80)
        print("  Cases where xrs-numpy is SLOWER than rasterio")
        print("=" * 80)
        print()
        print(f"  {'#':>2}  {'pattern':<18} {'size':>10}"
              f"  {'xrs-np':>9}  {'rasterio':>9}  {'xrs slower by':>13}")
        print(f"  {'':->2}  {'':->18} {'':->10}"
              f"  {'':->9}  {'':->9}  {'':->13}")
        for i, (comp, ratio, gen, nx, ny, xrs_ms, comp_ms) in enumerate(
                top, 1):
            size = f"{nx}x{ny}"
            print(f"  {i:>2}  {gen:<18} {size:>10}"
                  f"  {xrs_ms:>7.1f}ms  {comp_ms:>7.1f}ms"
                  f"  {ratio:>11.1f}x")
        print()

    # Write markdown report
    md = generate_markdown(all_results, slower_cases, has_cupy)
    md_path = os.path.join(os.path.dirname(__file__), "POLYGONIZE_BENCHMARKS.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
