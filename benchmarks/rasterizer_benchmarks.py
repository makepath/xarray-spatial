"""
Benchmark: xarray-spatial vs datashader vs geocube vs rasterio rasterization.

Compares wall-clock time and output consistency across a range of geometry
types, output resolutions, and feature counts.  All libraries rasterize
the same geometries into the same pixel grid.

Usage:
    python benchmarks/rasterizer_benchmarks.py

Outputs:
    - Console: timing tables, consistency checks, top-10 regressions
    - benchmarks/RASTERIZER_BENCHMARKS.md: formatted markdown report
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import geopandas as gpd
import spatialpandas
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import from_bounds
from shapely.geometry import (
    Point, Polygon, MultiPolygon, LineString, MultiLineString, MultiPoint,
    box,
)

import datashader as ds
from geocube.api.core import make_geocube
from xrspatial.rasterize import rasterize as xrs_rasterize


# ---------------------------------------------------------------------------
# Geometry generators
# ---------------------------------------------------------------------------

def make_circles(n, rng):
    """Random buffered points (64-vertex circles)."""
    cx = rng.uniform(-150, 150, n)
    cy = rng.uniform(-70, 70, n)
    radii = rng.uniform(2, 15, n)
    geoms = [Point(x, y).buffer(r, resolution=16)
             for x, y, r in zip(cx, cy, radii)]
    vals = rng.uniform(1, 100, n)
    return geoms, vals


def make_irregular(n, rng, n_verts=128):
    """Irregular star-like polygons with many vertices."""
    geoms = []
    vals = rng.uniform(1, 100, n)
    for _ in range(n):
        cx = rng.uniform(-150, 150)
        cy = rng.uniform(-70, 70)
        r = rng.uniform(5, 20)
        angles = np.sort(rng.uniform(0, 2 * np.pi, n_verts))
        radii = r * (0.7 + 0.3 * rng.uniform(size=n_verts))
        xs = cx + radii * np.cos(angles)
        ys = cy + radii * np.sin(angles)
        coords = list(zip(xs, ys))
        coords.append(coords[0])
        geoms.append(Polygon(coords))
    return geoms, vals


def make_rectangles(n, rng):
    """Axis-aligned rectangles."""
    geoms = []
    vals = rng.uniform(1, 100, n)
    for _ in range(n):
        cx = rng.uniform(-150, 150)
        cy = rng.uniform(-70, 70)
        hw = rng.uniform(2, 20)
        hh = rng.uniform(2, 15)
        geoms.append(box(cx - hw, cy - hh, cx + hw, cy + hh))
    return geoms, vals


def make_stars(n, rng, n_points=5):
    """Concave star polygons with alternating inner/outer vertices."""
    geoms = []
    vals = rng.uniform(1, 100, n)
    for _ in range(n):
        cx = rng.uniform(-150, 150)
        cy = rng.uniform(-70, 70)
        r_outer = rng.uniform(5, 20)
        r_inner = r_outer * rng.uniform(0.3, 0.5)
        angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False)
        angles -= np.pi / 2
        coords = []
        for i, a in enumerate(angles):
            r = r_outer if i % 2 == 0 else r_inner
            coords.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        coords.append(coords[0])
        geoms.append(Polygon(coords))
    return geoms, vals


def make_donuts(n, rng):
    """Polygons with holes (annuli)."""
    geoms = []
    vals = rng.uniform(1, 100, n)
    for _ in range(n):
        cx = rng.uniform(-150, 150)
        cy = rng.uniform(-70, 70)
        r_outer = rng.uniform(8, 20)
        r_inner = r_outer * rng.uniform(0.3, 0.6)
        shell = Point(cx, cy).buffer(r_outer, resolution=16)
        hole = Point(cx, cy).buffer(r_inner, resolution=16)
        geoms.append(shell.difference(hole))
    return geoms, vals


def make_multipolygons(n, rng):
    """MultiPolygon collections (2-4 parts each)."""
    geoms = []
    vals = rng.uniform(1, 100, n)
    for _ in range(n):
        n_parts = rng.integers(2, 5)
        parts = []
        for _ in range(n_parts):
            cx = rng.uniform(-150, 150)
            cy = rng.uniform(-70, 70)
            r = rng.uniform(2, 8)
            parts.append(Point(cx, cy).buffer(r, resolution=8))
        geoms.append(MultiPolygon(parts))
    return geoms, vals


def make_lines(n, rng):
    """Random multi-segment LineStrings."""
    geoms = []
    vals = rng.uniform(1, 100, n)
    for _ in range(n):
        n_pts = rng.integers(3, 10)
        xs = np.cumsum(rng.uniform(-20, 20, n_pts)) + rng.uniform(-140, 140)
        ys = np.cumsum(rng.uniform(-10, 10, n_pts)) + rng.uniform(-60, 60)
        xs = np.clip(xs, -179, 179)
        ys = np.clip(ys, -89, 89)
        geoms.append(LineString(zip(xs, ys)))
    return geoms, vals


def make_multilines(n, rng):
    """MultiLineString collections (2-4 parts each)."""
    geoms = []
    vals = rng.uniform(1, 100, n)
    for _ in range(n):
        n_parts = rng.integers(2, 5)
        parts = []
        for _ in range(n_parts):
            n_pts = rng.integers(2, 6)
            xs = np.cumsum(rng.uniform(-15, 15, n_pts)) + rng.uniform(-140, 140)
            ys = np.cumsum(rng.uniform(-8, 8, n_pts)) + rng.uniform(-60, 60)
            xs = np.clip(xs, -179, 179)
            ys = np.clip(ys, -89, 89)
            parts.append(LineString(zip(xs, ys)))
        geoms.append(MultiLineString(parts))
    return geoms, vals


def make_points(n, rng):
    """Random points."""
    xs = rng.uniform(-170, 170, n)
    ys = rng.uniform(-80, 80, n)
    geoms = [Point(x, y) for x, y in zip(xs, ys)]
    vals = rng.uniform(1, 100, n)
    return geoms, vals


def make_multipoints(n, rng):
    """MultiPoint collections (3-8 points each)."""
    geoms = []
    vals = rng.uniform(1, 100, n)
    for _ in range(n):
        n_pts = rng.integers(3, 9)
        xs = rng.uniform(-170, 170, n_pts)
        ys = rng.uniform(-80, 80, n_pts)
        geoms.append(MultiPoint(list(zip(xs, ys))))
    return geoms, vals


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

BOUNDS = (-180, -90, 180, 90)

GEOM_JSON = json.dumps({
    "type": "Polygon",
    "coordinates": [[
        [BOUNDS[0], BOUNDS[1]],
        [BOUNDS[2], BOUNDS[1]],
        [BOUNDS[2], BOUNDS[3]],
        [BOUNDS[0], BOUNDS[3]],
        [BOUNDS[0], BOUNDS[1]],
    ]],
})


def run_xrspatial(pairs, width, height):
    return xrs_rasterize(
        pairs, width=width, height=height,
        bounds=BOUNDS, fill=0.0,
    )


def run_datashader(spd_gdf, width, height):
    cvs = ds.Canvas(
        plot_width=width, plot_height=height,
        x_range=(BOUNDS[0], BOUNDS[2]),
        y_range=(BOUNDS[1], BOUNDS[3]),
    )
    return cvs.polygons(spd_gdf, geometry="geometry", agg=ds.last("value"))


def run_geocube(gdf, width, height):
    x_res = (BOUNDS[2] - BOUNDS[0]) / width
    y_res = (BOUNDS[3] - BOUNDS[1]) / height
    return make_geocube(
        gdf,
        measurements=["value"],
        resolution=(-y_res, x_res),
        geom=GEOM_JSON,
        fill=0.0,
    )


def run_rasterio(shapes, width, height):
    transform = from_bounds(*BOUNDS, width, height)
    return rio_rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        dtype=np.float64,
    )


def run_xrspatial_cupy(pairs, width, height):
    import cupy
    result = xrs_rasterize(
        pairs, width=width, height=height,
        bounds=BOUNDS, fill=0.0, use_cuda=True,
    )
    cupy.cuda.Device().synchronize()
    return result


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def bench(fn, *args, warmup=1, repeats=5):
    """Return (result, mean_seconds, std_seconds) over *repeats* after *warmup*."""
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
# Output consistency helpers
# ---------------------------------------------------------------------------

def _to_numpy(result, label):
    """Extract a 2-D float64 numpy array from any rasterizer output."""
    if isinstance(result, np.ndarray):
        return result.astype(np.float64)
    try:
        import xarray as xr
        if isinstance(result, xr.Dataset):
            return result["value"].values.astype(np.float64)
    except Exception:
        pass
    data = getattr(result, "data", result)
    try:
        data = data.get()  # cupy
    except AttributeError:
        pass
    return np.asarray(data, dtype=np.float64)


def _coverage_mask(arr):
    """Boolean mask of covered (non-zero, non-NaN) pixels."""
    return np.isfinite(arr) & (arr != 0.0)


def check_consistency(results, width, height):
    """Compare outputs pairwise. Returns list of dicts with results."""
    arrays = {}
    for label, res in results.items():
        arr = _to_numpy(res, label)
        if arr.shape != (height, width):
            arr = arr[:height, :width]
        arrays[label] = arr

    labels = list(arrays.keys())
    checks = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a_lbl, b_lbl = labels[i], labels[j]
            a, b = arrays[a_lbl], arrays[b_lbl]

            mask_a = _coverage_mask(a)
            mask_b = _coverage_mask(b)

            intersection = np.sum(mask_a & mask_b)
            union = np.sum(mask_a | mask_b)
            iou = intersection / union if union > 0 else 1.0

            overlap = mask_a & mask_b
            if np.any(overlap):
                rmse = np.sqrt(np.mean((a[overlap] - b[overlap]) ** 2))
            else:
                rmse = float("nan")

            checks.append({
                "a": a_lbl, "b": b_lbl,
                "iou": iou, "rmse": rmse,
                "ok": iou > 0.85,
            })

    return checks


# ---------------------------------------------------------------------------
# Geometry type registry
# ---------------------------------------------------------------------------

GEN_LABELS = {
    "circles_64v":     ("Circles (64 vertices)",    "polygon"),
    "irregular_128v":  ("Irregular (128 vertices)", "polygon"),
    "rectangles":      ("Rectangles",               "polygon"),
    "stars_5pt":       ("Stars (5-point, concave)", "polygon"),
    "donuts":          ("Donuts (polygon + hole)",   "polygon"),
    "multipolygons":   ("MultiPolygons (2-4 parts)","polygon"),
    "lines":           ("LineStrings",              "line"),
    "multilines":      ("MultiLineStrings",         "line"),
    "points":          ("Points",                   "point"),
    "multipoints":     ("MultiPoints (3-8 pts)",    "point"),
}

GENERATORS = {
    "circles_64v": {
        "fn": lambda n, rng: make_circles(n, rng),
        "kind": "polygon",
    },
    "irregular_128v": {
        "fn": lambda n, rng: make_irregular(n, rng, n_verts=128),
        "kind": "polygon",
    },
    "rectangles": {
        "fn": lambda n, rng: make_rectangles(n, rng),
        "kind": "polygon",
    },
    "stars_5pt": {
        "fn": lambda n, rng: make_stars(n, rng, n_points=5),
        "kind": "polygon",
    },
    "donuts": {
        "fn": lambda n, rng: make_donuts(n, rng),
        "kind": "polygon",
    },
    "multipolygons": {
        "fn": lambda n, rng: make_multipolygons(n, rng),
        "kind": "polygon",
    },
    "lines": {
        "fn": lambda n, rng: make_lines(n, rng),
        "kind": "line",
    },
    "multilines": {
        "fn": lambda n, rng: make_multilines(n, rng),
        "kind": "line",
    },
    "points": {
        "fn": lambda n, rng: make_points(n, rng),
        "kind": "point",
    },
    "multipoints": {
        "fn": lambda n, rng: make_multipoints(n, rng),
        "kind": "point",
    },
}

GEN_ORDER = [
    "circles_64v", "irregular_128v", "rectangles", "stars_5pt",
    "donuts", "multipolygons", "lines", "multilines", "points", "multipoints",
]


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def _fmt_ms(seconds):
    ms = seconds * 1000
    if ms >= 100:
        return f"{ms:.0f} ms"
    if ms >= 10:
        return f"{ms:.1f} ms"
    return f"{ms:.2f} ms"


def generate_markdown(all_results, slower_cases, has_cupy):
    """Build the full markdown report string from collected results."""
    lines = []
    w = lines.append

    w("# Rasterizer Benchmarks")
    w("")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    w(f"Generated: {now}")
    w("")
    w("Compares xarray-spatial (numpy and cupy backends) against datashader, "
      "geocube, and rasterio across 10 geometry types, 3 feature counts "
      "(50/200/1000), and 5 output resolutions (100-4000 px wide).")
    w("")
    w("- **Polygon types** (circles, irregular, rectangles, stars, donuts, "
      "multipolygons): all 5 rasterizers compared")
    w("- **Line types** (lines, multilines): xrspatial, geocube, rasterio "
      "(datashader uses a different API for lines)")
    w("- **Point types** (points, multipoints): xrspatial, geocube, rasterio "
      "(datashader uses a different API for points)")
    w("")

    # Per-geometry sections
    for gen_key in GEN_ORDER:
        if gen_key not in all_results:
            continue
        label, kind = GEN_LABELS[gen_key]
        entries = all_results[gen_key]
        is_poly = kind == "polygon"

        w(f"## {label}")
        w("")

        # Timing table
        w("### Timings")
        w("")
        if is_poly:
            h = "| n | size | xrs-numpy | xrs-cupy | datashader | geocube | rasterio |"
            s = "|---:|---:|---:|---:|---:|---:|---:|"
        else:
            h = "| n | size | xrs-numpy | xrs-cupy | geocube | rasterio |"
            s = "|---:|---:|---:|---:|---:|---:|"
        w(h)
        w(s)

        for e in entries:
            size = f"{e['w']}x{e['h']}"
            xnp = _fmt_ms(e["xrs_mean"])
            xcu = _fmt_ms(e["cupy_mean"]) if e["cupy_mean"] is not None else "--"
            gc = _fmt_ms(e["gc_mean"])
            rio = _fmt_ms(e["rio_mean"])
            if is_poly:
                dsh = _fmt_ms(e["ds_mean"]) if e["ds_mean"] is not None else "--"
                w(f"| {e['n']} | {size} | {xnp} | {xcu} | {dsh} | {gc} | {rio} |")
            else:
                w(f"| {e['n']} | {size} | {xnp} | {xcu} | {gc} | {rio} |")

        w("")

        # Ratio table
        w("### Relative to xrs-numpy")
        w("")
        w("Values below 1.0 mean the competitor is faster than xrs-numpy.")
        w("")
        if is_poly:
            h = "| n | size | xrs-cupy | datashader | geocube | rasterio |"
            s = "|---:|---:|---:|---:|---:|---:|"
        else:
            h = "| n | size | xrs-cupy | geocube | rasterio |"
            s = "|---:|---:|---:|---:|---:|"
        w(h)
        w(s)

        for e in entries:
            size = f"{e['w']}x{e['h']}"
            xrs = e["xrs_mean"]
            cu_r = f"{e['cupy_mean']/xrs:.2f}x" if e["cupy_mean"] is not None and xrs > 0 else "--"
            gc_r = f"{e['gc_mean']/xrs:.2f}x" if xrs > 0 else "--"
            rio_r = f"{e['rio_mean']/xrs:.2f}x" if xrs > 0 else "--"
            if is_poly:
                ds_r = f"{e['ds_mean']/xrs:.2f}x" if e["ds_mean"] is not None and xrs > 0 else "--"
                w(f"| {e['n']} | {size} | {cu_r} | {ds_r} | {gc_r} | {rio_r} |")
            else:
                w(f"| {e['n']} | {size} | {cu_r} | {gc_r} | {rio_r} |")

        w("")

        # Consistency summary
        w("### Consistency")
        w("")
        pair_data = defaultdict(lambda: {"ious": [], "rmses": []})
        for e in entries:
            for c in e["consistency"]:
                key = f"{c['a']} vs {c['b']}"
                pair_data[key]["ious"].append(c["iou"])
                pair_data[key]["rmses"].append(c["rmse"])

        w("| pair | IoU min | IoU max | RMSE range |")
        w("|:-----|--------:|--------:|-----------:|")
        for pair, d in pair_data.items():
            imin = min(d["ious"])
            imax = max(d["ious"])
            rmin = min(r for r in d["rmses"] if not np.isnan(r)) if any(not np.isnan(r) for r in d["rmses"]) else float("nan")
            rmax = max(r for r in d["rmses"] if not np.isnan(r)) if any(not np.isnan(r) for r in d["rmses"]) else float("nan")
            w(f"| {pair} | {imin:.4f} | {imax:.4f} | {rmin:.1f} - {rmax:.1f} |")

        w("")

    # Top 10 slower cases
    if slower_cases:
        slower_cases.sort(key=lambda x: -x[1])
        top = slower_cases[:10]

        w("## Where xrs-numpy is slower")
        w("")
        w("Top 10 configurations where another rasterizer beats xrs-numpy.")
        w("")
        w("| # | geometry | n | size | faster lib | xrs-numpy | other | xrs slower by |")
        w("|--:|:---------|--:|-----:|:-----------|----------:|------:|--------------:|")
        for i, (comp, ratio, gen, n, ww, hh, xrs_ms, comp_ms) in enumerate(top, 1):
            size = f"{ww}x{hh}"
            w(f"| {i} | {gen} | {n} | {size} | {comp} | {xrs_ms:.1f} ms | {comp_ms:.1f} ms | {ratio:.1f}x |")
        w("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)

    resolutions = [100, 500, 1000, 2000, 4000]
    feature_counts = [50, 200, 1000]

    try:
        import cupy  # noqa: F401
        has_cupy = True
    except ImportError:
        has_cupy = False

    # Structured results: gen_key -> list of row dicts
    all_results = defaultdict(list)
    slower_cases = []

    header = (
        f"{'generator':<18} {'n':>5} {'width':>6} {'height':>6}"
        f"  {'xrs-numpy (s)':>14}"
        f"  {'xrs-cupy (s)':>14}"
        f"  {'datashader (s)':>14}"
        f"  {'geocube (s)':>14}"
        f"  {'rasterio (s)':>14}"
        f"  {'ds/xrs':>8}"
        f"  {'gc/xrs':>8}"
        f"  {'rio/xrs':>8}"
        f"  {'cupy/xrs':>8}"
    )
    print(header)
    print("-" * len(header))

    for gen_name, gen_cfg in GENERATORS.items():
        gen_fn = gen_cfg["fn"]
        kind = gen_cfg["kind"]
        use_datashader = kind == "polygon"

        for n in feature_counts:
            geoms, vals = gen_fn(n, rng)
            pairs = list(zip(geoms, vals))

            gdf = gpd.GeoDataFrame(
                {"value": vals}, geometry=geoms, crs="EPSG:4326")
            rio_shapes = [(g, v) for g, v in zip(geoms, vals)]

            if use_datashader:
                spd_gdf = spatialpandas.GeoDataFrame(gdf)

            for width in resolutions:
                height = width // 2

                xrs_res, xrs_mean, xrs_std = bench(
                    run_xrspatial, pairs, width, height)
                gc_res, gc_mean, gc_std = bench(
                    run_geocube, gdf, width, height)
                rio_res, rio_mean, rio_std = bench(
                    run_rasterio, rio_shapes, width, height)

                ds_mean = ds_std = ds_res = None
                if use_datashader:
                    ds_res, ds_mean, ds_std = bench(
                        run_datashader, spd_gdf, width, height)
                    ds_ratio = ds_mean / xrs_mean if xrs_mean > 0 else float("inf")
                    ds_col = f"  {ds_mean:>8.4f}+-{ds_std:.4f}"
                    ds_rat = f"  {ds_ratio:>7.2f}x"
                else:
                    ds_col = f"  {'--':>14}"
                    ds_rat = f"  {'--':>8}"

                cupy_mean = cupy_std = cupy_res = None
                if has_cupy:
                    cupy_res, cupy_mean, cupy_std = bench(
                        run_xrspatial_cupy, pairs, width, height)
                    cupy_ratio = (cupy_mean / xrs_mean
                                  if xrs_mean > 0 else float("inf"))
                    cupy_col = f"  {cupy_mean:>8.4f}+-{cupy_std:.4f}"
                    cupy_rat = f"  {cupy_ratio:>7.2f}x"
                else:
                    cupy_col = f"  {'n/a':>14}"
                    cupy_rat = f"  {'n/a':>8}"

                gc_ratio = gc_mean / xrs_mean if xrs_mean > 0 else float("inf")
                rio_ratio = rio_mean / xrs_mean if xrs_mean > 0 else float("inf")

                # Track slower-than cases
                if use_datashader and ds_mean is not None:
                    ds_r = ds_mean / xrs_mean if xrs_mean > 0 else float("inf")
                    if ds_r < 1.0:
                        slower_cases.append((
                            "datashader", 1.0 / ds_r, gen_name, n,
                            width, height, xrs_mean * 1000, ds_mean * 1000))
                if gc_ratio < 1.0:
                    slower_cases.append((
                        "geocube", 1.0 / gc_ratio, gen_name, n,
                        width, height, xrs_mean * 1000, gc_mean * 1000))
                if rio_ratio < 1.0:
                    slower_cases.append((
                        "rasterio", 1.0 / rio_ratio, gen_name, n,
                        width, height, xrs_mean * 1000, rio_mean * 1000))

                # Console output
                print(
                    f"{gen_name:<18} {n:>5} {width:>6} {height:>6}"
                    f"  {xrs_mean:>8.4f}+-{xrs_std:.4f}"
                    f"{cupy_col}"
                    f"{ds_col}"
                    f"  {gc_mean:>8.4f}+-{gc_std:.4f}"
                    f"  {rio_mean:>8.4f}+-{rio_std:.4f}"
                    f"{ds_rat}"
                    f"  {gc_ratio:>7.2f}x"
                    f"  {rio_ratio:>7.2f}x"
                    f"{cupy_rat}"
                )

                # Consistency checks
                consistency_inputs = {"xrs-numpy": xrs_res}
                if ds_res is not None:
                    consistency_inputs["datashader"] = ds_res
                consistency_inputs["geocube"] = gc_res
                consistency_inputs["rasterio"] = rio_res
                if cupy_res is not None:
                    consistency_inputs["xrs-cupy"] = cupy_res
                checks = check_consistency(consistency_inputs, width, height)

                for c in checks:
                    status = "OK" if c["ok"] else "WARN"
                    print(f"    {c['a']} vs {c['b']}: "
                          f"IoU={c['iou']:.4f}  RMSE={c['rmse']:.4f}  "
                          f"[{status}]")

                # Store structured result
                all_results[gen_name].append({
                    "n": n, "w": width, "h": height,
                    "xrs_mean": xrs_mean, "xrs_std": xrs_std,
                    "cupy_mean": cupy_mean, "cupy_std": cupy_std,
                    "ds_mean": ds_mean, "ds_std": ds_std,
                    "gc_mean": gc_mean, "gc_std": gc_std,
                    "rio_mean": rio_mean, "rio_std": rio_std,
                    "consistency": checks,
                })

    # Console: top 10 slower cases
    if slower_cases:
        slower_cases.sort(key=lambda x: -x[1])
        top = slower_cases[:10]
        print()
        print("=" * 95)
        print("  Top 10 cases where xrs-numpy is SLOWER than a competitor")
        print("=" * 95)
        print()
        print(f"  {'#':>2}  {'geometry':<18} {'n':>5}  {'size':>10}"
              f"  {'faster lib':<12} {'xrs-np':>9}  {'other':>9}"
              f"  {'xrs slower by':>13}")
        print(f"  {'':->2}  {'':->18} {'':->5}  {'':->10}"
              f"  {'':->12} {'':->9}  {'':->9}  {'':->13}")
        for i, (comp, ratio, gen, n, w, h, xrs_ms, comp_ms) in enumerate(
                top, 1):
            size = f"{w}x{h}"
            print(f"  {i:>2}  {gen:<18} {n:>5}  {size:>10}"
                  f"  {comp:<12} {xrs_ms:>7.1f}ms  {comp_ms:>7.1f}ms"
                  f"  {ratio:>11.1f}x")
        print()

    # Write markdown report
    md = generate_markdown(all_results, slower_cases, has_cupy)
    md_path = os.path.join(os.path.dirname(__file__), "RASTERIZER_BENCHMARKS.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
