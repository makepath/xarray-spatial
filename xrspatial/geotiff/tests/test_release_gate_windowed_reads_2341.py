"""Release gate: windowed-read coords + shifted-transform parity (epic #2341).

Epic #2341 names "windowed reads return unshifted transforms" as a
priority risk: a window that returns the file's full-extent transform
but a small array is a footgun that downstream spatial functions silently
trust. This file is the single release-gate citation for that risk.

For each file in a small representative corpus (integer dtype with a
nodata sentinel, float dtype with NaN nodata, float dtype without a
sentinel, uint8 MinIsWhite stripped without GeoTIFF tags) and for
both eager (``open_geotiff(..., window=...)``) and dask
(``read_geotiff_dask(..., window=...)``) read paths the test asserts:

* the returned shape equals the window's ``(height, width)``;
* ``coords['y']`` / ``coords['x']`` equal the matching slice of the
  full-file coords (bit-exact);
* ``attrs['transform']`` equals
  ``T_full * Affine.translation(window.col_off, window.row_off)``
  computed by hand from ``T_full``, with no float drift;
* the canonical non-transform release attrs (``crs``, ``crs_wkt``,
  ``nodata``, ``masked_nodata``, ``georef_status``, ``raster_type``)
  match the unwindowed read.

Out of scope here (sibling PRs of epic #2341): overview / sidecar
metadata survival (PR 3 / #2359), stable-codec read/write/read
round-trip (PR 4 / #2360), negative tests for ambiguous metadata
(PR 5 / #2361).

Assertions are inlined per-file inside this module so the file is the
single citation in ``docs/source/reference/release_gate_geotiff.rst``
and is self-contained against the four sibling PRs landing in parallel.
"""
from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, read_geotiff_dask, to_geotiff


_HAS_TIFFFILE = importlib.util.find_spec("tifffile") is not None
_skip_no_tifffile = pytest.mark.skipif(
    not _HAS_TIFFFILE, reason="tifffile required for MinIsWhite fixture")


# ---------------------------------------------------------------------------
# Window geometry under test
# ---------------------------------------------------------------------------
# Strictly interior to the 256x256 fixture so the test fails on any
# off-by-one in row/col offsets and so a wrong window cannot silently
# coincide with the full-extent read. Width != height so a swapped axis
# also fails the shape assertion. Two cases:
#
# * ``aligned``: offsets are a multiple of the dask ``chunks=32`` size,
#   so the window starts on a chunk boundary.
# * ``chunk-misaligned``: offsets are NOT a multiple of 32 and the
#   window's height / width are also not chunk-aligned, so the dask
#   reader has to split chunks. A reader that off-by-ones the
#   chunk math at the window origin fails this case but not the
#   aligned one.
_FULL_H = 256
_FULL_W = 256


# ``window=`` kwarg ordering (per the ``open_geotiff`` /
# ``read_geotiff_dask`` docstrings): ``(row_start, col_start,
# row_stop, col_stop)``.
_WINDOWS = (
    pytest.param((32, 64, 96, 192),
                 id="aligned-row32-col64-h64-w128"),
    pytest.param((33, 65, 95, 191),
                 id="chunk-misaligned-row33-col65-h62-w126"),
)


# Pixel geometry pinned on every fixture. Non-square pixels and a
# fractional origin catch any window path that accidentally uses
# integer arithmetic or drops the fractional part of the origin when
# shifting. ``_ORIGIN_X = 500123.5`` is intentionally fractional so a
# windowed reader that internally rounds (or that re-derives the origin
# from int pixel indices) fails the exact-tuple equality on transform.
_PIXEL_WIDTH = 30.0
_PIXEL_HEIGHT = -25.0
_ORIGIN_X = 500123.5
_ORIGIN_Y = 4001987.25


# Canonical non-transform release attrs from the epic #2341 list.
# ``transform`` is asserted separately by the algebraic spec; the
# remaining canonical keys must equal the unwindowed read's values
# exactly because a window is not allowed to silently rewrite them.
#
# ``masked_nodata`` is on the canonical list but is data-dependent by
# design (see ``_attrs.py``: "True iff the in-memory array is float
# dtype and the reader's sentinel-to-NaN step ran"). A window that
# happens to contain no sentinel pixels legitimately flips the flag
# from True to False on an integer source. We assert structure (the
# key is present iff ``nodata`` is present and the value is a bool)
# rather than exact equality so a real attr-drop is still caught and
# a correct content-dependent flip does not flag as silent wrongness.
_NON_TRANSFORM_ATTRS_VALUE_EQUAL = (
    "crs",
    "crs_wkt",
    "nodata",
    "georef_status",
    "raster_type",
)
_NON_TRANSFORM_ATTRS_STRUCTURAL = ("masked_nodata",)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _build_da(arr: np.ndarray, nodata=None) -> xr.DataArray:
    """Wrap ``arr`` as a DataArray with the pinned pixel geometry.

    ``to_geotiff`` infers the on-disk transform from the y/x coords. The
    coords are pixel centers; ``origin = pixel_corner``, so the centers
    are offset by half a pixel from the origin. The on-disk transform
    that comes back through ``attrs['transform']`` is then exactly
    ``(_PIXEL_WIDTH, 0, _ORIGIN_X, 0, _PIXEL_HEIGHT, _ORIGIN_Y)``.
    """
    h, w = arr.shape
    x_centers = _ORIGIN_X + _PIXEL_WIDTH * 0.5 + _PIXEL_WIDTH * np.arange(w)
    y_centers = _ORIGIN_Y + _PIXEL_HEIGHT * 0.5 + _PIXEL_HEIGHT * np.arange(h)
    attrs = {"crs": 32610}
    if nodata is not None:
        attrs["nodata"] = nodata
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={"y": y_centers.astype(np.float64),
                "x": x_centers.astype(np.float64)},
        attrs=attrs,
    )


def _write_int16_with_nodata(path: Path) -> None:
    rng = np.random.default_rng(2341)
    arr = rng.integers(
        -1000, 1000, size=(_FULL_H, _FULL_W), dtype=np.int16
    )
    # Sprinkle the sentinel so the read path actually carries it.
    arr[10, 10] = -9999
    arr[200, 5] = -9999
    da = _build_da(arr, nodata=-9999)
    to_geotiff(da, str(path), compression="deflate", tiled=False)


def _write_float32_with_nan_nodata(path: Path) -> None:
    rng = np.random.default_rng(2342)
    arr = (rng.standard_normal((_FULL_H, _FULL_W)) * 100).astype(np.float32)
    # Drop a NaN inside the window region so masked_nodata is exercised.
    arr[40, 80] = np.nan
    da = _build_da(arr, nodata=np.float32("nan"))
    to_geotiff(da, str(path), compression="deflate", tiled=True,
               tile_size=64)


def _write_float32_no_nodata(path: Path) -> None:
    rng = np.random.default_rng(2343)
    arr = (rng.standard_normal((_FULL_H, _FULL_W)) * 50).astype(np.float32)
    da = _build_da(arr, nodata=None)
    to_geotiff(da, str(path), compression="none", tiled=False)


def _write_uint8_miniswhite(path: Path) -> None:
    """Write a MinIsWhite (photometric=0) uint8 stripped TIFF via tifffile.

    Matches the miniswhite cell in ``test_backend_pixel_parity_matrix_1813.py``
    so this release-gate row covers the same representative layout.
    """
    import tifffile  # local import: gated by ``_skip_no_tifffile``
    rng = np.random.default_rng(2344)
    arr = rng.integers(0, 256, size=(_FULL_H, _FULL_W), dtype=np.uint8)
    tifffile.imwrite(
        str(path), arr, photometric="miniswhite",
        compression="none", metadata=None,
    )


_CORPUS = (
    pytest.param(_write_int16_with_nodata, id="int16-deflate-stripped-nodata"),
    pytest.param(_write_float32_with_nan_nodata,
                 id="float32-deflate-tiled-nan-nodata"),
    pytest.param(_write_float32_no_nodata,
                 id="float32-none-stripped-no-nodata"),
    pytest.param(_write_uint8_miniswhite,
                 id="uint8-miniswhite-stripped",
                 marks=_skip_no_tifffile),
)


@pytest.fixture
def corpus_file(tmp_path, request):
    """Write a single fixture file and return its on-disk path.

    Each invocation uses a unique filename (issue tag + uuid + builder id)
    so sibling rockout worktrees and parallel test runs cannot collide on
    the same tmp file. The builder is parametrized at test-call time.
    """
    builder = request.param
    tag = uuid.uuid4().hex[:8]
    path = tmp_path / f"release_gate_2341_{builder.__name__[1:]}_{tag}.tif"
    builder(path)
    return path


# ---------------------------------------------------------------------------
# Algebraic spec for the windowed transform
# ---------------------------------------------------------------------------

def _expected_window_transform(t_full, col_off, row_off):
    """Return ``T_full * Affine.translation(col_off, row_off)`` by hand.

    The spec from epic #2341 is the algebraic identity, not whatever
    rasterio's window math happens to compute. For an axis-aligned
    transform ``(a, b, c, d, e, f)`` this simplifies to

        T_window = (a, b, c + a*col_off + b*row_off,
                    d, e, f + d*col_off + e*row_off)

    The b / d terms are kept in the algebra so a future rotated-grid
    fixture that exercises this same release gate inherits the right
    answer without the test needing to be rewritten.
    """
    a, b, c, d, e, f = (float(x) for x in t_full)
    return (
        a,
        b,
        c + a * col_off + b * row_off,
        d,
        e,
        f + d * col_off + e * row_off,
    )


# ---------------------------------------------------------------------------
# Backend matrix
# ---------------------------------------------------------------------------

def _open_eager(path, *, window=None):
    return open_geotiff(str(path), window=window)


def _open_dask(path, *, window=None):
    # ``chunks`` is required for ``read_geotiff_dask``; pick a value
    # that produces more than one chunk over the window so the chunk
    # math along the window origin is exercised.
    return read_geotiff_dask(str(path), window=window, chunks=32)


_READERS = (
    pytest.param(_open_eager, id="eager"),
    pytest.param(_open_dask, id="dask"),
)


# ---------------------------------------------------------------------------
# The release-gate assertions
# ---------------------------------------------------------------------------

def _assert_shape(out, *, expected_h, expected_w):
    assert out.shape == (expected_h, expected_w), (
        f"release gate: windowed read shape {out.shape} does not equal "
        f"the requested window shape {(expected_h, expected_w)}; a window "
        f"that returns the wrong shape is silently wrong, not noisily wrong"
    )


def _assert_coords_slice(windowed, full,
                         *, row_off, col_off, height, width):
    """The window's coords must be the bit-exact slice of the full coords.

    Computing the windowed coords from the shifted transform and
    comparing to the slice of the full coords pins both routes: any
    drift between ``transform`` shift and coord arithmetic shows up
    here, not at the next downstream call site.
    """
    full_y = np.asarray(full.coords["y"].values)
    full_x = np.asarray(full.coords["x"].values)
    win_y = np.asarray(windowed.coords["y"].values)
    win_x = np.asarray(windowed.coords["x"].values)
    np.testing.assert_array_equal(
        win_y,
        full_y[row_off:row_off + height],
        err_msg=(
            "release gate: windowed read y-coords are not a slice of the "
            "unwindowed read's y-coords; downstream callers that join on "
            "y will silently mismatch"
        ),
    )
    np.testing.assert_array_equal(
        win_x,
        full_x[col_off:col_off + width],
        err_msg=(
            "release gate: windowed read x-coords are not a slice of the "
            "unwindowed read's x-coords"
        ),
    )


def _assert_transform_shifted(windowed, full, *, col_off, row_off):
    """``attrs['transform']`` must equal ``T_full * translation(col, row)``.

    Asserted via exact tuple equality (not ``approx``) because the spec
    says no float drift: the contract is the algebraic identity, and
    any tolerance here would let a buggy windowed reader pass by
    re-deriving the origin from the y/x coord arrays (where floating
    rounding can creep in).

    When the source has no georef tags, ``transform`` is absent from
    both reads (see issue #1710 / ``_coords.py`` -- the reader drops the
    synthesised unit transform on non-georef sources). In that case the
    contract is that *neither* read has a transform; introducing one on
    the windowed side would be the silent-wrongness this gate exists
    to catch.
    """
    if "transform" not in full.attrs:
        assert "transform" not in windowed.attrs, (
            f"release gate: source has no georef and the unwindowed read "
            f"emits no ``transform``, but the windowed read invented one: "
            f"{windowed.attrs.get('transform')!r}"
        )
        return
    t_full = tuple(full.attrs["transform"])
    assert "transform" in windowed.attrs, (
        f"release gate: unwindowed read carries ``transform`` "
        f"({t_full!r}) but the windowed read dropped it"
    )
    t_win = tuple(windowed.attrs["transform"])
    assert len(t_full) == 6, (
        f"release gate: full-read transform is not a 6-tuple: {t_full!r}"
    )
    assert len(t_win) == 6, (
        f"release gate: windowed-read transform is not a 6-tuple: {t_win!r}"
    )
    expected = _expected_window_transform(t_full, col_off, row_off)
    assert t_win == expected, (
        f"release gate: windowed transform does not equal "
        f"T_full * Affine.translation(col_off={col_off}, row_off={row_off})\n"
        f"  T_full   = {t_full!r}\n"
        f"  T_window = {t_win!r}\n"
        f"  expected = {expected!r}"
    )


def _assert_canonical_attrs_unchanged(windowed, full):
    """The non-transform canonical attrs must match the unwindowed read.

    A window slices pixels; it does not redefine CRS, nodata sentinel,
    georef status, or raster type. Drift on any of these is the exact
    silent-wrongness the epic calls out. ``masked_nodata`` is checked
    structurally because the flag is by-design data-dependent.
    """
    for key in _NON_TRANSFORM_ATTRS_VALUE_EQUAL:
        if key not in full.attrs:
            assert key not in windowed.attrs, (
                f"release gate: windowed read introduced attrs[{key!r}] "
                f"that the unwindowed read does not have"
            )
            continue
        assert key in windowed.attrs, (
            f"release gate: windowed read dropped attrs[{key!r}] that the "
            f"unwindowed read carries"
        )
        full_val = full.attrs[key]
        win_val = windowed.attrs[key]
        # NaN-aware compare. Try ``np.isnan`` on a scalar before
        # falling back to ``==``: a Python float NaN and a
        # ``np.float32`` NaN both report True under ``np.isnan`` but
        # only the Python float passes ``isinstance(x, float)``, and
        # we don't want a future backend that returns a numpy-scalar
        # nodata to silently slip into the ``==`` branch where
        # NaN != NaN flips the test from "checked equal" to "always
        # fails".
        try:
            full_is_nan = bool(np.isnan(full_val))
        except (TypeError, ValueError):
            full_is_nan = False
        if full_is_nan:
            try:
                win_is_nan = bool(np.isnan(win_val))
            except (TypeError, ValueError):
                win_is_nan = False
            assert win_is_nan, (
                f"release gate: NaN-valued attrs[{key!r}] did not survive "
                f"the windowed read: full={full_val!r} window={win_val!r}"
            )
        else:
            assert win_val == full_val, (
                f"release gate: windowed read changed attrs[{key!r}]: "
                f"full={full_val!r} window={win_val!r}"
            )
    for key in _NON_TRANSFORM_ATTRS_STRUCTURAL:
        full_present = key in full.attrs
        win_present = key in windowed.attrs
        assert full_present == win_present, (
            f"release gate: windowed read changed presence of "
            f"attrs[{key!r}]: full_has={full_present} "
            f"window_has={win_present}"
        )
        if full_present:
            assert isinstance(windowed.attrs[key], bool), (
                f"release gate: attrs[{key!r}] is not a bool on the "
                f"windowed read: {windowed.attrs[key]!r}"
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.release_gate
@pytest.mark.parametrize("window", _WINDOWS)
@pytest.mark.parametrize("corpus_file", _CORPUS, indirect=True)
@pytest.mark.parametrize("reader", _READERS)
def test_release_gate_windowed_read_shape(corpus_file, reader, window):
    """The returned shape equals the window's ``(height, width)``."""
    row_off, col_off, row_stop, col_stop = window
    out = reader(corpus_file, window=window)
    _assert_shape(out,
                  expected_h=row_stop - row_off,
                  expected_w=col_stop - col_off)


@pytest.mark.release_gate
@pytest.mark.parametrize("window", _WINDOWS)
@pytest.mark.parametrize("corpus_file", _CORPUS, indirect=True)
@pytest.mark.parametrize("reader", _READERS)
def test_release_gate_windowed_read_coords_slice(corpus_file, reader, window):
    """``coords['y'/'x']`` equals the matching slice of the full coords."""
    row_off, col_off, row_stop, col_stop = window
    full = reader(corpus_file)
    out = reader(corpus_file, window=window)
    _assert_coords_slice(
        out, full,
        row_off=row_off, col_off=col_off,
        height=row_stop - row_off, width=col_stop - col_off,
    )


@pytest.mark.release_gate
@pytest.mark.parametrize("window", _WINDOWS)
@pytest.mark.parametrize("corpus_file", _CORPUS, indirect=True)
@pytest.mark.parametrize("reader", _READERS)
def test_release_gate_windowed_read_transform_shifted(
    corpus_file, reader, window,
):
    """``attrs['transform']`` equals ``T_full * translation(col, row)``."""
    row_off, col_off, _row_stop, _col_stop = window
    full = reader(corpus_file)
    out = reader(corpus_file, window=window)
    _assert_transform_shifted(out, full, col_off=col_off, row_off=row_off)


@pytest.mark.release_gate
@pytest.mark.parametrize("window", _WINDOWS)
@pytest.mark.parametrize("corpus_file", _CORPUS, indirect=True)
@pytest.mark.parametrize("reader", _READERS)
def test_release_gate_windowed_read_canonical_attrs_unchanged(
    corpus_file, reader, window,
):
    """The non-transform canonical attrs match the unwindowed read."""
    full = reader(corpus_file)
    out = reader(corpus_file, window=window)
    _assert_canonical_attrs_unchanged(out, full)
