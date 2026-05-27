"""Strict pixel-equality and kwarg-threading parity across read backends.

The strictest mode of backend parity. Four concerns share the file
because they fail in the same ways:

* Pixel-byte parity across (numpy / dask+numpy / cupy / dask+cupy) on a
  representative dtype + compression + layout matrix, plus VRT, COG,
  BigTIFF, and MinIsWhite fixtures.
* Cross-entry-point parity: ``read_geotiff_dask``, ``read_geotiff_gpu``,
  and ``read_vrt`` agree with ``open_geotiff`` for the same source.
* Kwarg threading through the dispatcher: ``open_geotiff`` and
  ``to_geotiff`` forward window / band / max_pixels / tiled / etc. to
  the backend-specific entry points instead of silently dropping them.
* MinIsWhite photometric handling: local / dask / HTTP / GPU read
  paths agree on the inverted pixel domain.

If a refactor moves an entry-point body to a new module, this file is
the first thing that breaks when the move drops a kwarg, inverts a
photometric, or reorders an attrs-population step.
"""
from __future__ import annotations

import http.server
import importlib.util
import socketserver
import threading
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (open_geotiff, read_geotiff_dask, read_geotiff_gpu, read_vrt,
                               to_geotiff, write_vrt)

from .._helpers.markers import gpu_available, requires_gpu, requires_loopback

# ---------------------------------------------------------------------------
# Environment gating
# ---------------------------------------------------------------------------

_HAS_GPU = gpu_available()
_HAS_TIFFFILE = importlib.util.find_spec("tifffile") is not None

# Use the shared marker from ``_helpers/markers.py`` for the GPU gate.
_skip_no_gpu = requires_gpu
_skip_no_tifffile = pytest.mark.skipif(
    not _HAS_TIFFFILE, reason="tifffile required for MinIsWhite fixture")


_BACKENDS = [
    pytest.param({}, id="numpy"),
    pytest.param({"chunks": 32}, id="dask+numpy"),
    pytest.param({"gpu": True}, id="cupy", marks=_skip_no_gpu),
    pytest.param({"gpu": True, "chunks": 32}, id="dask+cupy", marks=_skip_no_gpu),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _materialise(da: xr.DataArray) -> np.ndarray:
    """Return a numpy view of the data regardless of backend."""
    raw = da.data
    if hasattr(raw, "compute"):
        raw = raw.compute()
    if hasattr(raw, "get"):
        raw = raw.get()
    return np.asarray(raw)


def _comparable_attrs(attrs: dict) -> dict:
    """Strip attr keys that vary between reads of the same file by design.

    ``source`` carries the input path or a path-like; round-trip writers
    leave it under different keys across backends. Drop it before
    comparison. Same for ``_band_summary`` which is a debug aid.
    """
    skip = {"source", "_band_summary"}
    return {k: v for k, v in attrs.items() if k not in skip}


def _coord_view(da: xr.DataArray, name: str) -> np.ndarray:
    return np.asarray(da.coords[name].values)


def _make_reference_data(dtype: np.dtype, height: int = 64, width: int = 64,
                         seed: int = 1813) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    if dtype.kind == "f":
        return (rng.standard_normal((height, width)) * 100).astype(dtype)
    if dtype.kind == "u":
        info = np.iinfo(dtype)
        return rng.integers(0, info.max, size=(height, width), dtype=dtype)
    if dtype.kind == "i":
        info = np.iinfo(dtype)
        return rng.integers(info.min, info.max, size=(height, width), dtype=dtype)
    raise NotImplementedError(f"unsupported dtype: {dtype}")


def _wrap(arr: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        arr, dims=["y", "x"],
        coords={
            "y": np.arange(arr.shape[0], dtype=np.float64),
            "x": np.arange(arr.shape[1], dtype=np.float64),
        },
        attrs={"crs": 4326},
    )


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _write_stripped(path: Path, dtype: np.dtype, compression: str) -> None:
    to_geotiff(_wrap(_make_reference_data(dtype)), str(path),
               compression=compression, tiled=False)


def _write_tiled(path: Path, dtype: np.dtype, compression: str) -> None:
    to_geotiff(_wrap(_make_reference_data(dtype)), str(path),
               compression=compression, tiled=True, tile_size=32)


def _write_cog(path: Path) -> None:
    to_geotiff(_wrap(_make_reference_data(np.dtype("float32"))), str(path),
               compression="deflate", cog=True, tile_size=16,
               overview_levels=[2, 4])


def _write_bigtiff(path: Path) -> None:
    to_geotiff(_wrap(_make_reference_data(np.dtype("float32"))), str(path),
               compression="deflate", bigtiff=True)


def _write_miniswhite(path: Path) -> None:
    """Write a MinIsWhite (photometric=0) uint8 stripped TIFF via tifffile."""
    import tifffile  # local import: only the miniswhite cell needs this
    arr = _make_reference_data(np.dtype("uint8")).astype(np.uint8)
    tifffile.imwrite(
        str(path), arr, photometric="miniswhite",
        compression="none", metadata=None,
    )


def _write_vrt_mosaic(dir_path: Path) -> Path:
    """Write a 2x2 mosaic of float32 stripped tiles plus a VRT pointing to them."""
    tile_paths: list[str] = []
    tile_h = tile_w = 32
    for r in range(2):
        for c in range(2):
            arr = np.full((tile_h, tile_w),
                          float(r * 2 + c + 1), dtype=np.float32)
            origin_x = float(c * tile_w)
            origin_y = -float(r * tile_h)
            da = xr.DataArray(
                arr, dims=["y", "x"],
                coords={
                    "y": np.arange(origin_y, origin_y - tile_h, -1, dtype=np.float64),
                    "x": np.arange(origin_x, origin_x + tile_w, dtype=np.float64),
                },
                attrs={"crs": 4326},
            )
            p = dir_path / f"vrt_tile_{r}_{c}_1813.tif"
            to_geotiff(da, str(p), compression="none", tiled=False)
            tile_paths.append(str(p))
    vrt_path = dir_path / "mosaic_1813.vrt"
    write_vrt(str(vrt_path), tile_paths, relative=False, crs=4326)
    return vrt_path


@pytest.fixture(scope="session")
def _parity_fixture_dir(tmp_path_factory):
    """Session-scoped directory holding every parity fixture file.

    Sharing one dir + caching by id keeps fixture build cost flat at one
    write per fixture instead of one write per test cell (192 in this
    matrix), which was a real IO hit on Windows runners.
    """
    return tmp_path_factory.mktemp("parity_1813")


@pytest.fixture
def fixture_factory(_parity_fixture_dir):
    """Return a builder that resolves a fix_id to its on-disk path.

    Files are cached across the test session: a fixture already present
    on disk is returned without rewriting.
    """
    dir_path = _parity_fixture_dir

    def _build(fix_id: str) -> Path:
        if fix_id == "vrt":
            vrt_path = dir_path / "mosaic_1813.vrt"
            if vrt_path.exists():
                return vrt_path
            return _write_vrt_mosaic(dir_path)

        safe_id = fix_id.replace("/", "-")
        path = dir_path / f"parity_{safe_id}_1813.tif"
        if path.exists():
            return path
        if fix_id.startswith("stripped/"):
            _, dtype_name, comp = fix_id.split("/")
            _write_stripped(path, np.dtype(dtype_name), comp)
        elif fix_id.startswith("tiled/"):
            _, dtype_name, comp = fix_id.split("/")
            _write_tiled(path, np.dtype(dtype_name), comp)
        elif fix_id == "cog":
            _write_cog(path)
        elif fix_id == "bigtiff":
            _write_bigtiff(path)
        elif fix_id == "miniswhite":
            _write_miniswhite(path)
        else:
            raise ValueError(f"unknown fixture id: {fix_id}")
        return path
    return _build


# Representative matrix: five dtypes, three compressions, both layouts, plus
# COG/BigTIFF/MinIsWhite/VRT. Kept compact to keep CI cost bounded; entry-point
# parity (numpy ↔ dask ↔ gpu) is the same regardless of dtype, so a few
# representatives per axis catches drift. The miniswhite cell needs tifffile;
# it skips cleanly when tifffile is unavailable instead of taking down the
# whole matrix.
_TIFF_FIXTURES = [
    pytest.param("stripped/uint8/none", id="stripped-uint8-none"),
    pytest.param("stripped/uint16/deflate", id="stripped-uint16-deflate"),
    pytest.param("stripped/int16/lzw", id="stripped-int16-lzw"),
    pytest.param("stripped/float32/deflate", id="stripped-float32-deflate"),
    pytest.param("stripped/float64/none", id="stripped-float64-none"),
    pytest.param("tiled/uint8/deflate", id="tiled-uint8-deflate"),
    pytest.param("tiled/uint16/lzw", id="tiled-uint16-lzw"),
    pytest.param("tiled/float32/none", id="tiled-float32-none"),
    pytest.param("tiled/float64/deflate", id="tiled-float64-deflate"),
    pytest.param("cog", id="cog-float32-deflate"),
    pytest.param("bigtiff", id="bigtiff-float32-deflate"),
    pytest.param("miniswhite", id="miniswhite-uint8-none",
                 marks=_skip_no_tifffile),
]


# ---------------------------------------------------------------------------
# Pixel / coord / attrs parity across backends via open_geotiff
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fix_id", _TIFF_FIXTURES)
@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
def test_open_geotiff_pixel_bytes_match(fixture_factory, fix_id, backend_kwargs):
    """Pixels are byte-identical across backends for the same file."""
    path = fixture_factory(fix_id)
    ref_arr = _materialise(open_geotiff(str(path)))
    actual_arr = _materialise(open_geotiff(str(path), **backend_kwargs))

    assert ref_arr.tobytes() == actual_arr.tobytes(), (
        f"fixture={fix_id} backend={backend_kwargs}: pixel bytes differ "
        f"(ref_dtype={ref_arr.dtype}, actual_dtype={actual_arr.dtype})"
    )


@pytest.mark.parametrize("fix_id", _TIFF_FIXTURES)
@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
def test_open_geotiff_coords_match(fixture_factory, fix_id, backend_kwargs):
    """y/x coord arrays are bitwise-equal across backends."""
    path = fixture_factory(fix_id)
    ref = open_geotiff(str(path))
    actual = open_geotiff(str(path), **backend_kwargs)
    for axis in ("y", "x"):
        ref_c = _coord_view(ref, axis)
        actual_c = _coord_view(actual, axis)
        assert ref_c.dtype == actual_c.dtype, (
            f"fixture={fix_id} backend={backend_kwargs}: {axis} coord dtype "
            f"differs: ref={ref_c.dtype} actual={actual_c.dtype}"
        )
        assert ref_c.tobytes() == actual_c.tobytes(), (
            f"fixture={fix_id} backend={backend_kwargs}: {axis} coord bytes differ"
        )


@pytest.mark.parametrize("fix_id", _TIFF_FIXTURES)
@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
def test_open_geotiff_attrs_match(fixture_factory, fix_id, backend_kwargs):
    """attrs equal across backends after stripping transient keys."""
    path = fixture_factory(fix_id)
    ref = open_geotiff(str(path))
    actual = open_geotiff(str(path), **backend_kwargs)

    ref_attrs = _comparable_attrs(ref.attrs)
    actual_attrs = _comparable_attrs(actual.attrs)

    assert set(ref_attrs.keys()) == set(actual_attrs.keys()), (
        f"fixture={fix_id} backend={backend_kwargs}: attrs key set differs.\n"
        f"  only-in-ref:    {sorted(set(ref_attrs) - set(actual_attrs))}\n"
        f"  only-in-actual: {sorted(set(actual_attrs) - set(ref_attrs))}"
    )
    for key in ref_attrs:
        ref_v = ref_attrs[key]
        actual_v = actual_attrs[key]
        if isinstance(ref_v, np.ndarray):
            assert isinstance(actual_v, np.ndarray) and \
                np.array_equal(ref_v, actual_v), (
                    f"fixture={fix_id} backend={backend_kwargs}: "
                    f"attrs[{key!r}] arrays differ"
                )
        else:
            assert ref_v == actual_v, (
                f"fixture={fix_id} backend={backend_kwargs}: "
                f"attrs[{key!r}]: ref={ref_v!r} actual={actual_v!r}"
            )


# ---------------------------------------------------------------------------
# Cross-entry-point parity: open_geotiff vs the direct backend functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fix_id", _TIFF_FIXTURES)
def test_read_geotiff_dask_matches_open_geotiff(fixture_factory, fix_id):
    """``read_geotiff_dask(p)`` byte-matches ``open_geotiff(p, chunks=N)``."""
    path = fixture_factory(fix_id)
    via_open = open_geotiff(str(path), chunks=32)
    via_direct = read_geotiff_dask(str(path), chunks=32)
    a = _materialise(via_open).tobytes()
    b = _materialise(via_direct).tobytes()
    assert a == b, (
        f"fixture={fix_id}: read_geotiff_dask diverges from "
        f"open_geotiff(chunks=32)"
    )


@_skip_no_gpu
@pytest.mark.parametrize("fix_id", _TIFF_FIXTURES)
def test_read_geotiff_gpu_matches_open_geotiff(fixture_factory, fix_id):
    """``read_geotiff_gpu(p)`` byte-matches ``open_geotiff(p, gpu=True)``."""
    path = fixture_factory(fix_id)
    via_open = open_geotiff(str(path), gpu=True)
    via_direct = read_geotiff_gpu(str(path))
    a = _materialise(via_open).tobytes()
    b = _materialise(via_direct).tobytes()
    assert a == b, (
        f"fixture={fix_id}: read_geotiff_gpu diverges from "
        f"open_geotiff(gpu=True)"
    )


# ---------------------------------------------------------------------------
# VRT cross-backend and cross-entry-point parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
def test_read_vrt_pixel_bytes_match(fixture_factory, backend_kwargs):
    path = fixture_factory("vrt")
    ref = read_vrt(str(path))
    actual = read_vrt(str(path), **backend_kwargs)
    assert _materialise(ref).tobytes() == _materialise(actual).tobytes(), (
        f"read_vrt backend={backend_kwargs}: pixel bytes differ"
    )


@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
def test_read_vrt_coords_match(fixture_factory, backend_kwargs):
    path = fixture_factory("vrt")
    ref = read_vrt(str(path))
    actual = read_vrt(str(path), **backend_kwargs)
    for axis in ("y", "x"):
        ref_c = _coord_view(ref, axis)
        actual_c = _coord_view(actual, axis)
        assert ref_c.dtype == actual_c.dtype, (
            f"read_vrt backend={backend_kwargs}: {axis} coord dtype "
            f"differs: ref={ref_c.dtype} actual={actual_c.dtype}"
        )
        assert ref_c.tobytes() == actual_c.tobytes(), (
            f"read_vrt backend={backend_kwargs}: {axis} coord bytes differ"
        )


@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
def test_open_geotiff_dot_vrt_routes_to_read_vrt(fixture_factory, backend_kwargs):
    """``open_geotiff(path.vrt)`` byte-matches ``read_vrt(path)``."""
    path = fixture_factory("vrt")
    via_open = open_geotiff(str(path), **backend_kwargs)
    via_direct = read_vrt(str(path), **backend_kwargs)
    assert _materialise(via_open).tobytes() == _materialise(via_direct).tobytes(), (
        f"open_geotiff(.vrt) diverges from read_vrt: backend={backend_kwargs}"
    )


# ---------------------------------------------------------------------------
# Sanity check: the fixture builders produce readable files at all
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fix_id", _TIFF_FIXTURES)
def test_fixture_builders_produce_readable_files(fixture_factory, fix_id):
    path = fixture_factory(fix_id)
    da = open_geotiff(str(path))
    assert da.ndim in (2, 3)
    assert da.size > 0


# ===========================================================================
# Kwarg threading through the dispatcher (originally test_backend_kwarg_parity)
# ===========================================================================
#
# ``open_geotiff`` and ``to_geotiff`` route to backend-specific entry
# points (``read_geotiff_dask``, ``write_geotiff_gpu``) whose kwarg sets
# were narrower than the dispatcher's. The dispatcher silently dropped
# the missing kwargs when it routed to the smaller-API backend; the
# fix pins them through. These tests gate that contract.


@pytest.fixture
def small_tiff_path(tmp_path):
    """4x6 single-band tiled tiff with a small CRS+transform."""
    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        coords={
            'y': np.array([0.5, 1.5, 2.5, 3.5]),
            'x': np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'kwarg_parity_small.tif'
    to_geotiff(da, str(p), tile_size=16)
    return str(p), arr


@pytest.fixture
def small_multiband_tiff_path(tmp_path):
    """4x6 three-band tiled tiff."""
    arr = np.arange(72, dtype=np.float32).reshape(4, 6, 3)
    da = xr.DataArray(
        arr,
        dims=['y', 'x', 'band'],
        coords={
            'y': np.array([0.5, 1.5, 2.5, 3.5]),
            'x': np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
            'band': [0, 1, 2],
        },
        attrs={'crs': 4326},
    )
    p = tmp_path / 'kwarg_parity_mb.tif'
    to_geotiff(da, str(p), tile_size=16)
    return str(p), arr


def test_read_geotiff_dask_window_clips_region(small_tiff_path):
    """``window=`` restricts the lazy region; chunks span only the window."""
    path, arr = small_tiff_path
    da = read_geotiff_dask(path, chunks=2, window=(1, 2, 4, 6))
    assert da.shape == (3, 4)
    np.testing.assert_array_equal(da.values, arr[1:4, 2:6])


def test_read_geotiff_dask_window_via_dispatcher(small_tiff_path):
    """``open_geotiff(window=..., chunks=...)`` keeps the window."""
    path, arr = small_tiff_path
    da = open_geotiff(path, window=(0, 1, 3, 4), chunks=2)
    assert da.shape == (3, 3)
    np.testing.assert_array_equal(da.values, arr[0:3, 1:4])


def test_read_geotiff_dask_band_selects_single_band(small_multiband_tiff_path):
    """``band=`` produces a 2D DataArray with the selected band."""
    path, arr = small_multiband_tiff_path
    da = read_geotiff_dask(path, chunks=4, band=1)
    assert da.ndim == 2
    np.testing.assert_array_equal(da.values, arr[:, :, 1])


def test_read_geotiff_dask_band_via_dispatcher(small_multiband_tiff_path):
    """``open_geotiff(band=..., chunks=...)`` keeps the band."""
    path, arr = small_multiband_tiff_path
    da = open_geotiff(path, band=2, chunks=4)
    assert da.ndim == 2
    np.testing.assert_array_equal(da.values, arr[:, :, 2])


def test_read_geotiff_dask_max_pixels_bounds_chunk_not_full_image(
        small_tiff_path):
    """``max_pixels`` bounds each chunk, not the full lazy region (#2501).

    The 4x6 fixture is 24 pixels in total. With ``chunks=2`` each chunk
    is at most 2x2 = 4 pixels, so ``max_pixels=10`` permits the read
    even though the full image exceeds the cap. With ``chunks=4`` a
    chunk reaches 4x4 = 16 pixels, which trips the per-chunk guard at
    ``.compute()`` time.
    """
    path, arr = small_tiff_path

    # Per-chunk cap is satisfied; full image is not. Should succeed.
    da = read_geotiff_dask(path, chunks=2, max_pixels=10)
    np.testing.assert_array_equal(da.values, arr)

    # Per-chunk cap is exceeded. The graph builds, but the chunk task
    # raises the safety-limit error on compute.
    da = read_geotiff_dask(path, chunks=4, max_pixels=10)
    with pytest.raises(ValueError, match="exceed the safety limit"):
        da.compute()


def test_read_geotiff_dask_max_pixels_chunk_includes_band_count(
        small_multiband_tiff_path):
    """Multi-band: per-chunk cap is ``chunk_h * chunk_w * samples`` (#2501).

    The 4x6x3 fixture is 72 pixels. With ``chunks=2`` each chunk
    materialises 2x2x3 = 12 pixels, fitting under ``max_pixels=20``
    even though the full image (72) does not. With ``chunks=4`` the
    largest chunk is 4x4x3 = 48 pixels and the per-chunk decode trips
    the cap on compute.
    """
    path, arr = small_multiband_tiff_path

    # Per-chunk cap satisfied (12 px <= 20) but full image (72) is not.
    da = read_geotiff_dask(path, chunks=2, max_pixels=20)
    np.testing.assert_array_equal(da.values, arr)

    # Per-chunk cap exceeded (48 px > 20). Graph builds, compute raises.
    da = read_geotiff_dask(path, chunks=4, max_pixels=20)
    with pytest.raises(ValueError, match="exceed the safety limit"):
        da.compute()


def test_read_geotiff_dask_window_band_combined(small_multiband_tiff_path):
    """``window`` and ``band`` cooperate."""
    path, arr = small_multiband_tiff_path
    da = read_geotiff_dask(path, chunks=2, window=(1, 1, 4, 5), band=0)
    assert da.shape == (3, 4)
    np.testing.assert_array_equal(da.values, arr[1:4, 1:5, 0])


def test_read_geotiff_dask_invalid_window_raises(small_tiff_path):
    """Out-of-bounds windows fail loudly instead of silently clipping."""
    path, _ = small_tiff_path
    with pytest.raises(ValueError, match="window=.* is outside"):
        read_geotiff_dask(path, chunks=2, window=(0, 0, 100, 100))


def test_read_geotiff_dask_invalid_band_raises(small_multiband_tiff_path):
    """Out-of-range band indexes fail with IndexError."""
    path, _ = small_multiband_tiff_path
    with pytest.raises(IndexError, match="band=5 out of range"):
        read_geotiff_dask(path, chunks=4, band=5)


def test_write_geotiff_gpu_rejects_tiled_false(tmp_path):
    """The GPU writer is tiled-only; ``tiled=False`` must fail loudly."""
    from xrspatial.geotiff import write_geotiff_gpu

    dummy = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="tiled=True"):
        write_geotiff_gpu(dummy, str(tmp_path / 'never.tif'), tiled=False)


def test_write_geotiff_gpu_rejects_nonzero_max_z_error(tmp_path):
    """LERC budget is not implementable on the GPU path."""
    from xrspatial.geotiff import write_geotiff_gpu

    dummy = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="max_z_error is not supported"):
        write_geotiff_gpu(dummy, str(tmp_path / 'never.tif'), max_z_error=1.0)


@_skip_no_gpu
def test_write_geotiff_gpu_accepts_streaming_buffer_bytes_as_noop(tmp_path):
    """``streaming_buffer_bytes`` is accepted for API parity (no-op)."""
    import cupy

    from xrspatial.geotiff import write_geotiff_gpu

    arr = cupy.arange(16, dtype=cupy.float32).reshape(4, 4)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': np.arange(4, dtype=np.float64),
                              'x': np.arange(4, dtype=np.float64)})
    p = tmp_path / 'kwarg_parity_streaming.tif'
    write_geotiff_gpu(da, str(p), streaming_buffer_bytes=4096, tile_size=16)
    rd = open_geotiff(str(p))
    np.testing.assert_array_equal(rd.values, arr.get())


@_skip_no_gpu
def test_to_geotiff_threads_tiled_false_into_gpu_dispatcher(tmp_path):
    """``to_geotiff(..., gpu=True, tiled=False)`` rejects, no silent flip."""
    import cupy

    arr = cupy.zeros((2, 2), dtype=cupy.float32)
    da = xr.DataArray(arr, dims=['y', 'x'],
                      coords={'y': [0.0, 1.0], 'x': [0.0, 1.0]})
    with pytest.raises(ValueError, match="tiled=False"):
        to_geotiff(da, str(tmp_path / 'never.tif'),
                   gpu=True, tiled=False)


# ===========================================================================
# MinIsWhite photometric backend parity (originally test_miniswhite_backend_parity)
# ===========================================================================
#
# MinIsWhite (photometric=0) inverts pixel intensity at decode time.
# The local, dask, HTTP, and GPU read paths must all agree on the
# inverted pixel domain.


class _MwRangeHandler(http.server.BaseHTTPRequestHandler):
    payload: bytes = b''

    def do_GET(self):  # noqa: N802
        rng = self.headers.get('Range')
        if rng and rng.startswith('bytes='):
            spec = rng[len('bytes='):]
            start_s, _, end_s = spec.partition('-')
            start = int(start_s)
            end = int(end_s) if end_s else len(self.payload) - 1
            chunk = self.payload[start:end + 1]
            self.send_response(206)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header(
                'Content-Range',
                f'bytes {start}-{start + len(chunk) - 1}/{len(self.payload)}',
            )
            self.send_header('Content-Length', str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_args, **_kwargs):
        pass


def _mw_serve(payload: bytes):
    handler_cls = type(
        'MwRangeHandler', (_MwRangeHandler,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


@pytest.fixture
def miniswhite_http_url(tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    stored = np.array([[0, 1, 2], [10, 128, 255]], dtype=np.uint8)
    path = tmp_path / "tmp_miniswhite.tif"
    tifffile.imwrite(str(path), stored, photometric='miniswhite')
    httpd, port = _mw_serve(path.read_bytes())
    try:
        yield f'http://127.0.0.1:{port}/tmp_miniswhite.tif', stored
    finally:
        httpd.shutdown()
        httpd.server_close()


@requires_loopback
def test_miniswhite_http_matches_local_reader(miniswhite_http_url):
    """HTTP read of a MinIsWhite TIFF returns the inverted pixel domain."""
    url, stored = miniswhite_http_url
    got = open_geotiff(url)
    np.testing.assert_array_equal(got.values, np.iinfo(stored.dtype).max - stored)


@requires_loopback
def test_miniswhite_http_dask_matches_local_reader(miniswhite_http_url):
    """Dask HTTP read agrees with the eager HTTP read on inversion."""
    url, stored = miniswhite_http_url
    got = open_geotiff(url, chunks=2).compute()
    np.testing.assert_array_equal(got.values, np.iinfo(stored.dtype).max - stored)


@_skip_no_gpu
def test_miniswhite_gpu_matches_cpu_reader(tmp_path):
    """GPU MinIsWhite read agrees with the CPU reader.

    The writer pre-inverts MinIsWhite pixels so the reader's
    unconditional inversion restores the user-domain values; the
    round-trip is the identity for both backends.
    """
    from xrspatial.geotiff._writer import write

    stored = np.array([[0, 1, 2], [10, 128, 255]], dtype=np.uint8)
    path = str(tmp_path / "tmp_miniswhite_gpu.tif")
    write(stored, path, compression='deflate', tiled=True, tile_size=16,
          photometric='miniswhite')

    cpu = open_geotiff(path)
    gpu = open_geotiff(path, gpu=True)

    np.testing.assert_array_equal(cpu.values, stored)
    np.testing.assert_array_equal(gpu.data.get(), cpu.values)
