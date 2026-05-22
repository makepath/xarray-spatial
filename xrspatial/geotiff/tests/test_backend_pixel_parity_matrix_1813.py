"""End-to-end pixel-byte-parity matrix across read backends and entry points.

Locks in the no-regression contract for issue #1813's multi-PR refactor of
``xrspatial/geotiff/__init__.py``. Every read entry point (``open_geotiff``,
``read_geotiff_dask``, ``read_geotiff_gpu``, ``read_vrt``) must produce
byte-identical pixels, bitwise-equal coords, and matching ``attrs`` across
the four backends (numpy, dask+numpy, cupy, dask+cupy) for a representative
matrix of dtypes, compressions, and layouts (stripped, tiled, COG,
BigTIFF, MinIsWhite, VRT).

When a subsequent PR in #1813 moves an entry-point body to a new module,
this matrix is the first thing that breaks if the move drops a kwarg,
inverts a photometric, or reorders an attrs-population step.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (open_geotiff, read_geotiff_dask, read_geotiff_gpu, read_vrt,
                               to_geotiff, write_vrt)

# ---------------------------------------------------------------------------
# Environment gating
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_HAS_TIFFFILE = importlib.util.find_spec("tifffile") is not None

_skip_no_gpu = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")
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
