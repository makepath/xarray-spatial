"""Tests for ``PlanarConfiguration`` handling in CPU and GPU multi-band reads.

Covers two bugs surfaced by the geotiff accuracy audit:

A2 (HIGH)
    GPU tiled multi-band reads returned wrong pixel values because the
    tile-assembly kernel iterated a single ``tile_offsets`` list with
    ``bytes_per_pixel = itemsize * samples``. Planar=2 files (one tile
    sequence per band, contiguous in TileOffsets) silently produced
    garbage because the kernel had no notion of separate band slabs.

A3 (MEDIUM)
    The CPU stripped reader returned planar=2 multi-band shape with
    axes mismatched against the assigned dims. Already fixed in the CPU
    path; this module pins the contract so future refactors do not
    regress it.

Each combination -- planar config x layout x band count x dtype --
round-trips through the writer (or tifffile, where we want to control
planar config and layout independently) and asserts that the CPU read,
the GPU read, and the original numpy buffer all agree pixel-for-pixel.

GPU tests skip cleanly when CUDA is unavailable.
"""
from __future__ import annotations

import importlib.util
import os
import tempfile

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")


def _gpu_available() -> bool:
    """True if cupy is importable and CUDA is initialised."""
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy

        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU,
    reason="cupy + CUDA required",
)


def _make_data(bands: int, height: int, width: int, dtype) -> np.ndarray:
    """Deterministic random multi-band raster shaped (bands, height, width)."""
    rng = np.random.RandomState(0xA2A3 + bands * 1000 + height + np.dtype(dtype).itemsize)
    info = np.iinfo(dtype)
    high = min(int(info.max), 60_000) + 1
    return rng.randint(0, high, size=(bands, height, width)).astype(dtype)


def _write_tiff(path: str, data: np.ndarray, *, planar: str, tiled: bool):
    """Write *data* (bands, height, width) with the requested planar config + layout.

    tifffile defaults: when bands are first axis, planar='separate'.
    Pass planar='contig' to interleave samples (data is transposed to
    (height, width, bands) before write).
    """
    kwargs = {"photometric": "rgb" if data.shape[0] == 3 else "minisblack"}
    if data.shape[0] not in (1, 3):
        # tifffile rejects 'rgb' for non-3-band; use a generic photometric
        # tag and tell it to skip extra-sample warnings.
        kwargs = {"photometric": "minisblack"}
    if tiled:
        kwargs["tile"] = (32, 32)
    if planar == "separate":
        kwargs["planarconfig"] = "separate"
        tifffile.imwrite(path, data, **kwargs)
    elif planar == "contig":
        kwargs["planarconfig"] = "contig"
        tifffile.imwrite(path, np.transpose(data, (1, 2, 0)), **kwargs)
    else:
        raise ValueError(f"unknown planar={planar!r}")


@pytest.mark.parametrize("planar", ["separate", "contig"])
@pytest.mark.parametrize("tiled", [True, False])
@pytest.mark.parametrize("bands", [2, 3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_cpu_planar_multiband(planar, tiled, bands, dtype):
    """CPU ``open_geotiff`` returns (y, x, band) for every planar+layout combo."""
    from xrspatial.geotiff import open_geotiff

    height, width = 64, 96
    data = _make_data(bands, height, width, dtype)
    expected = np.transpose(data, (1, 2, 0))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.tif")
        _write_tiff(path, data, planar=planar, tiled=tiled)
        out = open_geotiff(path)

    assert out.dims == ("y", "x", "band"), (
        f"got dims {out.dims} for planar={planar} tiled={tiled} "
        f"bands={bands} dtype={np.dtype(dtype).name}"
    )
    assert out.shape == (height, width, bands)
    arr = np.asarray(out.data)
    assert arr.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(arr, expected)


@_gpu_only
@pytest.mark.parametrize("planar", ["separate", "contig"])
@pytest.mark.parametrize("tiled", [True, False])
@pytest.mark.parametrize("bands", [2, 3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_gpu_matches_cpu_planar_multiband(planar, tiled, bands, dtype):
    """GPU read agrees with CPU read AND with the source array."""
    from xrspatial.geotiff import open_geotiff, read_geotiff_gpu

    height, width = 64, 96
    data = _make_data(bands, height, width, dtype)
    expected = np.transpose(data, (1, 2, 0))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.tif")
        _write_tiff(path, data, planar=planar, tiled=tiled)
        cpu = np.asarray(open_geotiff(path).data)
        gpu_da = read_geotiff_gpu(path)

    assert gpu_da.dims == ("y", "x", "band")
    assert gpu_da.shape == (height, width, bands)
    gpu = gpu_da.data.get()

    np.testing.assert_array_equal(cpu, expected)
    np.testing.assert_array_equal(gpu, expected)
    np.testing.assert_array_equal(gpu, cpu)


@pytest.mark.parametrize("tiled", [True, False])
def test_cpu_singleband_sanity(tiled):
    """Single-band reads stay 2-D regardless of layout."""
    from xrspatial.geotiff import open_geotiff

    rng = np.random.RandomState(7)
    data = rng.randint(0, 200, size=(48, 80)).astype(np.uint8)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.tif")
        kwargs = {"photometric": "minisblack"}
        if tiled:
            kwargs["tile"] = (32, 32)
        tifffile.imwrite(path, data, **kwargs)
        out = open_geotiff(path)
    assert out.dims == ("y", "x")
    assert out.shape == (48, 80)
    np.testing.assert_array_equal(np.asarray(out.data), data)


@_gpu_only
@pytest.mark.parametrize("tiled", [True, False])
def test_gpu_singleband_sanity(tiled):
    """Single-band GPU reads stay 2-D regardless of layout."""
    from xrspatial.geotiff import read_geotiff_gpu

    rng = np.random.RandomState(7)
    data = rng.randint(0, 200, size=(48, 80)).astype(np.uint8)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.tif")
        kwargs = {"photometric": "minisblack"}
        if tiled:
            kwargs["tile"] = (32, 32)
        tifffile.imwrite(path, data, **kwargs)
        out = read_geotiff_gpu(path)
    assert out.dims == ("y", "x")
    assert out.shape == (48, 80)
    np.testing.assert_array_equal(out.data.get(), data)


def test_a3_repro_stripped_planar_separate():
    """Spec-level guard for A3: stripped planar=2 must yield (y, x, band)."""
    from xrspatial.geotiff import open_geotiff

    data = _make_data(3, 64, 96, np.uint8)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sp.tif")
        tifffile.imwrite(path, data, photometric="rgb")  # default planar=2
        out = open_geotiff(path)
    assert out.dims == ("y", "x", "band")
    assert out.shape == (64, 96, 3), (
        f"got shape {out.shape} for dims {out.dims} -- A3 regressed"
    )
