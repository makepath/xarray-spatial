"""External-interop compliance suite for BigTIFF COG output.

Issue #2303 (part of #2286 -- production-readiness wave D). Sibling of
``test_cog_writer_compliance.py`` (issue #2292): same shape, same
assertions, but the matrix is scoped to the BigTIFF-specific layout
rather than re-validating every codec.

Scope
-----

These tests force BigTIFF via ``bigtiff=True`` on a small raster so the
output exercises:

- The BigTIFF header (magic ``43`` + 8-byte offsets).
- BigTIFF IFD entry layout (8-byte counts, 20-byte entry stride,
  8-byte next-IFD pointer).
- Tile / overview offset and byte-count arrays in LONG8 form.

A small raster is enough because the on-disk layout machinery flips
the same switches whether the file is 50 KB or 5 GB. Allocating a true
multi-gigabyte buffer is out of scope here -- the auto-BigTIFF threshold
is exercised separately by monkeypatching the IFD-overhead helper, so
the decision logic gets coverage without the allocation.

Tier decision rationale
-----------------------

Promotion of ``writer.bigtiff_cog`` to ``stable`` is deliberately out
of scope for this PR. ``SUPPORTED_FEATURES['writer.bigtiff_cog']`` is
set to ``advanced`` and stays there even when every row of this suite
passes -- promotion happens after the gate has lived in CI for a
release cycle (same rule the rest of the #2286 wave follows). If a row
uncovers a real BigTIFF COG bug, mark that row ``xfail`` with a linked
follow-up issue rather than fixing the writer here.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._header import parse_all_ifds, parse_header

rasterio = pytest.importorskip(
    "rasterio",
    reason="rasterio is required for the BigTIFF COG compliance suite",
)


# ---------------------------------------------------------------------------
# Matrix definitions
# ---------------------------------------------------------------------------

# One lossless integer codec, one lossless float codec. The BigTIFF
# layout is codec-agnostic; the codec axis here just confirms the
# codec/dtype combinations land cleanly inside the BigTIFF wrapper.
INT_CODEC = "deflate"
FLOAT_CODEC = "zstd"

ROWS = [
    pytest.param(INT_CODEC, np.uint16, 1, id="deflate-uint16-1band"),
    pytest.param(INT_CODEC, np.uint16, 3, id="deflate-uint16-3band"),
    pytest.param(FLOAT_CODEC, np.float32, 1, id="zstd-float32-1band"),
    pytest.param(FLOAT_CODEC, np.float32, 3, id="zstd-float32-3band"),
]


# ---------------------------------------------------------------------------
# Helpers (mirror ``test_cog_writer_compliance.py``)
# ---------------------------------------------------------------------------


def _make_data(
    dtype: np.dtype,
    *,
    bands: int = 1,
    height: int = 64,
    width: int = 64,
    rng_seed: int = 23,
) -> np.ndarray:
    dt = np.dtype(dtype)
    rng = np.random.RandomState(rng_seed + bands)
    if dt.kind == "f":
        base = rng.uniform(-100.0, 100.0, size=(height, width)).astype(dt)
    else:
        info = np.iinfo(dt)
        high = min(info.max, 1000)
        base = rng.randint(0, high, size=(height, width)).astype(dt)
    if bands == 1:
        return base
    layers = [base]
    for b in range(1, bands):
        layers.append((base + b * 7).astype(dt))
    return np.stack(layers, axis=-1)


def _build_da(arr: np.ndarray, *, crs: int | None = 4326) -> xr.DataArray:
    if arr.ndim == 2:
        h, w = arr.shape
        dims = ("y", "x")
    else:
        h, w, _b = arr.shape
        dims = ("y", "x", "band")
    y = np.linspace(45.0, 44.0, h, dtype=np.float64)
    x = np.linspace(-120.0, -119.0, w, dtype=np.float64)
    attrs: dict = {}
    if crs is not None:
        attrs["crs"] = crs
    return xr.DataArray(arr, dims=dims, coords={"y": y, "x": x}, attrs=attrs)


def _arrange_for_rasterio(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    return np.transpose(arr, (2, 0, 1))


def _is_tiled(src) -> bool:
    shapes = src.block_shapes
    if not shapes:
        return False
    bh, bw = shapes[0]
    return bh == bw and bh < src.height and bw < src.width


def _assert_bigtiff_header(path: str) -> None:
    """Confirm the on-disk file carries the BigTIFF magic and 8-byte offset."""
    with open(path, "rb") as f:
        head = f.read(16)
    assert head[:2] in (b"II", b"MM"), f"missing TIFF byte-order marker: {head[:2]!r}"
    bo = "<" if head[:2] == b"II" else ">"
    magic = struct.unpack_from(f"{bo}H", head, 2)[0]
    assert magic == 43, (
        f"expected BigTIFF magic 43, got {magic} -- writer did not emit "
        f"BigTIFF despite bigtiff=True"
    )
    # BigTIFF: bytes 4-5 = offset size (8), bytes 6-7 = 0, bytes 8-15 = first IFD offset.
    offset_size = struct.unpack_from(f"{bo}H", head, 4)[0]
    reserved = struct.unpack_from(f"{bo}H", head, 6)[0]
    assert offset_size == 8, f"BigTIFF offset size should be 8, got {offset_size}"
    assert reserved == 0, f"BigTIFF reserved field should be 0, got {reserved}"


def _assert_ifds_before_data(path: str) -> None:
    """COG layout contract: every IFD sits before any tile data block.

    Same invariant ``test_cog_writer_compliance.py`` enforces. Also
    asserts the parsed header reports ``is_bigtiff=True`` so the rest of
    the IFD-walk is going through the BigTIFF code path.
    """
    with open(path, "rb") as f:
        data = f.read()
    header = parse_header(data)
    assert header.is_bigtiff, "parse_header should report BigTIFF for these files"
    ifds = parse_all_ifds(data, header)
    assert len(ifds) >= 2, (
        f"expected at least 2 IFDs (full res + overview), got {len(ifds)}"
    )
    tile_offsets: list[int] = []
    for ifd in ifds:
        offs = ifd.tile_offsets
        if offs:
            tile_offsets.extend(offs)
    assert tile_offsets, "no tile offsets found; output is not tiled"
    first_data = min(tile_offsets)
    assert header.first_ifd_offset < first_data, (
        f"first IFD offset {header.first_ifd_offset} >= first tile data "
        f"offset {first_data}; IFDs must come before image data in a COG"
    )


def _try_cog_validate(path: str) -> None:
    """Optional rio-cogeo / GDAL validator. Same skip semantics as #2292."""
    try:
        from rio_cogeo.cogeo import cog_validate
    except ImportError:
        cog_validate = None  # type: ignore[assignment]

    if cog_validate is not None:
        valid, errors, _warns = cog_validate(path, strict=False)
        assert valid, f"rio_cogeo cog_validate failed: errors={errors}"
        return

    try:
        from osgeo_utils.samples import validate_cloud_optimized_geotiff
    except ImportError:
        pytest.skip(
            "neither rio-cogeo nor GDAL validate_cloud_optimized_geotiff "
            "is installed; skipping external COG validator step"
        )
        return

    _warns, errors, _details = validate_cloud_optimized_geotiff.validate(
        path, full_check=True,
    )
    assert not errors, f"GDAL validator errors: {errors}"


# ---------------------------------------------------------------------------
# Forced BigTIFF: matrix of (codec, dtype, band-count) x one overview level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codec,dtype,bands", ROWS)
def test_bigtiff_cog_roundtrip(tmp_path, codec, dtype, bands):
    """Force-BigTIFF COG round-trip: base pixels byte-exact, overviews survive,
    georef survives, BigTIFF header on disk, IFDs precede tile data.

    Mirrors the per-row contract of ``test_codec_dtype_bands_roundtrip``
    in ``test_cog_writer_compliance.py``, narrowed to the BigTIFF axis.
    """
    arr = _make_data(dtype, bands=bands, height=64, width=64)
    da = _build_da(arr, crs=4326)

    path = str(
        tmp_path / f"2303_bigtiff_cog_{codec}_{np.dtype(dtype).name}_b{bands}.tif"
    )
    to_geotiff(
        da, path,
        compression=codec, cog=True, tile_size=16,
        overview_levels=[2], bigtiff=True,
    )

    # Header sanity first -- if this fails the rest of the file is suspect.
    _assert_bigtiff_header(path)

    expected = _arrange_for_rasterio(arr)
    with rasterio.open(path) as src:
        assert _is_tiled(src), (
            f"{codec} {dtype} b{bands}: COG output must be tiled"
        )
        assert src.count == bands, (
            f"band count mismatch: expected {bands}, got {src.count}"
        )
        assert src.dtypes == tuple([np.dtype(dtype).name] * bands), (
            f"dtype tuple mismatch: expected "
            f"{tuple([np.dtype(dtype).name] * bands)}, got {src.dtypes}"
        )
        actual = src.read()
        assert actual.shape == expected.shape, (
            f"shape mismatch: expected {expected.shape}, got {actual.shape}"
        )
        # Lossless codecs -> byte-exact at full resolution.
        np.testing.assert_array_equal(
            actual, expected,
            err_msg=f"base pixels diverged for codec={codec} dtype={dtype}",
        )
        for b in range(1, bands + 1):
            ovs = src.overviews(b)
            assert ovs == [2], (
                f"band {b}: expected overview factors [2], got {ovs}"
            )
        # Confirm overview shape matches the 2x decimation factor.
        with rasterio.open(path, OVERVIEW_LEVEL=0) as ov:
            assert ov.shape == (arr.shape[0] // 2, arr.shape[1] // 2), (
                f"overview shape mismatch: expected "
                f"{(arr.shape[0] // 2, arr.shape[1] // 2)}, got {ov.shape}"
            )
        assert src.crs is not None and src.crs.to_epsg() == 4326, (
            f"CRS round-trip failed: got {src.crs}"
        )
        assert not src.transform.is_identity, (
            "transform should not be identity for a georeferenced raster"
        )

    _assert_ifds_before_data(path)


# ---------------------------------------------------------------------------
# Nodata under BigTIFF: float NaN sentinel survives the COG round-trip.
# ---------------------------------------------------------------------------


def test_bigtiff_cog_nodata_nan_survives(tmp_path):
    """NaN sentinel round-trips through a BigTIFF COG."""
    arr = _make_data(np.float32, bands=1, height=64, width=64)
    arr[0, 0] = np.nan
    arr[3, 9] = np.nan
    da = _build_da(arr, crs=4326)

    path = str(tmp_path / "2303_bigtiff_cog_nodata_nan.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2], nodata=float("nan"), bigtiff=True,
    )

    _assert_bigtiff_header(path)
    with rasterio.open(path) as src:
        assert src.nodata is not None and np.isnan(src.nodata), (
            f"nodata tag should be NaN, got {src.nodata}"
        )
        actual = src.read(1)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(arr))
        finite = ~np.isnan(arr)
        np.testing.assert_array_equal(actual[finite], arr[finite])


# ---------------------------------------------------------------------------
# External validator (optional)
# ---------------------------------------------------------------------------


def test_bigtiff_cog_external_validator(tmp_path):
    """Run rio-cogeo / GDAL's COG validator against a forced-BigTIFF COG."""
    arr = _make_data(np.float32, bands=1, height=256, width=256)
    da = _build_da(arr, crs=4326)

    path = str(tmp_path / "2303_bigtiff_cog_validator.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=64,
        overview_levels=[2, 4], bigtiff=True,
    )

    _assert_bigtiff_header(path)
    _try_cog_validate(path)


# ---------------------------------------------------------------------------
# Auto-BigTIFF threshold row
# ---------------------------------------------------------------------------


def test_auto_bigtiff_threshold_promotes_for_cog(tmp_path, monkeypatch):
    """The COG writer auto-promotes to BigTIFF when the classic-TIFF
    estimate exceeds UINT32_MAX, even without ``bigtiff=True``.

    Allocating an actual >4 GiB raster would dominate CI runtime and
    memory budgets, so this row drives the decision boundary by
    monkeypatching ``_compute_classic_ifd_overhead`` to return a value
    just past UINT32_MAX. Mirrors the strategy used by
    ``test_eager_writer_promotes_to_bigtiff_when_overhead_dominates`` in
    ``test_eager_bigtiff_overhead_exact_1905.py`` -- the writer's
    auto-decision pipes ``estimated_file_size > UINT32_MAX`` through the
    same helper for both the GeoTIFF and COG layouts. If a future refactor
    decouples the COG estimate from this helper this row will surface
    that drift loudly.
    """
    arr = _make_data(np.float32, bands=1, height=64, width=64)
    da = _build_da(arr, crs=4326)

    from xrspatial.geotiff import _writer as writer_mod

    real_overhead = writer_mod._compute_classic_ifd_overhead

    def _huge_overhead(tags):
        # Push the estimate well past UINT32_MAX (0xFFFFFFFF).
        return real_overhead(tags) + 0x100000000

    monkeypatch.setattr(
        writer_mod, "_compute_classic_ifd_overhead", _huge_overhead,
    )

    path = str(tmp_path / "2303_auto_bigtiff_cog.tif")
    # No ``bigtiff=`` arg -> auto-decision path. ``cog=True`` to confirm
    # the COG layout branch participates in the same decision.
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2],
    )

    # File must be BigTIFF, must still be a valid COG, must still
    # round-trip pixels.
    _assert_bigtiff_header(path)
    with rasterio.open(path) as src:
        assert src.count == 1, f"expected 1 band, got {src.count}"
        assert src.overviews(1) == [2], (
            f"expected overview [2], got {src.overviews(1)}"
        )
        np.testing.assert_array_equal(src.read(1), arr)
    _assert_ifds_before_data(path)
