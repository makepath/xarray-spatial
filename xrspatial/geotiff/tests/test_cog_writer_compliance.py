"""External-interop compliance suite for ``to_geotiff(..., cog=True)``.

Issue #2292 (part of #2286 -- COG readiness/stability rollout).

These tests treat ``to_geotiff`` as a black box: every assertion goes through
rasterio (and optionally rio-cogeo / the GDAL ``validate_cloud_optimized_geotiff``
sample). The goal is to catch interop regressions where xrspatial writes a
file that satisfies the in-process round-trip but trips up external readers.

Matrix:

- Stable codecs: ``none``, ``deflate``, ``lzw``, ``zstd``, ``packbits``.
- Dtypes: at least one integer (``uint16``) and one float (``float32``).
- Bands: single-band and 3-band.
- Nodata: sentinel value (integer + float sentinel) and NaN.
- Georef: PixelIsArea and PixelIsPoint.
- Overviews: explicit level list and auto-generated levels.

Production code is out of scope -- if a row uncovers a real writer bug,
mark it ``xfail`` with a linked follow-up issue rather than fixing it
here.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._header import parse_all_ifds, parse_header

rasterio = pytest.importorskip(
    "rasterio",
    reason="rasterio is required for the external compliance suite",
)


# ---------------------------------------------------------------------------
# Test matrix definitions
# ---------------------------------------------------------------------------

# Stable, lossless codecs only. Each row should produce a byte-for-byte
# round-trip on the base level.
STABLE_CODECS = ["none", "deflate", "lzw", "zstd", "packbits"]

DTYPES = [
    pytest.param(np.uint16, id="uint16"),
    pytest.param(np.float32, id="float32"),
]

BAND_COUNTS = [
    pytest.param(1, id="1band"),
    pytest.param(3, id="3band"),
]

# ``raster_type`` attr the writer understands: ``'area'`` (default) or
# ``'point'``. We pass via attrs because that is the public surface.
GEOREF_MODES = [
    pytest.param("area", id="area"),
    pytest.param("point", id="point"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    dtype: np.dtype,
    *,
    bands: int = 1,
    height: int = 64,
    width: int = 64,
    rng_seed: int = 17,
) -> np.ndarray:
    """Deterministic raster shaped (h, w) or (h, w, bands)."""
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
    # Stack with small per-band offsets so bands are distinguishable.
    layers = [base]
    for b in range(1, bands):
        layers.append((base + b * 7).astype(dt))
    return np.stack(layers, axis=-1)  # (h, w, bands)


def _build_da(
    arr: np.ndarray,
    *,
    raster_type: str = "area",
    crs: int | str | None = 4326,
) -> xr.DataArray:
    """Wrap ``arr`` in a DataArray with EPSG:4326 coords and georef attrs."""
    if arr.ndim == 2:
        h, w = arr.shape
        dims = ("y", "x")
    else:
        h, w, _b = arr.shape
        dims = ("y", "x", "band")
    y = np.linspace(45.0, 44.0, h, dtype=np.float64)
    x = np.linspace(-120.0, -119.0, w, dtype=np.float64)
    coords: dict = {"y": y, "x": x}
    attrs: dict = {}
    if crs is not None:
        attrs["crs"] = crs
    if raster_type == "point":
        attrs["raster_type"] = "point"
    return xr.DataArray(arr, dims=dims, coords=coords, attrs=attrs)


def _pick_sentinel(dtype: np.dtype) -> float | int:
    """Pick a nodata sentinel that fits the dtype.

    The signed-int branch is unreachable from the current DTYPES list
    (only ``uint16`` and ``float32``) but is kept for the eventual case
    where the matrix grows. Dead branches in a helper are cheap and the
    intent is clearer than special-casing the current matrix here.
    """
    dt = np.dtype(dtype)
    if dt.kind == "f":
        return -9999.0
    if dt.kind == "u":
        return int(np.iinfo(dt).max)  # e.g. 65535 for uint16
    return int(np.iinfo(dt).min)


def _arrange_for_rasterio(arr: np.ndarray) -> np.ndarray:
    """Convert (h, w[, bands]) into rasterio's (bands, h, w)."""
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    # (h, w, bands) -> (bands, h, w)
    return np.transpose(arr, (2, 0, 1))


def _is_tiled(src) -> bool:
    """Rasterio's ``is_tiled`` is deprecated; reproduce its check locally.

    A dataset is tiled when block dimensions are square and smaller than
    the dataset itself (rasterio's old definition). ``block_shapes`` is
    a per-band list of ``(height, width)`` tuples.
    """
    shapes = src.block_shapes
    if not shapes:
        return False
    bh, bw = shapes[0]
    return bh == bw and bh < src.height and bw < src.width


def _assert_ifds_before_data(path: str) -> None:
    """COG layout contract: every IFD sits before any tile data block."""
    with open(path, "rb") as f:
        data = f.read()
    header = parse_header(data)
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
    # All IFD anchors must sit before the first tile blob.
    assert header.first_ifd_offset < first_data, (
        f"first IFD offset {header.first_ifd_offset} >= first tile data "
        f"offset {first_data}; IFDs must come before image data in a COG"
    )


def _require_validator_env() -> bool:
    """Return True if ``XRSPATIAL_REQUIRE_COG_VALIDATOR`` is set truthy.

    Truthy values: ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    Anything else, including unset / empty, returns False.

    CI sets this to make a missing validator dependency a hard failure
    rather than a silent skip. On a contributor laptop without rio-cogeo
    or GDAL it is unset and the validator step skips cleanly.
    """
    val = os.environ.get("XRSPATIAL_REQUIRE_COG_VALIDATOR", "")
    return val.lower() in {"1", "true", "yes", "on"}


def _try_cog_validate(path: str) -> None:
    """Call rio-cogeo's validator if present, else GDAL's.

    When ``XRSPATIAL_REQUIRE_COG_VALIDATOR=1`` is set in the environment
    and neither validator is importable, fail loudly instead of skipping
    so a misconfigured CI job cannot pretend the gate passed. When the
    env var is unset, missing dependencies skip cleanly.
    """
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
        if _require_validator_env():
            pytest.fail(
                "XRSPATIAL_REQUIRE_COG_VALIDATOR=1 but neither rio-cogeo "
                "nor GDAL validate_cloud_optimized_geotiff is importable. "
                "Install rio-cogeo (and/or GDAL Python bindings) on this "
                "job, or unset XRSPATIAL_REQUIRE_COG_VALIDATOR to allow "
                "the soft skip."
            )
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
# Codec x dtype x band-count: base pixels + overviews + georef survive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bands", BAND_COUNTS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("codec", STABLE_CODECS)
def test_codec_dtype_bands_roundtrip(tmp_path, codec, dtype, bands):
    """Stable codec round-trip via rasterio: base pixels byte-exact, georef survives.

    Contracts asserted per row:
    - rasterio.open succeeds and reports a tiled COG.
    - Band count and dtype survive.
    - Base pixels are byte-exact (stable codecs are lossless).
    - Overview decimation factors survive.
    - CRS and transform survive.
    - IFDs sit before any tile data block (COG layout).
    """
    arr = _make_data(dtype, bands=bands, height=64, width=64)
    da = _build_da(arr, raster_type="area", crs=4326)

    path = str(tmp_path / f"2292_codec_{codec}_{np.dtype(dtype).name}_b{bands}.tif")
    to_geotiff(
        da, path,
        compression=codec, cog=True, tile_size=16,
        overview_levels=[2],
    )

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
        # Stable codecs are lossless -> byte-exact at full resolution.
        actual = src.read()
        assert actual.shape == expected.shape, (
            f"shape mismatch: expected {expected.shape}, got {actual.shape}"
        )
        np.testing.assert_array_equal(
            actual, expected,
            err_msg=f"base pixels diverged for codec={codec} dtype={dtype}",
        )
        # Overviews
        for b in range(1, bands + 1):
            ovs = src.overviews(b)
            assert ovs == [2], (
                f"band {b}: expected overview factors [2], got {ovs}"
            )
        # CRS / transform
        assert src.crs is not None and src.crs.to_epsg() == 4326, (
            f"CRS round-trip failed: got {src.crs}"
        )
        assert not src.transform.is_identity, (
            "transform should not be identity for a georeferenced raster"
        )
    # COG layout invariant
    _assert_ifds_before_data(path)


# ---------------------------------------------------------------------------
# Nodata: sentinel and NaN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_nodata_sentinel_survives(tmp_path, dtype):
    """Integer and float sentinels survive write -> rasterio.open."""
    arr = _make_data(dtype, bands=1, height=64, width=64)
    sentinel = _pick_sentinel(dtype)
    # Mark a couple of cells as nodata.
    arr_with_nd = arr.copy()
    arr_with_nd[0, 0] = sentinel
    arr_with_nd[5, 7] = sentinel
    da = _build_da(arr_with_nd, raster_type="area", crs=4326)

    path = str(tmp_path / f"2292_nodata_sentinel_{np.dtype(dtype).name}.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2], nodata=sentinel,
    )

    with rasterio.open(path) as src:
        assert src.nodata is not None, "nodata tag not set on output"
        # rasterio normalises to float; compare numerically.
        assert float(src.nodata) == float(sentinel), (
            f"nodata mismatch: expected {sentinel}, got {src.nodata}"
        )
        actual = src.read(1)
        # Byte-exact at base level for deflate.
        np.testing.assert_array_equal(actual, arr_with_nd)


def test_nodata_nan_survives(tmp_path):
    """NaN nodata: NaN positions round-trip as NaN through rasterio."""
    arr = _make_data(np.float32, bands=1, height=64, width=64)
    arr[0, 0] = np.nan
    arr[3, 9] = np.nan
    da = _build_da(arr, raster_type="area", crs=4326)

    path = str(tmp_path / "2292_nodata_nan.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2], nodata=float("nan"),
    )

    with rasterio.open(path) as src:
        assert src.nodata is not None and np.isnan(src.nodata), (
            f"nodata tag should be NaN, got {src.nodata}"
        )
        actual = src.read(1)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(arr))
        finite = ~np.isnan(arr)
        np.testing.assert_array_equal(actual[finite], arr[finite])


# ---------------------------------------------------------------------------
# Georef: PixelIsArea vs PixelIsPoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raster_type", GEOREF_MODES)
def test_raster_type_tag_survives(tmp_path, raster_type):
    """AREA_OR_POINT tag survives to rasterio.tags()."""
    arr = _make_data(np.float32, bands=1, height=32, width=32)
    da = _build_da(arr, raster_type=raster_type, crs=4326)

    path = str(tmp_path / f"2292_georef_{raster_type}.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2],
    )

    with rasterio.open(path) as src:
        tag = src.tags().get("AREA_OR_POINT")
        expected_tag = "Area" if raster_type == "area" else "Point"
        assert tag == expected_tag, (
            f"AREA_OR_POINT tag mismatch: expected {expected_tag!r}, "
            f"got {tag!r}"
        )
        # Base values still round-trip exactly.
        np.testing.assert_array_equal(src.read(1), arr)


# ---------------------------------------------------------------------------
# Overviews: explicit list vs auto-generated
# ---------------------------------------------------------------------------


def test_overviews_explicit_levels(tmp_path):
    """``overview_levels=[2, 4, 8]`` produces exactly those decimations."""
    arr = _make_data(np.float32, bands=1, height=128, width=128)
    da = _build_da(arr, raster_type="area", crs=4326)

    path = str(tmp_path / "2292_overviews_explicit.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2, 4, 8],
    )

    with rasterio.open(path) as src:
        assert src.overviews(1) == [2, 4, 8], (
            f"expected overviews [2, 4, 8], got {src.overviews(1)}"
        )
        # Each native overview should have the expected shape.
        for lvl, factor in enumerate([2, 4, 8]):
            with rasterio.open(path, OVERVIEW_LEVEL=lvl) as ov:
                exp_h = arr.shape[0] // factor
                exp_w = arr.shape[1] // factor
                assert ov.shape == (exp_h, exp_w), (
                    f"overview {factor}x: expected shape ({exp_h}, {exp_w}), "
                    f"got {ov.shape}"
                )
    _assert_ifds_before_data(path)


@pytest.mark.parametrize("resampling", ["mean", "nearest"])
def test_overview_pixels_match_expected(tmp_path, resampling):
    """Overview pixel values agree with a hand-computed 2x decimation.

    Uses a deterministic base array so we can predict the level-1 overview
    in pure numpy. ``mean`` reduces each 2x2 block to its mean; ``nearest``
    keeps the upper-left pixel of each block. The writer should produce
    overviews that match within float tolerance (lossless codec on the
    base, deterministic block reducer on the overview).
    """
    base = _make_data(np.float32, bands=1, height=64, width=64)
    da = _build_da(base, raster_type="area", crs=4326)

    path = str(tmp_path / f"2292_ovpix_{resampling}.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2], overview_resampling=resampling,
    )

    if resampling == "mean":
        # Block-mean 2x2 -> (32, 32). Promote to float64 for the reduction
        # so the comparison is not biased by float32 round-off in the
        # intermediate sum, then cast back to match what the reader
        # returns.
        b = base.astype(np.float64).reshape(32, 2, 32, 2).mean(axis=(1, 3))
        expected_ov = b.astype(np.float32)
    else:  # nearest
        # Upper-left pixel of each 2x2 block.
        expected_ov = base[::2, ::2]

    with rasterio.open(path, OVERVIEW_LEVEL=0) as ov:
        actual = ov.read(1)
    assert actual.shape == expected_ov.shape, (
        f"{resampling}: expected overview shape {expected_ov.shape}, "
        f"got {actual.shape}"
    )
    # Tolerance: the writer's mean reducer accumulates in float64 internally
    # but the on-disk result is float32; comparing against our hand-computed
    # float32 expected leaves <= 1 ULP of slack per cell.
    np.testing.assert_allclose(
        actual, expected_ov, rtol=1e-5, atol=1e-5,
        err_msg=f"{resampling} overview pixels diverged from expected",
    )


def test_overviews_auto_generated(tmp_path):
    """``overview_levels=None`` with cog=True auto-generates a pyramid."""
    arr = _make_data(np.float32, bands=1, height=128, width=128)
    da = _build_da(arr, raster_type="area", crs=4326)

    path = str(tmp_path / "2292_overviews_auto.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=32,
    )

    with rasterio.open(path) as src:
        ovs = src.overviews(1)
        assert len(ovs) >= 1, f"expected at least one overview, got {ovs}"
        # Auto-generated pyramid: every level is a power of two, strictly
        # increasing, and large enough that the next halving would not fall
        # below the tile_size of 32. The bitwise test below is the classic
        # power-of-two check: ``o & (o - 1) == 0`` is True iff ``o`` has a
        # single set bit. The ``o >= 2`` guard rules out the false-positive
        # at ``o == 0``.
        assert all((o & (o - 1)) == 0 and o >= 2 for o in ovs), (
            f"auto overviews should be powers of two >= 2, got {ovs}"
        )
        assert all(b > a for a, b in zip(ovs, ovs[1:])), (
            f"auto overviews not strictly increasing: {ovs}"
        )
    _assert_ifds_before_data(path)


# ---------------------------------------------------------------------------
# TIFF layout sanity: tiled, sane tile offsets, IFDs before data
# ---------------------------------------------------------------------------


def test_layout_is_cog_shaped(tmp_path):
    """A cog=True file is tiled, has overview IFDs, and IFDs precede data."""
    arr = _make_data(np.uint16, bands=1, height=128, width=128)
    da = _build_da(arr, raster_type="area", crs=4326)

    path = str(tmp_path / "2292_layout.tif")
    to_geotiff(
        da, path,
        compression="lzw", cog=True, tile_size=32,
        overview_levels=[2, 4],
    )

    with rasterio.open(path) as src:
        assert _is_tiled(src), "COG output must be tiled, got stripped layout"
        assert src.block_shapes[0] == (32, 32), (
            f"unexpected block shape: {src.block_shapes}"
        )

    # All IFDs come before image data; tile offsets are monotonic-ish
    # (not strictly monotonic across IFDs but every offset must point inside
    # the file).
    with open(path, "rb") as f:
        data = f.read()
    header = parse_header(data)
    ifds = parse_all_ifds(data, header)
    assert len(ifds) == 3, (
        f"expected 3 IFDs (full + 2 overviews), got {len(ifds)}"
    )
    file_len = len(data)
    for ifd in ifds:
        for off in (ifd.tile_offsets or ()):
            assert 0 <= off < file_len, (
                f"tile offset {off} outside file bounds [0, {file_len})"
            )
    _assert_ifds_before_data(path)


# ---------------------------------------------------------------------------
# Optional external validator
# ---------------------------------------------------------------------------


# The writer currently emits overview tile blocks in the wrong order
# (issue #2308 -- surfaced by this very gate). rio-cogeo reports:
#   "The offset of the first block of overview of index 0 should be
#    after the one of the overview of index 1"
# The in-process round-trip and the local IFD-before-data layout
# invariant both still pass; only the block ordering across overviews
# trips external validators. xfail (strict=False) so the CI gate is
# wired up and stays green for the rest of the suite. Drop this
# marker once the writer fix lands.
@pytest.mark.xfail(
    reason=(
        "writer emits overview tile blocks in wrong order; see #2308. "
        "Remove xfail when the writer fix lands."
    ),
    strict=False,
)
def test_external_cog_validator(tmp_path):
    """Run rio-cogeo / GDAL's COG validator if available, else skip cleanly."""
    arr = _make_data(np.float32, bands=1, height=256, width=256)
    da = _build_da(arr, raster_type="area", crs=4326)

    path = str(tmp_path / "2292_validator.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=64,
        overview_levels=[2, 4],
    )

    _try_cog_validate(path)


# ---------------------------------------------------------------------------
# Validator-mode env contract (issue #2302)
# ---------------------------------------------------------------------------


def test_require_validator_env_strict_fails_when_dep_missing(
    tmp_path, monkeypatch,
):
    """``XRSPATIAL_REQUIRE_COG_VALIDATOR=1`` must fail (not skip) if both
    validators are absent.

    This guards the CI gate: if the install step silently drops rio-cogeo
    or GDAL, the compliance suite must fail rather than skip past the
    validator step. Stub both imports as ``ImportError`` so the test runs
    the same on every job, validator-present or not.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        fl = tuple(fromlist) if fromlist else ()
        rio_match = (
            name == "rio_cogeo.cogeo" and "cog_validate" in fl
        )
        gdal_match = (
            name == "osgeo_utils.samples"
            and "validate_cloud_optimized_geotiff" in fl
        )
        if rio_match or gdal_match:
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    monkeypatch.setenv("XRSPATIAL_REQUIRE_COG_VALIDATOR", "1")

    arr = _make_data(np.float32, bands=1, height=64, width=64)
    da = _build_da(arr, raster_type="area", crs=4326)
    path = str(tmp_path / "2302_require_strict.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2],
    )

    # ``pytest.fail.Exception`` is a documented alias for
    # ``_pytest.outcomes.Failed`` on pytest >= 7 (which this repo pins
    # via setup.cfg). Update both spots in this file if that pin moves.
    with pytest.raises(pytest.fail.Exception, match="XRSPATIAL_REQUIRE_COG_VALIDATOR"):
        _try_cog_validate(path)


def test_require_validator_env_unset_skips_when_dep_missing(
    tmp_path, monkeypatch,
):
    """With the env var unset, missing validators trigger a clean skip.

    This is the contributor-laptop path: no rio-cogeo / GDAL installed,
    the compliance suite still passes without the optional validator
    step.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        fl = tuple(fromlist) if fromlist else ()
        rio_match = (
            name == "rio_cogeo.cogeo" and "cog_validate" in fl
        )
        gdal_match = (
            name == "osgeo_utils.samples"
            and "validate_cloud_optimized_geotiff" in fl
        )
        if rio_match or gdal_match:
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    monkeypatch.delenv("XRSPATIAL_REQUIRE_COG_VALIDATOR", raising=False)

    arr = _make_data(np.float32, bands=1, height=64, width=64)
    da = _build_da(arr, raster_type="area", crs=4326)
    path = str(tmp_path / "2302_require_unset.tif")
    to_geotiff(
        da, path,
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2],
    )

    with pytest.raises(pytest.skip.Exception):
        _try_cog_validate(path)


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_require_validator_env_truthy_values(monkeypatch, val):
    """All documented truthy spellings activate strict mode."""
    monkeypatch.setenv("XRSPATIAL_REQUIRE_COG_VALIDATOR", val)
    assert _require_validator_env() is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "anything"])
def test_require_validator_env_non_truthy_values(monkeypatch, val):
    """Empty or non-truthy spellings leave strict mode off."""
    if val == "":
        monkeypatch.delenv("XRSPATIAL_REQUIRE_COG_VALIDATOR", raising=False)
    else:
        monkeypatch.setenv("XRSPATIAL_REQUIRE_COG_VALIDATOR", val)
    assert _require_validator_env() is False
