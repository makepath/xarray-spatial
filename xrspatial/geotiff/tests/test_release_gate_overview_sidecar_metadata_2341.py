"""Release-gate: overview / sidecar metadata survival (epic #2341, PR 3 of 5).

Epic #2341 flags "overview reads lose CRS/transform/nodata metadata" as a
priority risk. The existing overview tests under
``xrspatial/geotiff/tests/`` (``test_cog_overview_nodata_1613.py``,
``test_dask_overview_level.py``, ``test_cog_cubic_overview_nodata_1623.py``,
etc.) assert pixel correctness or specific nodata behaviour on the overview
output. They do not pin full metadata survival on every overview level, and
they do not pin parity between internal COG overviews and external
``.ovr`` sidecars.

This module pins that contract on both read paths (eager and dask):

* For an internal-overview COG with base + overview levels at factors 2 and
  4, the per-level ``attrs`` agree on ``crs``, ``crs_wkt``,
  ``georef_status``, ``raster_type``, ``nodata``, and ``masked_nodata``.
  The ``transform`` field scales pixel size by the level factor while
  keeping the origin fixed.

* For a file whose overviews live in an external ``.ovr`` sidecar at the
  same factors, the same metadata-survival contract holds.

Both fixtures are constructed in-test so the test exercises the writer
path that produces them. If the writer regresses, this gate breaks too --
that is the intended coupling for a release gate.

Out of scope (covered elsewhere): pixel correctness of the resampling
kernels (``test_cog_overview_nodata_1613.py``,
``test_cog_cubic_overview_nodata_1623.py``); VRT mosaics (epic #2342).
"""
from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
import xarray as xr

# ``rasterio`` is the only writer that emits a ``.tif.ovr`` sidecar through
# its ``TIFF_USE_OVR`` env hint. The internal-overview path uses
# ``to_geotiff(cog=True, overview_levels=...)`` so it does not depend on
# rasterio.
rasterio = pytest.importorskip("rasterio")
pytest.importorskip("dask.array")

from rasterio.enums import Resampling  # noqa: E402
from xrspatial.geotiff import (  # noqa: E402
    open_geotiff,
    read_geotiff_dask,
    to_geotiff,
)


# ---------------------------------------------------------------------------
# Constants for the in-test fixtures.
# ---------------------------------------------------------------------------

# Base raster is 64x64 so factors of 2 and 4 give clean 32x32 and 16x16
# overviews.
_BASE_SIZE = 64
_OVERVIEW_FACTORS = (2, 4)
# Pixel size of 1 in projected units; origin at (-120, 45) so the test
# also catches a regression where the origin is silently rewritten as the
# overview is read.
_BASE_TRANSFORM = (1.0, 0.0, -120.0, 0.0, -1.0, 45.0)
_BASE_CRS = 4326
_NODATA = -9999.0

# Metadata keys the release-gate contract requires equal across all
# overview levels. Any drift on these keys means a downstream caller that
# branches on them will see a different file at the overview level than
# at the base.
_EQUAL_KEYS = (
    "crs",
    "crs_wkt",
    "georef_status",
    "raster_type",
    "nodata",
    "masked_nodata",
)


def _make_raster() -> xr.DataArray:
    """Return a 64x64 float32 DataArray with one NaN cell.

    The NaN is rewritten to the sentinel on write, exercising the
    ``masked_nodata`` lifecycle so the readback restores NaN and stamps
    ``masked_nodata=True`` on every level.
    """
    arr = np.arange(_BASE_SIZE * _BASE_SIZE,
                    dtype=np.float32).reshape(_BASE_SIZE, _BASE_SIZE)
    arr[0, 0] = np.nan
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        attrs={"transform": _BASE_TRANSFORM, "crs": _BASE_CRS},
    )


def _unique_tmp_path(tmp_path, label: str) -> str:
    """Return a unique path inside ``tmp_path`` tagged with the issue number.

    Parallel sibling agents share the pytest tmp root in CI worker
    layouts; include the issue number plus a uuid so a stray collision
    does not cross-pollute between PRs of epic #2341.
    """
    return str(tmp_path / f"release_gate_2341_{label}_{uuid.uuid4().hex}.tif")


def _write_internal_overview_cog(path: str) -> None:
    """Write a COG with base + internal overviews at factors 2 and 4.

    Asserts the writer actually emitted the requested overview IFDs.
    Without this guard, a regression where ``to_geotiff(cog=True,
    overview_levels=[2, 4])`` silently drops the overview chain would
    only surface downstream as a shape-mismatch in the reader, far
    from the writer call that caused it.
    """
    da = _make_raster()
    to_geotiff(
        da, path,
        nodata=_NODATA,
        cog=True,
        compression="deflate",
        tiled=True,
        tile_size=16,
        overview_levels=list(_OVERVIEW_FACTORS),
        overview_resampling="nearest",
    )
    with rasterio.open(path) as ds:
        assert ds.overviews(1) == list(_OVERVIEW_FACTORS), (
            f"writer did not emit the requested overview IFDs: "
            f"got {ds.overviews(1)}, expected {list(_OVERVIEW_FACTORS)}"
        )


def _write_external_sidecar(path: str) -> None:
    """Write a tiled TIFF + ``.ovr`` sidecar at factors 2 and 4.

    The xrspatial writer does not emit external sidecars; build the
    sidecar by reopening the tiled base file with ``TIFF_USE_OVR=YES`` so
    GDAL routes overview IFDs into ``<path>.ovr`` instead of appending
    them to the base file. This is the same path
    ``golden_corpus/generate.py`` uses for the bundled
    ``overview_external_ovr_uint16`` fixture.
    """
    da = _make_raster()
    to_geotiff(
        da, path,
        nodata=_NODATA,
        tiled=True,
        tile_size=16,
    )
    # Sanity: the base must not already carry internal overviews.
    with rasterio.open(path) as ds:
        assert ds.overviews(1) == [], (
            "base file must have no internal overviews before sidecar build"
        )

    with rasterio.Env(TIFF_USE_OVR="YES", COMPRESS_OVERVIEW="DEFLATE"):
        with rasterio.open(path, "r+") as ds:
            ds.build_overviews(list(_OVERVIEW_FACTORS), Resampling.nearest)
    assert os.path.exists(path + ".ovr"), (
        "TIFF_USE_OVR=YES must produce a .ovr sidecar next to the base file"
    )


# ---------------------------------------------------------------------------
# Inlined assertion helpers. The release-gate epic explicitly forbids a
# shared helper module between PRs of #2341 (four sibling agents work in
# parallel); keep these private to this module.
# ---------------------------------------------------------------------------

def _assert_metadata_equal_across_levels(attrs_by_level: dict) -> None:
    """Assert every key in ``_EQUAL_KEYS`` agrees across overview levels.

    Absent-on-every-level is also fine (the contract is "equal", not
    "present"). The reader does not stamp ``raster_type`` for the default
    RasterPixelIsArea, so this branch covers the common case where the
    key is absent on every level.

    Float / int equality on ``nodata`` is intentional: a downstream
    caller comparing ``attrs['nodata']`` to a sentinel uses ``==``, which
    treats ``-9999.0`` and ``-9999`` as equal. The release contract is
    "equality under ``==``", not "identical Python type".
    """
    base = attrs_by_level[0]
    for key in _EQUAL_KEYS:
        base_present = key in base
        base_val = base.get(key)
        for lvl, attrs in attrs_by_level.items():
            if lvl == 0:
                continue
            other_present = key in attrs
            other_val = attrs.get(key)
            assert other_present == base_present, (
                f"attrs[{key!r}] presence drifts: base={base_present}, "
                f"level={lvl}: {other_present}"
            )
            if base_present:
                assert other_val == base_val, (
                    f"attrs[{key!r}] differs across levels: "
                    f"base={base_val!r} level={lvl}: {other_val!r}"
                )


def _assert_transform_scales(attrs_by_level: dict, factors: dict) -> None:
    """Assert ``transform`` scales by ``factors[level]`` with the origin held.

    ``factors`` maps overview-level index -> decimation factor (the base
    level passes a factor of 1). Pixel sizes (``a`` and ``e``) multiply
    by the factor; the origin (``c`` and ``f``) is held fixed; the
    rotation terms (``b`` and ``d``) stay zero for the axis-aligned
    fixtures this test builds.
    """
    base = attrs_by_level[0]["transform"]
    base_a, base_b, base_c, base_d, base_e, base_f = base
    for lvl, attrs in attrs_by_level.items():
        factor = factors[lvl]
        t = attrs["transform"]
        a, b, c, d, e, f = t
        assert a == pytest.approx(base_a * factor), (
            f"level {lvl}: pixel width did not scale by {factor}: "
            f"got {a}, expected {base_a * factor}"
        )
        assert e == pytest.approx(base_e * factor), (
            f"level {lvl}: pixel height did not scale by {factor}: "
            f"got {e}, expected {base_e * factor}"
        )
        assert c == pytest.approx(base_c), (
            f"level {lvl}: origin x drifted: got {c}, expected {base_c}"
        )
        assert f == pytest.approx(base_f), (
            f"level {lvl}: origin y drifted: got {f}, expected {base_f}"
        )
        assert b == pytest.approx(0.0) and d == pytest.approx(0.0), (
            f"level {lvl}: axis-aligned transform must not gain rotation "
            f"terms, got b={b}, d={d}"
        )


def _read_levels_eager(path: str) -> dict:
    """Read base + each overview level via ``open_geotiff``.

    Returns a dict keyed by level index (0 = base, 1 = first overview,
    ...) where the value is the read ``DataArray``.
    """
    out = {0: open_geotiff(path)}
    for i, _ in enumerate(_OVERVIEW_FACTORS, start=1):
        out[i] = open_geotiff(path, overview_level=i)
    return out


def _read_levels_dask(path: str) -> dict:
    """Read base + each overview level via ``read_geotiff_dask``.

    The chunk size is intentionally small (8) so per-chunk reads
    cover at least one chunk boundary at every level; a regression where
    the dask graph drops attrs on assembly would surface here.
    """
    out = {0: read_geotiff_dask(path, chunks=8)}
    for i, _ in enumerate(_OVERVIEW_FACTORS, start=1):
        out[i] = read_geotiff_dask(path, chunks=8, overview_level=i)
    return out


def _factors_by_level() -> dict:
    """Map level index (0=base, 1, 2, ...) to its decimation factor."""
    factors = {0: 1}
    for i, f in enumerate(_OVERVIEW_FACTORS, start=1):
        factors[i] = f
    return factors


# ---------------------------------------------------------------------------
# Internal COG overviews.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reader", ["eager", "dask"])
def test_cog_internal_overview_metadata_survives(tmp_path, reader):
    """COG with internal overviews preserves the metadata contract.

    Asserts that ``crs``, ``crs_wkt``, ``georef_status``, ``raster_type``,
    ``nodata``, and ``masked_nodata`` agree across base + every overview
    level for both the eager and dask read paths.
    """
    path = _unique_tmp_path(tmp_path, f"cog_meta_{reader}")
    _write_internal_overview_cog(path)

    if reader == "eager":
        levels = _read_levels_eager(path)
    else:
        levels = _read_levels_dask(path)

    attrs_by_level = {lvl: da.attrs for lvl, da in levels.items()}
    _assert_metadata_equal_across_levels(attrs_by_level)

    # Sanity: the contract requires these keys to actually be present on
    # the base (otherwise "equal across levels" trivially holds with
    # everything absent). The constructed fixture carries CRS + nodata,
    # so the read must surface them.
    base = attrs_by_level[0]
    assert base.get("crs") == _BASE_CRS
    assert base.get("crs_wkt"), "crs_wkt must be set on a CRS-carrying COG"
    assert base.get("nodata") == _NODATA
    assert base.get("masked_nodata") is True
    assert base.get("georef_status") == "full"


@pytest.mark.parametrize("reader", ["eager", "dask"])
def test_cog_internal_overview_transform_scales(tmp_path, reader):
    """COG with internal overviews preserves transform origin and scales pixel size."""
    path = _unique_tmp_path(tmp_path, f"cog_xform_{reader}")
    _write_internal_overview_cog(path)

    if reader == "eager":
        levels = _read_levels_eager(path)
    else:
        levels = _read_levels_dask(path)

    attrs_by_level = {lvl: da.attrs for lvl, da in levels.items()}
    _assert_transform_scales(attrs_by_level, _factors_by_level())


def test_cog_internal_overview_shape_matches_factors(tmp_path):
    """Smoke: shapes follow the decimation factors so the test exercises real overview IFDs.

    Catches a regression where the reader silently returns the base
    image when asked for an overview level. Independent of the metadata
    contract: shape comes from the IFD, not ``attrs``.
    """
    path = _unique_tmp_path(tmp_path, "cog_shape")
    _write_internal_overview_cog(path)

    base = open_geotiff(path)
    assert base.shape == (_BASE_SIZE, _BASE_SIZE)
    for i, factor in enumerate(_OVERVIEW_FACTORS, start=1):
        da = open_geotiff(path, overview_level=i)
        expected = _BASE_SIZE // factor
        assert da.shape == (expected, expected), (
            f"overview_level={i} returned shape {da.shape}, "
            f"expected ({expected}, {expected})"
        )


# ---------------------------------------------------------------------------
# External `.ovr` sidecar.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reader", ["eager", "dask"])
def test_sidecar_overview_metadata_survives(tmp_path, reader):
    """External `.ovr` sidecar preserves the metadata contract.

    Same contract as the internal-overview test: CRS, georef status,
    nodata pair, raster_type all agree across base + sidecar overview
    levels.
    """
    path = _unique_tmp_path(tmp_path, f"sidecar_meta_{reader}")
    _write_external_sidecar(path)

    if reader == "eager":
        levels = _read_levels_eager(path)
    else:
        levels = _read_levels_dask(path)

    attrs_by_level = {lvl: da.attrs for lvl, da in levels.items()}
    _assert_metadata_equal_across_levels(attrs_by_level)

    # Same presence sanity check as the COG test: the fixture is built
    # with CRS + nodata so the gate covers a real read, not an
    # everything-absent vacuum.
    base = attrs_by_level[0]
    assert base.get("crs") == _BASE_CRS
    assert base.get("crs_wkt"), (
        "crs_wkt must be set when the base file carries an EPSG code"
    )
    assert base.get("nodata") == _NODATA
    assert base.get("masked_nodata") is True
    assert base.get("georef_status") == "full"


@pytest.mark.parametrize("reader", ["eager", "dask"])
def test_sidecar_overview_transform_scales(tmp_path, reader):
    """External `.ovr` sidecar scales pixel size by 2 per level, origin held."""
    path = _unique_tmp_path(tmp_path, f"sidecar_xform_{reader}")
    _write_external_sidecar(path)

    if reader == "eager":
        levels = _read_levels_eager(path)
    else:
        levels = _read_levels_dask(path)

    attrs_by_level = {lvl: da.attrs for lvl, da in levels.items()}
    _assert_transform_scales(attrs_by_level, _factors_by_level())


def test_sidecar_overview_shape_matches_factors(tmp_path):
    """Smoke: sidecar reads return the right shape per level.

    Mirrors the COG smoke test so a "reader silently returns base
    pixels for any overview level" regression also surfaces here, on
    the external-sidecar path.
    """
    path = _unique_tmp_path(tmp_path, "sidecar_shape")
    _write_external_sidecar(path)

    base = open_geotiff(path)
    assert base.shape == (_BASE_SIZE, _BASE_SIZE)
    for i, factor in enumerate(_OVERVIEW_FACTORS, start=1):
        da = open_geotiff(path, overview_level=i)
        expected = _BASE_SIZE // factor
        assert da.shape == (expected, expected), (
            f"sidecar overview_level={i} returned shape {da.shape}, "
            f"expected ({expected}, {expected})"
        )


# ---------------------------------------------------------------------------
# Cross-source parity: internal COG vs external sidecar at matching factors.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reader", ["eager", "dask"])
def test_internal_vs_sidecar_metadata_agree(tmp_path, reader):
    """Internal COG and external sidecar agree on the metadata contract.

    The release-gate epic specifically calls out parity between the two
    paths -- if a downstream caller switches a deployment from inline
    overviews to a sidecar (or vice versa), the read contract must not
    change. The two fixtures share base raster, CRS, transform, and
    nodata, so the per-level attrs must agree key-by-key.
    """
    cog_path = _unique_tmp_path(tmp_path, f"parity_cog_{reader}")
    sidecar_path = _unique_tmp_path(tmp_path, f"parity_sidecar_{reader}")
    _write_internal_overview_cog(cog_path)
    _write_external_sidecar(sidecar_path)

    if reader == "eager":
        cog_levels = _read_levels_eager(cog_path)
        sidecar_levels = _read_levels_eager(sidecar_path)
    else:
        cog_levels = _read_levels_dask(cog_path)
        sidecar_levels = _read_levels_dask(sidecar_path)

    assert set(cog_levels) == set(sidecar_levels)
    for lvl in cog_levels:
        for key in _EQUAL_KEYS:
            cog_attrs = cog_levels[lvl].attrs
            sidecar_attrs = sidecar_levels[lvl].attrs
            assert (key in cog_attrs) == (key in sidecar_attrs), (
                f"level {lvl}: attrs[{key!r}] presence differs between "
                f"internal-COG and sidecar reads "
                f"(cog={key in cog_attrs}, sidecar={key in sidecar_attrs})"
            )
            if key in cog_attrs:
                assert cog_attrs[key] == sidecar_attrs[key], (
                    f"level {lvl}: attrs[{key!r}] differs between "
                    f"internal-COG and sidecar reads: "
                    f"cog={cog_attrs[key]!r}, sidecar={sidecar_attrs[key]!r}"
                )
        # Transform parity at every level: the two fixtures use the
        # same base transform, so every level's transform must match.
        assert cog_levels[lvl].attrs["transform"] == pytest.approx(
            sidecar_levels[lvl].attrs["transform"]
        ), (
            f"level {lvl}: transform differs between internal-COG and "
            f"sidecar reads"
        )
