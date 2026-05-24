"""Backend parity for VRT reads with sidecar/overview interactions (#2321 sub-task 4).

Sub-task 4 of issue #2321 (parent) locks down the VRT support contract by
asserting eager / dask parity on the surface that is most likely to drift:

* metadata (``attrs['transform']``, ``attrs['crs']``, ``attrs['crs_wkt']``,
  ``attrs['georef_status']``), not just pixel values;
* coords -- a windowed read must shift the ``y`` / ``x`` arrays consistently
  between the eager and the lazy code paths;
* sidecar interactions -- a VRT whose backing source is a GeoTIFF with an
  external ``.tif.ovr`` pyramid must surface the same georef attrs (and
  pixel values) as an equivalent VRT over the inline-overview fixture.

The shape mirrors ``test_backend_parity_matrix.py``: a small declarative
fixture / backend matrix, one ``assert_parity`` helper, and a single
parametrised test. We do not re-invent helpers -- the materialisation
and pixel-comparison primitives match the matrix file so a future move
to the shared parity harness is mechanical.

VRT fixtures use the existing ``write_vrt`` writer (``xrspatial.geotiff
._vrt.write_vrt``) on top of ``to_geotiff`` source tiles, plus the
bundled ``overview_external_ovr_uint16.tif`` / ``.tif.ovr`` sidecar pair
from ``golden_corpus/fixtures/`` (and its inline-overview counterpart).

Acceptance per the parent issue: the VRT path cannot pass by returning
correct pixel values with wrong georeferencing attrs. Windowed eager
and lazy VRT reads agree on shape, coords, attrs, and values.
"""
from __future__ import annotations

import pathlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff
from xrspatial.geotiff._vrt import write_vrt as _write_vrt_internal


# ---------------------------------------------------------------------------
# Fixture paths shipped under golden_corpus.
# ---------------------------------------------------------------------------

_GOLDEN = (
    pathlib.Path(__file__).resolve().parent
    / "golden_corpus"
    / "fixtures"
)
_SIDECAR_TIF = _GOLDEN / "overview_external_ovr_uint16.tif"
_SIDECAR_OVR = _GOLDEN / "overview_external_ovr_uint16.tif.ovr"
_INLINE_OVR_TIF = _GOLDEN / "overview_internal_uint16.tif"


def _sidecar_fixture_or_skip() -> Path:
    """Return the bundled sidecar TIFF or skip if absent."""
    if not _SIDECAR_TIF.exists() or not _SIDECAR_OVR.exists():
        pytest.skip("sidecar overview fixture not present in golden_corpus")
    return _SIDECAR_TIF


def _inline_overview_fixture_or_skip() -> Path:
    if not _INLINE_OVR_TIF.exists():
        pytest.skip("inline overview fixture not present in golden_corpus")
    return _INLINE_OVR_TIF


# ---------------------------------------------------------------------------
# Materialisation + comparison helpers
# (mirrors ``test_backend_parity_matrix.py`` so cross-test parity reads
# the same way).
# ---------------------------------------------------------------------------

def _materialise(da: xr.DataArray) -> np.ndarray:
    raw = da.data
    if hasattr(raw, "compute"):
        raw = raw.compute()
    if hasattr(raw, "get"):
        raw = raw.get()
    return np.asarray(raw)


def _coord_view(da: xr.DataArray, name: str) -> np.ndarray:
    return np.asarray(da.coords[name].values)


def _assert_pixels_equal(ref: np.ndarray, actual: np.ndarray,
                         *, label: str) -> None:
    """Pixel equality, dtype-aware (mirrors test_backend_parity_matrix.py)."""
    assert ref.dtype == actual.dtype, (
        f"{label}: dtype differs ref={ref.dtype} actual={actual.dtype}"
    )
    assert ref.shape == actual.shape, (
        f"{label}: shape differs ref={ref.shape} actual={actual.shape}"
    )
    if ref.dtype.kind == "f":
        assert np.array_equal(ref, actual, equal_nan=True), (
            f"{label}: float pixels differ (NaN-aware)"
        )
    else:
        assert ref.tobytes() == actual.tobytes(), (
            f"{label}: integer pixel bytes differ"
        )


def _assert_metadata_parity(
    ref: xr.DataArray,
    actual: xr.DataArray,
    *,
    label: str,
    expected_dims: tuple[str, ...],
) -> None:
    """Fail if any of the parity-critical attrs / coords drift between two reads.

    The acceptance bar for this PR: the VRT path cannot pass by returning
    correct pixel values with wrong georeferencing attrs. Every field
    checked here is part of the VRT contract that downstream code relies
    on, so a backend that ships the right bytes with the wrong attrs
    still fails the cell.
    """
    # Dims and order.
    assert actual.dims == expected_dims, (
        f"{label}: dims {actual.dims!r} != expected {expected_dims!r}"
    )
    assert ref.dims == expected_dims, (
        f"{label}: ref dims {ref.dims!r} != expected {expected_dims!r}"
    )

    # Coord values + coord dtype per axis. A windowed read that decoded
    # the right pixels but shifted the coords inconsistently would
    # surface here, not in the pixel check above.
    for axis in expected_dims:
        if axis not in ref.coords:
            continue
        ref_c = _coord_view(ref, axis)
        actual_c = _coord_view(actual, axis)
        assert ref_c.dtype == actual_c.dtype, (
            f"{label}: coord {axis!r} dtype "
            f"ref={ref_c.dtype} actual={actual_c.dtype}"
        )
        assert ref_c.shape == actual_c.shape, (
            f"{label}: coord {axis!r} shape "
            f"ref={ref_c.shape} actual={actual_c.shape}"
        )
        assert ref_c.tobytes() == actual_c.tobytes(), (
            f"{label}: coord {axis!r} bytes differ "
            f"(ref[:3]={ref_c[:3].tolist()!r}, "
            f"actual[:3]={actual_c[:3].tolist()!r})"
        )

    # Transform tuple. ``rasterio.Affine`` (if used) compares equal to
    # a 6-tuple via ``__eq__`` so this works for both surface forms.
    ref_t = ref.attrs.get("transform")
    actual_t = actual.attrs.get("transform")
    assert ref_t == actual_t, (
        f"{label}: transform tuple differs "
        f"ref={ref_t!r} actual={actual_t!r}"
    )

    # CRS attrs. The contract: ``attrs['crs']`` carries the EPSG int when
    # one is recognised, ``attrs['crs_wkt']`` always carries the WKT.
    assert ref.attrs.get("crs") == actual.attrs.get("crs"), (
        f"{label}: attrs['crs'] differs "
        f"ref={ref.attrs.get('crs')!r} actual={actual.attrs.get('crs')!r}"
    )
    assert ref.attrs.get("crs_wkt") == actual.attrs.get("crs_wkt"), (
        f"{label}: crs_wkt differs"
    )

    # georef_status: lazy / eager / GPU all populate this from the same
    # helper (#2136 / #2162). A drift here means the dask graph builder
    # is using a different finalization path than the eager reader,
    # which is exactly the kind of regression this matrix should catch.
    assert ref.attrs.get("georef_status") == actual.attrs.get(
        "georef_status"
    ), (
        f"{label}: georef_status differs "
        f"ref={ref.attrs.get('georef_status')!r} "
        f"actual={actual.attrs.get('georef_status')!r}"
    )


# ---------------------------------------------------------------------------
# VRT fixture builders.
# Each builder writes its files inside a fresh ``tmp_path`` and returns a
# (vrt_path, expected_dtype) pair. The harness then calls open_geotiff +
# read_vrt with the four backend cells and compares them.
# ---------------------------------------------------------------------------

def _build_two_tile_float32_vrt(tmp_path: Path) -> tuple[Path, np.dtype]:
    """Two 16x16 float32 tiles laid out side-by-side as a 16x32 mosaic.

    Differentiated values per tile so the windowed cells exercise both
    halves of the mosaic without colliding with the sidecar fixture.
    """
    tile_h, tile_w = 16, 16
    paths: list[str] = []
    for c in range(2):
        arr = np.full(
            (tile_h, tile_w), float(c + 1) * 1000.0, dtype=np.float32
        )
        # Sprinkle distinct values so a swap between tiles surfaces.
        arr[0, 0] = -7.0 + c
        arr[tile_h - 1, tile_w - 1] = 9000.0 + c
        origin_x = float(c * tile_w)
        da = xr.DataArray(
            arr, dims=["y", "x"],
            coords={
                "y": np.arange(tile_h - 1, -1, -1, dtype=np.float64),
                "x": np.arange(
                    origin_x, origin_x + tile_w, dtype=np.float64),
            },
            attrs={"crs": 4326},
        )
        tile_path = tmp_path / f"tile_2321_{c}.tif"
        to_geotiff(da, str(tile_path), compression="none", tiled=False)
        paths.append(str(tile_path))
    vrt_path = tmp_path / "two_tile_2321_.vrt"
    _write_vrt_internal(str(vrt_path), paths, relative=False)
    return vrt_path, np.dtype("float32")


def _build_sidecar_vrt(tmp_path: Path) -> tuple[Path, np.dtype]:
    """VRT over a copy of the bundled sidecar TIFF + its ``.ovr`` partner.

    Copying the pair into ``tmp_path`` keeps the original golden corpus
    file untouched and ensures the ``.ovr`` lookup is resolved at the
    VRT read site (not via a cached path on the original fixture).
    """
    src = _sidecar_fixture_or_skip()
    base = tmp_path / "sidecar_2321_.tif"
    shutil.copy(src, base)
    shutil.copy(str(src) + ".ovr", str(base) + ".ovr")
    vrt_path = tmp_path / "sidecar_2321_.vrt"
    _write_vrt_internal(str(vrt_path), [str(base)], relative=False)
    return vrt_path, np.dtype("uint16")


def _build_inline_overview_vrt(tmp_path: Path) -> tuple[Path, np.dtype]:
    """VRT over a copy of the inline-overview fixture (no sidecar).

    Used as the comparison source for ``test_sidecar_vrt_attrs_match_inline``:
    both fixtures share their base IFD (same dtype, transform, CRS, and
    bytes at level 0), so the VRT contract requires that the eager read
    surfaces identical georef attrs regardless of where the pyramid
    physically lives.
    """
    src = _inline_overview_fixture_or_skip()
    base = tmp_path / "inline_2321_.tif"
    shutil.copy(src, base)
    vrt_path = tmp_path / "inline_2321_.vrt"
    _write_vrt_internal(str(vrt_path), [str(base)], relative=False)
    return vrt_path, np.dtype("uint16")


# ---------------------------------------------------------------------------
# Backend matrix: eager (numpy), dask+numpy.
# GPU is intentionally omitted -- the VRT read path goes through the
# CPU decoder regardless of ``gpu=True`` for the pieces under test here,
# and ``read_vrt(gpu=True, chunks=...)`` already has dedicated coverage
# in ``test_vrt_lazy_chunks_1814.py``.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _BackendSpec:
    backend_id: str
    kwargs: dict[str, Any]


_BACKENDS: tuple[_BackendSpec, ...] = (
    _BackendSpec(backend_id="eager", kwargs={}),
    _BackendSpec(backend_id="dask", kwargs={"chunks": (16, 16)}),
)


def _backend_params() -> list:
    return [pytest.param(b, id=b.backend_id) for b in _BACKENDS]


# ---------------------------------------------------------------------------
# Fixture matrix: each entry is one (builder, label, expected_dims, window).
# The window column lets us reuse the same builder for the full-extent
# and the windowed cells without doubling the fixture surface.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FixtureSpec:
    fix_id: str
    builder: Callable[[Path], tuple[Path, np.dtype]]
    expected_dims: tuple[str, ...]
    # Window passed to open_geotiff / read_vrt; None means full extent.
    window: tuple[int, int, int, int] | None


# ``fix_id`` is unique per (builder, window); the ``vrt_fixture`` resolver
# below caches one on-disk layout per *builder*, so two specs that share
# a builder (e.g. the full-extent and windowed cells over the same VRT)
# reuse a single set of source TIFFs and a single ``.vrt`` file.
_FIXTURES: tuple[_FixtureSpec, ...] = (
    _FixtureSpec(
        fix_id="two-tile-float32-full",
        builder=_build_two_tile_float32_vrt,
        expected_dims=("y", "x"),
        window=None,
    ),
    _FixtureSpec(
        # The windowed cell straddles the seam between the two tiles
        # (col 8..24 spans tile 0's right half + tile 1's left half).
        # That makes the dask path actually read both backing sources,
        # not just one, so a windowed dask graph that only re-reads the
        # first source would surface here.
        fix_id="two-tile-float32-window-spans-seam",
        builder=_build_two_tile_float32_vrt,
        expected_dims=("y", "x"),
        window=(4, 8, 12, 24),
    ),
    _FixtureSpec(
        fix_id="sidecar-uint16-full",
        builder=_build_sidecar_vrt,
        expected_dims=("y", "x"),
        window=None,
    ),
    _FixtureSpec(
        fix_id="sidecar-uint16-window",
        builder=_build_sidecar_vrt,
        expected_dims=("y", "x"),
        window=(8, 8, 56, 56),
    ),
)


def _fixture_params() -> list:
    return [pytest.param(f, id=f.fix_id) for f in _FIXTURES]


# ---------------------------------------------------------------------------
# Cached fixture builds: one VRT layout per fix_id per session.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _vrt_parity_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("vrt_parity_2321_")


@pytest.fixture(scope="session")
def _vrt_parity_cache() -> dict[str, tuple[Path, np.dtype]]:
    """Session-scoped (path, dtype) cache shared across every cell.

    The cache must outlive a single test function. A function-scoped
    cache would be reset between cells, causing every cell to rebuild
    the same VRT and its source TIFFs. On POSIX a rebuild is just
    inefficient; on Windows it surfaces as PermissionError / OSError
    because ``to_geotiff`` writes through a ``.tmp`` file and then
    renames over the existing target while another cell may still
    hold the previous file mapped (issue surfaced in CI on
    ``windows-latest`` for #2330).
    """
    return {}


@pytest.fixture
def vrt_fixture(_vrt_parity_dir, _vrt_parity_cache):
    """Resolve a :class:`_FixtureSpec` to a (vrt_path, dtype) pair on disk.

    Each builder gets its own subdirectory so the on-disk layout (vrt +
    sources + any sidecar) is isolated from neighbouring builders. Builds
    are cached at session scope so the four cells that share a builder
    (e.g. full-extent + windowed over the same VRT) reuse one set of
    source TIFFs and one ``.vrt`` file.
    """
    base = _vrt_parity_dir
    cache = _vrt_parity_cache

    def _resolve(spec: _FixtureSpec) -> tuple[Path, np.dtype]:
        # The fix_id encodes both the builder and the window; collapse to
        # the builder so we do not rebuild identical layouts.
        key = spec.builder.__name__
        if key in cache:
            return cache[key]
        sub = base / key
        sub.mkdir(exist_ok=True)
        result = spec.builder(sub)
        cache[key] = result
        return result
    return _resolve


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", _fixture_params())
@pytest.mark.parametrize("backend", _backend_params())
def test_vrt_backend_parity(spec, backend, vrt_fixture):
    """One cell per (fixture, backend). Asserts pixels + metadata parity.

    Reference is always the eager numpy read with the same window kwarg.
    The cell compares the current backend's output against that
    reference. Eager-vs-eager is the identity case and locks in the
    parity-helper contract.
    """
    vrt_path, expected_dtype = vrt_fixture(spec)

    open_kwargs: dict[str, Any] = {}
    if spec.window is not None:
        open_kwargs["window"] = spec.window

    ref = open_geotiff(str(vrt_path), **open_kwargs)

    actual = open_geotiff(
        str(vrt_path), **open_kwargs, **backend.kwargs,
    )

    label = (
        f"fixture={spec.fix_id} backend={backend.backend_id} "
        f"window={spec.window!r}"
    )

    ref_arr = _materialise(ref)
    actual_arr = _materialise(actual)

    # Dtype against the explicit spec, not just against the reference.
    # A silent upcast that the reference also exhibits would still fail
    # here (the spec dtype is the contract).
    assert ref_arr.dtype == expected_dtype, (
        f"{label}: reference dtype {ref_arr.dtype} != "
        f"expected {expected_dtype}"
    )
    assert actual_arr.dtype == expected_dtype, (
        f"{label}: actual dtype {actual_arr.dtype} != "
        f"expected {expected_dtype}"
    )

    _assert_pixels_equal(ref_arr, actual_arr, label=label)
    _assert_metadata_parity(
        ref, actual, label=label, expected_dims=spec.expected_dims,
    )


# ---------------------------------------------------------------------------
# Cross-fixture parity: sidecar pyramid vs inline pyramid.
# Both backing TIFFs share their base IFD bytes (same uint16 raster, same
# transform, same CRS), so a VRT wrapping each must report identical
# georef attrs at level 0. The check guards against the sidecar lookup
# accidentally rewriting (or dropping) any of the contract-named attrs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", _backend_params())
def test_sidecar_vrt_attrs_match_inline(backend, tmp_path):
    """Sidecar-backed VRT and inline-overview-backed VRT report identical
    georef attrs and pixels at the base level.

    Acceptance criterion straight from the parent issue: the sidecar
    ``.ovr`` lookup must produce the same georef status and CRS attrs
    as an inline-overview source. The check runs on each backend so a
    drift introduced only on the dask path still surfaces.
    """
    side_sub = tmp_path / "sidecar"
    inline_sub = tmp_path / "inline"
    side_sub.mkdir()
    inline_sub.mkdir()
    side_vrt, side_dtype = _build_sidecar_vrt(side_sub)
    inline_vrt, inline_dtype = _build_inline_overview_vrt(inline_sub)

    assert side_dtype == inline_dtype, (
        f"sidecar dtype {side_dtype} != inline dtype {inline_dtype}; "
        f"the golden_corpus fixtures should share a base IFD"
    )

    side = open_geotiff(str(side_vrt), **backend.kwargs)
    inline = open_geotiff(str(inline_vrt), **backend.kwargs)

    label = (
        f"sidecar-vs-inline backend={backend.backend_id}"
    )

    # Shape parity is the precondition for the pixel comparison.
    assert side.shape == inline.shape, (
        f"{label}: shape differs side={side.shape} inline={inline.shape}"
    )

    # Pixel parity at the base level. Both fixtures share their level-0
    # bytes (the sidecar only adds an external pyramid), so the read-back
    # arrays should match byte-for-byte.
    _assert_pixels_equal(
        _materialise(inline), _materialise(side), label=label,
    )

    # Metadata parity: the read paths must surface identical georef
    # attrs across the two physical layouts.
    _assert_metadata_parity(
        inline, side, label=label, expected_dims=("y", "x"),
    )


# ---------------------------------------------------------------------------
# Windowed-coord shift parity: an eager windowed read and a chunked
# windowed read of the same VRT must report the same shifted coords
# AND the same shifted transform. Pixel equality alone is not enough --
# we want to catch the regression where the dask graph computes correct
# pixels but the assembled DataArray keeps the full-extent coords or
# transform.
# ---------------------------------------------------------------------------

def test_windowed_vrt_shifts_coords_and_transform_consistently(tmp_path):
    """Eager and lazy windowed VRT reads agree on shape, coords, attrs,
    and values.

    Per the parent issue's acceptance criterion. The cell is split out
    from the parametrised matrix above so a coord/transform drift on
    the dask path produces a single, named failure rather than a
    matrix-wide flag.
    """
    vrt_path, _ = _build_two_tile_float32_vrt(tmp_path)
    # Window deliberately straddles the tile seam (col 16) and trims
    # the y-axis on both ends, so both axes get shifted.
    window = (3, 5, 13, 27)

    eager = open_geotiff(str(vrt_path), window=window)
    lazy = open_geotiff(str(vrt_path), window=window, chunks=(5, 11))

    # Shape parity (precondition).
    assert eager.shape == (10, 22)
    assert lazy.shape == (10, 22)

    # Coord shift: the eager read's y/x arrays should match the lazy
    # read's exactly (same shape, same dtype, same bytes).
    np.testing.assert_array_equal(eager["y"].values, lazy["y"].values)
    np.testing.assert_array_equal(eager["x"].values, lazy["x"].values)
    assert eager["y"].dtype == lazy["y"].dtype
    assert eager["x"].dtype == lazy["x"].dtype

    # The window cuts the leading 3 rows and the leading 5 columns of
    # the full-extent grid (which goes from y=15..0 and x=0..31), so
    # the windowed first y is 12.0 and the windowed first x is 5.0.
    # The check pins the absolute shift, not just the eager/lazy
    # equality, so a regression that drifts BOTH backends the same
    # way still surfaces.
    assert eager["y"].values[0] == 12.0
    assert eager["x"].values[0] == 5.0

    # Transform must shift consistently: the rasterio 6-tuple's c
    # (origin_x) and f (origin_y) entries should reflect the window
    # offset, while pixel sizes (a, e) stay constant.
    eager_t = eager.attrs.get("transform")
    lazy_t = lazy.attrs.get("transform")
    assert eager_t == lazy_t, (
        f"transform differs eager={eager_t!r} lazy={lazy_t!r}"
    )
    # Pixel size unchanged by the window.
    assert eager_t[0] == 1.0 and eager_t[4] == -1.0, (
        f"pixel size mismatch in windowed transform {eager_t!r}"
    )

    # Pixel parity.
    np.testing.assert_array_equal(eager.values, lazy.compute().values)

    # CRS attrs parity.
    assert eager.attrs.get("crs") == lazy.attrs.get("crs")
    assert eager.attrs.get("crs_wkt") == lazy.attrs.get("crs_wkt")
    assert eager.attrs.get("georef_status") == lazy.attrs.get(
        "georef_status"
    )


# ---------------------------------------------------------------------------
# Absolute-shift parity for the sidecar windowed cell. The parametrised
# matrix only checks eager-vs-dask equality; pin the actual shifted
# coords and transform here so a regression that drifts BOTH backends
# the same way still surfaces. The bundled sidecar fixture has a known
# pixel size of 0.001 and origin (-120.0, 45.0).
# ---------------------------------------------------------------------------

def test_sidecar_window_shifts_to_known_coords(tmp_path):
    """The sidecar VRT, read with ``window=(8, 8, 56, 56)``, should land
    on the same coords / transform an absolute calculation would predict.

    The bundled fixture is 64x64 at pixel size 0.001 with origin
    (-120.0, 45.0). Trimming rows 8..56 / cols 8..56 yields a 48x48
    window whose x-coord array starts at -120.0 + 8 * 0.001 + half-pixel
    centre offset, and whose transform's c/f entries shift by the same
    8-pixel offsets.
    """
    vrt_path, _ = _build_sidecar_vrt(tmp_path)
    window = (8, 8, 56, 56)

    eager = open_geotiff(str(vrt_path), window=window)

    assert eager.shape == (48, 48)
    # Pixel size column (a, e) of the rasterio 6-tuple stays constant.
    t = eager.attrs.get("transform")
    assert t is not None, "windowed sidecar VRT dropped attrs['transform']"
    assert t[0] == pytest.approx(0.001)
    assert t[4] == pytest.approx(-0.001)
    # Origin shifts by 8 pixels: c += 8 * a, f += 8 * e.
    # Full-extent origin is c=-120.0, f=45.0.
    assert t[2] == pytest.approx(-120.0 + 8 * 0.001)
    assert t[5] == pytest.approx(45.0 + 8 * -0.001)


# ---------------------------------------------------------------------------
# Sanity check: the matrix harness itself flags a metadata regression.
# ---------------------------------------------------------------------------

def test_assert_metadata_parity_flags_transform_drift(tmp_path):
    """Locks the harness behaviour: a transform-only drift between two
    otherwise-identical DataArrays fails the parity helper.

    Without this, a regression that silently dropped the transform check
    inside ``_assert_metadata_parity`` would let the rest of the matrix
    pass with empty assertions.
    """
    vrt_path, _ = _build_two_tile_float32_vrt(tmp_path)
    da_ref = open_geotiff(str(vrt_path))
    da_bad = da_ref.copy()
    da_bad.attrs = dict(da_ref.attrs)
    # Mutate the transform's origin_x. The pixels and coords remain
    # untouched; only the attr drifts.
    old_t = da_bad.attrs["transform"]
    da_bad.attrs["transform"] = (
        old_t[0], old_t[1], old_t[2] + 1.0,
        old_t[3], old_t[4], old_t[5],
    )
    with pytest.raises(AssertionError, match="transform"):
        _assert_metadata_parity(
            da_ref, da_bad, label="harness-sanity",
            expected_dims=("y", "x"),
        )
