"""Required backend parity matrix per high-risk fixture (issue #1985).

Single source of truth for "does backend X still match the reference on
fixture Z." Existing scattered parity files (``test_attrs_parity_1548.py``,
``test_backend_pixel_parity_matrix_1813.py``, etc.) stay in place as
named regression markers for their bug numbers; new parity assertions go
here.

Harness contract
----------------

Every cell calls a single :func:`assert_parity` helper that checks the
same set of fields on the same fixture across every wired-up backend:

* pixel array (byte-equal for int, NaN-aware closeness for float)
* dtype
* dims and dim order
* coord values and coord dtype (per axis)
* transform tuple (rasterio 6-tuple)
* CRS as EPSG int when present, plus ``crs_wkt`` string
* declared nodata sentinel
* masking state (``attrs.get('masked_nodata')`` once issue #1988 lands;
  until then we accept absence as "unspecified" rather than asserting a
  value)
* a small subset of canonical attrs whose round-trip semantics are
  already settled in the module (``raster_type``, ``transform``,
  ``crs``, ``crs_wkt``). The "selected canonical attrs" list from issue
  #1985 will be tightened in a follow-up once issue #1984's PR 4 lands
  the contract version stamp.

Scope of this PR (matrix scaffold)
----------------------------------

PR 1 of the issue #1985 plan: harness end-to-end on one fixture (int16
single-band, no nodata) against one backend (eager numpy). Follow-up
PRs add fixtures and backends without touching the harness itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

# Each backend entry maps a stable id to the ``open_geotiff`` kwargs that
# select it. Follow-up PRs add dask+numpy, gpu, dask+gpu, http, and vrt
# entries. The matrix is parametrized over this list so wiring up a new
# backend means appending one row, not editing every test.
_BACKENDS = [
    pytest.param({}, id="numpy"),
    pytest.param({"chunks": 16}, id="dask+numpy"),
]


# ---------------------------------------------------------------------------
# Fixture descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FixtureSpec:
    """Declarative description of one high-risk fixture.

    Attributes
    ----------
    fix_id
        Stable id used in the parametrize call. Appears in test names.
    dtype
        Pixel dtype of the underlying array (and the on-disk SampleFormat).
    expected_dims
        Tuple of dim names in expected order.
    expected_crs_epsg
        EPSG int the read path should emit under ``attrs['crs']``.
    expected_nodata
        Declared nodata sentinel that the read path should surface under
        ``attrs['nodata']``. ``None`` means the fixture has no declared
        nodata; the harness then asserts ``'nodata' not in attrs``.
    expected_masked
        Tri-valued. ``True`` / ``False`` pin ``attrs['masked_nodata']``
        once issue #1988 has landed and a backend opts in. ``None``
        means "do not assert" -- used for fixtures without nodata or
        until the masking-state attr is wired up.
    builder
        Callable receiving a directory ``Path`` and the resolved target
        ``Path`` (cache-key filename). Writes the file at ``target`` and
        returns the final on-disk path. Most builders just return
        ``target`` unchanged; sidecar-producing builders (e.g. a
        ``.vrt`` over auxiliary tiles) may write multiple files and
        return the entry path.
    """

    fix_id: str
    dtype: np.dtype
    expected_dims: tuple[str, ...]
    expected_crs_epsg: int | None
    expected_nodata: object
    expected_masked: bool | None
    builder: Callable[[Path, Path], Path]


def _wrap_2d(arr: np.ndarray, *, crs: int | None) -> xr.DataArray:
    """Wrap a 2-D numpy array as a writer-ready DataArray.

    Uses unit-pixel descending-y coords (``y = height-1 .. 0``,
    ``x = 0 .. width-1``). The read-back transform tuple for a height-H
    fixture is ``(1.0, 0.0, -0.5, 0.0, -1.0, H - 0.5)`` -- the half-pixel
    offsets come from the PixelIsArea convention (origin is the pixel
    edge, coords are pixel centres) that the writer round-trips.
    """
    height, width = arr.shape
    da = xr.DataArray(
        arr, dims=["y", "x"],
        coords={
            "y": np.arange(height - 1, -1, -1, dtype=np.float64),
            "x": np.arange(width, dtype=np.float64),
        },
        attrs={},
    )
    if crs is not None:
        da.attrs["crs"] = crs
    return da


def _build_int16_single_band(dir_path: Path, target: Path) -> Path:
    """High-risk fixture: int16 single-band stripped TIFF, EPSG:4326, no nodata.

    Integer dtype catches sign-extension and predictor bugs that float
    fixtures hide. Seeded so the byte-equal pixel assertion is reproducible.
    """
    del dir_path  # builder writes directly to the resolved target path
    rng = np.random.default_rng(seed=19850)
    arr = rng.integers(-30000, 30000, size=(32, 32), dtype=np.int16)
    to_geotiff(
        _wrap_2d(arr, crs=4326), str(target),
        compression="none", tiled=False,
    )
    return target


_FIXTURES: list[_FixtureSpec] = [
    _FixtureSpec(
        fix_id="int16-single-band",
        dtype=np.dtype("int16"),
        expected_dims=("y", "x"),
        expected_crs_epsg=4326,
        expected_nodata=None,
        expected_masked=None,
        builder=_build_int16_single_band,
    ),
]


@pytest.fixture(scope="session")
def _parity_matrix_dir(tmp_path_factory):
    """Session-scoped scratch dir, one write per fixture id.

    Tests reuse files across cells (12 fixtures times 6 backends would
    otherwise be 72 redundant writes per pytest run).
    """
    return tmp_path_factory.mktemp("parity_matrix_1985")


@pytest.fixture
def parity_fixture(_parity_matrix_dir):
    """Resolve a :class:`_FixtureSpec` to an on-disk path.

    Files are cached across the session: a fixture already present on
    disk is returned without rewriting.
    """
    dir_path = _parity_matrix_dir

    def _resolve(spec: _FixtureSpec) -> Path:
        # ``fix_id`` may contain ``/`` once dtype/compression-keyed
        # fixtures land (cf. ``test_backend_pixel_parity_matrix_1813.py``
        # ids like ``stripped/int16/none``). Flatten to a single
        # filename so the resolver never creates subdirectories.
        safe_id = spec.fix_id.replace("/", "-")
        path = dir_path / f"parity_1985_{safe_id}.tif"
        if path.exists():
            return path
        return spec.builder(dir_path, path)
    return _resolve


# ---------------------------------------------------------------------------
# Materialisation + comparison helpers
# ---------------------------------------------------------------------------

def _materialise(da: xr.DataArray) -> np.ndarray:
    """Return a numpy view of ``da.data`` regardless of backend.

    Handles dask (``.compute``) and cupy (``.get``) without forcing the
    caller to know which backend produced the DataArray.
    """
    raw = da.data
    if hasattr(raw, "compute"):
        raw = raw.compute()
    if hasattr(raw, "get"):
        raw = raw.get()
    return np.asarray(raw)


def _coord_view(da: xr.DataArray, name: str) -> np.ndarray:
    return np.asarray(da.coords[name].values)


def _assert_pixels_equal(ref: np.ndarray, actual: np.ndarray, *, label: str) -> None:
    """Pixel equality, dtype-aware.

    Integer arrays must be byte-identical; float arrays compare NaN-aware
    with ``equal_nan=True``. Diverging dtypes always fail -- a backend
    that silently upcasts has a bug.
    """
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


# ---------------------------------------------------------------------------
# The matrix cell
# ---------------------------------------------------------------------------

def assert_parity(
    da: xr.DataArray,
    spec: _FixtureSpec,
    *,
    path: Path,
    label: str,
) -> None:
    """Assert every parity field for one (fixture, backend) cell.

    Run against an already-read DataArray rather than re-opening here so
    the same helper applies to both ``open_geotiff(path, **kwargs)`` and
    the explicit ``read_geotiff_dask`` / ``read_geotiff_gpu`` /
    ``read_vrt`` entry points wired up in follow-up PRs. ``path`` is the
    on-disk fixture, used to build the eager-numpy reference.

    The eager-numpy read of the same file is the reference for the pixel
    array, coord values, dims, and transform tuple. ``spec.dtype`` and
    ``spec.expected_crs_epsg`` / ``spec.expected_nodata`` are asserted
    against the actual independently of the reference, so a bug that
    silently changes them in *every* backend still fails this cell.
    """
    ref = open_geotiff(str(path))

    # Pixel array, dtype, shape.
    actual_arr = _materialise(da)
    _assert_pixels_equal(
        _materialise(ref), actual_arr, label=label,
    )

    # Dtype against the spec, not just against the reference. Catches a
    # silent upcast that the reference would also exhibit.
    assert actual_arr.dtype == spec.dtype, (
        f"{label}: dtype {actual_arr.dtype} != spec dtype {spec.dtype}"
    )

    # Dims + order.
    assert da.dims == spec.expected_dims, (
        f"{label}: dims {da.dims!r} != expected {spec.expected_dims!r}"
    )

    # Coord values and coord dtype, per axis.
    for axis in spec.expected_dims:
        if axis not in ref.coords:
            continue
        ref_c = _coord_view(ref, axis)
        actual_c = _coord_view(da, axis)
        assert ref_c.dtype == actual_c.dtype, (
            f"{label}: coord {axis!r} dtype "
            f"ref={ref_c.dtype} actual={actual_c.dtype}"
        )
        assert ref_c.tobytes() == actual_c.tobytes(), (
            f"{label}: coord {axis!r} bytes differ"
        )

    # Transform tuple.
    ref_t = ref.attrs.get("transform")
    actual_t = da.attrs.get("transform")
    assert ref_t == actual_t, (
        f"{label}: transform tuple differs ref={ref_t!r} actual={actual_t!r}"
    )

    # CRS: EPSG int + WKT string.
    if spec.expected_crs_epsg is not None:
        assert da.attrs.get("crs") == spec.expected_crs_epsg, (
            f"{label}: attrs['crs'] {da.attrs.get('crs')!r} != "
            f"expected {spec.expected_crs_epsg!r}"
        )
    ref_wkt = ref.attrs.get("crs_wkt")
    actual_wkt = da.attrs.get("crs_wkt")
    assert ref_wkt == actual_wkt, (
        f"{label}: crs_wkt differs ref={ref_wkt!r} actual={actual_wkt!r}"
    )

    # Nodata sentinel + masking state.
    if spec.expected_nodata is None:
        assert "nodata" not in da.attrs, (
            f"{label}: fixture declares no nodata but attrs['nodata']="
            f"{da.attrs.get('nodata')!r}"
        )
    else:
        assert da.attrs.get("nodata") == spec.expected_nodata, (
            f"{label}: attrs['nodata'] {da.attrs.get('nodata')!r} != "
            f"expected {spec.expected_nodata!r}"
        )

    # Masking state. ``attrs['masked_nodata']`` is the post-#1988 attr.
    # Until it ships, ``None`` means "do not assert" and the matrix
    # tolerates absence. The follow-up PR that wires #1988-aware
    # fixtures will tighten this.
    if spec.expected_masked is not None:
        actual_masked = da.attrs.get("masked_nodata")
        assert actual_masked == spec.expected_masked, (
            f"{label}: attrs['masked_nodata'] {actual_masked!r} != "
            f"expected {spec.expected_masked!r}"
        )

    # Selected canonical attrs: the reference and the actual agree on
    # presence and value. The list is intentionally narrow until issue
    # #1984's PR 4 (contract version stamp) lands and we can extend it.
    canonical_keys = ("raster_type", "transform", "crs", "crs_wkt")
    for key in canonical_keys:
        ref_v = ref.attrs.get(key)
        actual_v = da.attrs.get(key)
        assert ref_v == actual_v, (
            f"{label}: canonical attr {key!r} differs "
            f"ref={ref_v!r} actual={actual_v!r}"
        )


# ---------------------------------------------------------------------------
# The single matrix test entry point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spec", _FIXTURES, ids=lambda s: s.fix_id)
@pytest.mark.parametrize("backend_kwargs", _BACKENDS)
def test_backend_parity_matrix(parity_fixture, spec, backend_kwargs):
    """One cell per (fixture, backend). Asserts every parity field.

    A new backend or fixture lights up automatically on the next pytest
    run -- no per-cell test function needed.
    """
    path = parity_fixture(spec)
    da = open_geotiff(str(path), **backend_kwargs)
    label = f"fixture={spec.fix_id} backend={backend_kwargs}"
    assert_parity(da, spec, path=path, label=label)
