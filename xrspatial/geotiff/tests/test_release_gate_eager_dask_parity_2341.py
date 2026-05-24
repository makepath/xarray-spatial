"""Release gate: eager-vs-dask raster equivalence (PR 1 of 5 of epic #2341).

Epic #2341 calls out the highest release risk for the GeoTIFF surface:
pixels matching while ``attrs``, ``coords``, or ``dims`` silently disagree
between the eager (``open_geotiff``) and lazy (``read_geotiff_dask``)
entry points. Today both paths are documented as ``stable``, but no
single regression test asserts full raster equivalence -- pixels + dims +
coords + the seven release-attr keys -- across the two paths on the same
files.

This module reads each fixture in a representative corpus list once
through ``open_geotiff`` (eager) and once through ``read_geotiff_dask``
(materialised via ``.compute()``), then asserts:

* ``.values`` bit-exact (NaN-aware via ``np.array_equal(..., equal_nan=True)``)
* ``.dims`` equal
* ``.coords`` element-wise equal (dtype + bytes match per axis)
* seven release-attr keys equal:
  ``transform``, ``crs``, ``crs_wkt``, ``nodata``, ``masked_nodata``,
  ``georef_status``, ``raster_type``

The assertions are inlined as small helpers in this module. The four
sibling PRs of epic #2341 (windowed-shifted-transform, overview / sidecar
metadata, stable-codec round-trip, ambiguous-metadata negatives) ship
independently; consolidating the helpers into a shared module is a
follow-up once all five have landed and the common shape has settled.

The corpus covers the four scenarios called out in the issue:

* integer dtype with explicit integer nodata sentinel
* float dtype with NaN nodata
* MinIsWhite photometric (no explicit nodata tag)
* masked-nodata lifecycle: the same integer-sentinel fixture read with
  ``mask_nodata=False`` so the raw uint sentinel branch is pinned in
  parity against the default ``mask_nodata=True`` branch (which the
  integer-nodata row above already covers)

Out of scope (sibling PRs of epic #2341):

* Windowed-read shifted-transform parity (PR 2 of 5).
* Overview / sidecar metadata survival (PR 3 of 5).
* Stable-codec round-trip (PR 4 of 5).
* Negative tests for ambiguous metadata (PR 5 of 5).
"""
from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("dask")

from xrspatial.geotiff import open_geotiff, read_geotiff_dask  # noqa: E402

# Corpus fixtures live under ``golden_corpus/fixtures``; the same
# directory the wider parity matrix and the per-backend golden tests
# already use.
_FIXTURES_DIR = (
    pathlib.Path(__file__).resolve().parent / "golden_corpus" / "fixtures"
)

# Chunk size for the dask reads. The corpus fixtures used here are
# 64x64 or smaller, so a chunk of 32 produces either a 2x2 chunk grid
# or a single chunk depending on the fixture. Either way the dask
# plumbing fires.
_CHUNK_SIZE = 32

# The seven release-attr keys the parity contract pins. Drift on any
# of these between the eager and dask paths is a release blocker; see
# the module docstring for the rationale.
_RELEASE_ATTR_KEYS: tuple[str, ...] = (
    "transform",
    "crs",
    "crs_wkt",
    "nodata",
    "masked_nodata",
    "georef_status",
    "raster_type",
)


# ---------------------------------------------------------------------------
# Corpus selection
# ---------------------------------------------------------------------------

# One ``pytest.param`` per fixture scenario. ``open_kwargs`` carries any
# extra kwargs (e.g. ``mask_nodata=True``) applied to both the eager and
# dask reads so the masked-nodata-lifecycle row exercises the same
# masking semantics on both paths.
_CORPUS = [
    pytest.param(
        "nodata_int_sentinel_uint16",
        {},
        id="int-dtype-nodata",
    ),
    pytest.param(
        "nodata_nan_float32",
        {},
        id="float-dtype-nan-nodata",
    ),
    pytest.param(
        "nodata_miniswhite_uint8",
        {},
        id="miniswhite",
    ),
    # ``mask_nodata=False`` is the contrast cell to the first row's
    # default ``mask_nodata=True``: the raw uint16 sentinel is preserved
    # and ``masked_nodata`` flips to ``False``. Together the two cells
    # pin both sides of the nodata lifecycle on the same fixture, which
    # is the silent-disagreement case the issue calls out.
    pytest.param(
        "nodata_int_sentinel_uint16",
        {"mask_nodata": False},
        id="masked-nodata-lifecycle",
    ),
]


# ---------------------------------------------------------------------------
# Inlined helpers (per issue: no new shared helper module in this PR)
# ---------------------------------------------------------------------------

def _materialise(da: xr.DataArray) -> np.ndarray:
    """Return a host-side numpy view of ``da.values``.

    For an eager numpy-backed DataArray this is a straight ``np.asarray``;
    for a dask-backed DataArray ``.values`` triggers ``.compute()`` so
    the result is the materialised numpy array. The eager / lazy split
    is hidden here so the assertion call sites stay symmetric. Kept as
    a named helper (rather than inlined) so the sibling PRs of epic
    #2341 can copy the same shape when they land their own gates.
    """
    return np.asarray(da.values)


def _assert_values_equal(eager: xr.DataArray, lazy: xr.DataArray) -> None:
    """Bit-exact NaN-aware comparison of pixel values.

    Integer dtypes go through ``np.array_equal`` directly; float dtypes
    use ``equal_nan=True`` so a NaN-marked nodata cell compares equal to
    itself across paths. A dtype mismatch fails first with an explicit
    message because the float / int divergence is the single most
    informative diff when ``mask_nodata=True`` flips a row.
    """
    assert eager.dtype == lazy.dtype, (
        f"pixel dtype differs: eager={eager.dtype} lazy={lazy.dtype}"
    )
    eager_px = _materialise(eager)
    lazy_px = _materialise(lazy)
    assert eager_px.shape == lazy_px.shape, (
        f"pixel shape differs: eager={eager_px.shape} lazy={lazy_px.shape}"
    )
    equal_nan = eager_px.dtype.kind == "f"
    if not np.array_equal(eager_px, lazy_px, equal_nan=equal_nan):
        raise AssertionError(
            "pixel values differ between eager and dask reads "
            f"(dtype={eager_px.dtype}, equal_nan={equal_nan})"
        )


def _assert_dims_equal(eager: xr.DataArray, lazy: xr.DataArray) -> None:
    """Dims tuple matches exactly between the two paths."""
    assert eager.dims == lazy.dims, (
        f"dims differ: eager={eager.dims!r} lazy={lazy.dims!r}"
    )


def _assert_coords_equal(eager: xr.DataArray, lazy: xr.DataArray) -> None:
    """Per-axis coord dtype + byte-level equality.

    Coords drive transform reconstruction downstream, so a sub-ULP
    divergence still means a different transform. The bytewise compare
    catches a dtype-preserving rounding regression that ``allclose``
    would let through.
    """
    eager_coord_names = set(eager.coords)
    lazy_coord_names = set(lazy.coords)
    assert eager_coord_names == lazy_coord_names, (
        f"coord name set differs: "
        f"only-in-eager={sorted(eager_coord_names - lazy_coord_names)} "
        f"only-in-lazy={sorted(lazy_coord_names - eager_coord_names)}"
    )
    for axis in eager_coord_names:
        eager_c = np.asarray(eager.coords[axis].values)
        lazy_c = np.asarray(lazy.coords[axis].values)
        assert eager_c.dtype == lazy_c.dtype, (
            f"coord {axis!r} dtype differs: "
            f"eager={eager_c.dtype} lazy={lazy_c.dtype}"
        )
        assert eager_c.shape == lazy_c.shape, (
            f"coord {axis!r} shape differs: "
            f"eager={eager_c.shape} lazy={lazy_c.shape}"
        )
        assert eager_c.tobytes() == lazy_c.tobytes(), (
            f"coord {axis!r} bytes differ between eager and dask reads"
        )


def _is_nan_sentinel(value: Any) -> bool:
    """True when ``value`` is a NaN, regardless of scalar type.

    ``float('nan') != float('nan')`` by IEEE-754, so the nodata
    comparison needs an explicit NaN-aware branch. Accepts python
    floats, numpy scalars, and anything castable to ``float``; returns
    ``False`` for non-numeric values (including ``None``) so the
    caller falls through to the strict ``==`` branch.
    """
    if value is None:
        return False
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _attr_equal(a: Any, b: Any) -> bool:
    """Compare two attr values, treating NaN as equal to NaN.

    Notable divergence from ``test_backend_full_parity_2211.py``: the
    transform 6-tuple of floats is compared bit-exact here (via the
    tuple-recursion branch below), where the sibling gate allows a
    1e-9 ULP tolerance. Bit-exact is the contract the issue calls for
    on the same-file eager-vs-dask axis; the wider gate has to absorb
    a hypothetical future cross-backend float-rounding op (e.g. a GPU
    decode path) that does not exist on either of the two paths here.
    """
    if _is_nan_sentinel(a) and _is_nan_sentinel(b):
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and np.array_equal(a, b)
        )
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(_attr_equal(x, y) for x, y in zip(a, b))
    return a == b


def _assert_release_attrs_equal(
    eager: xr.DataArray, lazy: xr.DataArray,
) -> None:
    """Each of the seven release-attr keys agrees on presence + value.

    An attr absent on the eager read must also be absent on the dask
    read, and vice versa. This catches the silent-disagreement case the
    issue calls out: pixels and dims line up while one path stamps an
    attr the other omits.
    """
    for key in _RELEASE_ATTR_KEYS:
        in_eager = key in eager.attrs
        in_lazy = key in lazy.attrs
        assert in_eager == in_lazy, (
            f"release attr {key!r} presence differs: "
            f"eager={in_eager} lazy={in_lazy}"
        )
        if not in_eager:
            continue
        eager_v = eager.attrs[key]
        lazy_v = lazy.attrs[key]
        assert _attr_equal(eager_v, lazy_v), (
            f"release attr {key!r} value differs: "
            f"eager={eager_v!r} lazy={lazy_v!r}"
        )


# ---------------------------------------------------------------------------
# The parity gate
# ---------------------------------------------------------------------------

@pytest.mark.release_gate
@pytest.mark.parametrize("fixture_id, open_kwargs", _CORPUS)
def test_release_gate_eager_dask_full_parity(
    fixture_id: str, open_kwargs: dict,
) -> None:
    """Eager and dask reads of the same file agree on the full contract.

    Reads ``fixture_id`` once via ``open_geotiff`` and once via
    ``read_geotiff_dask``, then asserts pixel values, dims, coords, and
    the seven release-attr keys all match. The dask result is
    materialised via ``.values`` so the comparison is between concrete
    arrays, not between graph-vs-array.
    """
    path = _FIXTURES_DIR / f"{fixture_id}.tif"
    if not path.exists():
        pytest.skip(
            f"fixture {fixture_id!r} has no .tif on disk; run "
            f"`python -m xrspatial.geotiff.tests.golden_corpus.generate`"
        )

    eager = open_geotiff(str(path), **open_kwargs)
    lazy = read_geotiff_dask(str(path), chunks=_CHUNK_SIZE, **open_kwargs)

    _assert_values_equal(eager, lazy)
    _assert_dims_equal(eager, lazy)
    _assert_coords_equal(eager, lazy)
    _assert_release_attrs_equal(eager, lazy)


def test_release_gate_corpus_is_non_empty() -> None:
    """The corpus list must not silently shrink to zero rows.

    A parametrize argument list that empties out (e.g. a bad refactor
    that filters every entry) would cause pytest to collect zero cells
    and the matrix would pass vacuously. Pin the row count so a stale
    refactor surfaces here instead.
    """
    assert len(_CORPUS) == 4, (
        f"corpus row count drifted: expected 4 scenarios "
        f"(int-nodata, float-nan-nodata, miniswhite, masked-nodata-lifecycle), "
        f"got {len(_CORPUS)}"
    )
