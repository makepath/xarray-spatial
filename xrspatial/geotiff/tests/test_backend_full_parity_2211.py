"""Backend full-parity gate (PR-F of #2211 / closes #2229).

A single table-driven parity test that runs the same fixture through
every read backend ``xrspatial.geotiff`` ships and asserts the full
observable contract on each output: pixel values, dims and coords,
transform/georef attrs, CRS attrs, nodata sentinel and dtype, and a
curated set of canonical metadata attrs.

This module is the **contract gate** for the geotiff refactor epic
#2211. Earlier phases (PRs A through E) shared finalization helpers
across the eager and lazy backends so the four CPU/GPU x eager/dask
paths produce the same DataArray. PR-F locks that result in: any drift
between backends on the curated attrs set raises here, so a refactor
in one backend that silently changes an attr cannot land without an
update to this test.

Design notes
------------

* The ``eager_numpy`` read of each fixture is the reference every
  other backend is compared against. The reference is read once per
  fixture and cached for the module's lifetime so a 6-backend
  matrix does not multiply IO by 6x.
* Backends covered (per #2229):
    - ``eager_numpy``  -- ``open_geotiff(path)`` (also the reference)
    - ``dask_numpy``   -- ``open_geotiff(path, chunks=...)``
    - ``gpu``          -- ``open_geotiff(path, gpu=True, on_gpu_failure='strict')``
    - ``dask_gpu``     -- ``open_geotiff(path, gpu=True, chunks=..., on_gpu_failure='strict')``
    - ``vrt_eager``    -- ``open_geotiff(vrt_path)`` (one-source VRT
                          generated per fixture)
    - ``http_fsspec``  -- ``open_geotiff('memory:///corpus/<id>.tif')``
                          through fsspec's in-process memory filesystem
* Fixtures come from the existing golden corpus
  (``xrspatial/geotiff/tests/golden_corpus/``) so the fixture surface
  is the same one Phase 3 (#1930) exercised. The set is filtered to
  fixtures the full parity matrix can compare cleanly today; known
  divergences (e.g. JPEG-YCbCr axis order, MinIsWhite inversion) live
  in ``_INTENTIONAL_SKIPS`` with citations.
* The reference is the ``eager_numpy`` read of the same fixture, taken
  once per fixture and reused across backends. Pixels are compared
  ``allclose`` for float dtypes (NaN-aware) and bit-exact for ints.
* GPU and Dask+GPU rows skip with a **loud** reason when cupy or CUDA
  is missing: the issue text calls this out explicitly so a silent
  collection of zero GPU tests in CI is itself a bug.

Adding a new backend or a new fixture
-------------------------------------

To add a backend: append a ``_Backend`` to ``_BACKENDS`` with its
``read`` callable, an availability predicate, and the list of fixture
ids it cannot handle (with citations). The parametrize matrix lights
it up on the next run.

To add a fixture: drop it into the manifest at
``golden_corpus/manifest.yaml`` and regenerate. The matrix picks it
up automatically.
"""
from __future__ import annotations

import importlib.util
import pathlib
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("yaml")
pytest.importorskip("rasterio")

from xrspatial.geotiff import open_geotiff, write_vrt  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus import generate  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus._marks import (  # noqa: E402
    fast_slow_marks_for,
)


FIXTURES_DIR = (
    pathlib.Path(generate.__file__).resolve().parent / "fixtures"
)

# Chunk size for the dask backends. Most corpus fixtures are 64x64 or
# smaller, so 32 produces either a 2x2 chunk grid or a single chunk;
# either way the dask plumbing fires.
_CHUNK_SIZE = 32


# ---------------------------------------------------------------------------
# Environment gating
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    """True iff cupy is importable AND a CUDA device is actually usable.

    The two-step check matches ``conftest.gpu_available``: some sandboxes
    ship cupy without a working CUDA runtime, so ``import cupy`` alone
    is not enough.
    """
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


_HAS_GPU = _gpu_available()
_HAS_FSSPEC = importlib.util.find_spec("fsspec") is not None
_HAS_DASK = importlib.util.find_spec("dask") is not None


# ---------------------------------------------------------------------------
# Curated rich metadata attrs (the contract this gate locks in)
# ---------------------------------------------------------------------------

# Geo / CRS / nodata core. Every fixture-carrying georef ought to agree
# on these across every backend.
_GEOREF_KEYS: tuple[str, ...] = (
    "transform",
    "crs",
    "crs_wkt",
)

# Nodata semantics. ``masked_nodata`` is the flag from #2092 that says
# "the reader rewrote sentinel pixels to NaN"; pinning it across
# backends catches a backend that silently disagreed on the mask.
_NODATA_KEYS: tuple[str, ...] = (
    "nodata",
    "masked_nodata",
)

# Canonical metadata attrs from the #1984 contract. ``raster_type`` is
# absent when implicit-area; ``x_resolution`` / ``y_resolution`` /
# ``resolution_unit`` only fire when the fixture set them. The check
# below compares presence + value; an attr absent on the reference
# must also be absent on the candidate, and vice versa.
_CANONICAL_METADATA_KEYS: tuple[str, ...] = (
    "raster_type",
    "x_resolution",
    "y_resolution",
    "resolution_unit",
    "georef_status",
)


# ---------------------------------------------------------------------------
# Skip / xfail taxonomy
# ---------------------------------------------------------------------------

# Fixtures the entire parity matrix skips. Each entry cites the source
# of the divergence (an issue link or a sibling test that asserts the
# divergent behaviour). Skips here are **intentional**: they describe
# behaviour the matrix would compare unequal across backends *by
# design*, not bugs.
_INTENTIONAL_SKIPS: dict[str, str] = {
    "nodata_miniswhite_uint8": (
        "MinIsWhite photometric inversion: xrspatial inverts pixels per "
        "#1797; rasterio leaves them raw. The matrix would compare "
        "inverted-vs-raw and fail on every row. Covered by "
        "test_miniswhite_backend_parity_1797.py."
    ),
    "compression_jpeg_uint8_ycbcr": (
        "JPEG-YCbCr is lossy and exposes a (bands, y, x) vs (y, x, band) "
        "axis-order divergence that the golden-corpus oracle handles "
        "via _normalise_axis_order but this gate's dims/coords check "
        "cannot, because the dims tuple itself differs. Covered by "
        "test_golden_corpus_eager_numpy_1930.py et al., which use the "
        "axis-normalising oracle."
    ),
}


# Per-backend skips. Outer key is the backend id; inner maps fixture id
# to a reason. Entries describe backend-specific limitations that
# cannot be resolved at this layer.
_BACKEND_SKIPS: dict[str, dict[str, str]] = {
    "vrt_eager": {
        # The VRT writer requires a CRS to lay out the mosaic; the
        # citation-only fixture's user-defined WKT does not survive
        # the round trip through rasterio's VRT serializer cleanly
        # enough for a bit-exact parity check. Eager numpy and dask
        # still parity-check this fixture directly.
        "crs_citation_only": (
            "VRT round-trip mutates user-defined CRS WKT; covered by "
            "test_user_defined_crs_wkt_1632.py."
        ),
        # External sidecar .ovr does not survive a VRT wrap. Base-IFD
        # coverage stays in the per-backend corpus tests.
        "overview_external_ovr_uint16": (
            "External .ovr sidecar is not preserved through VRT wrap."
        ),
        # Sparse tiles with explicit holes are not a clean VRT input
        # (the writer would have to invent fill bytes). Covered by
        # test_sparse_cog.py and test_vrt_holes_attr_1734.py.
        "sparse_tiled_uint16": (
            "Sparse-tile holes are not preserved through VRT wrap."
        ),
        # VRT wrap does not surface the source TIFF's XResolution /
        # YResolution / ResolutionUnit tags, nor extra_tags. The base
        # IFD parity is covered by the eager / dask backends directly;
        # the VRT cell would compare missing-vs-present canonical
        # attrs and fail by design.
        "extra_tags_uint16": (
            "VRT wrap does not propagate source TIFF resolution tags or "
            "extra_tags; covered by test_extra_tags_safe_filter_1657.py."
        ),
    },
    "http_fsspec": {
        # External sidecar .ovr fetch is not implemented for the cloud
        # source path; the base IFD still reads cleanly, but the gate
        # checks transform/coords that depend on the overview chain
        # surfacing the same way as on the eager path. Per-backend
        # corpus tests cover the no-overview case.
        "overview_external_ovr_uint16": (
            "External .ovr sidecar reader is not wired into the cloud "
            "source path."
        ),
    },
}


# ---------------------------------------------------------------------------
# Backend table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Backend:
    """One row of the parity matrix.

    ``read`` opens the fixture and returns a DataArray. ``available``
    reports whether the row can run in this environment; when it
    cannot, the parametrize layer attaches a ``pytest.skip`` with the
    ``unavailable_reason`` text. ``skips`` lists per-fixture skips
    keyed by fixture id.
    """

    backend_id: str
    read: Callable[[pathlib.Path, str], xr.DataArray]
    available: bool
    unavailable_reason: str
    skips: dict[str, str] = field(default_factory=dict)


def _read_eager_numpy(path: pathlib.Path, _fixture_id: str) -> xr.DataArray:
    return open_geotiff(str(path))


def _read_dask_numpy(path: pathlib.Path, _fixture_id: str) -> xr.DataArray:
    return open_geotiff(str(path), chunks=_CHUNK_SIZE)


def _read_gpu(path: pathlib.Path, _fixture_id: str) -> xr.DataArray:
    return open_geotiff(str(path), gpu=True, on_gpu_failure="strict")


def _read_dask_gpu(path: pathlib.Path, _fixture_id: str) -> xr.DataArray:
    return open_geotiff(
        str(path), gpu=True, chunks=_CHUNK_SIZE, on_gpu_failure="strict",
    )


def _read_vrt_eager(path: pathlib.Path, fixture_id: str) -> xr.DataArray:
    """Wrap ``path`` in a one-source VRT and read it back.

    Lives behind a session-scoped cache so the per-fixture VRT
    materialisation runs once even though every backend test
    parametrizes over fixtures independently. The wrap uses
    ``write_vrt`` directly so the test exercises xrspatial's own VRT
    plumbing, not rasterio's.

    The source ``.tif`` is **copied** into the VRT cache directory
    rather than referenced in place. The VRT reader's path-containment
    guard (#1671) refuses to follow a SourceFilename that resolves
    outside the VRT directory unless ``XRSPATIAL_VRT_ALLOWED_ROOTS``
    is set; copying keeps the test self-contained instead of mutating
    a global env var for every cell.
    """
    import shutil
    cache_dir = _vrt_cache_dir(path.parent)
    local_src = cache_dir / f"{fixture_id}.tif"
    if not local_src.exists():
        shutil.copy2(path, local_src)
    vrt_path = cache_dir / f"{fixture_id}.vrt"
    if not vrt_path.exists():
        write_vrt(str(vrt_path), [str(local_src)])
    return open_geotiff(str(vrt_path))


def _read_http_fsspec(path: pathlib.Path, fixture_id: str) -> xr.DataArray:
    """Serve the fixture bytes via fsspec's in-process memory FS.

    The ``memory://`` scheme drives the same ``_CloudSource`` path an
    ``s3://`` or ``gs://`` URL would, minus the network and
    credentials. Matches how
    ``test_golden_corpus_fsspec_1930.py`` exercises the cloud-eager
    read path in restricted-sandbox CI.

    The fsspec memory filesystem is a process-global singleton, so the
    fixture's bytes are written under a unique key and the key is
    deleted before returning. The cloud-eager read path materialises
    pixels inline (``_CloudSource`` downloads under the
    ``max_cloud_bytes`` budget; see #1928), so the DataArray no longer
    needs the memory entry once ``open_geotiff`` returns.
    """
    import fsspec
    fs = fsspec.filesystem("memory")
    key = f"/corpus_full_parity_2211/{fixture_id}.tif"
    with open(path, "rb") as f:
        fs.pipe(key, f.read())
    try:
        da = open_geotiff(f"memory://{key}")
    finally:
        # Best-effort cleanup; fsspec memory store deletions are
        # idempotent. The cloud-eager path has already pulled the
        # bytes into the DataArray by this point.
        try:
            fs.rm(key)
        except FileNotFoundError:
            pass
    return da


def _vrt_cache_dir(fixtures_dir: pathlib.Path) -> pathlib.Path:
    """Return the per-session VRT scratch directory.

    Materialising one-source VRTs into the fixtures directory itself
    would pollute the corpus; ``tmp_path`` is fixture-scoped and not
    shared across parametrize cells. A module-level cache directory
    under the system tmp keeps the VRTs around for the duration of the
    pytest process and lets every backend cell hit the same .vrt.
    """
    import hashlib
    import tempfile
    base = pathlib.Path(tempfile.gettempdir()) / "xrspatial_2229_vrt_cache"
    base.mkdir(parents=True, exist_ok=True)
    # Pin the cache directory name to a stable digest of the
    # fixtures path. ``hashlib.sha1`` is deterministic across
    # processes (unlike ``hash()`` which uses ``PYTHONHASHSEED``
    # salting), so two pytest runs against the same fixtures dir
    # reuse the same cache. The digest is truncated to 12 hex chars
    # for path readability; collisions are not security-relevant
    # because the cache directory is per-user-tmp.
    digest = hashlib.sha1(str(fixtures_dir).encode()).hexdigest()[:12]
    sub = base / f"fix_{digest}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


_GPU_UNAVAILABLE_REASON = (
    "GPU backend skipped LOUDLY: cupy + CUDA are not available in this "
    "environment. Per issue #2229, GPU and Dask+GPU rows must skip "
    "explicitly rather than silently collect zero tests. To exercise "
    "this row, install cupy and ensure a CUDA device is reachable."
)

_DASK_UNAVAILABLE_REASON = (
    "dask backend skipped: dask is not installed. Install xarray-spatial "
    "with the dask extra to exercise this row."
)

_FSSPEC_UNAVAILABLE_REASON = (
    "http_fsspec backend skipped: fsspec is not installed. Install "
    "fsspec to exercise the cloud-source dispatch."
)


_BACKENDS: list[_Backend] = [
    _Backend(
        backend_id="eager_numpy",
        read=_read_eager_numpy,
        available=True,
        unavailable_reason="",
    ),
    _Backend(
        backend_id="dask_numpy",
        read=_read_dask_numpy,
        available=_HAS_DASK,
        unavailable_reason=_DASK_UNAVAILABLE_REASON,
    ),
    _Backend(
        backend_id="gpu",
        read=_read_gpu,
        available=_HAS_GPU,
        unavailable_reason=_GPU_UNAVAILABLE_REASON,
    ),
    _Backend(
        backend_id="dask_gpu",
        read=_read_dask_gpu,
        available=_HAS_GPU and _HAS_DASK,
        unavailable_reason=(
            _GPU_UNAVAILABLE_REASON
            if not _HAS_GPU
            else _DASK_UNAVAILABLE_REASON
        ),
        skips=dict(_BACKEND_SKIPS.get("dask_gpu", {})),
    ),
    _Backend(
        backend_id="vrt_eager",
        read=_read_vrt_eager,
        available=True,
        unavailable_reason="",
        skips=dict(_BACKEND_SKIPS["vrt_eager"]),
    ),
    _Backend(
        backend_id="http_fsspec",
        read=_read_http_fsspec,
        available=_HAS_FSSPEC,
        unavailable_reason=_FSSPEC_UNAVAILABLE_REASON,
        skips=dict(_BACKEND_SKIPS["http_fsspec"]),
    ),
]


# ---------------------------------------------------------------------------
# Manifest / fixture helpers
# ---------------------------------------------------------------------------

def _resolved_fixtures() -> list[dict[str, Any]]:
    """Return manifest entries with defaults merged, sorted by id."""
    manifest = generate.load_manifest()
    entries = generate.validate(manifest)
    entries.sort(key=lambda e: e["id"])
    return entries


def _fixture_path(entry: dict[str, Any]) -> pathlib.Path:
    return FIXTURES_DIR / f"{entry['id']}.tif"


def _is_lossy(entry: dict[str, Any]) -> bool:
    tol = entry.get("tolerance") or {}
    return bool(tol.get("lossy", False))


_FIXTURES = _resolved_fixtures()


# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------

def _build_fixture_params() -> list[pytest.param]:
    """One ``pytest.param`` per manifest entry, with slow/skip marks."""
    out: list[pytest.param] = []
    for entry in _FIXTURES:
        fid = entry["id"]
        marks = list(fast_slow_marks_for(entry))
        if fid in _INTENTIONAL_SKIPS:
            marks.append(pytest.mark.skip(reason=_INTENTIONAL_SKIPS[fid]))
        out.append(pytest.param(entry, id=fid, marks=marks))
    return out


def _build_backend_params() -> list[pytest.param]:
    """One ``pytest.param`` per backend; unavailable ones carry skip marks."""
    out: list[pytest.param] = []
    for backend in _BACKENDS:
        marks = []
        if not backend.available:
            marks.append(pytest.mark.skip(reason=backend.unavailable_reason))
        out.append(pytest.param(backend, id=backend.backend_id, marks=marks))
    return out


_FIXTURE_PARAMS = _build_fixture_params()
_BACKEND_PARAMS = _build_backend_params()


# ---------------------------------------------------------------------------
# Pixel + attrs assertions
# ---------------------------------------------------------------------------

def _is_nan_sentinel(value: Any) -> bool:
    """True when ``value`` is a NaN sentinel, regardless of scalar type.

    Accepts Python floats, numpy scalars, and any object castable to
    ``float``. Non-numeric values (including ``None``) return False so
    the caller can fall through to a strict ``==`` comparison and get
    a useful diff message.
    """
    if value is None:
        return False
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _materialise(da: xr.DataArray) -> np.ndarray:
    """Return a host-side numpy view regardless of backend.

    Mirrors the corpus oracle's ``_candidate_pixels``. Order matters:
    a dask-of-cupy array needs ``.compute()`` before ``.get()`` because
    the wrapping is graph-on-device, not device-on-graph.
    """
    raw = da.data
    if hasattr(raw, "compute"):
        raw = raw.compute()
    if hasattr(raw, "get"):
        raw = raw.get()
    return np.asarray(raw)


def _assert_pixels_close(
    ref: np.ndarray, cand: np.ndarray, *, lossy: bool, label: str,
) -> None:
    """Compare reference and candidate pixels.

    Float dtypes use NaN-aware ``allclose`` so a NaN-marked nodata
    cell compares equal to itself. Integer dtypes go bit-exact. Lossy
    fixtures (JPEG today) drop to a shape-only assertion because the
    codec is intrinsically inexact.
    """
    assert ref.shape == cand.shape, (
        f"{label}: shape mismatch ref={ref.shape} cand={cand.shape}"
    )
    if lossy:
        return
    assert ref.dtype == cand.dtype, (
        f"{label}: dtype mismatch ref={ref.dtype} cand={cand.dtype}"
    )
    if ref.dtype.kind == "f":
        # ``allclose`` with ``equal_nan=True`` and a relative tolerance:
        # every shipped fixture compares bit-exact today (the four
        # eager/dask/gpu/dask+gpu backends share decode primitives and
        # the VRT path leaves pixel bytes alone). The relative
        # tolerance is here to absorb a hypothetical future codec /
        # predictor that does a real floating-point op rather than as
        # documented headroom for an existing drift. ``rtol=1e-12``
        # tracks data magnitude so a small-magnitude fixture is not
        # secretly held to a slacker bar than a large-magnitude one;
        # ``atol=0`` keeps zero values strict. The strict-zero
        # behaviour is deliberate: a future fixture comparing an
        # exact 0.0 against a sub-ULP non-zero is a real decode
        # divergence at the smallest representable value and should
        # fail rather than slip past under a generous absolute floor.
        # Do not loosen ``atol`` without confirming the new fixture's
        # expectations.
        ok = np.allclose(ref, cand, rtol=1e-12, atol=0.0, equal_nan=True)
        if not ok:
            # Report the worst offender so a regression is debuggable.
            diff = np.abs(np.where(
                np.isnan(ref) & np.isnan(cand), 0.0, ref - cand
            ))
            raise AssertionError(
                f"{label}: pixel allclose failed; max abs diff="
                f"{np.nanmax(diff)!r}"
            )
    else:
        if not np.array_equal(ref, cand):
            raise AssertionError(
                f"{label}: integer pixels differ (bit-exact comparison "
                f"failed) ref.dtype={ref.dtype}"
            )


def _assert_dims_and_coords(
    ref: xr.DataArray, cand: xr.DataArray, *, label: str,
) -> None:
    """Dims tuple and per-axis coord values + dtype agree.

    The matrix only compares fixtures whose dims tuple is identical
    across backends -- multi-band axis-order divergences (JPEG-YCbCr)
    live in ``_INTENTIONAL_SKIPS`` for that reason. So the dim
    equality is a strict tuple compare.
    """
    assert ref.dims == cand.dims, (
        f"{label}: dims mismatch ref={ref.dims!r} cand={cand.dims!r}"
    )
    for axis in ref.dims:
        if axis not in ref.coords:
            # Reference does not carry this coord (e.g. band index
            # dropped); only fail if the candidate disagrees on
            # presence.
            assert axis not in cand.coords, (
                f"{label}: candidate has coord {axis!r} that the "
                f"reference does not"
            )
            continue
        assert axis in cand.coords, (
            f"{label}: candidate is missing coord {axis!r} that the "
            f"reference carries"
        )
        ref_c = np.asarray(ref.coords[axis].values)
        cand_c = np.asarray(cand.coords[axis].values)
        assert ref_c.dtype == cand_c.dtype, (
            f"{label}: coord {axis!r} dtype ref={ref_c.dtype} "
            f"cand={cand_c.dtype}"
        )
        # ``allclose`` for floating-point coords, bit-exact for ints.
        # Coords drive transform reconstruction so the tolerance is
        # tight; a difference at the ULP level still means a different
        # transform.
        if ref_c.dtype.kind == "f":
            assert np.allclose(ref_c, cand_c, rtol=0.0, atol=1e-9), (
                f"{label}: coord {axis!r} values differ"
            )
        else:
            assert np.array_equal(ref_c, cand_c), (
                f"{label}: coord {axis!r} values differ"
            )


def _assert_transform_attrs(
    ref: xr.DataArray, cand: xr.DataArray, *, label: str,
) -> None:
    """Transform 6-tuple agrees, allowing a ULP for float rounding."""
    ref_t = ref.attrs.get("transform")
    cand_t = cand.attrs.get("transform")
    if ref_t is None and cand_t is None:
        return
    assert ref_t is not None and cand_t is not None, (
        f"{label}: transform presence differs ref={ref_t!r} cand={cand_t!r}"
    )
    ref_tup = tuple(float(v) for v in ref_t)
    cand_tup = tuple(float(v) for v in cand_t)
    assert len(ref_tup) == 6 and len(cand_tup) == 6, (
        f"{label}: transform must be a 6-tuple, got ref={ref_tup} "
        f"cand={cand_tup}"
    )
    for i, (a, b) in enumerate(zip(ref_tup, cand_tup)):
        assert abs(a - b) <= 1e-9, (
            f"{label}: transform[{i}] differs ref={a!r} cand={b!r}"
        )


def _assert_crs_attrs(
    ref: xr.DataArray, cand: xr.DataArray, *, label: str,
) -> None:
    """``crs`` (EPSG int or string) and ``crs_wkt`` agree by value."""
    for key in ("crs", "crs_wkt"):
        ref_v = ref.attrs.get(key)
        cand_v = cand.attrs.get(key)
        assert ref_v == cand_v, (
            f"{label}: attr {key!r} differs ref={ref_v!r} cand={cand_v!r}"
        )


def _assert_nodata_attrs(
    ref: xr.DataArray, cand: xr.DataArray, *, label: str,
) -> None:
    """nodata sentinel and the masked-nodata flag agree (NaN-aware)."""
    ref_nd = ref.attrs.get("nodata")
    cand_nd = cand.attrs.get("nodata")
    if ref_nd is None and cand_nd is None:
        pass
    else:
        # NaN sentinel equality: float('nan') != float('nan'), but the
        # two are the same nodata for our purposes. The float cast
        # below accepts numpy scalars (``numpy.float32(nan)``) and
        # python floats alike; a non-numeric sentinel (e.g. None on
        # one side only) falls through to the ``==`` branch which
        # surfaces the mismatch with a useful message.
        ref_is_nan = _is_nan_sentinel(ref_nd)
        cand_is_nan = _is_nan_sentinel(cand_nd)
        if not (ref_is_nan and cand_is_nan):
            assert ref_nd == cand_nd, (
                f"{label}: nodata differs ref={ref_nd!r} cand={cand_nd!r}"
            )
    ref_masked = ref.attrs.get("masked_nodata")
    cand_masked = cand.attrs.get("masked_nodata")
    assert ref_masked == cand_masked, (
        f"{label}: masked_nodata differs ref={ref_masked!r} "
        f"cand={cand_masked!r}"
    )

    # Pixel dtype must match too: the masked-nodata contract upcasts
    # an integer raster to float so NaN can live in it, so a drift here
    # is a drift in the nodata semantics, not just in pixel storage.
    ref_dtype = np.dtype(ref.dtype)
    cand_dtype = np.dtype(cand.dtype)
    assert ref_dtype == cand_dtype, (
        f"{label}: pixel dtype differs ref={ref_dtype} cand={cand_dtype}"
    )


def _assert_canonical_metadata_attrs(
    ref: xr.DataArray, cand: xr.DataArray, *, label: str,
) -> None:
    """The #1984 canonical-metadata subset agrees on presence + value.

    An attr absent on the reference must also be absent on the
    candidate, and vice versa. This catches a backend that silently
    stamps an attr the others omit, or omits an attr the others stamp.
    """
    for key in _CANONICAL_METADATA_KEYS:
        in_ref = key in ref.attrs
        in_cand = key in cand.attrs
        assert in_ref == in_cand, (
            f"{label}: canonical attr {key!r} presence differs "
            f"ref={in_ref} cand={in_cand}"
        )
        if in_ref:
            ref_v = ref.attrs[key]
            cand_v = cand.attrs[key]
            assert ref_v == cand_v, (
                f"{label}: canonical attr {key!r} value differs "
                f"ref={ref_v!r} cand={cand_v!r}"
            )


# ---------------------------------------------------------------------------
# Eager-numpy reference cache (one read per fixture, reused per backend)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _reference_cache() -> dict[str, xr.DataArray]:
    """Cache eager-numpy reads keyed by fixture id.

    Reading the same fixture once per backend in a 6-backend x 30+
    fixture matrix would multiply IO by 6x; cache the eager-numpy
    candidate (which is also the reference) once per session.
    """
    return {}


def _reference_for(
    entry: dict[str, Any], cache: dict[str, xr.DataArray],
) -> xr.DataArray:
    fid = entry["id"]
    if fid not in cache:
        cache[fid] = open_geotiff(str(_fixture_path(entry)))
    return cache[fid]


# ---------------------------------------------------------------------------
# The single parity-gate test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", _BACKEND_PARAMS)
@pytest.mark.parametrize("manifest_entry", _FIXTURE_PARAMS)
def test_backend_full_parity(
    manifest_entry: dict[str, Any],
    backend: _Backend,
    _reference_cache: dict[str, xr.DataArray],
) -> None:
    """Single contract gate for every (backend, fixture) cell.

    Each cell:

    1. Looks up (or reads) the eager-numpy reference for the fixture.
    2. Reads the same fixture through ``backend.read``.
    3. Asserts pixel values, dims + coords, transform/georef, CRS,
       nodata, and the curated canonical metadata attrs.

    Per-backend skips (``backend.skips``) and intentional fixture
    skips (``_INTENTIONAL_SKIPS``, applied via the parametrize marks)
    cover cells the matrix legitimately cannot compare.
    """
    fixture_id = manifest_entry["id"]
    path = _fixture_path(manifest_entry)
    if not path.exists():
        pytest.skip(
            f"fixture {fixture_id!r} has no .tif on disk; run "
            f"`python -m xrspatial.geotiff.tests.golden_corpus.generate`"
        )

    # Per-backend fixture skip with a loud reason.
    if fixture_id in backend.skips:
        pytest.skip(
            f"backend={backend.backend_id} cannot read fixture="
            f"{fixture_id}: {backend.skips[fixture_id]}"
        )

    reference = _reference_for(manifest_entry, _reference_cache)
    try:
        candidate = backend.read(path, fixture_id)
    except Exception as exc:
        # Surface the read failure in the assertion message so a CI
        # log can be triaged without re-running locally.
        raise AssertionError(
            f"backend={backend.backend_id} failed to read fixture="
            f"{fixture_id}: {type(exc).__name__}: {exc}"
        ) from exc

    label = f"fixture={fixture_id} backend={backend.backend_id}"

    # 1. Pixel values.
    ref_px = _materialise(reference)
    cand_px = _materialise(candidate)
    _assert_pixels_close(
        ref_px, cand_px, lossy=_is_lossy(manifest_entry), label=label,
    )

    # 2. Dims + coords.
    _assert_dims_and_coords(reference, candidate, label=label)

    # 3. Transform / georef.
    _assert_transform_attrs(reference, candidate, label=label)

    # 4. CRS.
    _assert_crs_attrs(reference, candidate, label=label)

    # 5. Nodata + dtype.
    _assert_nodata_attrs(reference, candidate, label=label)

    # 6. Curated canonical metadata attrs.
    _assert_canonical_metadata_attrs(reference, candidate, label=label)


# ---------------------------------------------------------------------------
# Taxonomy hygiene: skip tables must reference real fixture ids
# ---------------------------------------------------------------------------

def test_taxonomy_ids_are_in_manifest() -> None:
    """Every fixture id in a skip table must exist in the manifest.

    Catches typos and stale entries left behind when a fixture is
    renamed or removed: a stale entry would silently keep a known-
    incompatible cell skipped even after the underlying fixture has
    been replaced.
    """
    manifest_ids = {e["id"] for e in _FIXTURES}
    referenced: set[str] = set(_INTENTIONAL_SKIPS)
    for backend in _BACKENDS:
        referenced.update(backend.skips)
    stale = referenced - manifest_ids
    assert not stale, (
        f"skip tables reference unknown fixture ids: {sorted(stale)}"
    )


def test_gpu_skip_reason_is_loud() -> None:
    """Per issue #2229: GPU + Dask+GPU skips must be explicit, not silent.

    Sanity-check the reason strings carry the LOUD marker and cite the
    issue. If a future refactor downgrades the reason to a generic
    string, this test fails to point the developer at the contract.
    """
    for backend_id in ("gpu", "dask_gpu"):
        backend = next(b for b in _BACKENDS if b.backend_id == backend_id)
        if backend.available:
            continue
        # Read the reason; it must mention the contract for clarity.
        reason = backend.unavailable_reason
        assert "skipped LOUDLY" in reason or "skipped" in reason, (
            f"{backend_id} unavailable_reason is not explicit enough: "
            f"{reason!r}"
        )
        assert "#2229" in reason or "cupy" in reason or "dask" in reason, (
            f"{backend_id} unavailable_reason does not cite the contract "
            f"or the missing dep: {reason!r}"
        )


# ---------------------------------------------------------------------------
# Storage-type sanity checks (the parity battery alone cannot see them)
# ---------------------------------------------------------------------------

def _first_eligible_fixture() -> dict[str, Any] | None:
    """Pick a fast, on-disk fixture none of the intentional-skip
    tables flag, so the storage-type sanity checks run against a
    fixture every backend reads cleanly.
    """
    for entry in _FIXTURES:
        if entry["id"] in _INTENTIONAL_SKIPS:
            continue
        if not _fixture_path(entry).exists():
            continue
        # Prefer a fast fixture; the sanity check just needs *any*
        # eligible fixture so falling back to slow is fine.
        if "fast" in (entry.get("tags") or []):
            return entry
    for entry in _FIXTURES:
        if entry["id"] not in _INTENTIONAL_SKIPS and _fixture_path(entry).exists():
            return entry
    return None


@pytest.mark.skipif(not _HAS_GPU, reason=_GPU_UNAVAILABLE_REASON)
def test_gpu_backend_returns_cupy_array() -> None:
    """Sanity check: the gpu row returns a cupy-backed DataArray.

    Catches the failure mode the parity battery cannot see: a silent
    fallback that returns a numpy array when the caller asked for
    ``gpu=True``. The pixels would still compare equal to the eager
    reference and the test would pass without exercising the GPU
    decode path at all.
    """
    import cupy
    entry = _first_eligible_fixture()
    if entry is None:
        pytest.skip("no eligible fixture on disk")
    da = _read_gpu(_fixture_path(entry), entry["id"])
    assert isinstance(da.data, cupy.ndarray), (
        f"gpu backend on fixture {entry['id']!r} returned "
        f"{type(da.data).__name__}, expected cupy.ndarray. "
        "A silent CPU fallback under gpu=True would let the parity "
        "matrix pass while exercising the wrong code path."
    )


@pytest.mark.skipif(not _HAS_DASK, reason=_DASK_UNAVAILABLE_REASON)
def test_dask_backend_returns_dask_array() -> None:
    """Sanity check: the dask_numpy row returns a dask-backed
    DataArray. Catches a regression where ``chunks=`` is silently
    dropped and the read goes through the eager path.
    """
    entry = _first_eligible_fixture()
    if entry is None:
        pytest.skip("no eligible fixture on disk")
    da = _read_dask_numpy(_fixture_path(entry), entry["id"])
    assert hasattr(da.data, "dask"), (
        f"dask_numpy backend on fixture {entry['id']!r} returned "
        f"data of type {type(da.data).__name__}, expected a "
        "dask-backed array."
    )


@pytest.mark.skipif(
    not (_HAS_GPU and _HAS_DASK),
    reason=(
        f"{_GPU_UNAVAILABLE_REASON} (or dask missing -- "
        f"{_DASK_UNAVAILABLE_REASON})"
    ),
)
def test_dask_gpu_backend_returns_dask_of_cupy() -> None:
    """Sanity check: the dask_gpu row returns a dask-graph-of-cupy
    DataArray. Both layers must be present; a regression that strips
    one would compute the right pixels via a different storage type.
    """
    import cupy
    entry = _first_eligible_fixture()
    if entry is None:
        pytest.skip("no eligible fixture on disk")
    da = _read_dask_gpu(_fixture_path(entry), entry["id"])
    assert hasattr(da.data, "dask"), (
        f"dask_gpu backend on fixture {entry['id']!r} dropped the "
        f"dask wrapping: data is {type(da.data).__name__}"
    )
    # Compute one chunk to peek at the device-side type without
    # materialising the whole array; ``_meta`` carries the chunk
    # prototype that dask uses for shape/dtype introspection.
    meta = getattr(da.data, "_meta", None)
    assert isinstance(meta, cupy.ndarray), (
        f"dask_gpu backend on fixture {entry['id']!r} carries a "
        f"non-cupy chunk prototype: {type(meta).__name__}. A "
        "dask-of-numpy graph would compute the right pixels but "
        "skip the GPU decode path."
    )
