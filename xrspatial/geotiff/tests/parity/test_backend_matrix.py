"""Matrix-style backend parity across high-risk fixtures.

Single source of truth for "does backend X still match the eager-numpy
reference on fixture Z." Four sections:

* High-risk fixture matrix plus an error sub-matrix: every
  (backend, fixture) cell runs through ``assert_parity``.
* Full-corpus parity over the golden corpus, using the manifest as
  the fixture set and the same ``open_geotiff`` entry-point across
  every backend.
* Canonical-attrs parity: each backend stamps the same canonical
  attrs for the same fixture, with a documented carve-out for
  backend-specific keys.
* Pass-through TIFF tag parity: ``x_resolution``, ``y_resolution``,
  ``resolution_unit``, ``image_description``, and ``extra_samples``
  agree across the four core backends.

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
* masking state (``attrs.get('masked_nodata')`` from #2092)
* a small subset of canonical attrs whose round-trip semantics are
  already settled in the module (``raster_type``, ``transform``,
  ``crs``, ``crs_wkt``).

Backends (issue #2132 plan)
---------------------------

The matrix is parametrised over up to 8 entries that span every
public dispatch path the reader supports:

* ``numpy`` -- eager local file
* ``dask+numpy`` -- chunked local file
* ``gpu`` -- eager local file via cupy
* ``dask+gpu`` -- chunked local file via cupy
* ``vrt-eager`` -- ``.vrt`` mosaic, eager
* ``vrt-dask`` -- ``.vrt`` mosaic, chunked
* ``http-cog`` -- HTTP range-read of a COG
* ``fsspec-memory`` -- ``memory://`` URI through fsspec

GPU rows skip when cupy + CUDA are missing. HTTP and fsspec rows skip
when their network or fsspec deps are absent. VRT rows are gated by
the writer being able to lay out the mosaic on disk -- always true on
local filesystems.

Cells that pair a backend with a source that physically cannot be
fed to it (e.g. a HTTP URL into ``vrt-eager``) skip via the
per-backend ``compat`` predicate on :class:`_BackendSpec`.
"""
from __future__ import annotations

import http.server
import importlib.util
import pathlib
import socketserver
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, read_vrt, to_geotiff, write_vrt
from xrspatial.geotiff._attrs import _finalize_eager_read, _finalize_lazy_read_attrs
from xrspatial.geotiff._errors import RotatedTransformError, UnparseableCRSError

from .._helpers.markers import gpu_available, requires_gpu, requires_loopback

# Alias so existing base-section signatures that say ``Path`` keep working.
Path = pathlib.Path


# ---------------------------------------------------------------------------
# Environment gating
# ---------------------------------------------------------------------------

_HAS_GPU = gpu_available()
_HAS_TIFFFILE = importlib.util.find_spec("tifffile") is not None
_HAS_FSSPEC = importlib.util.find_spec("fsspec") is not None
_HAS_DASK = importlib.util.find_spec("dask") is not None

# Use the shared marker from ``_helpers/markers.py`` for the GPU gate.
_skip_no_gpu = requires_gpu
_skip_no_tifffile = pytest.mark.skipif(
    not _HAS_TIFFFILE, reason="tifffile required for MinIsWhite fixture")
_skip_no_fsspec = pytest.mark.skipif(
    not _HAS_FSSPEC, reason="fsspec required for memory:// source")


# ---------------------------------------------------------------------------
# Source-type taxonomy
# ---------------------------------------------------------------------------

# Source types name how the fixture is delivered to ``open_geotiff``. The
# read backends accept a subset of source types; the compatibility matrix
# lives on :class:`_BackendSpec.compat`.
_SRC_LOCAL_TIFF = "local-tiff"
_SRC_LOCAL_VRT = "local-vrt"
_SRC_HTTP = "http"
_SRC_FSSPEC = "fsspec"


# ---------------------------------------------------------------------------
# Backend descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _BackendSpec:
    """Declarative description of one read backend.

    Attributes
    ----------
    backend_id
        Stable id used in the parametrize call. Appears in test names.
    kwargs
        Static ``open_geotiff`` kwargs that select this backend.
    compat
        Set of source-type ids this backend accepts. Cells with an
        incompatible (backend, source) pair skip with a clear reason.
    marks
        Pytest marks (e.g. skipif) applied to every cell using this
        backend. Used to gate GPU and fsspec backends behind their
        optional deps.
    source_type_override
        If set, the matrix dispatches the fixture path through this
        source type rather than the fixture's native type. Used by the
        HTTP and fsspec backends to deliver the same on-disk TIFF
        through a different transport.
    """

    backend_id: str
    kwargs: dict[str, Any]
    compat: frozenset[str]
    marks: tuple = field(default_factory=tuple)
    source_type_override: str | None = None


_BACKENDS: list[_BackendSpec] = [
    _BackendSpec(
        backend_id="numpy",
        kwargs={},
        # VRT fixtures are owned by the ``vrt-eager`` / ``vrt-dask``
        # rows below; routing them through ``numpy`` too would
        # duplicate identical cells.
        compat=frozenset({_SRC_LOCAL_TIFF, _SRC_HTTP, _SRC_FSSPEC}),
    ),
    _BackendSpec(
        backend_id="dask+numpy",
        kwargs={"chunks": 16},
        # Dask path supports fsspec URIs (#1749) but does not accept
        # raw BytesIO. VRT lives on the ``vrt-dask`` row.
        compat=frozenset({_SRC_LOCAL_TIFF, _SRC_FSSPEC}),
    ),
    _BackendSpec(
        backend_id="gpu",
        kwargs={"gpu": True},
        # GPU reader is local-file only. HTTP / fsspec deliver bytes
        # through code paths the GPU reader does not consume.
        compat=frozenset({_SRC_LOCAL_TIFF}),
        marks=(_skip_no_gpu,),
    ),
    _BackendSpec(
        backend_id="dask+gpu",
        kwargs={"gpu": True, "chunks": 16},
        compat=frozenset({_SRC_LOCAL_TIFF}),
        marks=(_skip_no_gpu,),
    ),
    _BackendSpec(
        backend_id="vrt-eager",
        kwargs={},
        # VRT-only backend: only the VRT fixture is in scope.
        compat=frozenset({_SRC_LOCAL_VRT}),
    ),
    _BackendSpec(
        backend_id="vrt-dask",
        kwargs={"chunks": 16},
        compat=frozenset({_SRC_LOCAL_VRT}),
    ),
    _BackendSpec(
        backend_id="http-cog",
        kwargs={},
        # HTTP backend re-routes any local TIFF fixture through a
        # loopback HTTP server. Not all fixtures are valid COGs but
        # the HTTP reader will still pull the bytes via range reads
        # for any TIFF that the local server can serve.
        compat=frozenset({_SRC_LOCAL_TIFF}),
        source_type_override=_SRC_HTTP,
    ),
    _BackendSpec(
        backend_id="fsspec-memory",
        kwargs={},
        # fsspec memory:// route accepts any local TIFF fixture
        # whose bytes can be uploaded into the in-memory filesystem.
        compat=frozenset({_SRC_LOCAL_TIFF}),
        source_type_override=_SRC_FSSPEC,
        marks=(_skip_no_fsspec,),
    ),
]


def _backend_params() -> list:
    """Build the pytest.param list for the backend matrix."""
    out = []
    for spec in _BACKENDS:
        out.append(pytest.param(spec, id=spec.backend_id, marks=spec.marks))
    return out


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
        Tri-valued. ``True`` / ``False`` pin ``attrs['masked_nodata']``.
        ``None`` means "do not assert" -- used for fixtures without
        nodata.
    source_type
        How the fixture is laid out on disk. Drives the
        backend-compatibility filter via :class:`_BackendSpec.compat`.
    read_kwargs
        Extra kwargs forwarded to every ``open_geotiff`` call for this
        fixture (e.g. ``mask_nodata=False``).
    marks
        Pytest marks applied to every cell using this fixture (e.g.
        ``_skip_no_tifffile`` for the MinIsWhite cell).
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
    source_type: str
    builder: Callable[[Path, Path], Path]
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    marks: tuple = field(default_factory=tuple)


def _wrap_2d(arr: np.ndarray, *, crs: int | None,
             nodata: object | None = None) -> xr.DataArray:
    """Wrap a 2-D numpy array as a writer-ready DataArray.

    Uses unit-pixel descending-y coords (``y = height-1 .. 0``,
    ``x = 0 .. width-1``). The read-back transform tuple for a height-H
    fixture is ``(1.0, 0.0, -0.5, 0.0, -1.0, H - 0.5)`` -- the half-pixel
    offsets come from the PixelIsArea convention (origin is the pixel
    edge, coords are pixel centres) that the writer round-trips.
    """
    height, width = arr.shape
    attrs: dict[str, Any] = {}
    if crs is not None:
        attrs["crs"] = crs
    if nodata is not None:
        attrs["nodata"] = nodata
    return xr.DataArray(
        arr, dims=["y", "x"],
        coords={
            "y": np.arange(height - 1, -1, -1, dtype=np.float64),
            "x": np.arange(width, dtype=np.float64),
        },
        attrs=attrs,
    )


def _wrap_3d(arr: np.ndarray, *, crs: int) -> xr.DataArray:
    """Wrap a 3-D (y, x, band) array as a writer-ready DataArray."""
    height, width, n_bands = arr.shape
    return xr.DataArray(
        arr, dims=["y", "x", "band"],
        coords={
            "y": np.arange(height - 1, -1, -1, dtype=np.float64),
            "x": np.arange(width, dtype=np.float64),
            "band": np.arange(n_bands),
        },
        attrs={"crs": crs},
    )


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _build_int16_single_band(dir_path: Path, target: Path) -> Path:
    """High-risk fixture: int16 single-band stripped TIFF, EPSG:4326, no nodata."""
    del dir_path
    rng = np.random.default_rng(seed=19850)
    arr = rng.integers(-30000, 30000, size=(32, 32), dtype=np.int16)
    to_geotiff(
        _wrap_2d(arr, crs=4326), str(target),
        compression="none", tiled=False,
    )
    return target


def _build_uint16_multiband_tiled(dir_path: Path, target: Path) -> Path:
    """Multiband tiled fixture: uint16, three bands, deflate-compressed."""
    del dir_path
    rng = np.random.default_rng(seed=21320)
    arr = rng.integers(0, 60000, size=(32, 32, 3), dtype=np.uint16)
    to_geotiff(
        _wrap_3d(arr, crs=4326), str(target),
        compression="deflate", tiled=True, tile_size=16,
    )
    return target


def _build_float32_with_nodata(dir_path: Path, target: Path) -> Path:
    """Float32 single-band fixture with a -9999.0 nodata sentinel."""
    del dir_path
    rng = np.random.default_rng(seed=21321)
    arr = (rng.standard_normal((32, 32)) * 100.0).astype(np.float32)
    # Sprinkle nodata sentinels into a few pixels so masking has work to do.
    arr[0, 0] = -9999.0
    arr[5, 7] = -9999.0
    arr[31, 31] = -9999.0
    to_geotiff(
        _wrap_2d(arr, crs=4326, nodata=-9999.0), str(target),
        compression="none", tiled=False,
    )
    return target


def _build_int8_unmasked(dir_path: Path, target: Path) -> Path:
    """Int8 single-band fixture with a -128 nodata sentinel.

    Read back with ``mask_nodata=False`` so the literal sentinel survives
    in the int8 buffer (locks the #2092 / #2127 masked-flag contract).
    """
    del dir_path
    rng = np.random.default_rng(seed=21322)
    arr = rng.integers(-100, 100, size=(32, 32), dtype=np.int8)
    arr[0, 0] = -128
    arr[4, 4] = -128
    to_geotiff(
        _wrap_2d(arr, crs=4326, nodata=-128), str(target),
        compression="none", tiled=False,
    )
    return target


def _build_cog(dir_path: Path, target: Path) -> Path:
    """COG fixture: float32 tiled with one overview level."""
    del dir_path
    rng = np.random.default_rng(seed=21323)
    arr = (rng.standard_normal((64, 64)) * 100.0).astype(np.float32)
    to_geotiff(
        _wrap_2d(arr, crs=4326), str(target),
        compression="deflate", cog=True, tile_size=16,
        overview_levels=[2],
    )
    return target


def _build_vrt_mosaic(dir_path: Path, target: Path) -> Path:
    """VRT fixture: 2-tile mosaic of float32 stripes laid out side by side."""
    tile_h, tile_w = 16, 16
    tile_paths: list[str] = []
    for c in range(2):
        arr = np.full((tile_h, tile_w),
                      float(c + 1), dtype=np.float32)
        origin_x = float(c * tile_w)
        da = xr.DataArray(
            arr, dims=["y", "x"],
            coords={
                "y": np.arange(tile_h - 1, -1, -1, dtype=np.float64),
                "x": np.arange(origin_x, origin_x + tile_w, dtype=np.float64),
            },
            attrs={"crs": 4326},
        )
        p = dir_path / f"{target.stem}_tile_{c}.tif"
        to_geotiff(da, str(p), compression="none", tiled=False)
        tile_paths.append(str(p))
    write_vrt(str(target), tile_paths, relative=False, crs=4326)
    return target


def _build_miniswhite(dir_path: Path, target: Path) -> Path:
    """MinIsWhite uint8 fixture written via tifffile (photometric=0)."""
    del dir_path
    import tifffile  # local import: only this builder needs tifffile
    rng = np.random.default_rng(seed=21324)
    arr = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    tifffile.imwrite(
        str(target), arr, photometric="miniswhite",
        compression="none", metadata=None,
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
        source_type=_SRC_LOCAL_TIFF,
        builder=_build_int16_single_band,
    ),
    _FixtureSpec(
        fix_id="uint16-multiband-tiled",
        dtype=np.dtype("uint16"),
        expected_dims=("y", "x", "band"),
        expected_crs_epsg=4326,
        expected_nodata=None,
        expected_masked=None,
        source_type=_SRC_LOCAL_TIFF,
        builder=_build_uint16_multiband_tiled,
    ),
    _FixtureSpec(
        fix_id="float32-nodata",
        dtype=np.dtype("float32"),
        expected_dims=("y", "x"),
        expected_crs_epsg=4326,
        expected_nodata=-9999.0,
        expected_masked=True,
        source_type=_SRC_LOCAL_TIFF,
        builder=_build_float32_with_nodata,
    ),
    _FixtureSpec(
        fix_id="int8-unmasked",
        dtype=np.dtype("int8"),
        expected_dims=("y", "x"),
        expected_crs_epsg=4326,
        expected_nodata=-128,
        expected_masked=False,
        source_type=_SRC_LOCAL_TIFF,
        builder=_build_int8_unmasked,
        read_kwargs={"mask_nodata": False},
    ),
    _FixtureSpec(
        fix_id="cog-float32",
        dtype=np.dtype("float32"),
        expected_dims=("y", "x"),
        expected_crs_epsg=4326,
        expected_nodata=None,
        expected_masked=None,
        source_type=_SRC_LOCAL_TIFF,
        builder=_build_cog,
    ),
    _FixtureSpec(
        fix_id="vrt-mosaic",
        dtype=np.dtype("float32"),
        expected_dims=("y", "x"),
        expected_crs_epsg=4326,
        expected_nodata=None,
        expected_masked=None,
        source_type=_SRC_LOCAL_VRT,
        builder=_build_vrt_mosaic,
    ),
    _FixtureSpec(
        fix_id="miniswhite",
        dtype=np.dtype("uint8"),
        expected_dims=("y", "x"),
        expected_crs_epsg=None,
        expected_nodata=None,
        expected_masked=None,
        source_type=_SRC_LOCAL_TIFF,
        builder=_build_miniswhite,
        marks=(_skip_no_tifffile,),
    ),
]


def _fixture_params() -> list:
    """Build the pytest.param list for the fixture matrix."""
    return [pytest.param(spec, id=spec.fix_id, marks=spec.marks)
            for spec in _FIXTURES]


@pytest.fixture(scope="session")
def _parity_matrix_dir(tmp_path_factory):
    """Session-scoped scratch dir, one write per fixture id.

    Tests reuse files across cells. The matrix has up to 8 backends
    x 7 fixtures; without caching every backend-row would rewrite the
    fixture from scratch.
    """
    return tmp_path_factory.mktemp("parity_matrix_2132")


@pytest.fixture
def parity_fixture(_parity_matrix_dir):
    """Resolve a :class:`_FixtureSpec` to an on-disk path.

    Files are cached across the session: a fixture already present on
    disk is returned without rewriting.
    """
    dir_path = _parity_matrix_dir

    def _resolve(spec: _FixtureSpec) -> Path:
        safe_id = spec.fix_id.replace("/", "-")
        suffix = ".vrt" if spec.source_type == _SRC_LOCAL_VRT else ".tif"
        path = dir_path / f"parity_2132_{safe_id}{suffix}"
        if path.exists():
            return path
        return spec.builder(dir_path, path)
    return _resolve


# ---------------------------------------------------------------------------
# Transport adapters for the HTTP and fsspec backend rows
# ---------------------------------------------------------------------------

class _MatrixRangeHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler with Range support, serving a payload dict by path.

    The dict ``payload_by_path`` is set by the server fixture and maps
    URL paths (``/parity_2132_int16-single-band.tif``) to bytes.
    """

    payload_by_path: dict[str, bytes] = {}

    def do_GET(self):  # noqa: N802
        payload = self.payload_by_path.get(self.path)
        if payload is None:
            self.send_response(404)
            self.end_headers()
            return
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes="):]
            start_s, _, end_s = spec.partition("-")
            start = int(start_s)
            end = int(end_s) if end_s else len(payload) - 1
            chunk = payload[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header(
                "Content-Range",
                f"bytes {start}-{start + len(chunk) - 1}/{len(payload)}",
            )
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args, **_kwargs):  # noqa: A003
        # Silence the default access log during tests.
        pass


@pytest.fixture(scope="session")
def _matrix_http_server_session():
    """Shared loopback HTTP server for the http-cog backend row.

    Started once per pytest session and torn down on session exit. The
    payload dict on the handler is cleared between tests by the
    function-scoped ``_matrix_http_server`` wrapper below; this fixture
    only owns the socket and the thread.
    """
    handler_cls = type(
        "MatrixRangeHandler", (_MatrixRangeHandler,),
        {"payload_by_path": dict(_MatrixRangeHandler.payload_by_path)},
    )
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", handler_cls
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def _matrix_http_server(_matrix_http_server_session):
    """Function-scoped HTTP server view: clears stale payloads after each test.

    Without this, the session-scoped ``payload_by_path`` dict accumulates
    one entry per cell and never releases the bytes. Keeping it
    function-scoped means a test only sees the URL paths it uploaded.
    """
    base_url, handler_cls = _matrix_http_server_session
    handler_cls.payload_by_path.clear()
    try:
        yield base_url, handler_cls
    finally:
        handler_cls.payload_by_path.clear()


def _deliver_via_http(spec: "_FixtureSpec | _ErrorFixtureSpec", on_disk: Path,
                      base_url: str, handler_cls,
                      monkeypatch) -> str:
    """Upload an on-disk fixture into the shared HTTP server and return URL.

    The success matrix passes a :class:`_FixtureSpec`; the error
    sub-matrix passes an :class:`_ErrorFixtureSpec`. Both expose
    ``fix_id`` so the function consumes either.
    """
    del spec  # the spec is unused; signature kept for symmetry with fsspec
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    with open(on_disk, "rb") as f:
        payload = f.read()
    url_path = f"/{on_disk.name}"
    handler_cls.payload_by_path[url_path] = payload
    return f"{base_url}{url_path}"


def _deliver_via_fsspec(spec: "_FixtureSpec | _ErrorFixtureSpec",
                        on_disk: Path) -> str:
    """Pipe an on-disk fixture into fsspec's memory:// filesystem.

    Returns the ``memory://`` URI the read path should consume. The
    memory filesystem persists for the pytest process, so the URI path
    is namespaced by the fixture id to avoid collisions across cells.
    """
    import fsspec
    fs = fsspec.filesystem("memory")
    safe_id = spec.fix_id.replace("/", "-")
    uri_path = f"/parity_2132_{safe_id}.tif"
    with open(on_disk, "rb") as f:
        payload = f.read()
    fs.pipe(uri_path, payload)
    return f"memory://{uri_path}"


# ---------------------------------------------------------------------------
# Materialisation + comparison helpers
# ---------------------------------------------------------------------------

def _materialise(da: xr.DataArray) -> np.ndarray:
    """Return a numpy view of ``da.data`` regardless of backend."""
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
    ref: xr.DataArray,
    label: str,
) -> None:
    """Assert every parity field for one (fixture, backend) cell.

    Run against an already-read DataArray rather than re-opening here so
    the same helper applies to both ``open_geotiff(path, **kwargs)`` and
    the explicit ``read_geotiff_dask`` / ``read_geotiff_gpu`` /
    ``read_vrt`` entry points wired up in follow-up PRs. ``ref`` is the
    eager-numpy read of the same fixture, used as the reference for the
    pixel array, coord values, dims, and transform tuple.

    ``spec.dtype`` and ``spec.expected_crs_epsg`` /
    ``spec.expected_nodata`` are asserted against the actual
    independently of the reference, so a bug that silently changes
    them in *every* backend still fails this cell.
    """
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

    # Coord values and coord dtype, per axis. Skip axes that the
    # reference does not carry as a coord (e.g. ``band`` for some
    # multiband layouts when the writer drops the index).
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

    # Transform tuple. The VRT path uses ``rasterio.Affine`` instances
    # which compare equal to 6-tuples via ``__eq__``.
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

    # Masking state: ``attrs['masked_nodata']`` reflects whether the
    # reader replaced sentinel pixels with NaN (#2092 / #2127). The
    # contract is fixed once a fixture declares a sentinel.
    if spec.expected_masked is not None:
        actual_masked = da.attrs.get("masked_nodata")
        assert actual_masked == spec.expected_masked, (
            f"{label}: attrs['masked_nodata'] {actual_masked!r} != "
            f"expected {spec.expected_masked!r}"
        )

    # Selected canonical attrs: reference and actual agree on presence
    # and value. The list is intentionally narrow until issue #1984's
    # contract version stamp lands.
    canonical_keys = ("raster_type", "transform", "crs", "crs_wkt")
    for key in canonical_keys:
        ref_v = ref.attrs.get(key)
        actual_v = da.attrs.get(key)
        assert ref_v == actual_v, (
            f"{label}: canonical attr {key!r} differs "
            f"ref={ref_v!r} actual={actual_v!r}"
        )


# ---------------------------------------------------------------------------
# Source-delivery wrapper: hands one fixture to a specific source type
# ---------------------------------------------------------------------------

def _resolve_source(
    spec: _FixtureSpec, on_disk: Path, backend: _BackendSpec,
    *,
    http_state, monkeypatch,
) -> object:
    """Return the value that should be passed as ``source`` to ``open_geotiff``.

    Most backends consume the on-disk path verbatim. The ``http-cog``
    and ``fsspec-memory`` backends override the source type, so the
    fixture bytes are re-served through the requested transport.
    """
    target_type = backend.source_type_override or spec.source_type
    if target_type == _SRC_LOCAL_TIFF or target_type == _SRC_LOCAL_VRT:
        return str(on_disk)
    if target_type == _SRC_HTTP:
        base_url, handler_cls = http_state
        return _deliver_via_http(spec, on_disk, base_url, handler_cls, monkeypatch)
    if target_type == _SRC_FSSPEC:
        return _deliver_via_fsspec(spec, on_disk)
    raise AssertionError(f"unknown source type: {target_type}")


# ---------------------------------------------------------------------------
# The single matrix test entry point
# ---------------------------------------------------------------------------

@requires_loopback
@pytest.mark.parametrize("spec", _fixture_params())
@pytest.mark.parametrize("backend", _backend_params())
def test_backend_parity_matrix(
    parity_fixture, spec, backend,
    _matrix_http_server, monkeypatch,
):
    """One cell per (fixture, backend). Asserts every parity field.

    A new backend or fixture lights up automatically on the next pytest
    run -- no per-cell test function needed. Incompatible (backend,
    source) pairs skip cleanly rather than failing.
    """
    if spec.source_type not in backend.compat:
        pytest.skip(
            f"backend={backend.backend_id} does not consume source_type="
            f"{spec.source_type} (fixture={spec.fix_id})"
        )

    path = parity_fixture(spec)

    # Eager-numpy reference: read the same on-disk fixture through the
    # default backend so the matrix compares like-for-like.
    ref = open_geotiff(str(path), **spec.read_kwargs)

    # Resolve the source the backend should actually consume (the
    # on-disk path for local backends, an HTTP URL for the HTTP row,
    # or a memory:// URI for the fsspec row).
    source = _resolve_source(
        spec, path, backend,
        http_state=_matrix_http_server, monkeypatch=monkeypatch,
    )

    da = open_geotiff(source, **backend.kwargs, **spec.read_kwargs)
    label = (
        f"fixture={spec.fix_id} backend={backend.backend_id} "
        f"kwargs={backend.kwargs}"
    )
    assert_parity(da, spec, ref=ref, label=label)


# ---------------------------------------------------------------------------
# Error-fixture sub-matrix: rotated ModelTransformationTag without opt-in
# ---------------------------------------------------------------------------

_ROTATED_M = (
    8.660254037844387, -5.0, 0.0, 100.0,   # x row (30 deg rotation, pix=10)
    5.0, 8.660254037844387, 0.0, 200.0,    # y row
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def _write_rotated_tiff(path: Path, arr: np.ndarray) -> None:
    """Hand-build a TIFF with a rotated ``ModelTransformationTag``.

    Mirrors the minimal writer used by
    ``test_allow_rotated_geotiff_2115.py`` so the matrix can assert
    error behaviour without depending on rasterio / GDAL.
    """
    import struct
    h, w = arr.shape
    arr = np.ascontiguousarray(arr.astype("<u2"))
    header_size = 8
    strip_size = h * w * 2
    transform_off = header_size + strip_size
    transform_size = 16 * 8
    ifd_off = transform_off + transform_size

    entries = [
        (256, 3, 1, w),
        (257, 3, 1, h),
        (258, 3, 1, 16),
        (259, 3, 1, 1),
        (262, 3, 1, 1),
        (273, 4, 1, header_size),
        (277, 3, 1, 1),
        (278, 3, 1, h),
        (279, 4, 1, strip_size),
        (339, 3, 1, 1),
        # 34264 is TAG_MODEL_TRANSFORMATION.
        (34264, 12, 16, transform_off),
    ]
    entries.sort(key=lambda e: e[0])
    ifd_bytes = struct.pack("<H", len(entries))
    for tag, type_id, count, val in entries:
        if type_id == 3:
            ifd_bytes += struct.pack("<HHIHH", tag, type_id, count, val, 0)
        else:
            ifd_bytes += struct.pack("<HHII", tag, type_id, count, val)
    ifd_bytes += struct.pack("<I", 0)

    with open(path, "wb") as f:
        f.write(struct.pack("<HHI", 0x4949, 42, ifd_off))
        f.write(arr.tobytes())
        f.write(struct.pack("<16d", *_ROTATED_M))
        f.write(ifd_bytes)


@dataclass(frozen=True)
class _ErrorFixtureSpec:
    """Declarative description of an error-only fixture.

    Attributes
    ----------
    fix_id
        Stable id for parametrization.
    exc
        Expected exception class. Each backend that consumes the fixture
        must raise this from ``open_geotiff``.
    match
        Substring the exception message must contain.
    source_type
        On-disk delivery type (constrains the backend matrix the same
        way :class:`_FixtureSpec.source_type` does).
    builder
        Writes the file at ``target`` and returns the resolved path.
    """

    fix_id: str
    exc: type[BaseException]
    match: str
    source_type: str
    builder: Callable[[Path, Path], Path]


def _build_rotated_no_optin(dir_path: Path, target: Path) -> Path:
    del dir_path
    arr = np.arange(20, dtype="<u2").reshape(4, 5)
    _write_rotated_tiff(target, arr)
    return target


_ERROR_FIXTURES: list[_ErrorFixtureSpec] = [
    _ErrorFixtureSpec(
        fix_id="rotated-no-allow_rotated",
        exc=RotatedTransformError,
        match="rotation",
        source_type=_SRC_LOCAL_TIFF,
        builder=_build_rotated_no_optin,
    ),
]


@pytest.fixture
def error_parity_fixture(_parity_matrix_dir):
    dir_path = _parity_matrix_dir

    def _resolve(spec: _ErrorFixtureSpec) -> Path:
        safe_id = spec.fix_id.replace("/", "-")
        path = dir_path / f"parity_2132_err_{safe_id}.tif"
        if path.exists():
            return path
        return spec.builder(dir_path, path)
    return _resolve


@requires_loopback
@pytest.mark.parametrize("error_spec", _ERROR_FIXTURES,
                         ids=lambda s: s.fix_id)
@pytest.mark.parametrize("backend", _backend_params())
def test_backend_parity_matrix_errors(
    error_parity_fixture, error_spec, backend,
    _matrix_http_server, monkeypatch,
):
    """Error fixtures raise the same exception on every compatible backend.

    Backends incompatible with the error fixture's source type skip;
    every remaining cell asserts the same ``pytest.raises`` contract.
    """
    if error_spec.source_type not in backend.compat:
        pytest.skip(
            f"backend={backend.backend_id} does not consume source_type="
            f"{error_spec.source_type}"
        )

    path = error_parity_fixture(error_spec)

    # Re-route the path through the requested transport (HTTP, fsspec)
    # so the error surfaces on the same code path as the success
    # matrix.
    source = _resolve_source(
        error_spec, path, backend,
        http_state=_matrix_http_server, monkeypatch=monkeypatch,
    )

    with pytest.raises(error_spec.exc, match=error_spec.match):
        # ``open_geotiff`` may return lazily for chunked reads, so
        # force a materialisation inside the ``pytest.raises`` block
        # so the error surfaces here regardless of laziness.
        out = open_geotiff(source, **backend.kwargs)
        _materialise(out)


# ===========================================================================
# Full-fixture parity gate over the golden corpus
# ===========================================================================
#
# Compares every read backend against the eager-numpy reference on every
# manifest fixture. Originally lived in
# ``test_backend_full_parity_2211.py``; merged here so a single file owns
# all matrix-style backend-parity assertions.

_HAS_YAML = importlib.util.find_spec("yaml") is not None
_HAS_RASTERIO = importlib.util.find_spec("rasterio") is not None

if _HAS_YAML and _HAS_RASTERIO:
    from xrspatial.geotiff.tests.golden_corpus import generate as _fp_generate
    from xrspatial.geotiff.tests.golden_corpus._marks import \
        fast_slow_marks_for as _fp_fast_slow_marks_for

    _FP_FIXTURES_DIR = (
        pathlib.Path(_fp_generate.__file__).resolve().parent / "fixtures"
    )
else:
    # Defined so attribute access in gated paths never raises NameError
    # under static analysis or a future refactor that drops a guard.
    _FP_FIXTURES_DIR = None

# Chunk size for the dask rows. Most corpus fixtures are 64x64 or
# smaller, so 32 produces either a 2x2 chunk grid or a single chunk.
_FP_CHUNK_SIZE = 32


_FP_CANONICAL_METADATA_KEYS: tuple[str, ...] = (
    "raster_type",
    "x_resolution",
    "y_resolution",
    "resolution_unit",
    "georef_status",
)


# Fixtures the full parity matrix skips outright. Each entry cites the
# source of the divergence.
_FP_INTENTIONAL_SKIPS: dict[str, str] = {
    "nodata_miniswhite_uint8": (
        "MinIsWhite photometric inversion: xrspatial inverts pixels per "
        "issue 1797; rasterio leaves them raw. The matrix would compare "
        "inverted-vs-raw and fail on every row. Covered by the dedicated "
        "miniswhite parity case in this file."
    ),
    "compression_jpeg_uint8_ycbcr": (
        "JPEG-YCbCr is lossy and exposes a (bands, y, x) vs (y, x, band) "
        "axis-order divergence that the golden-corpus oracle handles "
        "via _normalise_axis_order but this gate's dims/coords check "
        "cannot, because the dims tuple itself differs."
    ),
}


_FP_BACKEND_SKIPS: dict[str, dict[str, str]] = {
    "vrt_eager": {
        "crs_citation_only": (
            "VRT round-trip mutates user-defined CRS WKT."
        ),
        "overview_external_ovr_uint16": (
            "External .ovr sidecar is not preserved through VRT wrap."
        ),
        "sparse_tiled_uint16": (
            "Sparse-tile holes are not preserved through VRT wrap."
        ),
        "extra_tags_uint16": (
            "VRT wrap does not propagate source TIFF resolution tags or "
            "extra_tags."
        ),
    },
    "http_fsspec": {
        "overview_external_ovr_uint16": (
            "External .ovr sidecar reader is not wired into the cloud "
            "source path."
        ),
    },
}


@dataclass(frozen=True)
class _FpBackend:
    """One row of the full-parity matrix."""

    backend_id: str
    read: Callable[[pathlib.Path, str], xr.DataArray]
    available: bool
    unavailable_reason: str
    skips: dict[str, str] = field(default_factory=dict)


# Experimental and internal-only codecs require an explicit opt-in on
# the read side. The full parity matrix is orthogonal to the opt-in
# contract, so pass both flags through every opener.
_FP_OPTIN = {
    "allow_experimental_codecs": True,
    "allow_internal_only_jpeg": True,
}


def _fp_read_eager_numpy(path: pathlib.Path, *_: object) -> xr.DataArray:
    return open_geotiff(str(path), **_FP_OPTIN)


def _fp_read_dask_numpy(path: pathlib.Path, *_: object) -> xr.DataArray:
    return open_geotiff(str(path), chunks=_FP_CHUNK_SIZE, **_FP_OPTIN)


def _fp_read_gpu(path: pathlib.Path, *_: object) -> xr.DataArray:
    return open_geotiff(
        str(path), gpu=True, on_gpu_failure="strict", **_FP_OPTIN)


def _fp_read_dask_gpu(path: pathlib.Path, *_: object) -> xr.DataArray:
    return open_geotiff(
        str(path), gpu=True, chunks=_FP_CHUNK_SIZE,
        on_gpu_failure="strict", **_FP_OPTIN,
    )


def _fp_vrt_cache_dir(fixtures_dir: pathlib.Path) -> pathlib.Path:
    """Per-session VRT scratch directory, keyed by fixtures path digest."""
    import hashlib
    import tempfile
    base = pathlib.Path(tempfile.gettempdir()) / "xrspatial_parity_vrt_cache"
    base.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(fixtures_dir).encode()).hexdigest()[:12]
    sub = base / f"fix_{digest}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def _fp_read_vrt_eager(path: pathlib.Path, fixture_id: str) -> xr.DataArray:
    """Wrap ``path`` in a one-source VRT and read it back via xrspatial."""
    import shutil
    cache_dir = _fp_vrt_cache_dir(path.parent)
    local_src = cache_dir / f"{fixture_id}.tif"
    if not local_src.exists():
        shutil.copy2(path, local_src)
    vrt_path = cache_dir / f"{fixture_id}.vrt"
    if not vrt_path.exists():
        write_vrt(str(vrt_path), [str(local_src)])
    return open_geotiff(str(vrt_path), **_FP_OPTIN)


def _fp_read_http_fsspec(path: pathlib.Path, fixture_id: str) -> xr.DataArray:
    """Serve the fixture bytes through fsspec's in-process memory FS."""
    import fsspec
    fs = fsspec.filesystem("memory")
    key = f"/corpus_full_parity/{fixture_id}.tif"
    with open(path, "rb") as f:
        fs.pipe(key, f.read())
    try:
        da = open_geotiff(f"memory://{key}", **_FP_OPTIN)
    finally:
        try:
            fs.rm(key)
        except FileNotFoundError:
            pass
    return da


_FP_GPU_UNAVAILABLE_REASON = (
    "GPU backend skipped LOUDLY: cupy + CUDA are not available in this "
    "environment. GPU and Dask+GPU rows must skip explicitly rather "
    "than silently collect zero tests. To exercise these rows, install "
    "cupy and ensure a CUDA device is reachable."
)

_FP_DASK_UNAVAILABLE_REASON = (
    "dask backend skipped: dask is not installed."
)

_FP_FSSPEC_UNAVAILABLE_REASON = (
    "http_fsspec backend skipped: fsspec is not installed."
)


_FP_BACKENDS: list[_FpBackend] = [
    _FpBackend(
        backend_id="eager_numpy",
        read=_fp_read_eager_numpy,
        available=True,
        unavailable_reason="",
    ),
    _FpBackend(
        backend_id="dask_numpy",
        read=_fp_read_dask_numpy,
        available=_HAS_DASK,
        unavailable_reason=_FP_DASK_UNAVAILABLE_REASON,
    ),
    _FpBackend(
        backend_id="gpu",
        read=_fp_read_gpu,
        available=_HAS_GPU,
        unavailable_reason=_FP_GPU_UNAVAILABLE_REASON,
    ),
    _FpBackend(
        backend_id="dask_gpu",
        read=_fp_read_dask_gpu,
        available=_HAS_GPU and _HAS_DASK,
        unavailable_reason=(
            _FP_GPU_UNAVAILABLE_REASON if not _HAS_GPU
            else _FP_DASK_UNAVAILABLE_REASON
        ),
        skips=dict(_FP_BACKEND_SKIPS.get("dask_gpu", {})),
    ),
    _FpBackend(
        backend_id="vrt_eager",
        read=_fp_read_vrt_eager,
        available=_HAS_YAML and _HAS_RASTERIO,
        unavailable_reason="yaml + rasterio required",
        skips=dict(_FP_BACKEND_SKIPS["vrt_eager"]),
    ),
    _FpBackend(
        backend_id="http_fsspec",
        read=_fp_read_http_fsspec,
        available=_HAS_FSSPEC,
        unavailable_reason=_FP_FSSPEC_UNAVAILABLE_REASON,
        skips=dict(_FP_BACKEND_SKIPS["http_fsspec"]),
    ),
]


def _fp_resolved_fixtures() -> list[dict[str, Any]]:
    """Return manifest entries with defaults merged, sorted by id."""
    if not (_HAS_YAML and _HAS_RASTERIO):
        return []
    manifest = _fp_generate.load_manifest()
    entries = _fp_generate.validate(manifest)
    entries.sort(key=lambda e: e["id"])
    return entries


def _fp_fixture_path(entry: dict[str, Any]) -> pathlib.Path:
    return _FP_FIXTURES_DIR / f"{entry['id']}.tif"


def _fp_is_lossy(entry: dict[str, Any]) -> bool:
    tol = entry.get("tolerance") or {}
    return bool(tol.get("lossy", False))


_FP_FIXTURES = _fp_resolved_fixtures()


def _fp_build_fixture_params() -> list:
    """One ``pytest.param`` per manifest entry, with slow/skip marks."""
    if not (_HAS_YAML and _HAS_RASTERIO):
        return [pytest.param(
            None, id="no-manifest",
            marks=pytest.mark.skip(reason="yaml + rasterio required"),
        )]
    out = []
    for entry in _FP_FIXTURES:
        fid = entry["id"]
        marks = list(_fp_fast_slow_marks_for(entry))
        if fid in _FP_INTENTIONAL_SKIPS:
            marks.append(pytest.mark.skip(reason=_FP_INTENTIONAL_SKIPS[fid]))
        out.append(pytest.param(entry, id=fid, marks=marks))
    return out


def _fp_build_backend_params() -> list:
    """One ``pytest.param`` per backend; unavailable rows skip."""
    out = []
    for backend in _FP_BACKENDS:
        marks = []
        if not backend.available:
            marks.append(pytest.mark.skip(reason=backend.unavailable_reason))
        out.append(pytest.param(backend, id=backend.backend_id, marks=marks))
    return out


_FP_FIXTURE_PARAMS = _fp_build_fixture_params()
_FP_BACKEND_PARAMS = _fp_build_backend_params()


def _fp_is_nan_sentinel(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _fp_assert_pixels_close(
    ref: np.ndarray, cand: np.ndarray, *, lossy: bool, label: str,
) -> None:
    assert ref.shape == cand.shape, (
        f"{label}: shape mismatch ref={ref.shape} cand={cand.shape}"
    )
    if lossy:
        return
    assert ref.dtype == cand.dtype, (
        f"{label}: dtype mismatch ref={ref.dtype} cand={cand.dtype}"
    )
    if ref.dtype.kind == "f":
        # Bit-exact today across decode paths. ``rtol=1e-12`` tracks
        # data magnitude so small-magnitude fixtures aren't held to a
        # slacker bar. ``atol=0`` keeps zeros strict.
        ok = np.allclose(ref, cand, rtol=1e-12, atol=0.0, equal_nan=True)
        if not ok:
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


def _fp_assert_dims_and_coords(
    ref: xr.DataArray, cand: xr.DataArray, *, label: str,
) -> None:
    assert ref.dims == cand.dims, (
        f"{label}: dims mismatch ref={ref.dims!r} cand={cand.dims!r}"
    )
    for axis in ref.dims:
        if axis not in ref.coords:
            assert axis not in cand.coords, (
                f"{label}: candidate has coord {axis!r} that the "
                f"reference does not"
            )
            continue
        assert axis in cand.coords, (
            f"{label}: candidate is missing coord {axis!r}"
        )
        ref_c = np.asarray(ref.coords[axis].values)
        cand_c = np.asarray(cand.coords[axis].values)
        assert ref_c.dtype == cand_c.dtype, (
            f"{label}: coord {axis!r} dtype ref={ref_c.dtype} "
            f"cand={cand_c.dtype}"
        )
        if ref_c.dtype.kind == "f":
            assert np.allclose(ref_c, cand_c, rtol=0.0, atol=1e-9), (
                f"{label}: coord {axis!r} values differ"
            )
        else:
            assert np.array_equal(ref_c, cand_c), (
                f"{label}: coord {axis!r} values differ"
            )


def _fp_assert_transform_attrs(
    ref: xr.DataArray, cand: xr.DataArray, *, label: str,
) -> None:
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
        f"{label}: transform must be a 6-tuple"
    )
    for i, (a, b) in enumerate(zip(ref_tup, cand_tup)):
        assert abs(a - b) <= 1e-9, (
            f"{label}: transform[{i}] differs ref={a!r} cand={b!r}"
        )


def _fp_assert_crs_attrs(
    ref: xr.DataArray, cand: xr.DataArray, *, label: str,
) -> None:
    for key in ("crs", "crs_wkt"):
        ref_v = ref.attrs.get(key)
        cand_v = cand.attrs.get(key)
        assert ref_v == cand_v, (
            f"{label}: attr {key!r} differs ref={ref_v!r} cand={cand_v!r}"
        )


def _fp_assert_nodata_attrs(
    ref: xr.DataArray, cand: xr.DataArray, *, label: str,
) -> None:
    ref_nd = ref.attrs.get("nodata")
    cand_nd = cand.attrs.get("nodata")
    if ref_nd is None and cand_nd is None:
        pass
    else:
        ref_is_nan = _fp_is_nan_sentinel(ref_nd)
        cand_is_nan = _fp_is_nan_sentinel(cand_nd)
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
    ref_dtype = np.dtype(ref.dtype)
    cand_dtype = np.dtype(cand.dtype)
    assert ref_dtype == cand_dtype, (
        f"{label}: pixel dtype differs ref={ref_dtype} cand={cand_dtype}"
    )


def _fp_assert_canonical_metadata_attrs(
    ref: xr.DataArray, cand: xr.DataArray, *, label: str,
) -> None:
    for key in _FP_CANONICAL_METADATA_KEYS:
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


@pytest.fixture(scope="module")
def _fp_reference_cache() -> dict[str, xr.DataArray]:
    """Cache eager-numpy reads keyed by fixture id."""
    return {}


def _fp_reference_for(
    entry: dict[str, Any], cache: dict[str, xr.DataArray],
) -> xr.DataArray:
    fid = entry["id"]
    if fid not in cache:
        cache[fid] = open_geotiff(str(_fp_fixture_path(entry)), **_FP_OPTIN)
    return cache[fid]


@pytest.mark.parametrize("backend", _FP_BACKEND_PARAMS)
@pytest.mark.parametrize("manifest_entry", _FP_FIXTURE_PARAMS)
def test_backend_full_parity(
    manifest_entry,
    backend,
    _fp_reference_cache,
):
    """Full-corpus contract gate for every (backend, fixture) cell.

    1. Look up (or read) the eager-numpy reference for the fixture.
    2. Read the same fixture through ``backend.read``.
    3. Assert pixels, dims + coords, transform/georef, CRS, nodata,
       and the curated canonical metadata attrs.
    """
    if manifest_entry is None:
        pytest.skip("yaml + rasterio required for the manifest fixture set")

    fixture_id = manifest_entry["id"]
    path = _fp_fixture_path(manifest_entry)
    if not path.exists():
        pytest.skip(
            f"fixture {fixture_id!r} has no .tif on disk; run "
            f"`python -m xrspatial.geotiff.tests.golden_corpus.generate`"
        )

    if fixture_id in backend.skips:
        pytest.skip(
            f"backend={backend.backend_id} cannot read fixture="
            f"{fixture_id}: {backend.skips[fixture_id]}"
        )

    reference = _fp_reference_for(manifest_entry, _fp_reference_cache)
    try:
        candidate = backend.read(path, fixture_id)
    except Exception as exc:
        raise AssertionError(
            f"backend={backend.backend_id} failed to read fixture="
            f"{fixture_id}: {type(exc).__name__}: {exc}"
        ) from exc

    label = f"fixture={fixture_id} backend={backend.backend_id}"

    ref_px = _materialise(reference)
    cand_px = _materialise(candidate)
    _fp_assert_pixels_close(
        ref_px, cand_px, lossy=_fp_is_lossy(manifest_entry), label=label,
    )
    _fp_assert_dims_and_coords(reference, candidate, label=label)
    _fp_assert_transform_attrs(reference, candidate, label=label)
    _fp_assert_crs_attrs(reference, candidate, label=label)
    _fp_assert_nodata_attrs(reference, candidate, label=label)
    _fp_assert_canonical_metadata_attrs(reference, candidate, label=label)


def test_taxonomy_ids_are_in_manifest():
    """Every fixture id in a skip table must exist in the manifest."""
    if not (_HAS_YAML and _HAS_RASTERIO):
        pytest.skip("yaml + rasterio required")
    manifest_ids = {e["id"] for e in _FP_FIXTURES}
    referenced: set[str] = set(_FP_INTENTIONAL_SKIPS)
    for backend in _FP_BACKENDS:
        referenced.update(backend.skips)
    stale = referenced - manifest_ids
    assert not stale, (
        f"skip tables reference unknown fixture ids: {sorted(stale)}"
    )


def test_gpu_skip_reason_is_loud():
    """GPU + Dask+GPU skips must be explicit, not silent."""
    for backend_id in ("gpu", "dask_gpu"):
        backend = next(b for b in _FP_BACKENDS if b.backend_id == backend_id)
        if backend.available:
            continue
        reason = backend.unavailable_reason
        assert "skipped LOUDLY" in reason or "skipped" in reason, (
            f"{backend_id} unavailable_reason is not explicit enough: "
            f"{reason!r}"
        )


def _fp_first_eligible_fixture() -> dict[str, Any] | None:
    """Pick a fast, on-disk fixture that none of the skip tables flag."""
    if not (_HAS_YAML and _HAS_RASTERIO):
        return None
    for entry in _FP_FIXTURES:
        if entry["id"] in _FP_INTENTIONAL_SKIPS:
            continue
        if not _fp_fixture_path(entry).exists():
            continue
        if "fast" in (entry.get("tags") or []):
            return entry
    for entry in _FP_FIXTURES:
        if (entry["id"] not in _FP_INTENTIONAL_SKIPS
                and _fp_fixture_path(entry).exists()):
            return entry
    return None


@pytest.mark.skipif(not _HAS_GPU, reason=_FP_GPU_UNAVAILABLE_REASON)
def test_gpu_backend_returns_cupy_array():
    """Sanity check: the gpu row returns a cupy-backed DataArray."""
    import cupy
    entry = _fp_first_eligible_fixture()
    if entry is None:
        pytest.skip("no eligible fixture on disk")
    da = _fp_read_gpu(_fp_fixture_path(entry), entry["id"])
    assert isinstance(da.data, cupy.ndarray), (
        f"gpu backend on fixture {entry['id']!r} returned "
        f"{type(da.data).__name__}, expected cupy.ndarray"
    )


@pytest.mark.skipif(not _HAS_DASK, reason=_FP_DASK_UNAVAILABLE_REASON)
def test_dask_backend_returns_dask_array():
    """Sanity check: the dask_numpy row returns a dask-backed DataArray."""
    entry = _fp_first_eligible_fixture()
    if entry is None:
        pytest.skip("no eligible fixture on disk")
    da = _fp_read_dask_numpy(_fp_fixture_path(entry), entry["id"])
    assert hasattr(da.data, "dask"), (
        f"dask_numpy backend on fixture {entry['id']!r} returned "
        f"data of type {type(da.data).__name__}, expected a "
        "dask-backed array."
    )


@pytest.mark.skipif(
    not (_HAS_GPU and _HAS_DASK),
    reason=(
        f"{_FP_GPU_UNAVAILABLE_REASON} (or dask missing: "
        f"{_FP_DASK_UNAVAILABLE_REASON})"
    ),
)
def test_dask_gpu_backend_returns_dask_of_cupy():
    """Sanity check: the dask_gpu row returns a dask-graph-of-cupy DataArray."""
    import cupy
    entry = _fp_first_eligible_fixture()
    if entry is None:
        pytest.skip("no eligible fixture on disk")
    da = _fp_read_dask_gpu(_fp_fixture_path(entry), entry["id"])
    assert hasattr(da.data, "dask"), (
        f"dask_gpu backend on fixture {entry['id']!r} dropped the "
        f"dask wrapping: data is {type(da.data).__name__}"
    )
    meta = getattr(da.data, "_meta", None)
    assert isinstance(meta, cupy.ndarray), (
        f"dask_gpu backend on fixture {entry['id']!r} carries a "
        f"non-cupy chunk prototype: {type(meta).__name__}"
    )


# ===========================================================================
# Attrs-parity across backends (canonical attrs + key sets)
# ===========================================================================
#
# Two layers:
#
# 1. The canonical-attrs parity gate: every backend (eager, dask, GPU,
#    dask-GPU, VRT) stamps the same canonical attrs for the same
#    fixture, modulo documented backend-specific keys.
# 2. The pass-through TIFF tag parity: x_resolution, y_resolution,
#    resolution_unit, image_description, extra_samples now agree
#    across numpy / dask / cupy / dask+cupy on a TIFF that has those
#    tags set.

# Canonical fixture geometry shared by the attrs writers.
_AP_ORIGIN_X = -100.0
_AP_ORIGIN_Y = 40.0
_AP_PIXEL = 0.001
_AP_CRS_EPSG = 4326
_AP_HEIGHT = 32
_AP_WIDTH = 32


# Keys excluded from the cross-backend attrs comparison because they
# are documented as backend-specific:
#
# * ``vrt_holes`` is VRT-only.
# * ``nodata_pixels_present`` rides on eager + VRT paths but stays
#   absent on dask paths (lazy would have to force compute).
# * TIFF tag pass-through attrs are VRT-omitted (the VRT carries no
#   TIFF tags of its own). They are pinned for non-VRT backends in
#   ``test_pass_through_tags_match_across_backends``.
_AP_BACKEND_SPECIFIC_KEYS = frozenset({
    'vrt_holes',
    'nodata_pixels_present',
    'extra_tags',
    'image_description',
    'extra_samples',
    'gdal_metadata',
    'gdal_metadata_xml',
    'x_resolution',
    'y_resolution',
    'resolution_unit',
    'colormap',
})


_AP_TRANSFORM_RTOL = 1e-9
_AP_TRANSFORM_ATOL = 1e-9


def _ap_coord_array(arr: np.ndarray) -> xr.DataArray:
    """Wrap a 2-D ``arr`` with axis-aligned x/y coords and EPSG CRS."""
    assert arr.ndim == 2, "test fixtures only use 2-D arrays"
    h, w = arr.shape
    assert (h, w) == (_AP_HEIGHT, _AP_WIDTH), (
        "fixture geometry constants are out of sync with array shape"
    )
    y = np.linspace(_AP_ORIGIN_Y, _AP_ORIGIN_Y - _AP_PIXEL * (h - 1), h)
    x = np.linspace(_AP_ORIGIN_X, _AP_ORIGIN_X + _AP_PIXEL * (w - 1), w)
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x})
    da.attrs['crs'] = _AP_CRS_EPSG
    return da


def _ap_attrs_for_parity(attrs) -> dict:
    """Drop backend-specific keys before comparing attrs across paths."""
    return {k: v for k, v in dict(attrs).items()
            if k not in _AP_BACKEND_SPECIFIC_KEYS}


def _ap_attrs_close(a: dict, b: dict) -> bool:
    """Compare attrs dicts, allowing tiny numeric drift in ``transform``."""
    if set(a.keys()) != set(b.keys()):
        return False
    for k, va in a.items():
        vb = b[k]
        if k == 'transform' and isinstance(va, tuple) and isinstance(vb, tuple):
            if len(va) != len(vb):
                return False
            for x, y in zip(va, vb):
                if not np.isclose(
                    float(x), float(y),
                    rtol=_AP_TRANSFORM_RTOL, atol=_AP_TRANSFORM_ATOL,
                ):
                    return False
        else:
            if va != vb:
                return False
    return True


@dataclass(frozen=True)
class _ApFixture:
    """One row in the attrs-parity fixture set."""
    name: str
    writer: Callable[[str], '_ApFixtureMeta']
    vrt_compatible: bool = True


@dataclass(frozen=True)
class _ApFixtureMeta:
    """Layout facts the VRT helper needs to wrap the on-disk TIFF."""
    vrt_dtype: str
    nodata: Any = None


def _ap_write_plain_float(path) -> _ApFixtureMeta:
    arr = np.random.default_rng(seed=2227).random(
        (_AP_HEIGHT, _AP_WIDTH)).astype(np.float32)
    to_geotiff(_ap_coord_array(arr), path)
    return _ApFixtureMeta(vrt_dtype='Float32')


def _ap_write_float_with_nodata(path) -> _ApFixtureMeta:
    rng = np.random.default_rng(seed=2227)
    arr = rng.random((_AP_HEIGHT, _AP_WIDTH)).astype(np.float32)
    arr[0:4, 0:4] = -9999.0
    da = _ap_coord_array(arr)
    da.attrs['nodata'] = -9999.0
    to_geotiff(da, path)
    return _ApFixtureMeta(vrt_dtype='Float32', nodata=-9999.0)


def _ap_write_int_with_nodata(path) -> _ApFixtureMeta:
    rng = np.random.default_rng(seed=2227)
    arr = rng.integers(0, 1000, size=(_AP_HEIGHT, _AP_WIDTH), dtype=np.uint16)
    arr[0:4, 0:4] = 65535
    da = _ap_coord_array(arr)
    da.attrs['nodata'] = 65535
    to_geotiff(da, path)
    return _ApFixtureMeta(vrt_dtype='UInt16', nodata=65535)


def _ap_write_uint8_no_nodata(path) -> _ApFixtureMeta:
    rng = np.random.default_rng(seed=2227)
    arr = rng.integers(0, 256, size=(_AP_HEIGHT, _AP_WIDTH), dtype=np.uint8)
    to_geotiff(_ap_coord_array(arr), path)
    return _ApFixtureMeta(vrt_dtype='Byte')


_AP_FIXTURES = (
    _ApFixture('plain_float', _ap_write_plain_float),
    _ApFixture('float_with_nodata', _ap_write_float_with_nodata),
    _ApFixture('int_with_nodata', _ap_write_int_with_nodata),
    _ApFixture('uint8_no_nodata', _ap_write_uint8_no_nodata),
)


@dataclass(frozen=True)
class _ApBackend:
    name: str
    open_fn: Callable
    available: bool = True


def _ap_open_eager(path):
    return open_geotiff(path)


def _ap_open_dask(path):
    return open_geotiff(path, chunks=16)


def _ap_open_gpu(path):
    return open_geotiff(path, gpu=True)


def _ap_open_dask_gpu(path):
    return open_geotiff(path, gpu=True, chunks=16)


def _ap_open_vrt(path, meta):
    """Wrap the TIFF in a single-source VRT and read via ``read_vrt``.

    GDAL GeoTransform XML expects the upper-left CORNER as origin while
    ``_ap_coord_array`` uses center-based coords, so the corner is
    shifted by half a pixel here.
    """
    import os

    from pyproj import CRS

    height = _AP_HEIGHT
    width = _AP_WIDTH
    corner_x = _AP_ORIGIN_X - _AP_PIXEL / 2.0
    corner_y = _AP_ORIGIN_Y + _AP_PIXEL / 2.0
    geo_transform = (
        f"{corner_x:.6f}, {_AP_PIXEL:.6f}, 0.0, "
        f"{corner_y:.6f}, 0.0, {-_AP_PIXEL:.6f}"
    )
    crs_wkt = CRS.from_epsg(_AP_CRS_EPSG).to_wkt()
    nodata_xml = (f"<NoDataValue>{meta.nodata}</NoDataValue>"
                  if meta.nodata is not None else '')
    vrt_path = path + '.vrt'
    abs_src = os.path.abspath(path)
    xml = (
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{height}">'
        f'  <SRS>{crs_wkt}</SRS>'
        f'  <GeoTransform>{geo_transform}</GeoTransform>'
        f'  <VRTRasterBand dataType="{meta.vrt_dtype}" band="1">'
        f'    {nodata_xml}'
        f'    <SimpleSource>'
        f'      <SourceFilename relativeToVRT="0">{abs_src}</SourceFilename>'
        f'      <SourceBand>1</SourceBand>'
        f'      <SrcRect xOff="0" yOff="0" '
        f'               xSize="{width}" ySize="{height}"/>'
        f'      <DstRect xOff="0" yOff="0" '
        f'               xSize="{width}" ySize="{height}"/>'
        f'    </SimpleSource>'
        f'  </VRTRasterBand>'
        f'</VRTDataset>'
    )
    with open(vrt_path, 'w') as f:
        f.write(xml)
    return read_vrt(vrt_path)


_AP_BACKENDS = (
    _ApBackend('eager_numpy', lambda path, meta: _ap_open_eager(path)),
    _ApBackend('dask_numpy', lambda path, meta: _ap_open_dask(path)),
    _ApBackend('gpu', lambda path, meta: _ap_open_gpu(path), available=_HAS_GPU),
    _ApBackend('dask_gpu', lambda path, meta: _ap_open_dask_gpu(path),
               available=_HAS_GPU),
    _ApBackend('vrt', _ap_open_vrt),
)


_AP_AVAILABLE_BACKENDS = tuple(b for b in _AP_BACKENDS if b.available)


@pytest.mark.parametrize('fixture', _AP_FIXTURES, ids=lambda f: f.name)
def test_canonical_attrs_match_across_backends(tmp_path, fixture):
    """Every backend stamps the same canonical attrs for the same fixture.

    The eager numpy path is the canonical reference. Documented
    backend-specific keys (see ``_AP_BACKEND_SPECIFIC_KEYS``) are
    carved out before the comparison.
    """
    path = str(tmp_path / f'attrs_parity_{fixture.name}.tif')
    meta = fixture.writer(path)

    baseline = _ap_attrs_for_parity(open_geotiff(path).attrs)

    divergences = {}
    for backend in _AP_AVAILABLE_BACKENDS:
        if backend.name == 'vrt' and not fixture.vrt_compatible:
            continue
        if backend.name == 'eager_numpy':
            continue
        try:
            da = backend.open_fn(path, meta)
        except Exception as exc:  # pragma: no cover
            divergences[backend.name] = f"open failed: {exc!r}"
            continue
        candidate = _ap_attrs_for_parity(da.attrs)
        if not _ap_attrs_close(candidate, baseline):
            only_in_baseline = {
                k: baseline.get(k) for k in baseline
                if baseline.get(k) != candidate.get(k)
            }
            only_in_candidate = {
                k: candidate.get(k) for k in candidate
                if candidate.get(k) != baseline.get(k)
            }
            divergences[backend.name] = {
                'baseline_diff': only_in_baseline,
                'candidate_diff': only_in_candidate,
            }

    assert not divergences, (
        f"attrs diverged from eager-numpy baseline for fixture "
        f"{fixture.name!r}:\n  baseline: {baseline}\n  diffs: {divergences}"
    )


@pytest.mark.parametrize('fixture', _AP_FIXTURES, ids=lambda f: f.name)
def test_canonical_attrs_keys_match_across_backends(tmp_path, fixture):
    """Stronger contract: the set of canonical attr keys is identical."""
    path = str(tmp_path / f'attrs_parity_keys_{fixture.name}.tif')
    meta = fixture.writer(path)

    baseline_keys = set(_ap_attrs_for_parity(open_geotiff(path).attrs).keys())

    diffs = {}
    for backend in _AP_AVAILABLE_BACKENDS:
        if backend.name == 'vrt' and not fixture.vrt_compatible:
            continue
        if backend.name == 'eager_numpy':
            continue
        try:
            da = backend.open_fn(path, meta)
        except Exception as exc:  # pragma: no cover
            diffs[backend.name] = f"open failed: {exc!r}"
            continue
        keys = set(_ap_attrs_for_parity(da.attrs).keys())
        if keys != baseline_keys:
            diffs[backend.name] = {
                'missing': sorted(baseline_keys - keys),
                'extra': sorted(keys - baseline_keys),
            }

    assert not diffs, (
        f"canonical attrs keyset diverged from eager-numpy baseline for "
        f"fixture {fixture.name!r}:\n  baseline keys: {sorted(baseline_keys)}\n"
        f"  diffs: {diffs}"
    )


# Pass-through TIFF tag parity across the four core backends.
#
# Before the fix, the dask and cupy paths emitted a narrower attrs set
# than the eager numpy path. The fix factored a single helper that
# every backend calls; this section pins that contract.

_AP_PASS_THROUGH_KEYS = (
    'x_resolution',
    'y_resolution',
    'resolution_unit',
    'image_description',
    'extra_samples',
)


def _ap_write_tiff_with_pass_through_tags(path):
    """Write a tiled 2-band float32 TIFF with the pass-through TIFF tags.

    Uses tifffile's first-class ``resolution`` / ``resolutionunit`` /
    ``description`` kwargs. ``metadata=None`` suppresses tifffile's
    auto-generated shape JSON in ImageDescription so the fixture
    description survives.
    """
    tifffile = pytest.importorskip("tifffile")
    arr = np.random.default_rng(seed=1548).random(
        (64, 64, 2)).astype(np.float32)
    tifffile.imwrite(
        path, arr, photometric='minisblack', planarconfig='contig',
        tile=(32, 32), compression='deflate',
        resolution=(300, 300), resolutionunit=2,
        description='attrs parity fixture',
        metadata=None,
    )
    return arr


def _ap_attrs_subset(attrs, keys):
    return {k: attrs.get(k) for k in keys}


def test_pass_through_tags_eager_numpy_baseline(tmp_path):
    """Eager numpy is the canonical reference for the pass-through keys."""
    pytest.importorskip("tifffile")
    path = str(tmp_path / 'pass_through_baseline.tif')
    _ap_write_tiff_with_pass_through_tags(path)

    da = open_geotiff(path)
    for key in _AP_PASS_THROUGH_KEYS:
        assert key in da.attrs, (
            f"eager numpy is the canonical reference and should always "
            f"emit '{key}'; got attrs={sorted(da.attrs.keys())}"
        )


def test_pass_through_tags_dask_matches_numpy(tmp_path):
    """The dask read path emits the same pass-through attrs as numpy."""
    pytest.importorskip("tifffile")
    path = str(tmp_path / 'pass_through_dask.tif')
    _ap_write_tiff_with_pass_through_tags(path)

    np_da = open_geotiff(path)
    dk_da = open_geotiff(path, chunks=32)

    np_subset = _ap_attrs_subset(np_da.attrs, _AP_PASS_THROUGH_KEYS)
    dk_subset = _ap_attrs_subset(dk_da.attrs, _AP_PASS_THROUGH_KEYS)

    assert dk_subset == np_subset, (
        f"dask attrs diverge from numpy:\n"
        f"  numpy: {np_subset}\n"
        f"  dask : {dk_subset}"
    )


@_skip_no_gpu
def test_pass_through_tags_cupy_matches_numpy(tmp_path):
    """Cupy / GPU read emits the same pass-through attrs as numpy."""
    pytest.importorskip("tifffile")
    path = str(tmp_path / 'pass_through_cupy.tif')
    _ap_write_tiff_with_pass_through_tags(path)

    np_da = open_geotiff(path)
    gpu_da = open_geotiff(path, gpu=True)

    np_subset = _ap_attrs_subset(np_da.attrs, _AP_PASS_THROUGH_KEYS)
    gpu_subset = _ap_attrs_subset(gpu_da.attrs, _AP_PASS_THROUGH_KEYS)

    assert gpu_subset == np_subset, (
        f"cupy attrs diverge from numpy:\n"
        f"  numpy: {np_subset}\n"
        f"  cupy : {gpu_subset}"
    )


@_skip_no_gpu
def test_pass_through_tags_dask_cupy_matches_numpy(tmp_path):
    """Combined dask+cupy read still emits the pass-through attrs."""
    pytest.importorskip("tifffile")
    path = str(tmp_path / 'pass_through_dask_cupy.tif')
    _ap_write_tiff_with_pass_through_tags(path)

    np_da = open_geotiff(path)
    combined = open_geotiff(path, gpu=True, chunks=32)

    np_subset = _ap_attrs_subset(np_da.attrs, _AP_PASS_THROUGH_KEYS)
    combined_subset = _ap_attrs_subset(combined.attrs, _AP_PASS_THROUGH_KEYS)

    assert combined_subset == np_subset, (
        f"dask+cupy attrs diverge from numpy:\n"
        f"  numpy     : {np_subset}\n"
        f"  dask+cupy : {combined_subset}"
    )


def test_pass_through_tags_all_backend_keysets_equal(tmp_path):
    """The full set of attrs keys is identical across available backends.

    Guards against a future read path silently dropping a different
    attr that no per-key test happens to cover.
    """
    pytest.importorskip("tifffile")
    path = str(tmp_path / 'pass_through_keysets.tif')
    _ap_write_tiff_with_pass_through_tags(path)

    np_keys = set(open_geotiff(path).attrs.keys())
    dk_keys = set(open_geotiff(path, chunks=32).attrs.keys())

    backend_keys = {'numpy': np_keys, 'dask+numpy': dk_keys}
    if _HAS_GPU:
        backend_keys['cupy'] = set(open_geotiff(path, gpu=True).attrs.keys())
        backend_keys['dask+cupy'] = set(
            open_geotiff(path, gpu=True, chunks=32).attrs.keys())

    differences = {
        name: keys ^ np_keys
        for name, keys in backend_keys.items()
        if keys != np_keys
    }
    assert not differences, (
        f"backend attrs keysets diverge from numpy:\n"
        f"  numpy keys: {sorted(np_keys)}\n"
        f"  diffs    : "
        + "\n             ".join(
            f"{name}: symmetric_diff={sorted(diff)}"
            for name, diff in differences.items()
        )
    )

# ===========================================================================
# Read-finalization helper unit tests (#2162)
# Source: test_finalization_helpers_2162.py
# ===========================================================================


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeTransform:
    """Stand-in for ``GeoInfo.transform``. Axis-aligned by default."""

    def __init__(self, origin_x=0.0, origin_y=10.0,
                 pixel_width=1.0, pixel_height=-1.0,
                 rotated_affine=None):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.rotated_affine = rotated_affine


class _FakeGeoInfo:
    """Minimal ``GeoInfo`` stand-in covering the fields the helpers read.

    Mirrors the shape used in ``test_geotiff_metadata_2139.py`` so the
    two test files stay in sync if the ``GeoInfo`` field set ever grows.
    """

    def __init__(
        self,
        *,
        transform=None,
        crs_epsg=None,
        crs_wkt=None,
        raster_type=1,  # RASTER_PIXEL_IS_AREA
        has_georef=True,
        nodata=None,
        extra_tags=None,
        image_description=None,
        extra_samples=None,
        gdal_metadata=None,
        gdal_metadata_xml=None,
        x_resolution=None,
        y_resolution=None,
        resolution_unit=None,
    ):
        self.transform = transform if transform is not None else _FakeTransform()
        self.crs_epsg = crs_epsg
        self.crs_wkt = crs_wkt
        self.raster_type = raster_type
        self.has_georef = has_georef
        self.nodata = nodata
        self.extra_tags = extra_tags
        self.image_description = image_description
        self.extra_samples = extra_samples
        self.gdal_metadata = gdal_metadata
        self.gdal_metadata_xml = gdal_metadata_xml
        self.x_resolution = x_resolution
        self.y_resolution = y_resolution
        self.resolution_unit = resolution_unit


def _default_geo_info(**overrides):
    """A minimal georeferenced ``_FakeGeoInfo`` for happy-path tests."""
    base = dict(crs_epsg=4326, crs_wkt='EPSG:4326', nodata=-9999)
    base.update(overrides)
    return _FakeGeoInfo(**base)


# ---------------------------------------------------------------------------
# Eager helper: attrs surface
# ---------------------------------------------------------------------------


def test_eager_float_input_sets_nodata_lifecycle_attrs():
    arr = np.array([[1.0, -9999.0], [2.0, 3.0]], dtype=np.float32)
    gi = _default_geo_info()

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=-9999,
        mask_sentinel=-9999,
        mask_nodata=True,
        dtype=None,
        window=None,
        name='t',
    )

    assert da.attrs['nodata'] == -9999
    assert da.attrs['masked_nodata'] is True
    assert da.attrs['nodata_pixels_present'] is True
    # No explicit dtype= cast was requested.
    assert 'nodata_dtype_cast' not in da.attrs
    # georef_status comes from _populate_attrs_from_geo_info.
    assert da.attrs['georef_status'] == 'full'
    # The sentinel pixel is now NaN.
    assert np.isnan(np.asarray(da.values)[0, 1])


def test_eager_int_input_promotes_and_sets_attrs():
    arr = np.array([[1, 0], [0, -1]], dtype=np.int16)
    gi = _default_geo_info(nodata=0)

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=0,
        mask_sentinel=0,
        mask_nodata=True,
        dtype=None,
        window=None,
        name='t',
    )

    assert da.dtype.kind == 'f'
    assert da.attrs['nodata'] == 0
    assert da.attrs['masked_nodata'] is True
    assert da.attrs['nodata_pixels_present'] is True
    assert da.attrs['georef_status'] == 'full'


def test_eager_explicit_dtype_records_dtype_cast():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gi = _default_geo_info()

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=-9999,
        mask_sentinel=-9999,
        mask_nodata=True,
        dtype='float64',
        window=None,
        name='t',
    )

    assert da.dtype == np.float64
    assert da.attrs['nodata_dtype_cast'] == 'float64'
    # No sentinel matched, so pixels_present is False rather than absent.
    assert da.attrs['nodata_pixels_present'] is False


def test_eager_no_sentinel_pixels_present_is_false():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gi = _default_geo_info()

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=-9999,
        mask_sentinel=-9999,
        mask_nodata=True,
        dtype=None,
        window=None,
        name='t',
    )

    assert da.attrs['nodata_pixels_present'] is False
    assert da.attrs['masked_nodata'] is True


# ---------------------------------------------------------------------------
# Eager helper: mask_nodata=False opt-out (issue #2052)
# ---------------------------------------------------------------------------


def test_eager_mask_nodata_false_skips_mask_keeps_attr_surface():
    arr = np.array([[1.0, -9999.0], [2.0, 3.0]], dtype=np.float32)
    gi = _default_geo_info()

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=-9999,
        mask_sentinel=-9999,
        mask_nodata=False,
        dtype=None,
        window=None,
        name='t',
    )

    # The sentinel pixel is preserved literally.
    assert da.values[0, 1] == -9999.0
    # ``masked_nodata=False`` per the #2092 contract.
    assert da.attrs['masked_nodata'] is False
    # ``nodata_pixels_present`` still surfaces so callers know a sentinel
    # pixel exists in the buffer (#2135).
    assert da.attrs['nodata_pixels_present'] is True


def test_eager_mask_nodata_false_no_sentinel_present():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    gi = _default_geo_info()

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=-9999,
        mask_sentinel=-9999,
        mask_nodata=False,
        dtype=None,
        window=None,
        name='t',
    )

    assert da.attrs['nodata_pixels_present'] is False
    assert da.attrs['masked_nodata'] is False


# ---------------------------------------------------------------------------
# Eager helper: no declared sentinel
# ---------------------------------------------------------------------------


def test_eager_no_nodata_omits_nodata_attrs():
    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    gi = _default_geo_info(nodata=None)

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=None,
        mask_sentinel=None,
        mask_nodata=True,
        dtype=None,
        window=None,
        name='t',
    )

    assert 'nodata' not in da.attrs
    assert 'masked_nodata' not in da.attrs
    assert 'nodata_pixels_present' not in da.attrs
    assert 'nodata_dtype_cast' not in da.attrs


# ---------------------------------------------------------------------------
# Eager helper: mask_sentinel != nodata (GPU MinIsWhite inversion, #1809)
# ---------------------------------------------------------------------------


def test_eager_mask_sentinel_differs_from_nodata():
    # MinIsWhite inverts the sentinel: nodata=0 on disk but the in-memory
    # buffer has been inverted, so the masking value is 255 (8-bit) not 0.
    arr = np.array([[100.0, 255.0], [50.0, 0.0]], dtype=np.float32)
    gi = _default_geo_info(nodata=0)

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=0,           # canonical attrs['nodata'] keeps the on-disk sentinel
        mask_sentinel=255,  # actual value to match in the in-memory buffer
        mask_nodata=True,
        dtype=None,
        window=None,
        name='t',
    )

    # attrs['nodata'] keeps the file sentinel so a round-trip write
    # rebuilds the original GDAL_NODATA tag.
    assert da.attrs['nodata'] == 0
    # The pixel matching mask_sentinel=255 became NaN; the literal-0
    # pixel was left alone because mask_sentinel != 0.
    arr_out = np.asarray(da.values)
    assert np.isnan(arr_out[0, 1])
    assert arr_out[1, 1] == 0.0
    assert da.attrs['nodata_pixels_present'] is True


# ---------------------------------------------------------------------------
# Eager helper: validate-first ordering (no partial attrs leak)
# ---------------------------------------------------------------------------


def test_eager_raises_on_unparseable_crs_without_partial_attrs():
    # Try to bypass pyproj entirely: if pyproj is missing, the check
    # short-circuits and the test would be a no-op. Skip in that case.
    pytest.importorskip('pyproj')

    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    # An obviously-bad WKT string. The unparseable-CRS check raises
    # before any attrs population step runs.
    gi = _default_geo_info(crs_wkt='NOT-A-VALID-WKT-STRING')

    # ``attrs_in`` is mutated only via ``dict(attrs_in)`` copy, but the
    # important contract is "if validation raises, no DataArray is built
    # and the caller's seed dict is untouched." Pass a seed dict and
    # confirm it is unchanged afterwards.
    seed = {'sentinel_marker': True}
    with pytest.raises(UnparseableCRSError):
        _finalize_eager_read(
            arr,
            geo_info=gi,
            nodata=-9999,
            mask_sentinel=-9999,
            mask_nodata=True,
            dtype=None,
            window=None,
            name='t',
            attrs_in=seed,
        )
    # Seed dict was never written to; the validation failure raised
    # before any ``attrs[...]`` step ran. Check both the exact contents
    # AND the length so a future partial-leak that adds new keys is
    # caught even if the existing key still matches.
    assert seed == {'sentinel_marker': True}
    assert len(seed) == 1


def test_eager_allow_unparseable_crs_bypasses_check():
    pytest.importorskip('pyproj')

    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    gi = _default_geo_info(crs_wkt='NOT-A-VALID-WKT-STRING')

    # Opt-in bypass: the unparseable WKT lands on the attrs unchanged.
    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=-9999,
        mask_sentinel=-9999,
        mask_nodata=True,
        dtype=None,
        window=None,
        name='t',
        allow_unparseable_crs=True,
    )
    assert da.attrs['crs_wkt'] == 'NOT-A-VALID-WKT-STRING'


# ---------------------------------------------------------------------------
# Eager helper: attrs_in seed forwarded onto the DataArray
# ---------------------------------------------------------------------------


def test_eager_attrs_in_seed_is_copied_onto_dataarray():
    arr = np.array([[1.0]], dtype=np.float32)
    gi = _default_geo_info()

    seed = {'extra_user_attr': 'kept'}
    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=None,
        mask_sentinel=None,
        mask_nodata=True,
        dtype=None,
        window=None,
        name='t',
        attrs_in=seed,
    )

    assert da.attrs['extra_user_attr'] == 'kept'
    # The seed dict was not mutated -- the helper copies before writing.
    assert seed == {'extra_user_attr': 'kept'}


# ---------------------------------------------------------------------------
# Lazy helper: attrs surface (pixels_present is None per #2135 dask contract)
# ---------------------------------------------------------------------------


def test_lazy_float_dtype_sets_masked_true_no_pixels_present_attr():
    gi = _default_geo_info()

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype='float64',
        window=None,
    )

    assert attrs['nodata'] == -9999
    assert attrs['masked_nodata'] is True
    assert attrs['nodata_dtype_cast'] == 'float64'
    # pixels_present stays absent on the lazy path (#2135 dask contract).
    assert 'nodata_pixels_present' not in attrs
    assert attrs['georef_status'] == 'full'


def test_lazy_int_graph_dtype_keeps_masked_false():
    gi = _default_geo_info(nodata=0)

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=0,
        mask_nodata=True,
        graph_dtype='int16',
        caller_dtype=None,
        window=None,
    )

    # Integer graph -> the per-chunk mask cannot have run, so masked=False
    # mirrors the #2092 contract even though the caller asked for masking.
    assert attrs['masked_nodata'] is False
    # Post-#2206 split: no caller cast -> ``nodata_dtype_cast`` stays
    # absent. The auto-promoted graph dtype never leaks into the attr.
    assert 'nodata_dtype_cast' not in attrs


def test_lazy_int_caller_cast_records_dtype():
    # Companion to the test above: when the caller explicitly passes
    # ``dtype='int16'`` (an actual cast request), the attr surfaces.
    gi = _default_geo_info(nodata=0)

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=0,
        mask_nodata=True,
        graph_dtype='int16',
        caller_dtype='int16',
        window=None,
    )

    assert attrs['masked_nodata'] is False
    assert attrs['nodata_dtype_cast'] == 'int16'


def test_lazy_graph_and_caller_dtype_differ():
    # Pins which parameter drives which attr when they actually differ:
    # ``graph_dtype`` drives ``masked_nodata`` (float graph -> per-chunk
    # mask runs), ``caller_dtype`` drives ``nodata_dtype_cast`` (records
    # the user's cast request, not the graph dtype). An accidental swap
    # of the two arguments would flip both attrs at once.
    gi = _default_geo_info(nodata=0)

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=0,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype='int16',
        window=None,
    )

    assert attrs['masked_nodata'] is True
    assert attrs['nodata_dtype_cast'] == 'int16'


def test_lazy_mask_promoted_graph_no_caller_cast():
    # Mirrors the dask backend's int->float64 auto-promotion case: graph
    # dtype is float64 because masking forces it, but the caller did not
    # ask for any cast. ``masked_nodata`` follows the graph dtype;
    # ``nodata_dtype_cast`` follows caller intent (absent here).
    gi = _default_geo_info(nodata=0)

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=0,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype=None,
        window=None,
    )

    assert attrs['masked_nodata'] is True
    assert 'nodata_dtype_cast' not in attrs


def test_lazy_mask_nodata_false_sets_masked_false():
    gi = _default_geo_info()

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=False,
        graph_dtype='float64',
        caller_dtype=None,
        window=None,
    )

    assert attrs['masked_nodata'] is False
    assert 'nodata_pixels_present' not in attrs


def test_lazy_no_graph_dtype_resolves_to_masked_false():
    # ``graph_dtype=None`` means the caller didn't resolve a graph dtype
    # to compare against (matches the pre-#2135 dask paths in tests).
    gi = _default_geo_info()

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype=None,
        caller_dtype=None,
        window=None,
    )

    assert attrs['masked_nodata'] is False
    assert 'nodata_dtype_cast' not in attrs


def test_lazy_pixels_present_true_lands_when_caller_forwards(
):
    """PR-D of #2211: callers that already scanned for sentinel pixels
    (e.g. the eager VRT path's VRT-aware mask) can pass the result
    through the lazy helper's ``pixels_present`` kwarg so the attr is
    stamped via the same finalization helper rather than written
    ad-hoc by the backend.
    """
    gi = _default_geo_info()

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype=None,
        window=None,
        pixels_present=True,
    )

    assert attrs['nodata_pixels_present'] is True


def test_lazy_pixels_present_false_lands_when_caller_forwards():
    """Companion to the True case: a forwarded ``False`` lands as
    ``attrs['nodata_pixels_present'] is False`` rather than being
    treated as "absent". The presence-vs-absence distinction is what
    issue #2135 added the attr for.
    """
    gi = _default_geo_info()

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype=None,
        window=None,
        pixels_present=False,
    )

    assert attrs['nodata_pixels_present'] is False


def test_lazy_pixels_present_default_keeps_dask_contract():
    """Default ``pixels_present=None`` keeps the issue #2135 dask
    contract intact: the attr stays absent on lazy outputs because the
    dask backends cannot afford the eager ``.compute()`` a strict
    per-chunk scan would force.
    """
    gi = _default_geo_info()

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype=None,
        window=None,
    )

    assert 'nodata_pixels_present' not in attrs


def test_lazy_pixels_present_ignored_when_no_nodata():
    """When ``nodata is None`` the helper short-circuits all
    sentinel-lifecycle attrs (``masked_nodata``,
    ``nodata_dtype_cast``, and ``nodata_pixels_present``), regardless
    of whether the caller forwarded a ``pixels_present`` value. The
    sentinel-lifecycle attrs only make sense when a sentinel is
    declared, so a stray forward must not invent the attr.
    """
    gi = _default_geo_info(nodata=None)

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=None,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype=None,
        window=None,
        pixels_present=True,
    )

    assert 'nodata_pixels_present' not in attrs


def test_lazy_no_nodata_omits_nodata_attrs():
    gi = _default_geo_info(nodata=None)

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=None,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype='float64',
        window=None,
    )

    assert 'nodata' not in attrs
    assert 'masked_nodata' not in attrs
    assert 'nodata_pixels_present' not in attrs
    assert 'nodata_dtype_cast' not in attrs


# ---------------------------------------------------------------------------
# Lazy helper: validate-first ordering
# ---------------------------------------------------------------------------


def test_lazy_raises_on_unparseable_crs_without_partial_attrs():
    pytest.importorskip('pyproj')

    gi = _default_geo_info(crs_wkt='NOT-A-VALID-WKT-STRING')
    seed = {'sentinel_marker': True}

    with pytest.raises(UnparseableCRSError):
        _finalize_lazy_read_attrs(
            geo_info=gi,
            nodata=-9999,
            mask_nodata=True,
            graph_dtype='float64',
            caller_dtype='float64',
            window=None,
            attrs_in=seed,
        )
    assert seed == {'sentinel_marker': True}
    assert len(seed) == 1


def test_lazy_allow_unparseable_crs_bypasses_check():
    pytest.importorskip('pyproj')

    gi = _default_geo_info(crs_wkt='NOT-A-VALID-WKT-STRING')

    attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype='float64',
        window=None,
        allow_unparseable_crs=True,
    )
    assert attrs['crs_wkt'] == 'NOT-A-VALID-WKT-STRING'


# ---------------------------------------------------------------------------
# Both helpers: parity on the non-mask attrs surface
# ---------------------------------------------------------------------------


def test_lazy_and_eager_produce_same_georef_and_nodata_attrs():
    # Same input on both paths should produce the same attrs dict apart
    # from ``nodata_pixels_present`` (eager-only by design).
    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    gi = _default_geo_info()

    da = _finalize_eager_read(
        arr,
        geo_info=gi,
        nodata=-9999,
        mask_sentinel=-9999,
        mask_nodata=True,
        dtype='float64',
        window=None,
        name='t',
    )
    lazy_attrs = _finalize_lazy_read_attrs(
        geo_info=gi,
        nodata=-9999,
        mask_nodata=True,
        graph_dtype='float64',
        caller_dtype='float64',
        window=None,
    )

    keys_to_check = {
        'crs', 'crs_wkt', 'transform', 'georef_status',
        'nodata', 'masked_nodata', 'nodata_dtype_cast',
        '_xrspatial_geotiff_contract',
    }
    for key in keys_to_check:
        assert key in da.attrs, key
        assert key in lazy_attrs, key
        assert da.attrs[key] == lazy_attrs[key], key
    # And the lazy path does not carry the eager-only attr.
    assert 'nodata_pixels_present' in da.attrs
    assert 'nodata_pixels_present' not in lazy_attrs


# ---------------------------------------------------------------------------
# Lazy helper: mask_sentinel != nodata is a no-op (helper takes attrs only)
# ---------------------------------------------------------------------------


def test_lazy_helper_signature_omits_mask_sentinel():
    # The lazy helper deliberately does not accept ``mask_sentinel`` --
    # the dask graph applies masking per-chunk, and the per-chunk task
    # closes over the sentinel value separately. Pin the signature here
    # so a future refactor that adds ``mask_sentinel`` triggers a review.
    import inspect

    sig = inspect.signature(_finalize_lazy_read_attrs)
    assert 'mask_sentinel' not in sig.parameters


def test_eager_helper_signature_includes_mask_sentinel():
    # Mirror of the lazy check: the eager helper does take
    # ``mask_sentinel`` because the three GPU eager sites derive it three
    # different ways.
    import inspect

    sig = inspect.signature(_finalize_eager_read)
    assert 'mask_sentinel' in sig.parameters
