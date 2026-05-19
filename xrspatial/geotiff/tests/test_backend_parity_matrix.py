"""Required backend parity matrix per high-risk fixture (issue #1985, #2132).

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
import socketserver
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff, to_geotiff, write_vrt


# ---------------------------------------------------------------------------
# Environment gating
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    """True iff cupy is importable and the CUDA runtime is available.

    Mirrors the helper in ``test_backend_pixel_parity_matrix_1813.py``
    so GPU cells skip cleanly on hosts where cupy is installed but
    CUDA is not.
    """
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:  # pragma: no cover - defensive
        return False


_HAS_GPU = _gpu_available()
_HAS_TIFFFILE = importlib.util.find_spec("tifffile") is not None
_HAS_FSSPEC = importlib.util.find_spec("fsspec") is not None

_skip_no_gpu = pytest.mark.skipif(not _HAS_GPU, reason="cupy + CUDA required")
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
        exc=NotImplementedError,
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
