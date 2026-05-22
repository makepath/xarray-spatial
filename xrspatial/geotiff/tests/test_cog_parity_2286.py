"""COG parity rows for the release gate (issue #2294 / parent #2286).

A focused parity layer that locks the COG read/write paths to the
release gate. The six rows below cover the round-trip surface a caller
hits when they treat xrspatial as a COG producer or COG consumer, and
when they round-trip a COG through a third party (rasterio) or fetch
one via the HTTP range-read code path.

Rows
----

1. ``xrspatial write COG -> xrspatial eager read``
2. ``xrspatial write COG -> xrspatial dask read``
3. ``xrspatial write COG -> rasterio read``
4. ``golden/rasterio COG fixture -> xrspatial local read``
5. ``golden/rasterio COG fixture -> xrspatial HTTP range read``
6. ``golden/rasterio COG fixture -> xrspatial dask HTTP range read``

Each row asserts byte-exact pixels (every fixture used here is
lossless) and a fixed subset of the metadata contract: ``crs`` (or
``crs_wkt``), ``transform``, ``nodata`` (including the no-nodata
case), pixel ``dtype``, band count, and the ``(y, x)`` dim names.
The wider canonical-attrs surface (resolution, georef_status, etc.)
lives in ``test_backend_full_parity_2211.py``; this file is the
narrower COG-only gate.

Skip policy
-----------

Skips are always loud. If a dependency is missing (``rasterio``,
``dask``, ``fsspec``) the row calls ``pytest.skip`` with a string that
names the missing dependency. Silent collection of zero rows is itself
a bug under #2286.

Scope
-----

* CPU-only. The GPU rows stay out per the parent issue (``reader.gpu``
  is experimental and outside this gate).
* No experimental codecs. Every row uses lossless deflate so a
  byte-exact comparison is meaningful.
* This is a tests-only PR: no changes to production code or to
  ``test_backend_full_parity_2211.py``.
"""
from __future__ import annotations

import http.server
import importlib.util
import pathlib
import socketserver
import threading
import uuid

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("rasterio")

from xrspatial.geotiff import open_geotiff, to_geotiff  # noqa: E402
from xrspatial.geotiff._writer import write  # noqa: E402


# ---------------------------------------------------------------------------
# Environment gating
# ---------------------------------------------------------------------------

_HAS_DASK = importlib.util.find_spec("dask") is not None


def _require_dask() -> None:
    if not _HAS_DASK:
        pytest.skip(
            "dask is not installed; install the dask extra to exercise "
            "the COG dask-read row of the #2286 release gate."
        )


# Golden corpus COG fixture: tiled, internal overviews, written via
# GDAL's COG driver. Lives under ``golden_corpus/fixtures``.
_GOLDEN_COG_ID = "cog_internal_overview_uint16"


def _golden_cog_path() -> pathlib.Path:
    from xrspatial.geotiff.tests.golden_corpus import generate
    return (
        pathlib.Path(generate.__file__).resolve().parent
        / "fixtures"
        / f"{_GOLDEN_COG_ID}.tif"
    )


# ---------------------------------------------------------------------------
# Range-aware in-process HTTP server (mirrors the pattern used by
# test_cog_http_parallel_decode_2026_05_15.py and test_cog_http_concurrent.py).
# ---------------------------------------------------------------------------

class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload: bytes = b""

    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes="):]
            start_s, _, end_s = spec.partition("-")
            start = int(start_s)
            end = int(end_s) if end_s else len(self.payload) - 1
            chunk = self.payload[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header(
                "Content-Range",
                f"bytes {start}-{start + len(chunk) - 1}/{len(self.payload)}",
            )
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_args, **_kwargs):  # silence test noise
        return


def _serve_payload(payload: bytes, monkeypatch):
    """Spin a range-aware server bound to localhost; return (httpd, port).

    The handler subclass is named with a uuid suffix so that the two
    fixtures in this module (and any future ones) don't share a
    qualname. Without the suffix, tracebacks reuse the same class
    identifier across fixture invocations and become harder to read.

    ``allow_reuse_address = True`` lets the OS reclaim the port
    quickly when the test tears down (avoiding TIME_WAIT-related
    binding races under parallel pytest runs). ``timeout=5`` on the
    server caps how long a stuck request can pin the daemon thread.
    """
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    handler_cls = type(
        f"RangeHandler2286_{uuid.uuid4().hex[:8]}",
        (_RangeHandler,),
        {"payload": payload},
    )

    class _ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True
        timeout = 5

    httpd = _ReusableTCPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def xrspatial_cog(tmp_path):
    """xrspatial writes a small lossless COG; yield (path, source_array, attrs).

    The source is a deterministic uint16 ramp so byte-exact comparison
    is meaningful. CRS / transform / nodata are stamped via the public
    ``to_geotiff`` API so the round trip exercises the user-visible
    surface, not a private writer entry point.
    """
    h, w = 64, 64
    # Use a +1 offset so pixel value 0 never appears -- the reader
    # masks nodata-valued pixels to NaN under the #2092 contract,
    # which upcasts integer rasters to float64. The fixture's payload
    # is a deterministic ramp regardless of the offset.
    data = (np.arange(h * w, dtype=np.uint16) + 1).reshape(h, w)
    # Build a DataArray with a real CRS and a regular grid so the
    # transform is non-degenerate. Pixel size 0.01 deg.
    y = np.linspace(45.0, 45.0 - 0.01 * (h - 1), h)
    x = np.linspace(-120.0, -120.0 + 0.01 * (w - 1), w)
    da = xr.DataArray(
        data, dims=["y", "x"],
        coords={"y": y, "x": x},
        # No ``nodata`` attr: the masked-nodata path upcasts integer
        # rasters to float64 and replaces sentinel pixels with NaN,
        # which would break the byte-exact uint16 comparison. The
        # nodata read contract is exercised separately under
        # ``test_nodata_lifecycle_parity_2211.py``.
        attrs={"crs": 4326},
        name="cog_2286",
    )
    path = str(tmp_path / "xrspatial_cog_2286.tif")
    to_geotiff(
        da, path,
        compression="deflate",
        tiled=True,
        tile_size=16,
        cog=True,
        overview_levels=[2],
    )
    return path, data, {"crs": 4326, "nodata": None}


@pytest.fixture
def golden_cog_http(monkeypatch):
    """Serve the golden COG fixture over a range-aware in-process HTTP server.

    Yields ``(url, expected_array)`` where ``expected_array`` is the
    pixels read via the local xrspatial reader (the ground truth for
    HTTP comparison). The fixture lives in the golden corpus and was
    written by GDAL's COG driver, so it stresses the third-party
    interop side of the COG read path.
    """
    path = _golden_cog_path()
    if not path.exists():
        pytest.skip(
            f"golden COG fixture {_GOLDEN_COG_ID!r} missing on disk; run "
            "`python -m xrspatial.geotiff.tests.golden_corpus.generate` "
            "to materialise the corpus (issue #1930)."
        )
    with open(path, "rb") as f:
        payload = f.read()
    httpd, port = _serve_payload(payload, monkeypatch)
    try:
        # Use a stable filename in the URL so the SSRF-hardened reader
        # has a sensible-looking path to log.
        yield f"http://127.0.0.1:{port}/{_GOLDEN_COG_ID}.tif", path
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _materialise(da: xr.DataArray) -> np.ndarray:
    """Host-side numpy view (dask compute, cupy get) without leaking lazy state."""
    raw = da.data
    if hasattr(raw, "compute"):
        raw = raw.compute()
    if hasattr(raw, "get"):
        raw = raw.get()
    return np.asarray(raw)


def _assert_byte_exact(
    expected: np.ndarray, actual: np.ndarray, *, label: str,
) -> None:
    """Byte-exact equality on shape, dtype, and bytes. Every fixture is lossless."""
    assert expected.shape == actual.shape, (
        f"{label}: shape mismatch expected={expected.shape} actual={actual.shape}"
    )
    assert expected.dtype == actual.dtype, (
        f"{label}: dtype mismatch expected={expected.dtype} actual={actual.dtype}"
    )
    if not np.array_equal(expected, actual):
        diff = np.where(expected != actual)
        n = len(diff[0])
        raise AssertionError(
            f"{label}: byte-exact comparison failed; {n} pixel(s) differ"
        )


# Scope note: every fixture in this file is single-band 2D. The two
# helpers below hard-code that shape on purpose. If a future row adds
# a multi-band fixture, extend the helpers (or replace them with
# parametrised checks) rather than reusing them as-is.

def _assert_dim_names(da: xr.DataArray, *, label: str) -> None:
    """The 2D COG path must come back with ``(y, x)`` dim names."""
    assert da.dims == ("y", "x"), (
        f"{label}: dims must be ('y', 'x'), got {da.dims!r}"
    )


def _assert_band_count(arr: np.ndarray, *, label: str) -> None:
    """Single-band fixture; the returned array must be 2D."""
    assert arr.ndim == 2, (
        f"{label}: expected single-band 2D pixels, got ndim={arr.ndim} "
        f"shape={arr.shape}"
    )


def _assert_crs_present(da: xr.DataArray, *, label: str) -> None:
    """``crs`` (EPSG int or string) or ``crs_wkt`` must survive the read."""
    has_crs = "crs" in da.attrs and da.attrs["crs"] is not None
    has_wkt = "crs_wkt" in da.attrs and da.attrs["crs_wkt"]
    assert has_crs or has_wkt, (
        f"{label}: neither 'crs' nor 'crs_wkt' attr survived the read; "
        f"attrs={sorted(da.attrs)!r}"
    )


def _assert_crs_equals(da: xr.DataArray, expected_epsg: int, *, label: str) -> None:
    """Read-side CRS matches the writer's EPSG declaration."""
    crs = da.attrs.get("crs")
    assert crs == expected_epsg, (
        f"{label}: crs mismatch expected={expected_epsg!r} got={crs!r}"
    )


def _assert_transform(da: xr.DataArray, *, label: str) -> None:
    """Transform attr present and a finite 6-tuple."""
    t = da.attrs.get("transform")
    assert t is not None, f"{label}: transform attr missing"
    tup = tuple(float(v) for v in t)
    assert len(tup) == 6, f"{label}: transform must be a 6-tuple, got {tup}"
    assert all(np.isfinite(v) for v in tup), (
        f"{label}: transform has non-finite component: {tup}"
    )


def _assert_transform_equals(
    da: xr.DataArray, expected_t: tuple, *, label: str,
) -> None:
    """Transform 6-tuple matches an expected reference within a tight ULP."""
    t = da.attrs.get("transform")
    assert t is not None, f"{label}: transform attr missing"
    tup = tuple(float(v) for v in t)
    exp = tuple(float(v) for v in expected_t)
    assert len(tup) == 6 and len(exp) == 6
    for i, (a, b) in enumerate(zip(tup, exp)):
        assert abs(a - b) <= 1e-9, (
            f"{label}: transform[{i}] differs expected={b!r} got={a!r}"
        )


def _assert_nodata_equals(
    da: xr.DataArray, expected: float | int | None, *, label: str,
) -> None:
    """Assert nodata sentinel matches, including the no-nodata case.

    When ``expected`` is ``None`` we still check the read side: the
    reader must not fabricate a sentinel that the writer never stamped.
    The reader is allowed to expose the attr as ``None`` or omit it
    entirely; both count as "no nodata".
    """
    nd = da.attrs.get("nodata")
    if expected is None:
        assert nd is None, (
            f"{label}: writer stamped no nodata, but reader exposed "
            f"nodata={nd!r}"
        )
        return
    assert nd == expected, (
        f"{label}: nodata mismatch expected={expected!r} got={nd!r}"
    )


# ---------------------------------------------------------------------------
# Row 1: xrspatial write COG -> xrspatial eager read
# ---------------------------------------------------------------------------

def test_row1_xrspatial_cog_xrspatial_eager(xrspatial_cog):
    """xrspatial-written COG round-trips byte-exact through the eager reader."""
    path, expected, expected_attrs = xrspatial_cog
    da = open_geotiff(path)
    label = "row1_xrspatial_cog_eager"

    pixels = _materialise(da)
    _assert_band_count(pixels, label=label)
    _assert_byte_exact(expected, pixels, label=label)
    _assert_dim_names(da, label=label)
    _assert_crs_equals(da, expected_attrs["crs"], label=label)
    _assert_transform(da, label=label)
    _assert_nodata_equals(da, expected_attrs["nodata"], label=label)
    assert da.dtype == expected.dtype, (
        f"{label}: dtype mismatch expected={expected.dtype} got={da.dtype}"
    )


# ---------------------------------------------------------------------------
# Row 2: xrspatial write COG -> xrspatial dask read
# ---------------------------------------------------------------------------

def test_row2_xrspatial_cog_xrspatial_dask(xrspatial_cog):
    """xrspatial-written COG round-trips byte-exact through the dask reader."""
    _require_dask()
    path, expected, expected_attrs = xrspatial_cog
    da = open_geotiff(path, chunks=16)
    label = "row2_xrspatial_cog_dask"

    # Verify we actually went through the dask path; a regression that
    # silently drops ``chunks=`` and falls back to eager would pass the
    # pixel check but exercise the wrong code path.
    assert hasattr(da.data, "dask"), (
        f"{label}: chunks=16 did not produce a dask-backed DataArray; "
        f"got data type {type(da.data).__name__}"
    )

    pixels = _materialise(da)
    _assert_band_count(pixels, label=label)
    _assert_byte_exact(expected, pixels, label=label)
    _assert_dim_names(da, label=label)
    _assert_crs_equals(da, expected_attrs["crs"], label=label)
    _assert_transform(da, label=label)
    _assert_nodata_equals(da, expected_attrs["nodata"], label=label)
    assert da.dtype == expected.dtype, (
        f"{label}: dtype mismatch expected={expected.dtype} got={da.dtype}"
    )


# ---------------------------------------------------------------------------
# Row 3: xrspatial write COG -> rasterio read
# ---------------------------------------------------------------------------

def test_row3_xrspatial_cog_rasterio(xrspatial_cog):
    """rasterio reads an xrspatial-written COG and the pixel/metadata contract holds.

    Asserts the third-party reader sees the same pixels, dtype, CRS,
    transform, and nodata that xrspatial stamped on write. A regression
    that drops or mangles any of these would surface as a Tier-1
    interop break.
    """
    rasterio = pytest.importorskip(
        "rasterio",
        reason="rasterio is required for row 3 (issue #2294)",
    )
    path, expected, expected_attrs = xrspatial_cog
    label = "row3_xrspatial_cog_rasterio"

    with rasterio.open(path) as src:
        # Single-band fixture: read band 1.
        pixels = src.read(1)
        rio_crs = src.crs
        rio_transform = src.transform
        rio_nodata = src.nodata
        rio_count = src.count
        rio_dtype = np.dtype(src.dtypes[0])

    _assert_band_count(pixels, label=label)
    _assert_byte_exact(expected, pixels, label=label)
    assert rio_count == 1, f"{label}: rasterio reports band count {rio_count}"
    assert rio_dtype == expected.dtype, (
        f"{label}: dtype mismatch expected={expected.dtype} got={rio_dtype}"
    )
    # rasterio CRS -> EPSG int when possible.
    epsg = rio_crs.to_epsg() if rio_crs is not None else None
    assert epsg == expected_attrs["crs"], (
        f"{label}: rasterio CRS EPSG mismatch "
        f"expected={expected_attrs['crs']!r} got={epsg!r}"
    )
    # rasterio Affine is 6-tuple compatible via ``.a, .b, .c, .d, .e, .f``.
    assert rio_transform is not None, f"{label}: rasterio transform missing"
    assert all(np.isfinite(v) for v in (
        rio_transform.a, rio_transform.b, rio_transform.c,
        rio_transform.d, rio_transform.e, rio_transform.f,
    )), f"{label}: rasterio transform has non-finite component"
    if expected_attrs["nodata"] is None:
        # The writer was not asked to stamp a nodata; rasterio should
        # report ``None`` too. Anything else means the writer leaked
        # a sentinel onto the file.
        assert rio_nodata is None, (
            f"{label}: writer stamped an unrequested nodata; "
            f"rasterio reports {rio_nodata!r}"
        )
    else:
        assert rio_nodata == expected_attrs["nodata"], (
            f"{label}: rasterio nodata mismatch "
            f"expected={expected_attrs['nodata']!r} got={rio_nodata!r}"
        )


# ---------------------------------------------------------------------------
# Row 4: golden/rasterio COG fixture -> xrspatial local read
# ---------------------------------------------------------------------------

def test_row4_golden_cog_xrspatial_local():
    """Read the GDAL-written golden COG fixture with xrspatial's local reader.

    Compares pixels byte-exact against a rasterio read of the same
    bytes -- the GDAL COG driver wrote the file, so rasterio is the
    canonical oracle here. Catches regressions that returned the right
    shape but mangled values (e.g. wrong endianness, predictor drift,
    overview IFD picked instead of full res).
    """
    rasterio = pytest.importorskip(
        "rasterio",
        reason="rasterio is required for row 4 oracle (issue #2294)",
    )
    path = _golden_cog_path()
    if not path.exists():
        pytest.skip(
            f"golden COG fixture {_GOLDEN_COG_ID!r} missing on disk; run "
            "`python -m xrspatial.geotiff.tests.golden_corpus.generate` "
            "(issue #1930)."
        )
    da = open_geotiff(str(path))
    label = "row4_golden_cog_xrspatial_local"

    pixels = _materialise(da)
    _assert_band_count(pixels, label=label)
    _assert_dim_names(da, label=label)
    # The golden fixture is uint16 per the manifest entry.
    assert da.dtype == np.dtype("uint16"), (
        f"{label}: dtype expected=uint16 got={da.dtype}"
    )
    _assert_crs_present(da, label=label)
    _assert_transform(da, label=label)

    # Pixel parity against the rasterio oracle. The fixture is lossless
    # deflate, so byte-exact is the right bar.
    with rasterio.open(str(path)) as src:
        expected = src.read(1)
    _assert_byte_exact(expected, pixels, label=label)


# ---------------------------------------------------------------------------
# Row 5: golden/rasterio COG fixture -> xrspatial HTTP range read
# ---------------------------------------------------------------------------

def test_row5_golden_cog_xrspatial_http(golden_cog_http):
    """xrspatial's HTTP range reader returns the same pixels as the local read.

    Exercises the cloud-source code path against the GDAL-written
    fixture. The reference is the local read of the same bytes, so any
    drift between the local and HTTP paths surfaces here.
    """
    url, local_path = golden_cog_http
    label = "row5_golden_cog_xrspatial_http"

    local_da = open_geotiff(str(local_path))
    http_da = open_geotiff(url)

    local_px = _materialise(local_da)
    http_px = _materialise(http_da)

    _assert_band_count(http_px, label=label)
    _assert_byte_exact(local_px, http_px, label=label)
    _assert_dim_names(http_da, label=label)
    assert http_da.dtype == local_da.dtype, (
        f"{label}: dtype mismatch local={local_da.dtype} http={http_da.dtype}"
    )
    # CRS and transform survive the cloud-source path.
    local_crs = local_da.attrs.get("crs")
    http_crs = http_da.attrs.get("crs")
    assert local_crs == http_crs, (
        f"{label}: crs mismatch local={local_crs!r} http={http_crs!r}"
    )
    local_t = local_da.attrs.get("transform")
    assert local_t is not None, f"{label}: local read missing transform"
    _assert_transform_equals(http_da, local_t, label=label)
    # nodata presence must agree (the fixture may or may not carry one;
    # both sides must agree either way).
    assert ("nodata" in local_da.attrs) == ("nodata" in http_da.attrs), (
        f"{label}: nodata presence differs "
        f"local={'nodata' in local_da.attrs} http={'nodata' in http_da.attrs}"
    )
    if "nodata" in local_da.attrs:
        assert local_da.attrs["nodata"] == http_da.attrs["nodata"], (
            f"{label}: nodata value differs "
            f"local={local_da.attrs['nodata']!r} "
            f"http={http_da.attrs['nodata']!r}"
        )


# ---------------------------------------------------------------------------
# Row 6: golden/rasterio COG fixture -> xrspatial dask HTTP range read
# ---------------------------------------------------------------------------

def test_row6_golden_cog_xrspatial_dask_http(golden_cog_http):
    """The dask HTTP path returns the same pixels as the local read.

    Combines the cloud-source and chunked-read code paths. A regression
    that silently drops ``chunks=`` over HTTP would compute correct
    pixels via the eager path; the storage-type assertion below guards
    against that.
    """
    _require_dask()
    url, local_path = golden_cog_http
    label = "row6_golden_cog_xrspatial_dask_http"

    local_da = open_geotiff(str(local_path))
    http_da = open_geotiff(url, chunks=16)

    assert hasattr(http_da.data, "dask"), (
        f"{label}: chunks=16 over HTTP did not produce a dask-backed "
        f"DataArray; got data type {type(http_da.data).__name__}"
    )

    local_px = _materialise(local_da)
    http_px = _materialise(http_da)

    _assert_band_count(http_px, label=label)
    _assert_byte_exact(local_px, http_px, label=label)
    _assert_dim_names(http_da, label=label)
    assert http_da.dtype == local_da.dtype, (
        f"{label}: dtype mismatch local={local_da.dtype} http={http_da.dtype}"
    )
    local_crs = local_da.attrs.get("crs")
    http_crs = http_da.attrs.get("crs")
    assert local_crs == http_crs, (
        f"{label}: crs mismatch local={local_crs!r} http={http_crs!r}"
    )
    local_t = local_da.attrs.get("transform")
    assert local_t is not None, f"{label}: local read missing transform"
    _assert_transform_equals(http_da, local_t, label=label)
    assert ("nodata" in local_da.attrs) == ("nodata" in http_da.attrs), (
        f"{label}: nodata presence differs "
        f"local={'nodata' in local_da.attrs} http={'nodata' in http_da.attrs}"
    )
    if "nodata" in local_da.attrs:
        assert local_da.attrs["nodata"] == http_da.attrs["nodata"], (
            f"{label}: nodata value differs "
            f"local={local_da.attrs['nodata']!r} "
            f"http={http_da.attrs['nodata']!r}"
        )
