"""Regression tests for issue #2161.

Before the fix, ``open_geotiff(url, gpu=True)`` and
``read_geotiff_gpu(url)`` raised a raw ``FileNotFoundError`` on HTTP /
HTTPS / fsspec URLs because the eager path opened ``_FileSource(source)``
unconditionally and that path goes through ``open(real, 'rb')`` in the
mmap cache. The CPU eager path and the chunked GPU path both accept
those sources; the eager GPU path now does too via
``_read_geotiff_gpu_eager_via_cpu`` which reads the bytes with the
shared ``_read_to_array`` source dispatch and uploads the result via
``cupy.asarray``.

These tests pin:

1. Local paths still return a CuPy-backed DataArray (no regression).
2. ``http://`` URL returns a CuPy-backed DataArray whose values match
   the CPU eager path on the same URL.
3. ``memory://`` fsspec URI returns the same.
4. The error case (URL pointing at nothing valid) no longer bubbles a
   raw ``FileNotFoundError`` whose message is just the URL.
"""
from __future__ import annotations

import http.server
import importlib.util
import socketserver
import threading

import numpy as np
import pytest


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()
_gpu_only = pytest.mark.skipif(
    not _HAS_GPU, reason="cupy + CUDA required",
)


# ---------------------------------------------------------------------------
# Range-server harness (mirrors test_http_meta_buffer_1718)
# ---------------------------------------------------------------------------

class _RangeHandler2161(http.server.BaseHTTPRequestHandler):
    payload: bytes = b''

    def do_GET(self):  # noqa: N802
        rng = self.headers.get('Range')
        if rng and rng.startswith('bytes='):
            spec = rng[len('bytes='):]
            start_s, _, end_s = spec.partition('-')
            start = int(start_s)
            end = int(end_s) if end_s else len(self.payload) - 1
            chunk = self.payload[start:end + 1]
            self.send_response(206)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header(
                'Content-Range',
                f'bytes {start}-{start + len(chunk) - 1}/{len(self.payload)}',
            )
            self.send_header('Content-Length', str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_args, **_kwargs):
        pass


def _serve(payload: bytes):
    handler_cls = type(
        'RangeHandler2161Bound', (_RangeHandler2161,), {'payload': payload}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


@pytest.fixture
def small_tif_bytes_2161(tmp_path):
    """Build a small valid tiled COG and return its bytes + a reference
    numpy array."""
    from xrspatial.geotiff import to_geotiff
    arr = np.arange(32 * 32, dtype=np.float32).reshape(32, 32) * 0.5
    local = str(tmp_path / "src_2161.tif")
    to_geotiff(arr, local, compression="deflate", tiled=True, tile_size=16)
    with open(local, "rb") as f:
        return f.read(), arr, local


# ---------------------------------------------------------------------------
# 1. Local path regression check
# ---------------------------------------------------------------------------

@_gpu_only
def test_local_path_still_returns_cupy(small_tif_bytes_2161):
    """``read_geotiff_gpu(local.tif)`` must still take the disk-based
    eager path and return a CuPy-backed DataArray. The URL branch added
    in #2161 should not intercept local string paths."""
    import cupy

    from xrspatial.geotiff import read_geotiff_gpu

    _payload, arr_ref, local = small_tif_bytes_2161
    da = read_geotiff_gpu(local)
    assert isinstance(da.data, cupy.ndarray), (
        f"Local path no longer returns CuPy array (got {type(da.data)})"
    )
    np.testing.assert_array_equal(da.data.get(), arr_ref)


# ---------------------------------------------------------------------------
# 2. HTTP URL accepted, returns CuPy with values matching the CPU path
# ---------------------------------------------------------------------------

@_gpu_only
def test_http_url_returns_cupy_matching_cpu(small_tif_bytes_2161,
                                            monkeypatch):
    """HTTP URLs route through the CPU decode + GPU upload helper; the
    array values must match what the CPU eager path returns on the same
    URL, and the result must live on the device."""
    import cupy

    from xrspatial.geotiff import open_geotiff, read_geotiff_gpu

    payload, arr_ref, _local = small_tif_bytes_2161
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    httpd, _thread = _serve(payload)
    port = httpd.server_address[1]
    try:
        url = f'http://127.0.0.1:{port}/eager_2161.tif'

        # CPU reference via open_geotiff(gpu=False)
        da_cpu = open_geotiff(url)
        # GPU under test (direct entry point)
        da_gpu_direct = read_geotiff_gpu(url)
        # GPU via open_geotiff(gpu=True) -- the dispatcher entry
        da_gpu_open = open_geotiff(url, gpu=True)
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert isinstance(da_gpu_direct.data, cupy.ndarray)
    assert isinstance(da_gpu_open.data, cupy.ndarray)
    np.testing.assert_array_equal(da_gpu_direct.data.get(), np.asarray(da_cpu))
    np.testing.assert_array_equal(da_gpu_open.data.get(), np.asarray(da_cpu))
    np.testing.assert_array_equal(da_gpu_direct.data.get(), arr_ref)


# ---------------------------------------------------------------------------
# 3. fsspec memory:// URI accepted, returns CuPy with matching values
# ---------------------------------------------------------------------------

@_gpu_only
def test_memory_fsspec_uri_returns_cupy_matching_cpu(small_tif_bytes_2161):
    """An fsspec URI (``memory://``) was the same failure as HTTP: the
    eager GPU path opened it as a local file and raised
    ``FileNotFoundError``. After the fix it routes through the CPU read
    + GPU upload helper and matches the CPU eager path's values."""
    fsspec = pytest.importorskip("fsspec")
    import cupy

    from xrspatial.geotiff import open_geotiff, read_geotiff_gpu

    payload, arr_ref, _local = small_tif_bytes_2161
    fs = fsspec.filesystem("memory")
    mem_path = "/eager_url_2161.tif"
    fs.pipe(mem_path, payload)
    try:
        url = f"memory://{mem_path}"
        da_cpu = open_geotiff(url)
        da_gpu_direct = read_geotiff_gpu(url)
        da_gpu_open = open_geotiff(url, gpu=True)
    finally:
        try:
            fs.rm(mem_path)
        except FileNotFoundError:
            pass

    assert isinstance(da_gpu_direct.data, cupy.ndarray)
    assert isinstance(da_gpu_open.data, cupy.ndarray)
    np.testing.assert_array_equal(da_gpu_direct.data.get(), np.asarray(da_cpu))
    np.testing.assert_array_equal(da_gpu_open.data.get(), np.asarray(da_cpu))
    np.testing.assert_array_equal(da_gpu_direct.data.get(), arr_ref)


# ---------------------------------------------------------------------------
# 4. Bad URL no longer surfaces a bare FileNotFoundError
# ---------------------------------------------------------------------------

@_gpu_only
def test_unreachable_http_url_does_not_raise_filenotfound(monkeypatch):
    """Before the fix, ``read_geotiff_gpu("https://example.invalid/x.tif")``
    raised ``FileNotFoundError`` whose message was the URL itself
    (because ``_FileSource`` called ``open(url, 'rb')``). After the fix
    the URL routes through ``_read_to_array`` and any failure surfaces
    as an HTTP-layer / network error (URLError, ConnectionError, etc.)
    rather than a misleading filesystem error."""
    from xrspatial.geotiff import read_geotiff_gpu

    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    # Bind a port and immediately close it so the next connect attempt
    # fails fast. Picking a guaranteed-closed local port avoids hitting
    # the real internet from CI.
    import socket
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()

    url = f'http://127.0.0.1:{port}/nope_2161.tif'
    with pytest.raises(Exception) as excinfo:
        read_geotiff_gpu(url)

    # The message must not be the bare ``FileNotFoundError`` from
    # ``open(url, 'rb')``. The fix routes through the HTTP source so
    # the error is something networking-flavoured (URLError, OSError,
    # ConnectionRefusedError, MaxRetryError, etc.) and the URL string
    # should not be the *whole* of the error -- it should sit inside a
    # connect / read failure context.
    err = excinfo.value
    # Specifically: not FileNotFoundError pointing at the URL string.
    assert not (
        isinstance(err, FileNotFoundError)
        and str(err).rstrip("'").endswith(url)
    ), (
        f"URL surfaces as a bare FileNotFoundError again: {err!r}"
    )

    # And the error type should be networking-flavoured. ``urllib3``'s
    # MaxRetryError is the typical surface for a refused connect; bare
    # ``ConnectionError`` / ``OSError`` cover stdlib fallbacks. A
    # ``ValueError`` or ``RuntimeError`` here would mean the URL
    # branch is failing for the wrong reason (e.g. validation refused
    # the URL before the network was tried), which the loose check
    # above would not catch.
    import urllib3.exceptions as _u3
    assert isinstance(
        err,
        (_u3.MaxRetryError, _u3.HTTPError, ConnectionError, OSError),
    ), (
        f"Unreachable URL surfaced as {type(err).__name__} ({err!r}); "
        f"expected a networking exception."
    )


# ---------------------------------------------------------------------------
# 5. Helper is wired into the chunked-vs-eager dispatch correctly
# ---------------------------------------------------------------------------

@_gpu_only
def test_chunked_url_path_still_uses_chunked_helper(small_tif_bytes_2161,
                                                    monkeypatch):
    """``chunks=`` on a URL must still go through
    ``_read_geotiff_gpu_chunked`` (Dask + CuPy), not the eager helper.
    Smoke-checks that the eager-URL branch doesn't accidentally catch
    the chunked path."""
    import cupy
    import dask.array as da_mod

    from xrspatial.geotiff import open_geotiff

    payload, arr_ref, _local = small_tif_bytes_2161
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')
    httpd, _thread = _serve(payload)
    port = httpd.server_address[1]
    try:
        url = f'http://127.0.0.1:{port}/chunked_2161.tif'
        da_chunked = open_geotiff(url, gpu=True, chunks=16)
        # ``chunks=`` is lazy: confirm the dask graph is built and then
        # compute *before* shutting the server down so the per-chunk
        # CPU decode tasks can still reach the HTTP endpoint.
        assert isinstance(da_chunked.data, da_mod.Array), (
            "chunks= on a URL must produce a dask-backed result"
        )
        materialised = da_chunked.data.compute()
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert isinstance(materialised, cupy.ndarray), (
        "chunked URL path must materialise to CuPy blocks"
    )
    np.testing.assert_array_equal(materialised.get(), arr_ref)
