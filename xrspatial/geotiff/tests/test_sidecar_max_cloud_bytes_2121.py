"""Sidecar download honours ``max_cloud_bytes`` (issue #2121).

Before the fix, :func:`xrspatial.geotiff._sidecar.load_sidecar` downloaded
the sibling ``.tif.ovr`` over HTTP via ``_HTTPSource(path).read_all()`` and
over fsspec via ``fsspec.open(path, "rb").read()`` with no byte cap. The
base-file ``max_cloud_bytes`` budget that ``read_to_array`` and
``_CloudSource`` enforce was bypassed entirely, so a hostile server
serving a tiny base TIFF (which passes the cloud-budget check) plus a
multi-GB sidecar could OOM the reader on any ``overview_level >= 1``.

These tests pin the new contract:

* fsspec sidecar: declared size is checked against ``max_cloud_bytes``
  before any bytes are read, raising :class:`CloudSizeLimitError`.
* HTTP sidecar: ``max_bytes`` is threaded into the streaming read and
  the overshoot detector raises :class:`OSError` mid-download.
* ``max_cloud_bytes=None`` preserves the legacy unbounded behaviour.
* Local-file mmap path is unaffected (no byte budget needed -- mmap
  does not allocate the whole file).
"""
from __future__ import annotations

import pathlib
import shutil

import pytest

from xrspatial.geotiff._reader import CloudSizeLimitError
from xrspatial.geotiff._sidecar import load_sidecar


_FIXTURE = (
    pathlib.Path(__file__).resolve().parent
    / "golden_corpus"
    / "fixtures"
    / "overview_external_ovr_uint16.tif"
)


def _fixture_or_skip():
    if not _FIXTURE.exists():
        pytest.skip("sidecar fixture not present")
    sidecar = _FIXTURE.parent / "overview_external_ovr_uint16.tif.ovr"
    if not sidecar.exists():
        pytest.skip("sidecar .ovr file not present")
    return _FIXTURE


def _start_http_server(directory):
    """Spin up a loopback HTTP server serving *directory*. Returns (httpd, port)."""
    import http.server
    import socketserver
    import threading

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **kw):
            return  # silence

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

    httpd = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


# ---------------------------------------------------------------------------
# Local file: mmap path is not subject to the budget. The cap argument is
# accepted but ignored because no allocation happens.
# ---------------------------------------------------------------------------
def test_local_sidecar_ignores_max_cloud_bytes():
    src = _fixture_or_skip()
    sidecar_path = str(src) + ".ovr"
    # Pass a tiny budget: local mmap is not subject to it.
    sidecar = load_sidecar(sidecar_path, max_cloud_bytes=1)
    try:
        assert len(sidecar.ifds) == 2
    finally:
        closer = getattr(sidecar.data, "close", None)
        if closer is not None:
            closer()


# ---------------------------------------------------------------------------
# fsspec: declared size is checked against the budget before any read.
# ---------------------------------------------------------------------------
def test_fsspec_sidecar_rejects_when_exceeds_max_cloud_bytes(tmp_path):
    pytest.importorskip("fsspec")
    src = _fixture_or_skip()
    sidecar_src = str(src) + ".ovr"
    sidecar_size = pathlib.Path(sidecar_src).stat().st_size
    sidecar_copy = tmp_path / "x_2121.tif.ovr"
    shutil.copy(sidecar_src, sidecar_copy)

    uri = f"file://{sidecar_copy}"
    # Set the budget below the actual sidecar size: the size guard fires.
    with pytest.raises(CloudSizeLimitError) as exc:
        load_sidecar(uri, max_cloud_bytes=sidecar_size - 1)
    assert "exceeds max_cloud_bytes" in str(exc.value)
    assert str(sidecar_size) in str(exc.value).replace(",", "")


def test_fsspec_sidecar_succeeds_when_under_max_cloud_bytes(tmp_path):
    pytest.importorskip("fsspec")
    src = _fixture_or_skip()
    sidecar_src = str(src) + ".ovr"
    sidecar_size = pathlib.Path(sidecar_src).stat().st_size
    sidecar_copy = tmp_path / "y_2121.tif.ovr"
    shutil.copy(sidecar_src, sidecar_copy)

    uri = f"file://{sidecar_copy}"
    # Budget comfortably above the sidecar size: download proceeds.
    sidecar = load_sidecar(uri, max_cloud_bytes=sidecar_size * 10)
    assert len(sidecar.ifds) == 2


def test_fsspec_sidecar_max_cloud_bytes_none_is_unbounded(tmp_path):
    pytest.importorskip("fsspec")
    src = _fixture_or_skip()
    sidecar_src = str(src) + ".ovr"
    sidecar_copy = tmp_path / "z_2121.tif.ovr"
    shutil.copy(sidecar_src, sidecar_copy)

    uri = f"file://{sidecar_copy}"
    # max_cloud_bytes=None preserves the legacy unbounded behaviour: no
    # fsspec.size() call, no cap, sidecar reads through.
    sidecar = load_sidecar(uri, max_cloud_bytes=None)
    assert len(sidecar.ifds) == 2


# ---------------------------------------------------------------------------
# HTTP: streaming overshoot detector raises OSError when bytes exceed cap.
# ---------------------------------------------------------------------------
def test_http_sidecar_rejects_when_exceeds_max_cloud_bytes(
        tmp_path, monkeypatch):
    """Streaming download aborts when the body exceeds ``max_cloud_bytes``."""
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = _fixture_or_skip()
    sidecar_src = str(src) + ".ovr"
    sidecar_size = pathlib.Path(sidecar_src).stat().st_size
    shutil.copy(sidecar_src, tmp_path / "h_2121.tif.ovr")
    httpd, port = _start_http_server(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/h_2121.tif.ovr"
        # Streaming cap below the body size. The HTTP source surfaces the
        # over-shoot as OSError (Content-Length pre-check on
        # SimpleHTTPRequestHandler) or via the streaming probe -- either
        # way, the download is rejected before the full payload lands.
        with pytest.raises(OSError):
            load_sidecar(url, max_cloud_bytes=sidecar_size - 1)
    finally:
        httpd.shutdown()


def test_http_sidecar_succeeds_when_under_max_cloud_bytes(
        tmp_path, monkeypatch):
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = _fixture_or_skip()
    sidecar_src = str(src) + ".ovr"
    sidecar_size = pathlib.Path(sidecar_src).stat().st_size
    shutil.copy(sidecar_src, tmp_path / "ok_2121.tif.ovr")
    httpd, port = _start_http_server(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/ok_2121.tif.ovr"
        sidecar = load_sidecar(url, max_cloud_bytes=sidecar_size * 10)
        assert len(sidecar.ifds) == 2
    finally:
        httpd.shutdown()


def test_http_sidecar_max_cloud_bytes_none_is_unbounded(
        tmp_path, monkeypatch):
    """``max_cloud_bytes=None`` preserves the pre-#2121 unbounded read."""
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = _fixture_or_skip()
    sidecar_src = str(src) + ".ovr"
    shutil.copy(sidecar_src, tmp_path / "u_2121.tif.ovr")
    httpd, port = _start_http_server(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/u_2121.tif.ovr"
        sidecar = load_sidecar(url, max_cloud_bytes=None)
        assert len(sidecar.ifds) == 2
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------
# End-to-end: read_to_array's cloud_budget propagates into load_sidecar.
# A user-provided ``max_cloud_bytes`` set on the base file flows down to
# the sidecar fetch instead of being silently dropped.
# ---------------------------------------------------------------------------
def test_read_to_array_propagates_max_cloud_bytes_to_sidecar(
        tmp_path, monkeypatch):
    """A fsspec ``file://`` source with a tight budget rejects the sidecar.

    Pre-#2121, ``read_to_array(..., max_cloud_bytes=N)`` enforced ``N`` only
    on the base file and silently bypassed it for the sidecar fetch. This
    test inflates the sidecar (a normal ``.ovr`` is small relative to its
    base) so the cloud budget that admits the base file rejects the
    sidecar; the fix routes ``cloud_budget`` into ``load_sidecar`` so the
    sidecar fetch trips the same guard.
    """
    pytest.importorskip("fsspec")
    src = _fixture_or_skip()
    base_copy = tmp_path / "rt_2121.tif"
    sidecar_copy = tmp_path / "rt_2121.tif.ovr"
    shutil.copy(src, base_copy)
    shutil.copy(str(src) + ".ovr", sidecar_copy)
    # Pad the sidecar with trailing zeros so its on-disk size exceeds the
    # base. The TIFF parser ignores trailing bytes past the last IFD chain
    # entry, so the inflated sidecar still opens cleanly when the cap
    # admits it. The base file is left untouched.
    base_size = base_copy.stat().st_size
    target_sidecar_size = base_size + 4096
    with sidecar_copy.open("ab") as f:
        f.write(b"\x00" * (target_sidecar_size - sidecar_copy.stat().st_size))
    sidecar_size = sidecar_copy.stat().st_size
    assert sidecar_size > base_size

    # Budget that admits the base file (base_size <= budget) and rejects
    # the sidecar (sidecar_size > budget).
    budget = base_size + 1
    assert base_size <= budget < sidecar_size

    from xrspatial.geotiff._reader import read_to_array
    uri = f"file://{base_copy}"
    with pytest.raises(CloudSizeLimitError):
        read_to_array(uri, overview_level=1, max_cloud_bytes=budget)
