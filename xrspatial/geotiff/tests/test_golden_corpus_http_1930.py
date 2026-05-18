"""HTTP/COG backend cells against the golden-corpus oracle (issue #1930).

Phase 3 PR 5 of the corpus plan. The HTTP path in ``xrspatial.geotiff``
is COG-only: ``open_geotiff('http://...')`` routes to
``_reader._read_cog_http`` which uses range requests to fetch only
metadata and the tiles a window needs. The plain stripped / tiled
fixtures in the corpus would require a full-object download to read,
which is a different code path; this module focuses narrowly on the
COG fixture.

The in-process HTTP server is the same range-supporting
``BaseHTTPRequestHandler`` pattern used by
``test_http_meta_buffer_1718.py`` and friends. It serves the COG
fixture bytes on a random ``127.0.0.1`` port, and the test reads
``http://127.0.0.1:<port>/cog.tif`` through ``open_geotiff``. The
server is bound to localhost, which the HTTPSource's SSRF guard
normally rejects; the ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1`` env
var (the same hatch every other in-process HTTP test uses) opens the
loopback path for tests.

The current corpus has one COG fixture (``cog_internal_overview_uint16``)
which is not subject to any of the parity gaps the eager / dask / GPU
modules flag (no integer nodata, EPSG-coded CRS, single band), so this
module ships without a ``_PARITY_GAPS`` table. If a future COG fixture
hits one of the shared gaps, mirror the eager module's tables and the
``_build_param`` plumbing here.
"""
from __future__ import annotations

import http.server
import pathlib
import socketserver
import threading

import pytest

pytest.importorskip("yaml")
pytest.importorskip("rasterio")

from xrspatial.geotiff import open_geotiff  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus import generate  # noqa: E402
from xrspatial.geotiff.tests.golden_corpus._oracle import (  # noqa: E402
    compare_to_oracle,
)


FIXTURES_DIR = (
    pathlib.Path(generate.__file__).resolve().parent / "fixtures"
)


def _resolved_fixtures() -> list[dict]:
    """Return manifest entries with defaults merged, sorted by id."""
    manifest = generate.load_manifest()
    entries = generate.validate(manifest)
    entries.sort(key=lambda e: e["id"])
    return entries


def _cog_fixture_ids() -> list[str]:
    """Return manifest ids for fixtures the COG HTTP reader can serve.

    Only ``cog: true`` entries qualify. Returns sorted ids for stable
    parametrize output.
    """
    return [e["id"] for e in _resolved_fixtures() if e.get("cog")]


def _is_lossy(fixture_id: str) -> bool:
    """Look up the lossy flag for a fixture id from the manifest."""
    for e in _resolved_fixtures():
        if e["id"] == fixture_id:
            tol = e.get("tolerance") or {}
            return bool(tol.get("lossy", False))
    return False


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serve a single fixture's bytes with HTTP Range support.

    The bound subclass (built per-test via ``type(...)`` to bind a
    different payload) is what gets handed to ``socketserver.TCPServer``.
    """

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
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *_args, **_kwargs):
        # Quiet the per-request stderr noise that BaseHTTPRequestHandler
        # otherwise emits.
        pass


def _serve(payload: bytes) -> tuple[socketserver.TCPServer, threading.Thread]:
    handler_cls = type(
        "_RangeHandlerBound", (_RangeHandler,), {"payload": payload}
    )
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


@pytest.fixture
def allow_private_http(monkeypatch):
    """Open the SSRF guard for loopback addresses for the duration of one test.

    ``_HTTPSource`` normally rejects ``127.0.0.1`` and other private
    hosts. The same env var that every other in-process HTTP test in
    the repo uses (``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS``) reopens
    the loopback path.
    """
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")


_COG_IDS = _cog_fixture_ids()


@pytest.mark.skipif(
    not _COG_IDS,
    reason="no COG fixtures in the manifest; nothing to serve over HTTP",
)
@pytest.mark.parametrize("fixture_id", _COG_IDS, ids=_COG_IDS)
def test_http_cog_parity(fixture_id: str, allow_private_http) -> None:
    """``open_geotiff(http_url)`` for a COG fixture agrees with the oracle.

    Serves the fixture's bytes on an ephemeral 127.0.0.1 port, reads
    through the HTTP COG path, and compares against the rasterio
    reference read of the same on-disk file. The fixture and the
    served bytes are byte-identical, so any mismatch points at the
    HTTP plumbing rather than at the codec.
    """
    path = FIXTURES_DIR / f"{fixture_id}.tif"
    if not path.exists():
        pytest.skip(
            f"COG fixture {fixture_id!r} has no .tif on disk; run "
            f"`python -m xrspatial.geotiff.tests.golden_corpus.generate`"
        )

    with open(path, "rb") as f:
        payload = f.read()

    httpd, thread = _serve(payload)
    try:
        host, port = httpd.server_address
        url = f"http://{host}:{port}/{fixture_id}.tif"
        candidate = open_geotiff(url)
        # COG fixtures with internal overviews exercise the
        # overview-IFD code path through the HTTP range-request reader.
        # The factory points at the same served URL, so each overview
        # level is fetched through the same pipeline the full-
        # resolution candidate used.
        entry = next(
            (e for e in _resolved_fixtures() if e["id"] == fixture_id),
            None,
        )
        overviews = (entry or {}).get("overviews") or []
        factory = (
            (lambda lvl, u=url: open_geotiff(u, overview_level=lvl))
            if overviews
            else None
        )
        compare_to_oracle(
            path,
            candidate,
            lossy=_is_lossy(fixture_id),
            candidate_factory=factory,
        )
    finally:
        httpd.shutdown()
        httpd.server_close()
        # Daemon threads exit on process tear-down, but joining here
        # makes test-by-test cleanup deterministic and surfaces a hang
        # if ``shutdown()`` ever stops returning.
        thread.join(timeout=2.0)


def test_at_least_one_cog_fixture_exists() -> None:
    """Sanity check: the corpus has something for this backend to exercise.

    Today there is exactly one COG fixture (``cog_internal_overview_uint16``,
    phase 2 PR 7). If a future refactor accidentally drops the only
    COG entry, this test surfaces it so the HTTP cells do not silently
    cover nothing.
    """
    assert _COG_IDS, (
        "manifest has no COG fixtures; the HTTP backend has nothing to "
        "test against"
    )
