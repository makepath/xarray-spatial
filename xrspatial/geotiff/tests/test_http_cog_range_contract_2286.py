"""HTTP/range COG reader byte-budget and range-count contract (#2293).

Part of the COG readiness rollout (#2286). These tests pin the
transport behaviour of the HTTP COG reader with explicit byte-count
and range-count assertions, so a future refactor that re-introduces a
``read_all`` fallback, a per-chunk metadata fetch, or an unbounded
coalesced GET cannot land without flipping a test red.

This file is tests only -- no production code is touched. The fixtures
reuse the in-process loopback / mock-source patterns established by
the sibling files this module references:

* ``test_http_cog_coalesce.py`` -- ``_MockHTTPSource`` for in-process
  per-call recording, ``read_geotiff_dask`` once-per-graph IFD check.
* ``test_cog_http_close_on_error_1816.py`` -- ``_CloseTracker`` for
  close-on-error assertions and the loopback Range server.
* ``test_http_range_validation_1735.py`` -- misbehaving servers that
  trigger the ``read_range`` validation paths.
* ``test_http_window_band_planar_1669.py`` -- multi-band windowed read
  parity against the local path.
* ``test_http_stripped_window_max_pixels_issue_A_1842.py`` --
  ``_RecordingHTTPSource`` for byte/range bookkeeping and the
  intersecting-strip-only contract.

Each test names the contract it is pinning and the failure mode that
would put it red. If a row surfaces a real bug the prompt rules call
for filing a separate issue and marking the row ``xfail`` with the
link, rather than fixing the production code in this PR.
"""
from __future__ import annotations

import http.server
import socketserver
import threading

import numpy as np
import pytest

from xrspatial.geotiff import _reader as reader_mod
from xrspatial.geotiff import read_geotiff_dask
from xrspatial.geotiff._reader import (
    _HTTPSource,
    _read_cog_http,
    coalesce_ranges,
    split_coalesced_bytes,
)
from xrspatial.geotiff._sources import MAX_COALESCED_RANGE_BYTES_DEFAULT
from xrspatial.geotiff._writer import write


# ---------------------------------------------------------------------------
# Fixture: disable the .ovr sidecar discovery probe.
#
# Issue #2239 added a sibling-.ovr lookup to the HTTP read path. The
# in-memory mock servers below answer 200 for every path, so the probe
# sees a phantom sidecar and either adds an extra ``(0, 1)`` GET or
# attaches a bogus IFD chain. The contract tests in this file count
# specific GETs and assert bounded byte budgets; both are off by the
# probe's noise. Sidecar coverage lives in
# ``test_remote_sidecar_chunked_2239.py``; pin this file to the
# no-sidecar path so the budgets stay deterministic.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_sidecar_probe(monkeypatch):
    from xrspatial.geotiff import _sidecar as _sidecar_mod
    monkeypatch.setattr(_sidecar_mod, 'find_sidecar', lambda _src: None)
    # ``discover_remote_sidecar`` is invoked from the HTTP meta path;
    # short-circuit it too so the mock servers never see the probe.
    monkeypatch.setattr(
        _sidecar_mod,
        'discover_remote_sidecar',
        lambda src, ifds, **_kw: (ifds, None, set()),
    )


# ---------------------------------------------------------------------------
# Loopback Range HTTP server, copied from the sibling tests (verbatim).
# ---------------------------------------------------------------------------

class _RangeHandler(http.server.BaseHTTPRequestHandler):
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


def _stop(httpd):
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(autouse=True)
def _allow_loopback(monkeypatch):
    """The HTTP source rejects loopback by default after #1664."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS', '1')


# ---------------------------------------------------------------------------
# In-process recording HTTP source (no socket round-trip).
#
# Mirrors ``_MockHTTPSource`` / ``_RecordingHTTPSource`` from the sibling
# files: serves bytes from an in-memory buffer and tracks every
# ``read_range`` / ``read_all`` call so tests can assert how many GETs
# the reader issued and which byte ranges they covered.
# ---------------------------------------------------------------------------

class _RecordingHTTPSource(_HTTPSource):
    def __init__(self, buf: bytes):
        self._url = 'mock://2286/cog.tif'
        self._size = len(buf)
        self._pool = None
        self._buf = buf
        self.calls: list[tuple[int, int]] = []
        self.read_all_called = False
        self._closed_count = 0
        self._lock = threading.Lock()

    def read_range(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b''
        with self._lock:
            self.calls.append((start, length))
        return self._buf[start:start + length]

    def read_all(self) -> bytes:
        with self._lock:
            self.read_all_called = True
            self.calls.append((0, len(self._buf)))
        return self._buf

    def close(self):
        with self._lock:
            self._closed_count += 1

    # Helper accessors used by the test bodies.

    def total_bytes(self) -> int:
        return sum(le for _s, le in self.calls)

    def tile_or_strip_calls(self) -> list[tuple[int, int]]:
        """Calls past the header probe.

        The HTTP reader's first GET is a 16 KiB (or 64 KiB fallback)
        prefix anchored at offset 0 to fetch the IFD chain. Excluding
        that lets the test reason about pixel-data byte traffic alone.
        """
        return [
            (s, le) for (s, le) in self.calls
            if not (s == 0 and le in (16384, 65536))
        ]


# ---------------------------------------------------------------------------
# Small COG fixtures.
# ---------------------------------------------------------------------------

@pytest.fixture
def small_tiled_cog(tmp_path):
    """Single-band 256x256 float32 tiled COG with 32x32 tiles (64 tiles).

    Sized large enough that the whole file comfortably exceeds the
    header probe (16 KiB), so the "windowed read pulls less than the
    file" assertion is meaningful. Pseudo-random pixel values are used
    so deflate cannot squash the file below the header-probe size.
    """
    rng = np.random.RandomState(2293)
    arr = rng.standard_normal((256, 256)).astype(np.float32)
    path = str(tmp_path / 'tmp_2293_tiled.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=32,
          cog=True)
    with open(path, 'rb') as f:
        return f.read(), arr, path


@pytest.fixture
def cog_with_overviews(tmp_path):
    """256x256 float32 tiled COG with one overview level (factor 2).

    Pseudo-random pixels so deflate cannot collapse the level-0 IFD
    below the header probe, which would make the
    'overview pulls fewer bytes than base' assertion vacuous.
    """
    rng = np.random.RandomState(0x2293)
    arr = rng.standard_normal((256, 256)).astype(np.float32)
    path = str(tmp_path / 'tmp_2293_ovr.tif')
    write(arr, path, compression='deflate', tiled=True, tile_size=32,
          cog=True, overview_levels=[2])
    with open(path, 'rb') as f:
        return f.read(), arr, path


@pytest.fixture
def multiband_chunky_cog(tmp_path):
    """3-band tiled chunky (planar=1) COG, 128x128, 32x32 tiles."""
    h, w, bands = 128, 128, 3
    rng = np.random.RandomState(2293)
    expected = rng.randint(0, 200, size=(h, w, bands)).astype(np.uint8)
    path = str(tmp_path / 'tmp_2293_chunky.tif')
    write(expected, path, compression='deflate', tiled=True,
          tile_size=32, cog=True)
    with open(path, 'rb') as f:
        return f.read(), expected, path


# ===========================================================================
# 1. Windowed reads fetch only intersecting tiles -- bounded bytes + ranges
# ===========================================================================

def test_windowed_tile_read_bounded_bytes_and_range_count(
        small_tiled_cog, monkeypatch):
    """A 32x32 window aligned to one tile fetches a single tile's bytes,
    not the whole file.

    Pre-#1669/#1842 the HTTP path either ignored ``window=`` or fell
    back to ``read_all()`` and sliced. Either regression would push the
    total byte count past the per-tile budget.
    """
    buf, expected, _ = small_tiled_cog
    src = _RecordingHTTPSource(buf)
    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src)

    window = (32, 32, 64, 64)  # one whole 32x32 tile (tile [1, 1])
    arr, _ = _read_cog_http('http://mock/cog.tif', window=window)
    np.testing.assert_array_equal(arr, expected[32:64, 32:64])

    # The fallback ``read_all`` would pull the whole file. Pin against it.
    assert not src.read_all_called, (
        "windowed HTTP read fell back to read_all(); the windowed branch "
        "must fetch only intersecting tile byte ranges")

    pixel_calls = src.tile_or_strip_calls()
    # One window-aligned tile -> at most one coalesced pixel GET. The
    # coalescer may emit a single merged GET or a single per-tile GET;
    # both are fine. Two or more means the reader fetched neighbouring
    # tiles it did not need.
    assert len(pixel_calls) <= 1, (
        f"expected <=1 pixel GET for a single-tile window, got "
        f"{len(pixel_calls)}: {pixel_calls}")

    # The total byte count must be bounded by the windowed footprint
    # plus header slack, not the file size. 32x32 float32 + deflate
    # compression slack < 8 KiB; the file is >=128 KiB. The hard bound
    # is the file size; the soft bound (header + a few tiles' worth)
    # catches a regression where the reader pulls neighbouring tiles
    # it does not need.
    assert src.total_bytes() < len(buf), (
        f"windowed read consumed {src.total_bytes()} of {len(buf)} file "
        f"bytes; the windowed branch must not pull the whole file")
    assert src.total_bytes() <= 64 * 1024 + 32 * 1024, (
        f"windowed read consumed {src.total_bytes()} bytes; well above "
        f"the expected header + single-tile budget")


def test_windowed_multi_tile_read_range_count_bounded(
        small_tiled_cog, monkeypatch):
    """A window that touches 2x2=4 tiles must not fetch all 64 tiles
    in the file.

    Pins the intersect-only contract for windows that span multiple
    tiles. With coalescing on by default the four adjacent tiles may
    merge into one GET, so the assertion is on an upper bound, not an
    exact count.
    """
    buf, expected, _ = small_tiled_cog
    src = _RecordingHTTPSource(buf)
    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src)

    # Cover tiles (0,0), (0,1), (1,0), (1,1) -- top-left 2x2 block.
    window = (0, 0, 64, 64)
    arr, _ = _read_cog_http('http://mock/cog.tif', window=window)
    np.testing.assert_array_equal(arr, expected[0:64, 0:64])

    pixel_calls = src.tile_or_strip_calls()
    # 4 intersecting tiles -> coalescing collapses adjacent ones to a
    # small handful of GETs. A regression that fetched every tile in
    # the file would emit >=64 separate GETs or fall back to read_all.
    assert not src.read_all_called
    assert len(pixel_calls) <= 4, (
        f"expected <=4 pixel GETs for a 4-tile window, got "
        f"{len(pixel_calls)}: {pixel_calls}")
    # Hard bound: the byte count must not approach the file size.
    assert src.total_bytes() < len(buf), (
        f"4-tile window consumed {src.total_bytes()} of {len(buf)} bytes")


# ===========================================================================
# 2. Overview reads fetch overview IFD bytes, not full-res
# ===========================================================================

def test_overview_read_does_not_fetch_full_resolution_pixels(
        cog_with_overviews, monkeypatch):
    """An ``overview_level=1`` read must pull the overview IFD's tiles,
    not the level-0 tiles.

    Reads the file twice on the same recording mock -- once at the
    base level and once at level 1 -- and asserts the overview read
    consumed strictly fewer pixel bytes than the base read. A
    regression that ignored ``overview_level`` (or read level 0 and
    decimated) would land at byte parity with the base read.
    """
    buf, _expected, _ = cog_with_overviews

    src_base = _RecordingHTTPSource(buf)
    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src_base)
    base_arr, _ = _read_cog_http('http://mock/cog.tif', overview_level=0)
    assert base_arr.shape == (256, 256)

    src_ovr = _RecordingHTTPSource(buf)
    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src_ovr)
    ovr_arr, _ = _read_cog_http('http://mock/cog.tif', overview_level=1)
    # Overview decimation factor 2 -> 128x128 output.
    assert ovr_arr.shape == (128, 128)

    base_pixels = sum(le for _s, le in src_base.tile_or_strip_calls())
    ovr_pixels = sum(le for _s, le in src_ovr.tile_or_strip_calls())
    assert ovr_pixels < base_pixels, (
        f"overview read pulled {ovr_pixels} pixel bytes; base read pulled "
        f"{base_pixels}. The overview path must read the smaller IFD, "
        f"not the full-res IFD")

    # Sanity: the overview byte budget should be roughly a quarter of
    # base (factor-2 decimation on both axes). Allow generous slack for
    # codec overhead, tile padding, and metadata GETs. The hard contract
    # is "less than base"; this bound flags a future regression that
    # quietly grows the overview footprint past 75% of base.
    assert ovr_pixels < base_pixels * 0.75, (
        f"overview read consumed {ovr_pixels} of {base_pixels} base "
        f"pixel bytes ({ovr_pixels / base_pixels:.1%}); expected close "
        f"to 25% for a factor-2 overview")


# ===========================================================================
# 3. ``band=`` on multi-band COGs returns correct pixels with bounded reads
# ===========================================================================

def test_band_selection_multiband_chunky_bounded_reads(
        multiband_chunky_cog, monkeypatch):
    """Per-band reads of a planar=1 chunky COG must return the right
    pixels and stay within the file's byte budget.

    The chunky layout interleaves samples in each tile, so the HTTP
    path cannot avoid fetching every chunky tile that touches the
    requested rows -- but it must not exceed the file size, and the
    fetched bytes must decode to the same pixels as the local read.
    """
    buf, _expected, path = multiband_chunky_cog

    # Reference via the local-file path on the same buffer.
    from xrspatial.geotiff import open_geotiff
    local = open_geotiff(path, band=1)

    src = _RecordingHTTPSource(buf)
    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src)
    remote, _ = _read_cog_http('http://mock/cog.tif', band=1)
    np.testing.assert_array_equal(remote, np.asarray(local))
    assert remote.ndim == 2

    # Bounded-read contract: the HTTP path must not pull more than the
    # file's bytes, and must not fall through to ``read_all``.
    assert not src.read_all_called, (
        "band-selected HTTP read fell back to read_all(); band slicing "
        "must happen on the decoded array, not after a full download")
    assert src.total_bytes() <= len(buf) + 65536, (
        f"band-selected read consumed {src.total_bytes()} bytes against a "
        f"file of {len(buf)} bytes; the read must stay within the file's "
        f"byte budget plus a small header slack")


def test_band_selection_with_window_bounded_range_count(
        multiband_chunky_cog, monkeypatch):
    """``window=`` + ``band=`` on a multi-band COG: pixels match the
    local path, range count is bounded by the window footprint.
    """
    buf, _expected, path = multiband_chunky_cog
    from xrspatial.geotiff import open_geotiff
    window = (0, 0, 32, 32)
    local = open_geotiff(path, window=window, band=2)

    src = _RecordingHTTPSource(buf)
    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src)
    remote, _ = _read_cog_http('http://mock/cog.tif',
                               window=window, band=2)
    np.testing.assert_array_equal(remote, np.asarray(local))

    assert not src.read_all_called
    # 32x32 window aligned to one tile; one pixel GET, two at most if
    # the coalescer happens to split. Anything past that means the
    # reader fetched a tile outside the window or every band in turn.
    pixel_calls = src.tile_or_strip_calls()
    assert len(pixel_calls) <= 2, (
        f"expected <=2 pixel GETs for a single-tile window+band, got "
        f"{len(pixel_calls)}: {pixel_calls}")


# ===========================================================================
# 4. Dask COG reads parse metadata once per graph, not per chunk task
# ===========================================================================

def test_dask_read_parses_ifds_once_across_chunks(
        small_tiled_cog, monkeypatch):
    """An N-chunk dask graph must trigger at most one IFD-header GET
    across all chunk tasks.

    Mirrors ``test_dask_http_parses_ifds_once`` in
    ``test_http_cog_coalesce.py`` but exercises the explicit O(1)-in-
    chunk-count contract this PR is supposed to pin. A regression where
    each delayed task spins up a fresh ``_HTTPSource`` and reparses
    headers would scale header GETs with chunk count.
    """
    buf, expected, _ = small_tiled_cog
    src_holder: list[_RecordingHTTPSource] = []

    def _fake_http_source(url, *_a, **_kw):
        s = _RecordingHTTPSource(buf)
        src_holder.append(s)
        return s

    monkeypatch.setattr(reader_mod, '_HTTPSource', _fake_http_source)

    # 256x256 image; 32x32 chunks -> 64 chunks. If header parsing happens
    # per chunk task we should see ~64 header GETs. The contract says
    # at most one.
    da_arr = read_geotiff_dask('http://mock/cog.tif', chunks=32)
    n_chunks = da_arr.data.npartitions
    assert n_chunks >= 16, (
        f"expected >=16 chunks to make the count assertion meaningful, "
        f"got {n_chunks}")
    out = da_arr.compute()
    np.testing.assert_array_equal(np.asarray(out), expected)

    header_gets = 0
    for s in src_holder:
        for (start, length) in s.calls:
            # The header probe is exactly (0, 16384) or the doubled
            # fallback (0, 65536). Anything else is a pixel GET.
            if start == 0 and length in (16384, 65536):
                header_gets += 1
    assert header_gets <= 1, (
        f"expected <=1 header GET across {n_chunks} dask chunks; got "
        f"{header_gets} over {len(src_holder)} _HTTPSource instances. "
        f"Per-chunk header parsing would have produced ~{n_chunks}.")


def test_dask_header_gets_independent_of_chunk_count(
        small_tiled_cog, monkeypatch):
    """Doubling chunk count must not double header GETs (O(1) in chunks).

    Runs the same compute at two chunk granularities (32 and 64) and
    asserts neither pulls more than one header. Pinning the rate, not
    just an absolute count, catches a regression where the per-chunk
    GET is hidden under a small constant overhead at low chunk counts
    but grows linearly at higher ones.
    """
    buf, expected, _ = small_tiled_cog

    def _run(chunks):
        src_holder: list[_RecordingHTTPSource] = []

        def _fake(url, *_a, **_kw):
            s = _RecordingHTTPSource(buf)
            src_holder.append(s)
            return s

        monkeypatch.setattr(reader_mod, '_HTTPSource', _fake)
        out = read_geotiff_dask('http://mock/cog.tif', chunks=chunks).compute()
        np.testing.assert_array_equal(np.asarray(out), expected)
        return sum(
            1
            for s in src_holder
            for (start, length) in s.calls
            if start == 0 and length in (16384, 65536)
        )

    hdr_small = _run(32)   # 64 chunks on 256x256
    hdr_large = _run(64)   # 16 chunks on 256x256
    # Whatever the absolute count, both must be at most 1.
    assert hdr_small <= 1, (
        f"chunks=32 issued {hdr_small} header GETs; expected <=1")
    assert hdr_large <= 1, (
        f"chunks=64 issued {hdr_large} header GETs; expected <=1")


# ===========================================================================
# 5. Truncated / malformed COGs close HTTP resources reliably
# ===========================================================================

class _CloseCountingSource(_HTTPSource):
    """``_HTTPSource`` that counts ``close()`` invocations.

    Used to assert the HTTP read path closes the source on the error
    path (#1816), even when the IFD parse blows up on a truncated
    file. Unlike the wrapper in ``test_cog_http_close_on_error_1816``
    this subclass also serves real bytes so the failure can be driven
    by a malformed payload rather than a monkeypatched explosion.
    """

    def __init__(self, buf: bytes):
        self._url = 'mock://2286/bad.tif'
        self._size = len(buf)
        self._pool = None
        self._buf = buf
        self.close_count = 0
        self._lock = threading.Lock()

    def read_range(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b''
        return self._buf[start:start + length]

    def read_all(self) -> bytes:
        return self._buf

    def close(self):
        with self._lock:
            self.close_count += 1


def test_truncated_cog_closes_http_source(monkeypatch):
    """A truncated buffer must surface a clear exception and still
    close the HTTP source on the way out.

    The fixture serves only the first 32 bytes of what would be a
    valid TIFF. IFD parsing fails. The contract is:

    * the call raises (not a hang, not a silent zero-array return),
    * ``source.close()`` runs exactly once via the ``finally`` guard
      added for #1816.
    """
    bad = b'II\x2a\x00' + b'\x00' * 28  # valid magic, IFD pointer = 0
    src = _CloseCountingSource(bad)
    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src)

    with pytest.raises((ValueError, OSError)):
        _read_cog_http('http://mock/bad.tif')
    assert src.close_count == 1, (
        f"truncated COG read did not close the HTTP source: "
        f"close_count={src.close_count}")


def test_malformed_ifd_chain_closes_http_source(monkeypatch):
    """A file with a well-formed TIFF header but an IFD chain that
    points past the buffer raises a ``ValueError`` and still closes
    the source.

    Mirrors the #2050/#2266 'malformed pyramid metadata' scenarios:
    the header looks valid enough to start parsing, then the IFD
    extends past what any reasonable header prefetch will pull.
    """
    # Synthesize a tiny payload that crosses the parser-validation
    # threshold without being a real TIFF. The HTTP parser fetches
    # 16 KiB then expands. Without a valid IFD it raises after the
    # cap is hit; we just need the close-on-error contract to fire.
    payload = b'II\x2a\x00' + (0xFFFFFFF0).to_bytes(4, 'little') + b'\x00' * 64
    src = _CloseCountingSource(payload)
    monkeypatch.setattr(reader_mod, '_HTTPSource', lambda url: src)

    with pytest.raises((ValueError, OSError)):
        _read_cog_http('http://mock/bad.tif')
    assert src.close_count >= 1, (
        f"malformed IFD chain did not close the HTTP source: "
        f"close_count={src.close_count}")


def test_short_body_during_pixel_fetch_closes_source(
        small_tiled_cog, monkeypatch):
    """Header parses fine; the first pixel GET returns truncated bytes.
    The reader must raise (not hang) and close the source.

    Uses a real loopback server so the failure path runs through the
    actual urllib3 stack rather than a monkeypatched stub.
    """
    buf, _, _ = small_tiled_cog

    fail_after_header = {'tripped': False}

    class _ShortPixelHandler(_RangeHandler):
        payload = buf

        def do_GET(self):  # noqa: N802
            rng = self.headers.get('Range', '')
            if rng.startswith('bytes='):
                spec = rng[len('bytes='):]
                start_s, _, end_s = spec.partition('-')
                start = int(start_s)
                end = int(end_s) if end_s else len(self.payload) - 1
                # Header probe (start=0) goes through cleanly so the
                # IFD parses; any later GET returns a short body to
                # trip ``_HTTPSource.read_range``'s length check.
                if start != 0:
                    fail_after_header['tripped'] = True
                    advertised = end - start + 1
                    self.send_response(206)
                    self.send_header('Content-Length', str(advertised))
                    self.send_header(
                        'Content-Range',
                        f'bytes {start}-{end}/{len(self.payload)}',
                    )
                    self.end_headers()
                    # Send fewer bytes than advertised.
                    self.wfile.write(b'\x00' * max(1, advertised // 4))
                    return
                # Header probe -> serve normally.
                chunk = self.payload[start:end + 1]
                self.send_response(206)
                self.send_header('Content-Length', str(len(chunk)))
                self.send_header(
                    'Content-Range',
                    f'bytes {start}-{start + len(chunk) - 1}/'
                    f'{len(self.payload)}',
                )
                self.end_headers()
                self.wfile.write(chunk)

    httpd = socketserver.TCPServer(('127.0.0.1', 0), _ShortPixelHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{port}/cog.tif'

    trackers: list = []
    real_cls = reader_mod._HTTPSource

    class _Tracker:
        def __init__(self, real):
            self._real = real
            self.close_count = 0

        def __getattr__(self, name):
            return getattr(self._real, name)

        def close(self):
            self.close_count += 1
            return self._real.close()

    def factory(u, *a, **kw):
        t = _Tracker(real_cls(u, *a, **kw))
        trackers.append(t)
        return t

    monkeypatch.setattr(reader_mod, '_HTTPSource', factory)

    # ``urllib3.exceptions.ProtocolError`` is the underlying class
    # raised when the server short-bodies a chunked response. Newer
    # ``read_range`` paths convert this to ``OSError`` before it
    # escapes; older versions let urllib3's exception type bubble up.
    # Both are acceptable as long as a clear exception fires (not a
    # hang) and the source still gets closed.
    import urllib3.exceptions as _u3e

    try:
        with pytest.raises((OSError, ValueError, _u3e.ProtocolError,
                            _u3e.HTTPError)):
            _read_cog_http(url)
    finally:
        _stop(httpd)

    assert fail_after_header['tripped'], (
        "test scaffold mistake: the server never returned a short body, "
        "so the failure path under test never ran")
    # Every constructed source must have been closed exactly once.
    assert trackers, "no _HTTPSource was constructed"
    for t in trackers:
        assert t.close_count == 1, (
            f"truncated pixel-fetch path leaked a source: "
            f"close_count={t.close_count}")


# ===========================================================================
# 6. Coalescing bounded by the configured max-merged-range size
# ===========================================================================

def test_coalesce_does_not_silently_exceed_explicit_cap():
    """``coalesce_ranges`` must respect the explicit cap kwarg.

    Mirrors the pure-unit assertion in ``test_http_cog_coalesce.py``
    but folds it into this file as the canonical contract row for
    #2286: a future refactor that drops the cap (or treats it as an
    advisory) flips this test red.
    """
    one_mib = 1 << 20
    cap = 4 * one_mib
    # 8 ranges spaced 1 MiB apart -- without the cap the gap threshold
    # alone collapses them into a single ~7 MiB GET.
    ranges = [(i * one_mib, 1024) for i in range(8)]
    merged, mapping = coalesce_ranges(ranges, max_coalesced_range_bytes=cap)
    for _start, length in merged:
        assert length <= cap, (
            f"merged range of {length} bytes exceeds the {cap}-byte cap")
    assert len(merged) > 1, (
        "cap did not force any split; the contract is broken")
    assert len(mapping) == len(ranges)


def test_coalesce_default_cap_bounds_adversarial_input():
    """The default cap must bound an adversarial 'thousands of tiles
    spaced 1 MiB apart' input.

    This is the motivating #2266 scenario: a header with many tiny
    valid byte counts and sub-threshold gaps. Without the default cap
    the coalescer collapses the whole table into one multi-GiB GET.
    """
    one_mib = 1 << 20
    ranges = [(i * one_mib, 1024) for i in range(4096)]
    merged, _ = coalesce_ranges(ranges)
    for _start, length in merged:
        assert length <= MAX_COALESCED_RANGE_BYTES_DEFAULT, (
            f"merged range {length} bytes exceeds the default cap "
            f"{MAX_COALESCED_RANGE_BYTES_DEFAULT}; the safe-by-default "
            f"contract is broken")


def test_coalesced_get_size_capped_on_real_http_source():
    """The real ``_HTTPSource`` ``read_ranges_coalesced`` path must
    propagate the cap through to the wire-level GETs.

    Constructs an in-memory recording source (sharing the contract
    with the production class via subclassing), asks for ranges that
    would otherwise merge into one big GET, and asserts every actual
    GET respects the cap. Mirrors the dedicated row in
    ``test_http_cog_coalesce.py``; reproduced here as the contract
    anchor for #2293 so the cap survives any future refactor of the
    coalescer.
    """
    one_mib = 1 << 20
    buf = bytes((i * 13) & 0xFF for i in range(16 * one_mib))
    src = _RecordingHTTPSource(buf)

    ranges = [(i * one_mib, 1024) for i in range(8)]
    cap = 4 * one_mib
    out = src.read_ranges_coalesced(
        ranges, max_workers=2, max_coalesced_range_bytes=cap)

    # Bytes match the original per-range slices.
    for (off, length), tile in zip(ranges, out):
        assert tile == buf[off:off + length]

    # Every actual GET respects the cap.
    assert src.calls, "no GETs were issued"
    for _start, length in src.calls:
        assert length <= cap, (
            f"actual GET of {length} bytes exceeds the {cap}-byte cap")
    # The cap forced at least one split.
    assert len(src.calls) >= 2


def test_split_coalesced_bytes_round_trips_under_cap():
    """When the cap forces a split, ``split_coalesced_bytes`` still
    recovers every original byte range. The cap must not silently
    drop or duplicate bytes.
    """
    one_mib = 1 << 20
    payload_len = 8 * one_mib + 1024
    payload = bytes((i * 17) & 0xFF for i in range(payload_len))
    ranges = [(i * one_mib, 1024) for i in range(8)]

    merged, mapping = coalesce_ranges(
        ranges, max_coalesced_range_bytes=4 * one_mib)
    merged_bytes = [payload[s:s + le] for (s, le) in merged]
    out = split_coalesced_bytes(merged_bytes, mapping)
    for (off, length), tile in zip(ranges, out):
        assert tile == payload[off:off + length]


# ===========================================================================
# Bonus: end-to-end byte-budget check across the loopback server.
#
# The mocks above run in-process; the assertion below proves the same
# bounded contract holds when the bytes really do cross a socket.
# ===========================================================================

def test_loopback_end_to_end_windowed_byte_budget(small_tiled_cog):
    """End-to-end through the real loopback server: a windowed read
    returns the right pixels and the total payload returned across all
    206 responses is bounded by the windowed footprint.

    The loopback server does not let us count GETs from outside, but
    it does let us prove that the parts the reader pulled add up to
    less than the file size. This pins the contract against a real
    urllib3 stack, catching regressions that ride below the
    ``_HTTPSource`` abstraction (e.g. a transparent prefetch in the
    pool layer).
    """
    buf, expected, _ = small_tiled_cog
    served = {'bytes': 0}

    class _CountingHandler(_RangeHandler):
        payload = buf

        def do_GET(self):  # noqa: N802
            rng = self.headers.get('Range', '')
            if rng.startswith('bytes='):
                spec = rng[len('bytes='):]
                start_s, _, end_s = spec.partition('-')
                start = int(start_s)
                end = int(end_s) if end_s else len(self.payload) - 1
                chunk = self.payload[start:end + 1]
                served['bytes'] += len(chunk)
                self.send_response(206)
                self.send_header('Content-Length', str(len(chunk)))
                self.send_header(
                    'Content-Range',
                    f'bytes {start}-{start + len(chunk) - 1}/'
                    f'{len(self.payload)}',
                )
                self.end_headers()
                self.wfile.write(chunk)
                return
            served['bytes'] += len(self.payload)
            self.send_response(200)
            self.send_header('Content-Length', str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)

    httpd = socketserver.TCPServer(('127.0.0.1', 0), _CountingHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{port}/cog.tif'
    try:
        window = (32, 32, 64, 64)
        arr, _ = _read_cog_http(url, window=window)
        np.testing.assert_array_equal(arr, expected[32:64, 32:64])
    finally:
        _stop(httpd)

    # Hard upper bound: must be less than the whole file.
    assert served['bytes'] < len(buf), (
        f"loopback windowed read served {served['bytes']} of {len(buf)} "
        f"file bytes; the windowed path must not pull the whole file")
