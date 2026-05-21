"""Tests for parallel strip decode in ``_read_strips`` and
``_fetch_decode_cog_http_strips``.

Issue #2100. Both strip paths previously decoded strips sequentially in a
Python for-loop. The matching tile paths use a ``ThreadPoolExecutor``
gated on ``_PARALLEL_DECODE_PIXEL_THRESHOLD`` (64K pixels). These tests
codify the gate and verify correctness (parallel vs serial parity).
"""
from __future__ import annotations

import concurrent.futures
import http.server
import os
import socket
import tempfile
import threading
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff import _reader as _reader_mod
from xrspatial.geotiff._reader import read_to_array


def _make_stripped_uint16(height: int, width: int, *,
                          compression: str = "deflate") -> bytes:
    """Build a stripped TIFF in memory and return the bytes."""
    rng = np.random.default_rng(seed=12345)
    arr = rng.integers(0, 256, size=(height, width), dtype=np.uint16)
    da = xr.DataArray(arr, dims=["y", "x"])
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        path = f.name
    try:
        to_geotiff(da, path, compression=compression, tiled=False)
        with open(path, "rb") as f:
            return arr, f.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


class TestReadStripsParallelGate:
    """Local strip path: the parallel-decode gate engages on multi-strip."""

    def test_parallel_decode_matches_serial(self, tmp_path):
        """A wide stripped TIFF: parallel and serial decode return
        identical bytes.

        Uses pytest's ``tmp_path`` rather than
        ``tempfile.TemporaryDirectory()`` because ``read_to_array``
        leaves the mmap entry cached on close (see ``_MmapCache``);
        on Windows the cached mmap holds the file handle, so an
        eager ``TemporaryDirectory.__exit__`` rmtree races the mmap
        and raises ``WinError 32``. ``tmp_path`` defers cleanup to
        pytest's session teardown, which tolerates that race.
        """
        arr, blob = _make_stripped_uint16(2048, 4096)
        p = str(tmp_path / "s.tif")
        with open(p, "wb") as f:
            f.write(blob)
        par, _ = read_to_array(p)
        with patch.object(_reader_mod,
                          "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10**12):
            ser, _ = read_to_array(p)
        np.testing.assert_array_equal(par, ser)
        np.testing.assert_array_equal(par, arr)

    def test_parallel_pool_engages_on_multi_strip(self, tmp_path):
        """Confirm the parallel branch is taken: patch ThreadPoolExecutor
        and assert it is constructed when n_strips > 1 and strip pixels
        clear the gate."""
        arr, blob = _make_stripped_uint16(1024, 2048)
        p = str(tmp_path / "s.tif")
        with open(p, "wb") as f:
            f.write(blob)
        # Patch ``concurrent.futures.ThreadPoolExecutor`` rather than the
        # reader module binding because the strip decode lives in
        # ``_decode`` and re-imports the executor function-locally (PR-G,
        # issue #2246).
        with patch.object(concurrent.futures, "ThreadPoolExecutor",
                          wraps=concurrent.futures.ThreadPoolExecutor
                          ) as mock_pool:
            out, _ = read_to_array(p)
            # n_strips for a 1024-row file with default rps -> at
            # least 4 strips (TIFFs default rps=8KB / row).
            assert mock_pool.called
        np.testing.assert_array_equal(out, arr)

    def test_serial_path_used_for_small_strip(self, tmp_path):
        """A single-row image will produce a single strip; the parallel
        gate must short-circuit to the serial branch."""
        arr = np.arange(2 * 32, dtype=np.uint16).reshape(2, 32)
        da = xr.DataArray(arr, dims=["y", "x"])
        p = str(tmp_path / "tiny.tif")
        to_geotiff(da, p, compression="deflate", tiled=False)
        with patch.object(concurrent.futures, "ThreadPoolExecutor",
                          wraps=concurrent.futures.ThreadPoolExecutor
                          ) as mock_pool:
            out, _ = read_to_array(p)
            # Single-strip file => no pool.
            assert not mock_pool.called
        np.testing.assert_array_equal(out, arr)

    def test_windowed_strip_read_parallel(self, tmp_path):
        """A windowed read across multiple strips still matches the full
        decode."""
        arr, blob = _make_stripped_uint16(2048, 2048)
        p = str(tmp_path / "w.tif")
        with open(p, "wb") as f:
            f.write(blob)
        par, _ = read_to_array(p, window=(100, 100, 1500, 1500))
        with patch.object(_reader_mod,
                          "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10**12):
            ser, _ = read_to_array(p, window=(100, 100, 1500, 1500))
        np.testing.assert_array_equal(par, ser)
        np.testing.assert_array_equal(par, arr[100:1500, 100:1500])


# ----------------------------------------------------------------------
# HTTP COG strip path
# ----------------------------------------------------------------------

class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Serve a fixed byte blob with HTTP Range support."""
    blob: bytes = b""

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.blob)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            r0, r1 = rng[len("bytes="):].split("-")
            r0 = int(r0)
            r1 = int(r1) if r1 else len(self.blob) - 1
            r1 = min(r1, len(self.blob) - 1)
            body = self.blob[r0:r1 + 1]
            self.send_response(206)
            self.send_header("Content-Range",
                             f"bytes {r0}-{r1}/{len(self.blob)}")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.blob)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(self.blob)

    def log_message(self, format, *args):  # quiet
        return


def _start_server(blob: bytes):
    """Start an HTTP server serving the given blob on a free port."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    handler = type(
        "BlobHandler", (_RangeHandler,), {"blob": blob})
    server = http.server.HTTPServer(("127.0.0.1", port), handler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server, port


class TestHttpStripParallelDecode:
    def test_parallel_decode_matches_serial(self, monkeypatch):
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
        arr, blob = _make_stripped_uint16(1024, 2048, compression="deflate")
        server, port = _start_server(blob)
        try:
            url = f"http://127.0.0.1:{port}/s.tif"
            par, _ = read_to_array(url, window=(0, 0, 1024, 2048))
            with patch.object(_reader_mod,
                              "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10**12):
                ser, _ = read_to_array(url, window=(0, 0, 1024, 2048))
        finally:
            server.shutdown()
        np.testing.assert_array_equal(par, ser)
        np.testing.assert_array_equal(par, arr)

    def test_serial_gate_engages_on_single_strip(self, monkeypatch):
        """HTTP windowed-strip path with a single-strip source: the
        gate at ``_fetch_decode_cog_http_strips`` (n_decode_strips > 1)
        must short-circuit to the serial branch.

        ``ThreadPoolExecutor`` is also used by ``read_ranges`` to fan
        out fetches, but for a single-range fetch that path takes the
        ``len(ranges) == 1`` short-circuit and never instantiates a
        pool. So spying on ``_decode_strip_or_tile`` and asserting the
        thread identity catches a regression that fires the pool when
        it shouldn't, without false positives from the fetch layer.
        """
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
        # 2x32 uint16 → 1 strip with the writer's default rps.
        arr = np.arange(2 * 32, dtype=np.uint16).reshape(2, 32)
        da = xr.DataArray(arr, dims=["y", "x"])
        with tempfile.NamedTemporaryFile(suffix=".tif",
                                         delete=False) as f:
            tif_path = f.name
        try:
            to_geotiff(da, tif_path, compression="deflate",
                       tiled=False)
            with open(tif_path, "rb") as f:
                blob = f.read()
        finally:
            try:
                os.remove(tif_path)
            except OSError:
                pass

        threads_seen: list[int] = []
        real_decode = _reader_mod._decode_strip_or_tile

        def _spy(*args, **kwargs):
            threads_seen.append(threading.get_ident())
            return real_decode(*args, **kwargs)

        monkeypatch.setattr(
            _reader_mod, "_decode_strip_or_tile", _spy)

        server, port = _start_server(blob)
        try:
            url = f"http://127.0.0.1:{port}/tiny.tif"
            out, _ = read_to_array(url, window=(0, 0, 2, 32))
        finally:
            server.shutdown()

        main_tid = threading.get_ident()
        assert threads_seen, "_decode_strip_or_tile was never called"
        assert all(tid == main_tid for tid in threads_seen), (
            f"decode ran in worker thread(s) "
            f"{set(threads_seen) - {main_tid}}; main is {main_tid}. "
            f"The serial gate failed to short-circuit on a "
            f"single-strip HTTP read."
        )
        np.testing.assert_array_equal(out, arr)


# ----------------------------------------------------------------------
# Planar=2 multi-band stripped TIFF
# ----------------------------------------------------------------------


class TestPlanar2MultibandStripParallel:
    """Planar=2 multi-band stripped TIFF.

    ``_read_strips`` has a separate ``planar == 2 and samples > 1``
    branch in the strip-job collection loop (xrspatial/geotiff/
    _reader.py:1949-1963). The parallel pool itself is planar-agnostic,
    so a regression in band-ordering or per-band strip indexing would
    survive the chunky tests above; this class covers that branch.
    """

    def test_parallel_matches_serial_planar2(self, tmp_path):
        """Parallel and serial decode produce bit-identical output on
        a planar=2 multi-band stripped TIFF."""
        tifffile = pytest.importorskip("tifffile")
        rng = np.random.default_rng(seed=20260518)
        bands, height, width = 3, 1024, 1024
        arr = rng.integers(
            0, 1000, size=(bands, height, width), dtype=np.uint16)
        p = str(tmp_path / "planar2.tif")
        tifffile.imwrite(
            p, arr,
            photometric="rgb",
            planarconfig="separate",
            compression="deflate",
            rowsperstrip=128,
        )

        par, _ = read_to_array(p)
        with patch.object(
                _reader_mod, "_PARALLEL_DECODE_PIXEL_THRESHOLD",
                10 ** 12):
            ser, _ = read_to_array(p)

        np.testing.assert_array_equal(par, ser)
        # The reader returns (height, width, samples); the fixture is
        # (bands, height, width). Reorder for comparison.
        np.testing.assert_array_equal(par, np.moveaxis(arr, 0, -1))

    def test_windowed_planar2_parallel(self, tmp_path):
        """A window across multiple strips on a planar=2 multi-band
        stripped TIFF still matches the slice of the full decode."""
        tifffile = pytest.importorskip("tifffile")
        rng = np.random.default_rng(seed=20260519)
        bands, height, width = 3, 1024, 1024
        arr = rng.integers(
            0, 1000, size=(bands, height, width), dtype=np.uint16)
        p = str(tmp_path / "planar2_win.tif")
        tifffile.imwrite(
            p, arr,
            photometric="rgb",
            planarconfig="separate",
            compression="deflate",
            rowsperstrip=128,
        )

        par, _ = read_to_array(p, window=(100, 100, 900, 900))
        with patch.object(
                _reader_mod, "_PARALLEL_DECODE_PIXEL_THRESHOLD",
                10 ** 12):
            ser, _ = read_to_array(p, window=(100, 100, 900, 900))

        np.testing.assert_array_equal(par, ser)
        expected = np.moveaxis(arr, 0, -1)[100:900, 100:900]
        np.testing.assert_array_equal(par, expected)

    def test_http_windowed_planar2_parallel(self, monkeypatch):
        """HTTP windowed strip path on planar=2 multi-band: pins the
        per-band strip-job loop inside
        ``_fetch_decode_cog_http_strips`` (the local path's planar=2
        coverage above does not reach the HTTP fetch+decode branch
        because full-image HTTP reads dispatch back to
        ``_read_strips``)."""
        tifffile = pytest.importorskip("tifffile")
        monkeypatch.setenv(
            "XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
        rng = np.random.default_rng(seed=20260520)
        bands, height, width = 3, 1024, 1024
        arr = rng.integers(
            0, 1000, size=(bands, height, width), dtype=np.uint16)
        with tempfile.NamedTemporaryFile(
                suffix=".tif", delete=False) as f:
            tif_path = f.name
        try:
            tifffile.imwrite(
                tif_path, arr,
                photometric="rgb",
                planarconfig="separate",
                compression="deflate",
                rowsperstrip=128,
            )
            with open(tif_path, "rb") as f:
                blob = f.read()
        finally:
            try:
                os.remove(tif_path)
            except OSError:
                pass

        server, port = _start_server(blob)
        try:
            url = f"http://127.0.0.1:{port}/planar2.tif"
            par, _ = read_to_array(
                url, window=(100, 100, 900, 900))
            with patch.object(
                    _reader_mod, "_PARALLEL_DECODE_PIXEL_THRESHOLD",
                    10 ** 12):
                ser, _ = read_to_array(
                    url, window=(100, 100, 900, 900))
        finally:
            server.shutdown()

        np.testing.assert_array_equal(par, ser)
        expected = np.moveaxis(arr, 0, -1)[100:900, 100:900]
        np.testing.assert_array_equal(par, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
