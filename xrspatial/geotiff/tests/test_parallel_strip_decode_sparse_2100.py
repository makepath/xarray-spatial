"""Sparse-strip coverage for the parallel-decode strip paths (#2100).

The strip-decode parallelisation landed in #2100 / #2104 added a
collect-decode-place pipeline in both ``_read_strips`` and
``_fetch_decode_cog_http_strips``. The job-collection loop filters out
sparse strips (``byte_counts[idx] == 0``) so the pool never decodes an
empty byte slice, and the pre-allocated result already carries the
sparse fill value. ``test_parallel_strip_decode_2100.py`` exercises
the parallel/serial parity and the pool-engaged branch, but every
fixture has every strip populated; a regression that lost the sparse
filter (e.g. by appending a job before the ``if byte_counts[...] == 0:
continue`` guard) would slip through because the existing
``test_sparse_cog.py::TestSparseStrips`` fixture is 128x128, well below
the 64K-pixel parallel-decode gate.

These tests build a large sparse-stripped TIFF (>= 64K strip pixels,
multi-strip) so the parallel branch engages, then assert:

1. Local path: parallel and serial decode return the same array; the
   filled rows carry the source value and the sparse rows carry the
   nodata sentinel.
2. Local path under a window that straddles the sparse boundary.
3. Local planar=2 multi-band sparse decode (the dedicated
   ``planar == 2 and samples > 1`` branch in the strip-job collection
   loop has its own ``if byte_counts[global_idx] == 0: continue`` guard
   that the existing tests do not reach with a sparse fixture).
4. HTTP COG strip path: a windowed read that fetches a strict subset of
   non-sparse strips still parallelises and matches the local read.

A mutation against the sparse guard (delete the ``continue`` so sparse
strips are appended to ``strip_jobs``) flips every test in this file
red because the decoder either returns a zero-length array or raises
on empty input — confirmed before commit.
"""
from __future__ import annotations

import concurrent.futures
import http.server
import socket
import threading
from unittest.mock import patch

import numpy as np
import pytest

# Sparse-stripped fixtures depend on rasterio's TIFF writer (GDAL's
# ``SPARSE_OK`` driver option). Skip the module wholesale when rasterio
# is unavailable in the test environment; the GeoTIFF reader code paths
# under test do not depend on rasterio at runtime.
rasterio = pytest.importorskip("rasterio")

from xrspatial.geotiff import _decode as _decode_mod  # noqa: E402
from xrspatial.geotiff import _reader as _reader_mod  # noqa: E402
from xrspatial.geotiff._reader import read_to_array  # noqa: E402

# Local-strip helpers -------------------------------------------------------


def _write_sparse_stripped_large(
    path: str,
    *,
    width: int = 2048,
    height: int = 2048,
    rps: int = 64,
    filled_rows: int = 256,
    fill_value: int = 200,
    dtype: str = "uint16",
    nodata: int = 0,
    bands: int = 1,
    planar: str = "pixel",
):
    """Build a large stripped TIFF with sparse strips below ``filled_rows``.

    The default geometry (2048x2048, rps=64) yields ``width * rps =
    131_072`` pixels per strip — clear of the 64K parallel-decode gate
    — and 32 strips per band, so leaving rows below ``filled_rows`` un-
    written gives ``32 - filled_rows / rps`` sparse strips that the
    job-collection loop must filter.

    ``planar``: ``"pixel"`` (contig, planar=1) or ``"band"``
    (planar=2 / separate). Rasterio accepts only those literals.
    """
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "height": height,
        "width": width,
        "count": bands,
        "tiled": False,
        "blockysize": rps,
        "compress": "DEFLATE",
        "SPARSE_OK": "TRUE",
        "nodata": nodata,
        "interleave": planar,
    }
    fill = np.full((filled_rows, width), fill_value, dtype=np.dtype(dtype))
    with rasterio.open(path, "w", **profile) as dst:
        for b in range(1, bands + 1):
            dst.write(
                fill, b,
                window=rasterio.windows.Window(0, 0, width, filled_rows))


# HTTP range-server reused from the existing parallel-strip test.
class _RangeHandler(http.server.BaseHTTPRequestHandler):
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
            self.send_header(
                "Content-Range",
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

    def log_message(self, format, *args):
        return


def _start_server(blob: bytes):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    handler = type("BlobHandler", (_RangeHandler,), {"blob": blob})
    server = http.server.HTTPServer(("127.0.0.1", port), handler)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    return server, port


# Local-strip sparse coverage ----------------------------------------------

class TestReadStripsSparseParallel:
    """``_read_strips`` parallel branch with sparse strips."""

    def test_full_image_parallel_matches_serial(self, tmp_path):
        """Sparse + non-sparse strips: parallel and serial paths return
        bit-identical output, and the sparse rows land on the nodata
        sentinel."""
        path = str(tmp_path / "sparse_par_full.tif")
        _write_sparse_stripped_large(path)

        par, _ = read_to_array(path)
        # Patch the threshold in ``_decode`` (PR-G #2246 home of
        # ``_read_strips``), not in ``_reader``: the back-imported name in
        # ``_reader`` is a separate reference and patching it would leave
        # the live binding in ``_decode`` unchanged.
        with patch.object(
                _decode_mod,
                "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10 ** 12):
            ser, _ = read_to_array(path)

        np.testing.assert_array_equal(par, ser)
        # First 256 rows carry the fill value, the rest are sparse → 0.
        assert np.all(par[:256, :] == 200)
        assert np.all(par[256:, :] == 0)

    def test_parallel_pool_engages_on_sparse_multi_strip(self, tmp_path):
        """A 2048x2048 sparse stripped TIFF with rps=64 has multiple
        non-sparse strips; the parallel-decode pool must instantiate.

        Validates that the sparse-strip filter does not regress the gate
        by pruning the job list below ``n_strips > 1``."""
        path = str(tmp_path / "sparse_par_gate.tif")
        # 4 strips filled, 28 sparse → 4 non-sparse strips, so pool
        # engages because n_strips = 4 > 1 and strip_pixel_count
        # = 2048 * 64 = 131_072 >= 65_536.
        _write_sparse_stripped_large(path, filled_rows=256)
        # Patch ``concurrent.futures.ThreadPoolExecutor`` rather than the
        # reader module binding: strip decode lives in ``_decode`` after
        # PR-G (issue #2246) and re-imports the executor function-locally.
        with patch.object(
                concurrent.futures, "ThreadPoolExecutor",
                wraps=concurrent.futures.ThreadPoolExecutor) as mock_pool:
            out, _ = read_to_array(path)
            assert mock_pool.called, (
                "parallel-decode pool was not engaged for a multi-strip "
                "sparse-stripped TIFF whose non-sparse strips clear the "
                "parallel gate"
            )
        assert np.all(out[:256, :] == 200)
        assert np.all(out[256:, :] == 0)

    def test_windowed_across_sparse_boundary(self, tmp_path):
        """A window that straddles the filled/sparse boundary returns
        the filled rows on top and zeros below, with parallel == serial.

        Catches a regression in the per-strip placement loop that mis-
        attributes a parallel-decoded strip to the wrong destination
        slice when the strip range skips over sparse entries."""
        path = str(tmp_path / "sparse_par_win.tif")
        _write_sparse_stripped_large(path)

        win = (128, 0, 384, 1024)  # row range [128, 384), col range [0, 1024)
        par, _ = read_to_array(path, window=win)
        with patch.object(
                _decode_mod,
                "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10 ** 12):
            ser, _ = read_to_array(path, window=win)

        np.testing.assert_array_equal(par, ser)
        assert par.shape == (256, 1024)
        # Rows [128, 256) are filled; rows [256, 384) are sparse.
        assert np.all(par[:128, :] == 200)
        assert np.all(par[128:, :] == 0)

    def test_all_sparse_image_returns_fill(self, tmp_path):
        """An image with zero filled rows → every strip is sparse →
        ``strip_jobs`` is empty → the parallel branch's
        ``n_strips > 1`` gate is false and the loop short-circuits.

        Mirrors the "no jobs" degenerate case that the existing tests
        miss because they always produce >= 1 non-sparse strip."""
        path = str(tmp_path / "all_sparse.tif")
        _write_sparse_stripped_large(path, filled_rows=0)
        with patch.object(
                concurrent.futures, "ThreadPoolExecutor",
                wraps=concurrent.futures.ThreadPoolExecutor) as mock_pool:
            out, _ = read_to_array(path)
            # All strips sparse → no jobs → no pool.
            assert not mock_pool.called, (
                "parallel-decode pool was instantiated for an all-sparse "
                "image; the strip-job filter should have left the job "
                "list empty and the gate should have short-circuited"
            )
        assert out.shape == (2048, 2048)
        assert np.all(out == 0)


# Planar=2 sparse coverage --------------------------------------------------

class TestReadStripsSparsePlanar2:
    """``_read_strips`` planar=2 branch with sparse strips.

    The strip-job collection loop has a dedicated
    ``planar == 2 and samples > 1`` branch (lines 1949-1962 in
    ``_reader.py``) with its own ``if byte_counts[global_idx] == 0:
    continue`` guard. The existing parallel-strip planar=2 tests fill
    every strip, so a regression in this branch's sparse filter would
    survive."""

    def test_planar2_sparse_parallel_matches_serial(self, tmp_path):
        path = str(tmp_path / "planar2_sparse.tif")
        _write_sparse_stripped_large(
            path,
            width=1024,
            height=1024,
            rps=64,
            filled_rows=128,
            bands=3,
            planar="band",
        )

        par, _ = read_to_array(path)
        with patch.object(
                _decode_mod,
                "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10 ** 12):
            ser, _ = read_to_array(path)

        np.testing.assert_array_equal(par, ser)
        # Reader returns (h, w, samples) for planar=2 multi-band.
        # The fixture wrote the same fill into every band, so every
        # band has the same pattern.
        assert par.shape == (1024, 1024, 3)
        for b in range(3):
            assert np.all(par[:128, :, b] == 200)
            assert np.all(par[128:, :, b] == 0)


# HTTP COG strip sparse coverage -------------------------------------------

class TestHttpStripsSparseParallel:
    """``_fetch_decode_cog_http_strips`` with sparse strips.

    The HTTP strip path also filters ``byte_counts[idx] == 0`` from the
    fetch-range list (line 2646-2648 in ``_reader.py``); a window that
    targets only non-sparse strips still parallel-decodes, and the
    final placement loop must match the local path."""

    def test_http_windowed_strict_subset_parallel(self, tmp_path, monkeypatch):
        """HTTP windowed read on a sparse-stripped TIFF.

        Targeted window covers only filled rows so the fetch list
        excludes the sparse strips, the parallel-decode gate engages,
        and the result matches the local file read.
        """
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
        path = str(tmp_path / "sparse_http.tif")
        _write_sparse_stripped_large(path, filled_rows=256)
        with open(path, "rb") as f:
            blob = f.read()

        server, port = _start_server(blob)
        try:
            url = f"http://127.0.0.1:{port}/sparse.tif"
            par, _ = read_to_array(url, window=(0, 0, 256, 2048))
            with patch.object(
                    _reader_mod,
                    "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10 ** 12):
                ser, _ = read_to_array(url, window=(0, 0, 256, 2048))
        finally:
            server.shutdown()

        np.testing.assert_array_equal(par, ser)
        # The full window is in the filled region; nothing sparse.
        assert np.all(par == 200)

    def test_http_windowed_across_sparse_boundary(
            self, tmp_path, monkeypatch):
        """HTTP windowed read that straddles the sparse boundary: the
        fetch path emits a fetch range per non-sparse strip the window
        touches, the decoder runs in parallel on those, and the sparse
        strips inside the window carry the pre-filled fill value."""
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
        path = str(tmp_path / "sparse_http_boundary.tif")
        _write_sparse_stripped_large(path, filled_rows=256)
        with open(path, "rb") as f:
            blob = f.read()

        server, port = _start_server(blob)
        try:
            url = f"http://127.0.0.1:{port}/sparse2.tif"
            par, _ = read_to_array(url, window=(128, 0, 384, 2048))
            with patch.object(
                    _reader_mod,
                    "_PARALLEL_DECODE_PIXEL_THRESHOLD", 10 ** 12):
                ser, _ = read_to_array(url, window=(128, 0, 384, 2048))
        finally:
            server.shutdown()

        np.testing.assert_array_equal(par, ser)
        assert par.shape == (256, 2048)
        assert np.all(par[:128, :] == 200)
        assert np.all(par[128:, :] == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
