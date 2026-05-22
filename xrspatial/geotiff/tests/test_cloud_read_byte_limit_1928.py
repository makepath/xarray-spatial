"""Regression tests for issue #1928.

Eager reads from fsspec sources used to call ``_CloudSource.read_all()``
unconditionally, downloading the entire object before any TIFF header
parse or ``max_pixels`` guard could fire. A crafted ``s3://`` / ``gs://``
/ ``memory://`` object could exhaust memory or bandwidth before the
dimensions were checked.

The fix adds a ``max_cloud_bytes`` budget (default 256 MiB, env override
``XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES``) that runs against the compressed
object size before any bytes are fetched. ``_CloudSource`` already
fetches the size from fsspec at construction, so the check is free.
"""
from __future__ import annotations

import numpy as np
import pytest

fsspec = pytest.importorskip("fsspec")

from xrspatial.geotiff import open_geotiff, to_geotiff  # noqa: E402
from xrspatial.geotiff._reader import _MAX_CLOUD_BYTES_SENTINEL  # noqa: E402
from xrspatial.geotiff._reader import MAX_CLOUD_BYTES_DEFAULT  # noqa: E402
from xrspatial.geotiff._reader import (CloudSizeLimitError, _resolve_max_cloud_bytes,  # noqa: E402
                                       read_to_array)


def _put_in_memory_fs(path: str, payload: bytes) -> None:
    fs = fsspec.filesystem("memory")
    fs.pipe(path, payload)


def _drop_from_memory_fs(path: str) -> None:
    fs = fsspec.filesystem("memory")
    try:
        fs.rm(path)
    except FileNotFoundError:
        pass


def _make_small_tif_bytes(tmp_path) -> bytes:
    """Build a small valid TIFF via the public writer."""
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    local = str(tmp_path / "src_1928.tif")
    to_geotiff(arr, local, compression="none")
    with open(local, "rb") as f:
        return f.read()


class TestResolveMaxCloudBytes:
    """``_resolve_max_cloud_bytes`` precedence: kwarg > env > default."""

    def test_sentinel_returns_default(self):
        assert _resolve_max_cloud_bytes(
            _MAX_CLOUD_BYTES_SENTINEL
        ) == MAX_CLOUD_BYTES_DEFAULT

    def test_none_disables_check(self):
        assert _resolve_max_cloud_bytes(None) is None

    def test_int_kwarg_wins(self):
        assert _resolve_max_cloud_bytes(42) == 42

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES", "9999")
        assert _resolve_max_cloud_bytes(_MAX_CLOUD_BYTES_SENTINEL) == 9999

    def test_kwarg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES", "9999")
        assert _resolve_max_cloud_bytes(123) == 123
        assert _resolve_max_cloud_bytes(None) is None

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv(
            "XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES", "not-an-int"
        )
        assert _resolve_max_cloud_bytes(
            _MAX_CLOUD_BYTES_SENTINEL
        ) == MAX_CLOUD_BYTES_DEFAULT

    def test_zero_or_negative_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES", "0")
        assert _resolve_max_cloud_bytes(
            _MAX_CLOUD_BYTES_SENTINEL
        ) == MAX_CLOUD_BYTES_DEFAULT
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES", "-1")
        assert _resolve_max_cloud_bytes(
            _MAX_CLOUD_BYTES_SENTINEL
        ) == MAX_CLOUD_BYTES_DEFAULT


class TestCloudByteLimit:
    """End-to-end through ``read_to_array`` / ``open_geotiff``."""

    def test_small_cloud_object_under_budget_reads(self, tmp_path):
        """Default budget (256 MiB) does not block normal-sized files."""
        payload = _make_small_tif_bytes(tmp_path)
        path = "/under_budget_1928.tif"
        _put_in_memory_fs(path, payload)
        try:
            arr, _ = read_to_array(f"memory://{path}")
            assert arr.shape == (4, 4)
        finally:
            _drop_from_memory_fs(path)

    def test_oversized_cloud_object_rejected_before_read(self, tmp_path):
        """A file larger than ``max_cloud_bytes`` raises without reading.

        The TIFF itself is valid and small, but the explicit per-call
        ``max_cloud_bytes`` is set below the object size to force the
        guard to fire.
        """
        payload = _make_small_tif_bytes(tmp_path)
        path = "/over_budget_1928.tif"
        _put_in_memory_fs(path, payload)
        try:
            with pytest.raises(
                CloudSizeLimitError, match="exceeds max_cloud_bytes"
            ):
                read_to_array(f"memory://{path}", max_cloud_bytes=10)
        finally:
            _drop_from_memory_fs(path)

    def test_none_disables_limit(self, tmp_path):
        """``max_cloud_bytes=None`` restores pre-#1928 behaviour."""
        payload = _make_small_tif_bytes(tmp_path)
        path = "/disabled_check_1928.tif"
        _put_in_memory_fs(path, payload)
        try:
            arr, _ = read_to_array(
                f"memory://{path}", max_cloud_bytes=None
            )
            assert arr.shape == (4, 4)
        finally:
            _drop_from_memory_fs(path)

    def test_env_var_threshold_applied(self, tmp_path, monkeypatch):
        """Env override threads through when the kwarg is unspecified."""
        payload = _make_small_tif_bytes(tmp_path)
        path = "/env_budget_1928.tif"
        _put_in_memory_fs(path, payload)
        monkeypatch.setenv("XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES", "10")
        try:
            with pytest.raises(CloudSizeLimitError):
                read_to_array(f"memory://{path}")
        finally:
            _drop_from_memory_fs(path)

    def test_open_geotiff_plumbs_max_cloud_bytes(self, tmp_path):
        """The kwarg is reachable from the public ``open_geotiff`` entry
        point and reaches the eager path. Without it, the read succeeds;
        a tight limit rejects."""
        payload = _make_small_tif_bytes(tmp_path)
        path = "/open_geotiff_kwarg_1928.tif"
        _put_in_memory_fs(path, payload)
        try:
            da = open_geotiff(f"memory://{path}")
            assert da.shape == (4, 4)
            with pytest.raises(CloudSizeLimitError):
                open_geotiff(f"memory://{path}", max_cloud_bytes=8)
        finally:
            _drop_from_memory_fs(path)

    def test_local_file_unaffected(self, tmp_path):
        """The limit only applies to fsspec URIs. A local file with a
        tight ``max_cloud_bytes`` still reads (the kwarg is ignored).
        """
        arr = np.arange(16, dtype=np.float32).reshape(4, 4)
        local = str(tmp_path / "local_1928.tif")
        to_geotiff(arr, local, compression="none")
        # Tight limit must not fire on a local path.
        out, _ = read_to_array(local, max_cloud_bytes=1)
        np.testing.assert_array_equal(out, arr)

    def test_http_path_unaffected(self):
        """The HTTP path uses range requests, not ``read_all``, so the
        budget does not run there. We only check that the kwarg does not
        change the dispatch (no ``CloudSizeLimitError`` for http URLs).
        The HTTP code path is exercised by the loopback tests; here we
        just confirm dispatch.
        """
        # A clearly bogus HTTP URL should fail with a connection / DNS
        # style error, not a CloudSizeLimitError, since the cloud-byte
        # guard is not on the HTTP path.
        with pytest.raises(Exception) as exc_info:
            read_to_array(
                "http://127.0.0.1:1/nonexistent.tif",
                max_cloud_bytes=1,
            )
        assert not isinstance(exc_info.value, CloudSizeLimitError)
