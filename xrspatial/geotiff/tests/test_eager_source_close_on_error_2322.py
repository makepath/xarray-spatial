"""Regression tests for issue #2322.

``_read_to_array`` constructed a source (``_FileSource``,
``_BytesIOSource``, or ``_CloudSource``) and immediately called
``src.read_all()`` BEFORE entering the ``try/finally`` block that
calls ``src.close()``. If ``read_all()`` raised mid-read (a fsspec
network failure, a transient S3 error, a local I/O error), the
exception propagated up and ``src.close()`` was never called.

``_CloudSource.close()`` is a no-op today, so the leak was latent
rather than active, but the structural guard is needed so a future
resource-holding source (pinned credentials, persistent fsspec
sessions, cached file handles) does not silently leak on the failure
path.

These tests pin the contract by injecting fake source objects whose
``read_all()`` raises, then asserting ``close()`` was still called.
"""
from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._reader import _read_to_array


def _make_tiff_bytes() -> bytes:
    """Build a small valid TIFF in memory for happy-path baselines."""
    arr = np.arange(8 * 8, dtype=np.float32).reshape(8, 8)
    da = xr.DataArray(
        arr,
        dims=['y', 'x'],
        attrs={'crs': 4326,
               'transform': (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)},
    )
    buf = io.BytesIO()
    to_geotiff(da, buf, compression='none')
    return buf.getvalue()


class _FailingSource:
    """Fake source whose ``read_all()`` raises.

    Records the number of ``close()`` calls so the test can verify
    cleanup ran on the exception path. Mirrors the ``_CloseTracker``
    pattern in ``test_cog_http_close_on_error_1816.py`` but as a
    standalone fake rather than a delegating wrapper, because the
    failure happens inside ``read_all()`` itself (no real source ever
    succeeds in this scenario).
    """

    def __init__(self, exc: Exception):
        self._exc = exc
        self.close_count = 0
        self.read_all_count = 0

    def read_all(self):
        self.read_all_count += 1
        raise self._exc

    def close(self):
        self.close_count += 1


def test_file_source_closed_when_read_all_raises(tmp_path):
    """``_FileSource.read_all()`` failure path still runs ``close()``.

    Patches ``_FileSource`` so the constructor returns a fake whose
    ``read_all()`` raises. The fix wraps the eager read in the
    ``try/finally`` that calls ``src.close()``, so the close count
    must be exactly 1 even though ``read_all()`` raised.
    """
    path = str(tmp_path / "tmp_2322_cleanup_file.tif")
    # File must exist so ``_coerce_path`` does not bail early — even
    # though the patched ``_FileSource`` never reads from it.
    with open(path, 'wb') as f:
        f.write(_make_tiff_bytes())

    fake = _FailingSource(OSError("simulated local read failure"))

    with patch(
        'xrspatial.geotiff._reader._FileSource',
        return_value=fake,
    ):
        with pytest.raises(OSError, match="simulated local read failure"):
            _read_to_array(path)

    assert fake.read_all_count == 1, (
        "read_all() was not invoked; the test setup is wrong")
    assert fake.close_count == 1, (
        "src.close() was not called on the exception path; "
        "the try/finally guard around read_all() is missing")


def test_bytesio_source_closed_when_read_all_raises():
    """``_BytesIOSource.read_all()`` failure path still runs ``close()``.

    Same shape as the file-source test, but via the file-like
    branch (``_is_file_like(source)`` returns True for ``BytesIO``).
    """
    buf = io.BytesIO(_make_tiff_bytes())

    fake = _FailingSource(RuntimeError("simulated buffer read failure"))

    with patch(
        'xrspatial.geotiff._reader._BytesIOSource',
        return_value=fake,
    ):
        with pytest.raises(RuntimeError, match="simulated buffer read"):
            _read_to_array(buf)

    assert fake.read_all_count == 1
    assert fake.close_count == 1, (
        "src.close() was not called on the exception path for the "
        "BytesIO/file-like branch")


def test_cloud_source_closed_when_read_all_raises():
    """``_CloudSource.read_all()`` failure path still runs ``close()``.

    The cloud branch is the original motivating case: a fsspec read
    that fails mid-download must still tear down whatever state the
    source holds. Today ``_CloudSource.close()`` is a no-op, so this
    test exists primarily to lock in the structural guard before any
    real resource is added.

    Patches both ``_is_fsspec_uri`` (to route the input string into
    the cloud branch) and ``_CloudSource`` (to return a fake that
    raises in ``read_all()``). ``_resolve_max_cloud_bytes`` is
    bypassed by passing ``max_cloud_bytes=None`` so the
    pre-``read_all`` size check does not consume the close() call
    on its own error path (that branch is already exception-safe and
    is exercised separately in ``test_cloud_read_byte_limit_1928``).
    """
    fake = _FailingSource(OSError("simulated S3 mid-download failure"))

    with patch(
        'xrspatial.geotiff._reader._is_fsspec_uri',
        return_value=True,
    ), patch(
        'xrspatial.geotiff._reader._CloudSource',
        return_value=fake,
    ):
        with pytest.raises(OSError, match="simulated S3 mid-download"):
            _read_to_array("s3://fake-bucket/fake-key.tif",
                           max_cloud_bytes=None)

    assert fake.read_all_count == 1
    assert fake.close_count == 1, (
        "src.close() was not called on the exception path for the "
        "fsspec/cloud branch; this is the main case the fix targets")
