"""Corrupt ``.ovr`` sidecar must not break a base read (issue #2416).

The release contract classifies ``reader.local_file`` as stable and
``reader.sidecar_ovr`` as advanced. A stale, truncated, or malformed
sibling ``.ovr`` file -- typically dropped by external tools like
``gdaladdo``, partial downloads, or stray build artifacts -- should not
take the stable base read down.

These tests pin the contract:

* A base read against a valid TIFF succeeds when the sibling ``.ovr``
  bytes are garbage. A ``RuntimeWarning`` is emitted so the user can
  still investigate.
* Explicitly requesting ``overview_level=0`` is also a base read and
  must succeed.
* Explicitly requesting an external overview level (``overview_level=1``)
  surfaces the underlying parse error -- silent fallback only applies
  when the caller did not ask for the broken surface.
* ``CloudSizeLimitError`` still propagates because it represents a
  caller-set byte budget the caller asked to hear about.
* The same contract holds on the GPU eager path
  (``read_geotiff_gpu`` via ``open_geotiff(gpu=True)``) which also
  walks the sidecar before IFD selection.
* The metadata-only path (``_read_geo_info``) follows the same rule
  for the base level.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import open_geotiff
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._writers.eager import to_geotiff


# ---------------------------------------------------------------------------
# Fixtures: build a valid base TIFF and pair it with a chosen sidecar payload.
# ---------------------------------------------------------------------------
def _make_base(tmp_path) -> tuple:
    """Write a 4x4 uint16 base TIFF and return (path, expected array)."""
    arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
    xs = np.arange(4) + 0.5
    ys = np.arange(4) + 0.5
    da = xr.DataArray(arr, dims=("y", "x"), coords={"y": ys, "x": xs})
    da.attrs["crs"] = 4326
    path = tmp_path / "base_2416.tif"
    to_geotiff(da, str(path), tiled=False, compression="none")
    return path, arr


def _make_corrupt_sidecar(base_path, payload: bytes) -> None:
    side = str(base_path) + ".ovr"
    with open(side, "wb") as f:
        f.write(payload)


def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Eager CPU path: base read survives a broken sidecar.
# ---------------------------------------------------------------------------
def test_open_geotiff_base_read_survives_unreadable_sidecar(tmp_path):
    path, expected = _make_base(tmp_path)
    _make_corrupt_sidecar(path, b"not a tiff")

    with pytest.warns(RuntimeWarning, match="Ignoring unreadable sidecar"):
        da = open_geotiff(str(path))

    np.testing.assert_array_equal(da.values, expected)


def test_open_geotiff_overview_level_zero_survives_unreadable_sidecar(tmp_path):
    """``overview_level=0`` is also a base read; the sidecar is not needed."""
    path, expected = _make_base(tmp_path)
    _make_corrupt_sidecar(path, b"\x00\x00\x00\x00\x00\x00\x00\x00")

    with pytest.warns(RuntimeWarning, match="Ignoring unreadable sidecar"):
        da = open_geotiff(str(path), overview_level=0)

    np.testing.assert_array_equal(da.values, expected)


def test_read_to_array_base_read_survives_unreadable_sidecar(tmp_path):
    """The lower-level ``read_to_array`` entry point honours the same rule."""
    path, expected = _make_base(tmp_path)
    _make_corrupt_sidecar(path, b"this is not even close to a TIFF header")

    with pytest.warns(RuntimeWarning, match="Ignoring unreadable sidecar"):
        arr, _ = read_to_array(str(path))

    np.testing.assert_array_equal(arr, expected)


@pytest.mark.parametrize(
    "payload",
    [
        b"",                              # empty file
        b"x",                             # too short for any header parse
        b"not a tiff",                    # 10 bytes, looks like text
        b"GZIP\x1f\x8b\x08\x00",          # gzip magic, wrong file type
        b"\x89PNG\r\n\x1a\n",             # PNG magic, wrong file type
    ],
)
def test_open_geotiff_base_read_survives_various_sidecar_payloads(
    tmp_path, payload
):
    path, expected = _make_base(tmp_path)
    _make_corrupt_sidecar(path, payload)

    with pytest.warns(RuntimeWarning, match="Ignoring unreadable sidecar"):
        da = open_geotiff(str(path))

    np.testing.assert_array_equal(da.values, expected)


# ---------------------------------------------------------------------------
# Explicit external-overview requests still surface the error.
# ---------------------------------------------------------------------------
def test_open_geotiff_requesting_sidecar_level_still_raises(tmp_path):
    """Asking for ``overview_level=1`` is asking for the sidecar surface;
    a parse failure on that surface should not be silently swallowed."""
    path, _ = _make_base(tmp_path)
    _make_corrupt_sidecar(path, b"not a tiff")

    # The base IFD list has length 1, sidecar is unreadable -> level 1
    # falls outside the merged list. The reader's ``select_overview_ifd``
    # raises ``ValueError`` for an out-of-range level. The point is that
    # the error surfaces (warning + then explicit raise) rather than the
    # base read silently failing.
    with pytest.warns(RuntimeWarning, match="Ignoring unreadable sidecar"):
        with pytest.raises(ValueError, match="out of range"):
            open_geotiff(str(path), overview_level=1)


# ---------------------------------------------------------------------------
# CloudSizeLimitError still propagates: byte budget is a caller contract.
# ---------------------------------------------------------------------------
def test_cloud_size_limit_error_from_sidecar_is_not_silenced(tmp_path, monkeypatch):
    from xrspatial.geotiff import _sidecar as _sidecar_mod
    from xrspatial.geotiff._reader import CloudSizeLimitError

    path, _ = _make_base(tmp_path)
    _make_corrupt_sidecar(path, b"\x00")  # contents do not matter; we stub

    def _budget_breach(_path, *, max_cloud_bytes=None):
        raise CloudSizeLimitError("sidecar too big for budget")

    monkeypatch.setattr(_sidecar_mod, "load_sidecar", _budget_breach)

    with pytest.raises(CloudSizeLimitError):
        open_geotiff(str(path))


# ---------------------------------------------------------------------------
# Metadata-only path (used by the dask graph builder for local files).
# ---------------------------------------------------------------------------
def test_read_geo_info_base_survives_unreadable_sidecar(tmp_path):
    from xrspatial.geotiff import _read_geo_info

    path, _ = _make_base(tmp_path)
    _make_corrupt_sidecar(path, b"not a tiff")

    with pytest.warns(RuntimeWarning, match="Ignoring unreadable sidecar"):
        geo, h, w, dtype, _n = _read_geo_info(str(path))

    assert (h, w) == (4, 4)
    assert geo is not None


def test_read_geo_info_cloud_size_limit_error_is_not_silenced(
    tmp_path, monkeypatch
):
    """Symmetry with the eager CPU path: the metadata-only helper must
    re-raise ``CloudSizeLimitError`` rather than swallow it as a generic
    sidecar failure. Cannot fire on a local mmap source today, but the
    contract should not silently regress if a future patch widens the
    helper to a cloud source."""
    from xrspatial.geotiff import _read_geo_info
    from xrspatial.geotiff import _sidecar as _sidecar_mod
    from xrspatial.geotiff._reader import CloudSizeLimitError

    path, _ = _make_base(tmp_path)
    _make_corrupt_sidecar(path, b"\x00")

    def _budget_breach(_path, *, max_cloud_bytes=None):
        raise CloudSizeLimitError("sidecar too big for budget")

    monkeypatch.setattr(_sidecar_mod, "load_sidecar", _budget_breach)

    with pytest.raises(CloudSizeLimitError):
        _read_geo_info(str(path))


# ---------------------------------------------------------------------------
# GPU eager path: same contract.
# ---------------------------------------------------------------------------
@pytest.fixture
def _gpu_or_skip():
    if not _gpu_available():
        pytest.skip("cupy + CUDA required")


def test_open_geotiff_gpu_base_read_survives_unreadable_sidecar(
    tmp_path, _gpu_or_skip
):
    path, expected = _make_base(tmp_path)
    _make_corrupt_sidecar(path, b"not a tiff")

    with pytest.warns(RuntimeWarning, match="Ignoring unreadable sidecar"):
        gpu_da = open_geotiff(str(path), gpu=True)

    np.testing.assert_array_equal(gpu_da.data.get(), expected)
