"""Backend-parity coverage for bool / np.bool_ nodata rejection.

Issue #1911 added the ``isinstance(nodata, (bool, np.bool_)) -> TypeError``
guard at the ``to_geotiff`` entry point, with a belt-and-braces copy in
``_geotags.build_geo_tags``. The same parity check was not added to the
sibling writers:

* ``write_vrt`` -- currently writes ``<NoDataValue>True</NoDataValue>``
  into the VRT XML. No reader parses ``"True"`` as numeric, so the
  round-trip silently drops the sentinel (bug, see issue #1921).
* ``write_geotiff_gpu`` (direct call) -- currently raises ``TypeError``,
  but only because the deeper ``build_geo_tags`` guard fires; there is
  no explicit top-of-function check. A future refactor moving the
  ``build_geo_tags`` guard could regress this without anyone noticing.

This file pins both behaviours so the next refactor that touches either
writer surfaces the parity gap.

Found by ``/sweep-test-coverage`` (pass 15 / 2026-05-15).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff, write_vrt
from xrspatial.geotiff.tests.conftest import requires_gpu


@pytest.fixture
def uint8_da():
    """Small uint8 DataArray for nodata round-trip tests."""
    arr = np.zeros((4, 4), dtype=np.uint8)
    return xr.DataArray(arr, dims=['y', 'x'])


@pytest.fixture
def src_geotiff(uint8_da, tmp_path):
    """A real on-disk source GeoTIFF that write_vrt can point at."""
    path = str(tmp_path / "src_1921.tif")
    to_geotiff(uint8_da, path)
    return path


# ---------------------------------------------------------------------------
# write_vrt: the bug from issue #1921
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [True, False, np.bool_(True), np.bool_(False)],
)
@pytest.mark.xfail(
    reason="issue #1921: write_vrt currently writes str(bool) into "
           "<NoDataValue>; the fix should raise TypeError up front.",
    strict=True,
)
def test_write_vrt_rejects_bool_nodata(src_geotiff, tmp_path, bad):
    """``write_vrt`` should raise ``TypeError`` for any bool nodata.

    Will start passing once issue #1921 is fixed; flip the xfail to a
    plain assertion at that point.
    """
    vrt_path = str(tmp_path / "out_1921_bad.vrt")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        write_vrt(vrt_path, [src_geotiff], nodata=bad)


def test_write_vrt_with_bool_nodata_currently_emits_string(
        src_geotiff, tmp_path):
    """Pins the current (buggy) behaviour so the fix is visible as a diff.

    Today the VRT XML contains ``<NoDataValue>True</NoDataValue>``. Once
    issue #1921 is fixed, ``write_vrt`` will raise ``TypeError`` instead;
    this test is then expected to fail and gets removed in the same PR.
    """
    vrt_path = str(tmp_path / "out_1921_pin.vrt")
    try:
        write_vrt(vrt_path, [src_geotiff], nodata=True)
    except TypeError:
        pytest.skip("issue #1921 fixed: write_vrt now rejects bool nodata.")
    with open(vrt_path) as f:
        content = f.read()
    assert "<NoDataValue>True</NoDataValue>" in content, (
        "expected the current buggy str(True) emission")


@pytest.mark.parametrize(
    "good",
    [0, 0.0, -9999, 255, np.int16(-1), np.float32(0.5)],
)
def test_write_vrt_accepts_numeric_nodata(src_geotiff, tmp_path, good):
    """Numeric sentinels go through unchanged: the fix must not over-reject."""
    vrt_path = str(tmp_path / f"out_1921_numeric_{good!r}.vrt")
    write_vrt(vrt_path, [src_geotiff], nodata=good)
    with open(vrt_path) as f:
        content = f.read()
    # The exact format of the emitted nodata string is implementation
    # detail; we only assert no "True"/"False" leaked through.
    assert "<NoDataValue>True</NoDataValue>" not in content
    assert "<NoDataValue>False</NoDataValue>" not in content


def test_write_vrt_accepts_none_nodata(src_geotiff, tmp_path):
    """``nodata=None`` is the documented default and must keep working."""
    vrt_path = str(tmp_path / "out_1921_none.vrt")
    write_vrt(vrt_path, [src_geotiff], nodata=None)
    assert os.path.exists(vrt_path)


# ---------------------------------------------------------------------------
# write_geotiff_gpu: defense-in-depth parity
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize(
    "bad",
    [True, False, np.bool_(True), np.bool_(False)],
)
def test_write_geotiff_gpu_rejects_bool_nodata(uint8_da, tmp_path, bad):
    """Direct ``write_geotiff_gpu`` call rejects bool nodata.

    Currently raises ``TypeError`` only because the deeper
    ``build_geo_tags`` guard fires. Pinning the behaviour so a refactor
    that drops the deeper guard surfaces here, not in user code.
    """
    from xrspatial.geotiff import write_geotiff_gpu
    path = str(tmp_path / "gpu_1921_bad.tif")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        write_geotiff_gpu(uint8_da, path, nodata=bad)


@requires_gpu
def test_to_geotiff_gpu_dispatch_rejects_bool_nodata(uint8_da, tmp_path):
    """Auto-dispatch path: ``to_geotiff(gpu=True, nodata=True)``.

    The eager-side guard fires before dispatch, so the GPU writer never
    runs. Pin that ordering so a future refactor cannot accidentally
    skip the eager check on the GPU dispatch path.
    """
    path = str(tmp_path / "to_geotiff_gpu_1921.tif")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        to_geotiff(uint8_da, path, gpu=True, nodata=True)
