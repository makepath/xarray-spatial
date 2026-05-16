"""Backend-parity coverage for bool / np.bool_ nodata rejection.

Issue #1911 added the ``isinstance(nodata, (bool, np.bool_)) -> TypeError``
guard at the ``to_geotiff`` entry point, with a belt-and-braces copy in
``_geotags.build_geo_tags``. Issue #1921 extended the same parity to the
sibling writers:

* ``write_vrt`` -- now rejects bool nodata at the public wrapper via
  ``_validate_nodata_arg`` and again inside ``_vrt.write_vrt`` as
  defense-in-depth. Previously wrote ``<NoDataValue>True</NoDataValue>``
  into the VRT XML, which no reader parses as numeric, so the
  round-trip silently dropped the sentinel.
* ``write_geotiff_gpu`` (direct call) -- already routes through
  ``_validate_nodata_arg`` near the top of the function. Pinning the
  behaviour here so a refactor that drops that call surfaces the
  regression at the parity boundary, not inside ``build_geo_tags``.

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
def test_write_vrt_rejects_bool_nodata(src_geotiff, tmp_path, bad):
    """``write_vrt`` raises ``TypeError`` for any bool nodata.

    Fixed in issue #1921 by routing the public ``write_vrt`` wrapper
    through ``_validate_nodata_arg`` and adding a defense-in-depth check
    inside the internal ``_vrt.write_vrt``.
    """
    vrt_path = str(tmp_path / "out_1921_bad.vrt")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        write_vrt(vrt_path, [src_geotiff], nodata=bad)


@pytest.mark.parametrize(
    "bad",
    [True, False, np.bool_(True), np.bool_(False)],
)
def test_write_vrt_internal_rejects_bool_nodata(src_geotiff, tmp_path, bad):
    """Direct call to the internal ``_vrt.write_vrt`` also rejects bool.

    Defense-in-depth: the public wrapper's ``_validate_nodata_arg`` is
    skipped when callers reach the internal symbol directly (e.g. the
    multi-tile dask write path in ``_writers/eager.py`` that calls
    ``_vrt.write_vrt`` after writing per-tile GeoTIFFs, or a future
    split of the wrapper). Parametrize over both ``bool`` and
    ``np.bool_`` polarities so a refactor that narrows the internal
    guard to just ``bool`` surfaces here, not in user code. See #1921.
    """
    from xrspatial.geotiff._vrt import write_vrt as _internal_write_vrt
    vrt_path = str(tmp_path / "out_1921_internal.vrt")
    with pytest.raises(TypeError, match="nodata must be numeric"):
        _internal_write_vrt(vrt_path, [src_geotiff], nodata=bad)


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

    The top-of-function ``_validate_nodata_arg`` call (added by #1973)
    fires first; the deeper ``build_geo_tags`` guard is a second line
    of defense. Pinning the behaviour so a refactor that drops the
    top-of-function call surfaces here, not deep inside the geotag
    builder.
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
