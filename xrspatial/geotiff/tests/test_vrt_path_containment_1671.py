"""Regression tests for issue #1671: VRT path-traversal containment.

The previous fix for path traversal (#1185) called ``os.path.realpath``
on every ``SourceFilename`` but did not enforce that the resolved path
lived under the VRT directory. A crafted VRT could therefore read any
file the process had access to: ``../../etc/passwd``, a symlink that
escapes ``vrt_dir``, or an absolute path anywhere on disk.

The new behaviour rejects:

* relative sources (``relativeToVRT='1'``) that resolve outside the VRT
  directory
* absolute sources (``relativeToVRT='0'``) that resolve outside both
  the VRT directory and any allowlisted root

Operators that legitimately need cross-directory reads opt in via the
``XRSPATIAL_VRT_ALLOWED_ROOTS`` environment variable (colon-separated
list of directory paths).
"""
from __future__ import annotations

import os
import uuid

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import to_geotiff
from xrspatial.geotiff._vrt import parse_vrt, read_vrt as _read_vrt_internal


def _unique_dir(tmp_path, label: str) -> str:
    """Return a sub-path under ``tmp_path`` carrying the issue id + a
    uuid so parallel test workers cannot collide on the same name."""
    d = tmp_path / f"vrt_1671_{label}_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    return str(d)


def _write_minimal_tif(path: str) -> None:
    """Write a 4x4 float32 GeoTIFF the VRT can reference."""
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    y = np.linspace(1.0, 0.0, 4)
    x = np.linspace(0.0, 1.0, 4)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'crs': 4326},
    )
    to_geotiff(da, path, compression='none')


def _build_vrt(vrt_path: str, source_filename: str, relative: str) -> None:
    """Write a 4x4 single-band VRT pointing at *source_filename*.

    ``relative`` is the literal value of the ``relativeToVRT`` attribute
    ('0' or '1').
    """
    xml = (
        '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        '  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="{relative}">'
        f'{source_filename}</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )
    with open(vrt_path, 'w') as f:
        f.write(xml)


@pytest.fixture(autouse=True)
def _clear_allowlist_env(monkeypatch):
    """Make sure no stray XRSPATIAL_VRT_ALLOWED_ROOTS leaks across tests."""
    monkeypatch.delenv('XRSPATIAL_VRT_ALLOWED_ROOTS', raising=False)


# ---------------------------------------------------------------------------
# Relative-source traversal
# ---------------------------------------------------------------------------


def test_relative_source_with_dotdot_traversal_rejected(tmp_path):
    """A relative source resolving outside ``vrt_dir`` raises ValueError.

    ``parse_vrt`` is the resolution layer; the rejection must happen there
    so the dangerous path never reaches ``read_to_array``.
    """
    vrt_dir = _unique_dir(tmp_path, "trav_rel")
    xml = (
        '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        '      <SourceFilename relativeToVRT="1">'
        '../../../../../etc/passwd</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )
    with pytest.raises(ValueError, match="outside the VRT directory"):
        parse_vrt(xml, vrt_dir)


def test_relative_source_symlink_traversal_rejected(tmp_path):
    """A symlink under ``vrt_dir`` that points outside still gets rejected.

    The check uses ``realpath`` so a symlink-based escape is caught the
    same way ``../`` is.
    """
    vrt_dir = _unique_dir(tmp_path, "trav_sym")
    outside_dir = _unique_dir(tmp_path, "trav_sym_outside")
    outside_target = os.path.join(outside_dir, 'secret.tif')
    _write_minimal_tif(outside_target)

    # Plant a symlink inside vrt_dir that points to the outside file.
    sym = os.path.join(vrt_dir, 'inside.tif')
    os.symlink(outside_target, sym)

    vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
    _build_vrt(vrt_path, 'inside.tif', relative='1')

    with pytest.raises(ValueError, match="outside the VRT directory"):
        _read_vrt_internal(vrt_path)


# ---------------------------------------------------------------------------
# Absolute-source rejection by default
# ---------------------------------------------------------------------------


def test_absolute_source_outside_vrt_dir_rejected(tmp_path):
    """A SourceFilename pointing at an absolute path outside ``vrt_dir``
    is rejected by default."""
    vrt_dir = _unique_dir(tmp_path, "abs_outside")
    outside_dir = _unique_dir(tmp_path, "abs_outside_target")
    outside_tif = os.path.join(outside_dir, 'data.tif')
    _write_minimal_tif(outside_tif)

    vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
    _build_vrt(vrt_path, outside_tif, relative='0')

    with pytest.raises(ValueError, match="outside the VRT directory"):
        _read_vrt_internal(vrt_path)


def test_absolute_source_inside_vrt_dir_ok(tmp_path):
    """An absolute path that happens to resolve under ``vrt_dir`` passes.

    Mirrors the writer's ``relative=False`` round-trip case: the on-disk
    VRT carries the absolute path of a TIFF that still lives next to it.
    """
    vrt_dir = _unique_dir(tmp_path, "abs_inside")
    tif_path = os.path.join(vrt_dir, 'data.tif')
    _write_minimal_tif(tif_path)

    vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
    _build_vrt(vrt_path, tif_path, relative='0')

    arr, _ = _read_vrt_internal(vrt_path)
    assert arr.shape == (4, 4)


# ---------------------------------------------------------------------------
# Allowlist opt-in via XRSPATIAL_VRT_ALLOWED_ROOTS
# ---------------------------------------------------------------------------


def test_absolute_source_allowlisted_root_passes(tmp_path, monkeypatch):
    """Setting XRSPATIAL_VRT_ALLOWED_ROOTS allows cross-directory reads."""
    vrt_dir = _unique_dir(tmp_path, "allow_vrt")
    outside_dir = _unique_dir(tmp_path, "allow_data")
    outside_tif = os.path.join(outside_dir, 'data.tif')
    _write_minimal_tif(outside_tif)

    vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
    _build_vrt(vrt_path, outside_tif, relative='0')

    monkeypatch.setenv('XRSPATIAL_VRT_ALLOWED_ROOTS', outside_dir)
    arr, _ = _read_vrt_internal(vrt_path)
    assert arr.shape == (4, 4)


def test_allowlist_supports_multiple_roots(tmp_path, monkeypatch):
    """A colon-separated list permits sources under any listed root."""
    vrt_dir = _unique_dir(tmp_path, "multi_vrt")
    dir_a = _unique_dir(tmp_path, "multi_a")
    dir_b = _unique_dir(tmp_path, "multi_b")
    tif_b = os.path.join(dir_b, 'data.tif')
    _write_minimal_tif(tif_b)

    vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
    _build_vrt(vrt_path, tif_b, relative='0')

    monkeypatch.setenv(
        'XRSPATIAL_VRT_ALLOWED_ROOTS', os.pathsep.join([dir_a, dir_b]))
    arr, _ = _read_vrt_internal(vrt_path)
    assert arr.shape == (4, 4)


def test_allowlist_does_not_cover_traversal_via_relative_source(
    tmp_path, monkeypatch,
):
    """A relative source that escapes ``vrt_dir`` is rejected even if
    the resolved path happens to land under an allowlisted root.

    Relative paths declare intent to stay inside the VRT directory.
    Honouring that intent prevents an attacker from chaining an
    allowlist entry into a relative-source traversal.
    """
    vrt_dir = _unique_dir(tmp_path, "rel_with_allow")
    outside_dir = _unique_dir(tmp_path, "rel_with_allow_target")
    outside_tif = os.path.join(outside_dir, 'data.tif')
    _write_minimal_tif(outside_tif)

    # Build a relative path from vrt_dir to outside_tif.
    rel = os.path.relpath(outside_tif, vrt_dir)

    vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
    _build_vrt(vrt_path, rel, relative='1')

    monkeypatch.setenv('XRSPATIAL_VRT_ALLOWED_ROOTS', outside_dir)
    with pytest.raises(ValueError, match="outside the VRT directory"):
        _read_vrt_internal(vrt_path)


def test_allowlist_empty_entries_ignored(tmp_path, monkeypatch):
    """Empty entries in the allowlist (from stray colons) are skipped."""
    vrt_dir = _unique_dir(tmp_path, "empty_entry_vrt")
    outside_dir = _unique_dir(tmp_path, "empty_entry_data")
    outside_tif = os.path.join(outside_dir, 'data.tif')
    _write_minimal_tif(outside_tif)

    vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
    _build_vrt(vrt_path, outside_tif, relative='0')

    # Leading/trailing/embedded empty entries should not crash the parser
    # or accidentally grant access to ``/``.
    value = f":{outside_dir}::"
    monkeypatch.setenv('XRSPATIAL_VRT_ALLOWED_ROOTS', value)
    arr, _ = _read_vrt_internal(vrt_path)
    assert arr.shape == (4, 4)


# ---------------------------------------------------------------------------
# Happy-path regression: existing relative-source under vrt_dir still works
# ---------------------------------------------------------------------------


def test_normal_relative_source_under_vrt_dir(tmp_path):
    """A plain relative source under the VRT directory still reads fine."""
    vrt_dir = _unique_dir(tmp_path, "happy")
    tif_path = os.path.join(vrt_dir, 'data.tif')
    _write_minimal_tif(tif_path)

    vrt_path = os.path.join(vrt_dir, 'mosaic.vrt')
    _build_vrt(vrt_path, 'data.tif', relative='1')

    arr, _ = _read_vrt_internal(vrt_path)
    assert arr.shape == (4, 4)


def test_error_message_names_rejected_path(tmp_path):
    """The ValueError mentions the offending path so operators can diagnose."""
    vrt_dir = _unique_dir(tmp_path, "msg_check")
    xml = (
        '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <SimpleSource>\n'
        '      <SourceFilename relativeToVRT="1">'
        '../../etc/shadow</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )
    with pytest.raises(ValueError) as excinfo:
        parse_vrt(xml, vrt_dir)
    msg = str(excinfo.value)
    # The rejected resolved path should appear somewhere in the message.
    assert 'shadow' in msg
    # And the trusted root should also be cited.
    assert os.path.realpath(vrt_dir) in msg
