"""External `.tif.ovr` sidecar reads across the local and remote paths.

Each ``# Section:`` block below covers one sidecar scenario; helpers
that would otherwise collide across sections are namespaced per
section.

The sidecar fixtures live in ``../golden_corpus/fixtures``. Because this
file sits one directory deeper than the top-level tests location, the
fixture paths resolve against ``Path(__file__).resolve().parents[1]``
(the ``tests`` directory) rather than ``parent``.

HTTP / loopback-bind sections carry the shared ``@requires_loopback``
marker from ``.._helpers.markers``. Hermetic sections (fsspec
``memory://``, byte surgery, in-process parsing) run everywhere.
"""
from __future__ import annotations

import io
import os
import pathlib
import shutil
import struct
import uuid

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import _read_geo_info, open_geotiff
from xrspatial.geotiff._geotags import _ifd_has_georef_payload
from xrspatial.geotiff._header import parse_all_ifds, parse_header
from xrspatial.geotiff._reader import CloudSizeLimitError, read_to_array
from xrspatial.geotiff._sidecar import find_sidecar, load_sidecar
from xrspatial.geotiff._writers.eager import to_geotiff

from .._helpers.markers import requires_gpu, requires_loopback

# The bundled sidecar fixture, resolved from the ``tests`` directory
# (one level up from ``integration/``).
_FIXTURE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "golden_corpus"
    / "fixtures"
    / "overview_external_ovr_uint16.tif"
)


def _fixture_or_skip():
    if not _FIXTURE.exists():
        pytest.skip("sidecar fixture not present")
    if not (_FIXTURE.parent / "overview_external_ovr_uint16.tif.ovr").exists():
        pytest.skip("sidecar .ovr file not present")
    return _FIXTURE


# ============================================================================
# Section: sidecar_ovr
#
# External `.tif.ovr` sidecar overview reader.
#
# Before the fix, opening a GDAL/rasterio file whose overview pyramid
# lives in a sibling ``.tif.ovr`` worked at the base level but raised an
# ``overview_level out of range`` error for any non-zero level. The
# reader only walked the in-file IFD chain.
# ============================================================================


# ---------------------------------------------------------------------------
# Discovery helper: local file paths only.
# ---------------------------------------------------------------------------
def test_find_sidecar_returns_path_for_local_file_with_sidecar():
    src = str(_fixture_or_skip())
    assert find_sidecar(src) == src + ".ovr"


def test_find_sidecar_returns_none_when_sidecar_missing(tmp_path):
    src = tmp_path / "no_sidecar_2112.tif"
    src.write_bytes(b"\x00")
    assert find_sidecar(str(src)) is None


def test_find_sidecar_returns_none_for_file_like_object():
    assert find_sidecar(io.BytesIO(b"")) is None


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/path.tif",
        "gs://bucket/path.tif",
        "az://container/path.tif",
        "http://example.com/x.tif",
        "https://example.com/x.tif",
        "memory://buffer.tif",
    ],
)
def test_find_sidecar_returns_none_for_remote_uri(uri):
    assert find_sidecar(uri) is None


# ---------------------------------------------------------------------------
# Sidecar parser: returns a usable IFD list.
# ---------------------------------------------------------------------------
def test_load_sidecar_returns_two_ifds():
    src = _fixture_or_skip()
    sidecar = load_sidecar(str(src) + ".ovr")
    try:
        # The bundled fixture was written with two overview factors (2, 4).
        assert len(sidecar.ifds) == 2
        # First sidecar IFD is the 32x32 level, second is 16x16.
        assert (sidecar.ifds[0].width, sidecar.ifds[0].height) == (32, 32)
        assert (sidecar.ifds[1].width, sidecar.ifds[1].height) == (16, 16)
    finally:
        sidecar.data.close()


# ---------------------------------------------------------------------------
# End-to-end: each sidecar level reads through open_geotiff.
# ---------------------------------------------------------------------------
def test_open_geotiff_base_level_unchanged():
    src = _fixture_or_skip()
    da = open_geotiff(str(src))
    assert da.shape == (64, 64)
    assert da.dtype == np.uint16


def test_open_geotiff_sidecar_level_1():
    src = _fixture_or_skip()
    da = open_geotiff(str(src), overview_level=1)
    assert da.shape == (32, 32)
    assert da.dtype == np.uint16


def test_open_geotiff_sidecar_level_2():
    src = _fixture_or_skip()
    da = open_geotiff(str(src), overview_level=2)
    assert da.shape == (16, 16)
    assert da.dtype == np.uint16


def test_open_geotiff_out_of_range_after_sidecar_appended():
    src = _fixture_or_skip()
    with pytest.raises(ValueError, match="overview_level=3 is out of range"):
        open_geotiff(str(src), overview_level=3)


# ---------------------------------------------------------------------------
# Byte parity with rasterio: the production reference for sidecar overviews.
# ---------------------------------------------------------------------------
def test_sidecar_level_1_matches_rasterio():
    rasterio = pytest.importorskip("rasterio")
    src = _fixture_or_skip()
    with rasterio.open(str(src)) as ds:
        factor = ds.overviews(1)[0]
        out_shape = (ds.height // factor, ds.width // factor)
        rio_arr = ds.read(1, out_shape=out_shape)
    xr_arr = open_geotiff(str(src), overview_level=1).values
    np.testing.assert_array_equal(xr_arr, rio_arr)


def test_sidecar_level_2_matches_rasterio():
    rasterio = pytest.importorskip("rasterio")
    src = _fixture_or_skip()
    with rasterio.open(str(src)) as ds:
        factor = ds.overviews(1)[1]
        out_shape = (ds.height // factor, ds.width // factor)
        rio_arr = ds.read(1, out_shape=out_shape)
    xr_arr = open_geotiff(str(src), overview_level=2).values
    np.testing.assert_array_equal(xr_arr, rio_arr)


# ---------------------------------------------------------------------------
# Sidecar reaches the reader entry point and the metadata-only helper.
# ---------------------------------------------------------------------------
def test_read_to_array_sidecar_level_1(tmp_path):
    src = _fixture_or_skip()
    arr, geo = read_to_array(str(src), overview_level=1)
    assert arr.shape == (32, 32)
    assert arr.dtype == np.uint16


def test_metadata_only_includes_sidecar_levels(tmp_path):
    src = _fixture_or_skip()

    geo, h, w, dtype, _ = _read_geo_info(
        str(src), overview_level=2)
    assert (h, w) == (16, 16)
    # GeoTIFF tags inherit from the base IFD, so the metadata is non-empty.
    assert geo is not None


# ---------------------------------------------------------------------------
# Missing sidecar gracefully falls back to base-only behaviour.
# ---------------------------------------------------------------------------
def test_missing_sidecar_raises_overview_out_of_range(tmp_path):
    src = _fixture_or_skip()
    # Copy only the base file; the sidecar stays behind.
    copy_path = tmp_path / "no_sidecar_2112.tif"
    shutil.copy(src, copy_path)
    # Base level still works.
    assert open_geotiff(str(copy_path)).shape == (64, 64)
    # Asking for an overview level now fails the same way it did before
    # sidecar support was added.
    with pytest.raises(ValueError, match="out of range"):
        open_geotiff(str(copy_path), overview_level=1)


# ---------------------------------------------------------------------------
# GPU eager path: the sidecar levels read through the GPU branch too.
# ---------------------------------------------------------------------------
@requires_gpu
def test_gpu_eager_reads_sidecar_level_1():
    src = _fixture_or_skip()
    cpu = open_geotiff(str(src), overview_level=1)
    gpu = open_geotiff(str(src), overview_level=1, gpu=True)
    assert gpu.shape == cpu.shape
    np.testing.assert_array_equal(gpu.data.get(), cpu.values)


@requires_gpu
def test_gpu_eager_reads_sidecar_level_2():
    src = _fixture_or_skip()
    cpu = open_geotiff(str(src), overview_level=2)
    gpu = open_geotiff(str(src), overview_level=2, gpu=True)
    assert gpu.shape == cpu.shape
    np.testing.assert_array_equal(gpu.data.get(), cpu.values)


@requires_gpu
def test_gpu_eager_base_level_unchanged():
    src = _fixture_or_skip()
    gpu = open_geotiff(str(src), gpu=True)
    assert gpu.shape == (64, 64)


# ---------------------------------------------------------------------------
# HTTP sidecar discovery: tests use a tiny local HTTP server so the
# probe and download paths are exercised without a network round-trip.
# ---------------------------------------------------------------------------
def _start_http_server_2112(directory):
    import http.server
    import socketserver
    import threading

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **kw):
            return  # silence

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

    httpd = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


@requires_loopback
def test_find_sidecar_http_probe_returns_url_when_present(
        tmp_path, monkeypatch):
    # The sidecar probe now routes through ``_HTTPSource``, which
    # rejects loopback hostnames under the SSRF guard. Loopback is the
    # standard local-server pattern in this repo's HTTP tests (see
    # ``golden_corpus/test_http.py``); opt into the escape hatch the
    # production reader exposes.
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = _fixture_or_skip()
    shutil.copy(src, tmp_path / "x.tif")
    shutil.copy(str(src) + ".ovr", tmp_path / "x.tif.ovr")
    httpd, port = _start_http_server_2112(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/x.tif"
        assert find_sidecar(url) == url + ".ovr"
    finally:
        httpd.shutdown()


@requires_loopback
def test_find_sidecar_http_probe_returns_none_when_missing(
        tmp_path, monkeypatch):
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = _fixture_or_skip()
    shutil.copy(src, tmp_path / "x.tif")  # no .ovr copied
    httpd, port = _start_http_server_2112(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/x.tif"
        assert find_sidecar(url) is None
    finally:
        httpd.shutdown()


@requires_loopback
def test_find_sidecar_http_probe_rejects_loopback_without_env_override(
        tmp_path):
    """Without ``XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS=1``, the SSRF
    guard makes a loopback probe silently return ``None`` -- same
    silent-fail-to-base contract the rest of ``find_sidecar`` uses."""
    src = _fixture_or_skip()
    shutil.copy(src, tmp_path / "x.tif")
    shutil.copy(str(src) + ".ovr", tmp_path / "x.tif.ovr")
    httpd, port = _start_http_server_2112(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/x.tif"
        assert find_sidecar(url) is None
    finally:
        httpd.shutdown()


@requires_loopback
def test_load_sidecar_http_returns_ifds(tmp_path, monkeypatch):
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = _fixture_or_skip()
    shutil.copy(str(src) + ".ovr", tmp_path / "x.tif.ovr")
    httpd, port = _start_http_server_2112(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/x.tif.ovr"
        sidecar = load_sidecar(url)
        assert len(sidecar.ifds) == 2
        assert (sidecar.ifds[0].width, sidecar.ifds[0].height) == (32, 32)
        assert (sidecar.ifds[1].width, sidecar.ifds[1].height) == (16, 16)
    finally:
        httpd.shutdown()


def test_find_sidecar_fsspec_probe_returns_uri_when_present(tmp_path):
    pytest.importorskip("fsspec")
    src = _fixture_or_skip()
    shutil.copy(src, tmp_path / "y.tif")
    shutil.copy(str(src) + ".ovr", tmp_path / "y.tif.ovr")
    # file:// is a valid fsspec scheme that uses LocalFileSystem.
    uri = f"file://{tmp_path}/y.tif"
    assert find_sidecar(uri) == uri + ".ovr"


def test_file_like_source_reads_base_without_sidecar():
    src = _fixture_or_skip()
    with open(src, "rb") as f:
        buf = io.BytesIO(f.read())
    da = open_geotiff(buf)
    assert da.shape == (64, 64)


# ============================================================================
# Section: sidecar_own_geokeys
#
# Sidecar IFDs that declare their own georef payload.
#
# The local eager reader and the metadata-only path both swap pixel bytes
# over to the sidecar when the selected IFD lives there. A regression
# that always read georef tags from the base file would parse a
# sidecar's own GeoKeyDirectory / ModelPixelScale / ModelTiepoint /
# ModelTransformation against the wrong buffer.
# ============================================================================


def _make_dataarray_2315(values: np.ndarray, *, origin_x: float,
                         origin_y: float, pixel_w: float, pixel_h: float,
                         epsg: int = 4326) -> xr.DataArray:
    """Build a 2D DataArray whose attrs survive ``to_geotiff``."""
    h, w = values.shape
    xs = origin_x + (np.arange(w) + 0.5) * pixel_w
    ys = origin_y + (np.arange(h) + 0.5) * pixel_h
    da = xr.DataArray(values, dims=("y", "x"), coords={"y": ys, "x": xs})
    da.attrs["crs"] = epsg
    return da


def _write_pair(tmp_path, *, base_geo, sidecar_geo, sidecar_has_geokeys):
    """Write base.tif and (optionally geokey-bearing) sidecar.tif.ovr.

    ``base_geo`` and ``sidecar_geo`` are kwarg dicts forwarded to
    :func:`_make_dataarray_2315`. When ``sidecar_has_geokeys`` is False the
    sidecar file is rewritten to strip its GeoKey / Model* tags so the
    file represents the "GDAL convention" (no geokeys on the sidecar
    IFD). The byte-level strip operates on the parsed IFD entries: we
    rewrite the IFD's entry count and entries array to exclude the
    four georef tag IDs.
    """
    base_arr = np.full((8, 8), 1, dtype=np.uint16)
    sidecar_arr = np.full((4, 4), 2, dtype=np.uint16)

    base_path = tmp_path / "base_2315.tif"
    side_path = tmp_path / "base_2315.tif.ovr"

    base_da = _make_dataarray_2315(base_arr, **base_geo)
    to_geotiff(base_da, str(base_path), tiled=False, compression="none")

    side_da = _make_dataarray_2315(sidecar_arr, **sidecar_geo)
    to_geotiff(side_da, str(side_path), tiled=False, compression="none")

    # Real GDAL ``.tif.ovr`` files mark every IFD as a reduced-res
    # overview (NewSubfileType bit 0 set). ``to_geotiff`` writes a
    # standalone full-res TIFF so the tag is absent; insert one so the
    # sidecar IFD is treated as an overview by the inheritance helper.
    _mark_first_ifd_as_overview_2315(side_path)

    if not sidecar_has_geokeys:
        _strip_georef_tags_2315(side_path)

    return base_path, side_path


def _mark_first_ifd_as_overview_2315(path):
    """Set NewSubfileType=1 (reduced-resolution overview) on the first IFD.

    The tag is added or rewritten so the sidecar IFD looks like a real
    GDAL external overview rather than a freshly written full-res TIFF.
    Inline value, type LONG (3 = uint16 not enough -- the spec uses
    type 4 / uint32).
    """
    raw = bytearray(path.read_bytes())
    assert raw[:2] == b"II", "test fixture assumes little-endian classic TIFF"
    first_ifd_offset = struct.unpack_from("<I", raw, 4)[0]
    n_entries = struct.unpack_from("<H", raw, first_ifd_offset)[0]

    # Look for an existing NewSubfileType (254). If present, rewrite to
    # value=1. Otherwise insert a new entry and shift the trailing
    # entries forward, keeping the entries sorted by tag id.
    NSF = 254
    for i in range(n_entries):
        entry_off = first_ifd_offset + 2 + i * 12
        tag = struct.unpack_from("<H", raw, entry_off)[0]
        if tag == NSF:
            struct.pack_into("<HHII", raw, entry_off, NSF, 4, 1, 1)
            path.write_bytes(bytes(raw))
            return

    # Build the new entry.
    new_entry = struct.pack("<HHII", NSF, 4, 1, 1)

    # The IFD entries are sorted by tag id. 254 sorts before every tag
    # ``to_geotiff`` emits in practice (the smallest the writer uses is
    # 256 / ImageWidth), so the new entry goes at position 0. Assert
    # the invariant so a future writer change that emits a tag <= 254
    # fails this fixture loudly instead of silently producing an
    # out-of-order IFD that the reader could later reject.
    if n_entries > 0:
        first_tag = struct.unpack_from("<H", raw, first_ifd_offset + 2)[0]
        assert first_tag > NSF, (
            f"test fixture invariant: first emitted tag must be > {NSF}, "
            f"got {first_tag}"
        )
    insert_pos = first_ifd_offset + 2

    # next_ifd_off_pos = first_ifd_offset + 2 + n_entries * 12
    # block_end = next_ifd_off_pos + 4 (kept for reference; not used)

    # Slide every byte from ``insert_pos`` to end-of-file forward by 12
    # bytes. This makes room for the new entry and keeps value blocks
    # that lived past the entries array intact at their new offsets.
    raw[insert_pos:insert_pos] = b"\x00" * 12  # carve a 12-byte hole
    # raw is now 12 bytes longer; the hole sits at insert_pos..insert_pos+12.
    # The original tail bytes are at insert_pos+12..end now.

    struct.pack_into("<H", raw, first_ifd_offset, n_entries + 1)
    raw[insert_pos:insert_pos + 12] = new_entry

    # Every absolute byte offset stored in an IFD entry that points
    # at or past ``insert_pos`` is off by -12 now. Bump them.
    _bump_offsets_2315(raw, first_ifd_offset, n_entries + 1, insert_pos, 12)

    # Also bump the stripoffsets / tileoffsets if their value is inline
    # (count==1) but small enough to still point at a now-shifted
    # location. ``_bump_offsets_2315`` already handles the offset-vs-inline
    # case correctly via type_size*count > 4. For count==1 LONG (typ=4,
    # size=4) the value IS the byte offset and total=4 is inline, so
    # the helper above skipped it. Fix it explicitly here.
    _bump_inline_long_offsets_2315(raw, first_ifd_offset, n_entries + 1,
                                   insert_pos, 12)

    path.write_bytes(bytes(raw))


def _bump_inline_long_offsets_2315(raw, ifd_offset, n_entries, threshold,
                                   delta):
    """Bump inline LONG offsets stored in StripOffsets / TileOffsets.

    For these tags the *value* itself is a byte offset into the file.
    When count==1, the value sits inline in the entry's value field
    and ``_bump_offsets_2315`` skips it (because total <= 4). Patch it here.
    """
    OFFSET_TAGS = {273, 324}  # StripOffsets, TileOffsets
    for i in range(n_entries):
        entry_off = ifd_offset + 2 + i * 12
        tag = struct.unpack_from("<H", raw, entry_off)[0]
        if tag not in OFFSET_TAGS:
            continue
        type_id = struct.unpack_from("<H", raw, entry_off + 2)[0]
        count = struct.unpack_from("<I", raw, entry_off + 4)[0]
        if type_id == 4 and count == 1:
            val = struct.unpack_from("<I", raw, entry_off + 8)[0]
            if val >= threshold:
                struct.pack_into("<I", raw, entry_off + 8, val + delta)
        elif type_id == 3 and count == 1:
            val = struct.unpack_from("<H", raw, entry_off + 8)[0]
            if val >= threshold:
                struct.pack_into("<H", raw, entry_off + 8, val + delta)


def _bump_offsets_2315(raw, ifd_offset, n_entries, threshold, delta):
    """Add ``delta`` to every byte offset stored in the IFD >= threshold.

    Type sizes per TIFF 6.0: BYTE/SBYTE/ASCII/UNDEFINED=1, SHORT/SSHORT=2,
    LONG/SLONG/FLOAT=4, RATIONAL/SRATIONAL/DOUBLE=8.
    """
    TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1,
                 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
    for i in range(n_entries):
        entry_off = ifd_offset + 2 + i * 12
        type_id = struct.unpack_from("<H", raw, entry_off + 2)[0]
        count = struct.unpack_from("<I", raw, entry_off + 4)[0]
        ts = TYPE_SIZE.get(type_id, 1)
        total = ts * count
        if total <= 4:
            continue  # inline value, no offset to bump
        off = struct.unpack_from("<I", raw, entry_off + 8)[0]
        if off >= threshold:
            struct.pack_into("<I", raw, entry_off + 8, off + delta)

    # Bump the next-IFD offset if needed.
    next_ifd_off_pos = ifd_offset + 2 + n_entries * 12
    next_off = struct.unpack_from("<I", raw, next_ifd_off_pos)[0]
    if next_off >= threshold and next_off != 0:
        struct.pack_into("<I", raw, next_ifd_off_pos, next_off + delta)


def _strip_georef_tags_2315(path):
    """Rewrite ``path`` so its first IFD carries no georef tags.

    The four georef-bearing tag IDs (33550, 33922, 34264, 34735) are
    removed from the IFD entries array. The IFD's entry count field
    is decremented accordingly. The trailing value blocks the removed
    tags pointed at are left in place: bounds checking happens at the
    entry level, and any orphan bytes after the IFD are ignored by
    every IFD walker in this package.

    Little-endian classic TIFF assumed (matches what ``to_geotiff``
    writes by default).
    """
    raw = bytearray(path.read_bytes())
    if raw[:2] != b"II":
        raise AssertionError(
            "test fixture: expected little-endian TIFF from to_geotiff()")
    assert struct.unpack_from("<H", raw, 2)[0] == 42, "classic TIFF expected"
    first_ifd_offset = struct.unpack_from("<I", raw, 4)[0]

    n_entries = struct.unpack_from("<H", raw, first_ifd_offset)[0]
    GEOREF_TAGS = {33550, 33922, 34264, 34735}
    keep = []
    for i in range(n_entries):
        entry_off = first_ifd_offset + 2 + i * 12
        tag = struct.unpack_from("<H", raw, entry_off)[0]
        if tag in GEOREF_TAGS:
            continue
        keep.append(raw[entry_off:entry_off + 12])

    # Rewrite entry count + entries.  The post-entries 4-byte
    # ``next-IFD-offset`` field needs to stay where the truncated
    # entries array ends.
    next_ifd_off_pos = first_ifd_offset + 2 + n_entries * 12
    next_ifd_off = raw[next_ifd_off_pos:next_ifd_off_pos + 4]

    new_n = len(keep)
    struct.pack_into("<H", raw, first_ifd_offset, new_n)
    write_pos = first_ifd_offset + 2
    for entry in keep:
        raw[write_pos:write_pos + 12] = entry
        write_pos += 12
    raw[write_pos:write_pos + 4] = next_ifd_off
    # Zero out the trailing bytes from the original entries array so a
    # downstream walker that scans past ``next_ifd_off`` does not pick
    # up stale entry data.
    tail_start = write_pos + 4
    tail_end = next_ifd_off_pos + 4
    if tail_end > tail_start:
        raw[tail_start:tail_end] = b"\x00" * (tail_end - tail_start)

    path.write_bytes(bytes(raw))


# ---------------------------------------------------------------------------
# Helper unit test: the georef-payload detector recognises the four tags.
# ---------------------------------------------------------------------------
def test_ifd_has_georef_payload_true_for_pixel_scale(tmp_path):
    p = tmp_path / "p_2315.tif"
    da = _make_dataarray_2315(
        np.zeros((4, 4), dtype=np.uint16),
        origin_x=0.0, origin_y=0.0, pixel_w=1.0, pixel_h=-1.0,
    )
    to_geotiff(da, str(p), tiled=False, compression="none")
    data = p.read_bytes()
    hdr = parse_header(data)
    ifds = parse_all_ifds(data, hdr)
    assert _ifd_has_georef_payload(ifds[0])


def test_ifd_has_georef_payload_false_after_strip(tmp_path):
    p = tmp_path / "p_2315_strip.tif"
    da = _make_dataarray_2315(
        np.zeros((4, 4), dtype=np.uint16),
        origin_x=0.0, origin_y=0.0, pixel_w=1.0, pixel_h=-1.0,
    )
    to_geotiff(da, str(p), tiled=False, compression="none")
    _strip_georef_tags_2315(p)
    data = p.read_bytes()
    hdr = parse_header(data)
    ifds = parse_all_ifds(data, hdr)
    assert not _ifd_has_georef_payload(ifds[0])


# ---------------------------------------------------------------------------
# Sidecar carries its own geokeys -> the sidecar's georef wins.
# ---------------------------------------------------------------------------
def test_sidecar_with_own_geokeys_wins_eager(tmp_path):
    base_geo = dict(origin_x=100.0, origin_y=200.0,
                    pixel_w=10.0, pixel_h=-10.0, epsg=4326)
    side_geo = dict(origin_x=500.0, origin_y=800.0,
                    pixel_w=2.5, pixel_h=-2.5, epsg=3857)

    base, side = _write_pair(tmp_path,
                             base_geo=base_geo, sidecar_geo=side_geo,
                             sidecar_has_geokeys=True)

    da = open_geotiff(str(base), overview_level=1)
    # Sidecar transform wins.  Rasterio-style 6-tuple:
    # ``(pixel_width, 0, origin_x, 0, pixel_height, origin_y)``.
    t = da.attrs["transform"]
    assert t[0] == pytest.approx(2.5)
    assert t[2] == pytest.approx(500.0)
    assert t[4] == pytest.approx(-2.5)
    assert t[5] == pytest.approx(800.0)
    # Sidecar EPSG wins too (sidecar CRS is 3857, base is 4326).
    assert da.attrs.get("crs") == 3857


def test_sidecar_with_own_geokeys_wins_metadata_only(tmp_path):
    base_geo = dict(origin_x=100.0, origin_y=200.0,
                    pixel_w=10.0, pixel_h=-10.0, epsg=4326)
    side_geo = dict(origin_x=500.0, origin_y=800.0,
                    pixel_w=2.5, pixel_h=-2.5, epsg=3857)

    base, _ = _write_pair(tmp_path,
                          base_geo=base_geo, sidecar_geo=side_geo,
                          sidecar_has_geokeys=True)

    geo_info, h, w, _, _ = _read_geo_info(str(base), overview_level=1)
    assert (h, w) == (4, 4)
    assert geo_info.transform.origin_x == pytest.approx(500.0)
    assert geo_info.transform.origin_y == pytest.approx(800.0)
    assert geo_info.transform.pixel_width == pytest.approx(2.5)
    assert geo_info.transform.pixel_height == pytest.approx(-2.5)
    assert geo_info.crs_epsg == 3857


# ---------------------------------------------------------------------------
# Sidecar has no geokeys -> today's behavior is preserved (inherit base).
# ---------------------------------------------------------------------------
def test_sidecar_without_geokeys_inherits_from_base_eager(tmp_path):
    base_geo = dict(origin_x=100.0, origin_y=200.0,
                    pixel_w=10.0, pixel_h=-10.0, epsg=4326)
    # The sidecar's georef bytes will be stripped after write, so the
    # numbers in ``side_geo`` are irrelevant to the assertion.
    side_geo = dict(origin_x=500.0, origin_y=800.0,
                    pixel_w=2.5, pixel_h=-2.5, epsg=3857)

    base, _ = _write_pair(tmp_path,
                          base_geo=base_geo, sidecar_geo=side_geo,
                          sidecar_has_geokeys=False)

    da = open_geotiff(str(base), overview_level=1)
    # The sidecar IFD has no georef payload; the inheritance helper
    # pulls the transform from the base IFD and rescales for the
    # overview's reduction factor. Origin stays unchanged
    # (PixelIsArea), pixel size scales by 8/4 = 2.
    t = da.attrs["transform"]
    assert t[0] == pytest.approx(20.0)
    assert t[2] == pytest.approx(100.0)
    assert t[4] == pytest.approx(-20.0)
    assert t[5] == pytest.approx(200.0)
    assert da.attrs.get("crs") == 4326


def test_sidecar_without_geokeys_inherits_from_base_metadata_only(tmp_path):
    base_geo = dict(origin_x=100.0, origin_y=200.0,
                    pixel_w=10.0, pixel_h=-10.0, epsg=4326)
    side_geo = dict(origin_x=500.0, origin_y=800.0,
                    pixel_w=2.5, pixel_h=-2.5, epsg=3857)

    base, _ = _write_pair(tmp_path,
                          base_geo=base_geo, sidecar_geo=side_geo,
                          sidecar_has_geokeys=False)

    geo_info, h, w, _, _ = _read_geo_info(str(base), overview_level=1)
    assert (h, w) == (4, 4)
    # Base-inherited transform with rescaled pixel size for the 8->4
    # reduction factor.
    assert geo_info.transform.origin_x == pytest.approx(100.0)
    assert geo_info.transform.origin_y == pytest.approx(200.0)
    assert geo_info.transform.pixel_width == pytest.approx(20.0)
    assert geo_info.transform.pixel_height == pytest.approx(-20.0)
    assert geo_info.crs_epsg == 4326


# ============================================================================
# Section: sidecar_max_cloud_bytes
#
# Sidecar download honours ``max_cloud_bytes``.
#
# Before the fix, ``load_sidecar`` downloaded the sibling ``.tif.ovr``
# over HTTP / fsspec with no byte cap, bypassing the base-file budget so
# a hostile server serving a tiny base plus a multi-GB sidecar could OOM
# the reader on any ``overview_level >= 1``.
# ============================================================================


def _start_http_server_2121(directory):
    """Spin up a loopback HTTP server serving *directory*. Returns (httpd, port)."""
    import http.server
    import socketserver
    import threading

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a, **kw):
            return  # silence

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(directory), **kw)

    httpd = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


# ---------------------------------------------------------------------------
# Local file: mmap path is not subject to the budget. The cap argument is
# accepted but ignored because no allocation happens.
# ---------------------------------------------------------------------------
def test_local_sidecar_ignores_max_cloud_bytes():
    src = _fixture_or_skip()
    sidecar_path = str(src) + ".ovr"
    # Pass a tiny budget: local mmap is not subject to it.
    sidecar = load_sidecar(sidecar_path, max_cloud_bytes=1)
    try:
        assert len(sidecar.ifds) == 2
    finally:
        closer = getattr(sidecar.data, "close", None)
        if closer is not None:
            closer()


# ---------------------------------------------------------------------------
# fsspec: declared size is checked against the budget before any read.
# ---------------------------------------------------------------------------
def test_fsspec_sidecar_rejects_when_exceeds_max_cloud_bytes(tmp_path):
    pytest.importorskip("fsspec")
    src = _fixture_or_skip()
    sidecar_src = str(src) + ".ovr"
    sidecar_size = pathlib.Path(sidecar_src).stat().st_size
    sidecar_copy = tmp_path / "x_2121.tif.ovr"
    shutil.copy(sidecar_src, sidecar_copy)

    uri = f"file://{sidecar_copy}"
    # Set the budget below the actual sidecar size: the size guard fires.
    with pytest.raises(CloudSizeLimitError) as exc:
        load_sidecar(uri, max_cloud_bytes=sidecar_size - 1)
    assert "exceeds max_cloud_bytes" in str(exc.value)
    assert str(sidecar_size) in str(exc.value).replace(",", "")


def test_fsspec_sidecar_succeeds_when_under_max_cloud_bytes(tmp_path):
    pytest.importorskip("fsspec")
    src = _fixture_or_skip()
    sidecar_src = str(src) + ".ovr"
    sidecar_size = pathlib.Path(sidecar_src).stat().st_size
    sidecar_copy = tmp_path / "y_2121.tif.ovr"
    shutil.copy(sidecar_src, sidecar_copy)

    uri = f"file://{sidecar_copy}"
    # Budget comfortably above the sidecar size: download proceeds.
    sidecar = load_sidecar(uri, max_cloud_bytes=sidecar_size * 10)
    assert len(sidecar.ifds) == 2


def test_fsspec_sidecar_max_cloud_bytes_none_is_unbounded(tmp_path):
    pytest.importorskip("fsspec")
    src = _fixture_or_skip()
    sidecar_src = str(src) + ".ovr"
    sidecar_copy = tmp_path / "z_2121.tif.ovr"
    shutil.copy(sidecar_src, sidecar_copy)

    uri = f"file://{sidecar_copy}"
    # max_cloud_bytes=None preserves the legacy unbounded behaviour: no
    # fsspec.size() call, no cap, sidecar reads through.
    sidecar = load_sidecar(uri, max_cloud_bytes=None)
    assert len(sidecar.ifds) == 2


# ---------------------------------------------------------------------------
# HTTP: load_sidecar translates the underlying budget OSError into
# CloudSizeLimitError so both cloud transports raise the same type.
# ---------------------------------------------------------------------------
@requires_loopback
def test_http_sidecar_rejects_when_exceeds_max_cloud_bytes(
        tmp_path, monkeypatch):
    """Streaming download aborts when the body exceeds ``max_cloud_bytes``."""
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = _fixture_or_skip()
    sidecar_src = str(src) + ".ovr"
    sidecar_size = pathlib.Path(sidecar_src).stat().st_size
    shutil.copy(sidecar_src, tmp_path / "h_2121.tif.ovr")
    httpd, port = _start_http_server_2121(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/h_2121.tif.ovr"
        # Cap below the body size. Either the Content-Length pre-check
        # or the streaming overshoot probe inside ``_HTTPSource.read_all``
        # raises ``OSError``; ``load_sidecar`` re-raises it as
        # ``CloudSizeLimitError`` so callers see a single exception
        # type regardless of transport.
        with pytest.raises(CloudSizeLimitError) as exc:
            load_sidecar(url, max_cloud_bytes=sidecar_size - 1)
        assert "max_cloud_bytes" in str(exc.value)
        # The original OSError is preserved as the cause so debuggers
        # can still see the Content-Length / streaming detail.
        assert isinstance(exc.value.__cause__, OSError)
    finally:
        httpd.shutdown()


@requires_loopback
def test_http_sidecar_succeeds_when_under_max_cloud_bytes(
        tmp_path, monkeypatch):
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = _fixture_or_skip()
    sidecar_src = str(src) + ".ovr"
    sidecar_size = pathlib.Path(sidecar_src).stat().st_size
    shutil.copy(sidecar_src, tmp_path / "ok_2121.tif.ovr")
    httpd, port = _start_http_server_2121(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/ok_2121.tif.ovr"
        sidecar = load_sidecar(url, max_cloud_bytes=sidecar_size * 10)
        assert len(sidecar.ifds) == 2
    finally:
        httpd.shutdown()


@requires_loopback
def test_http_sidecar_max_cloud_bytes_none_is_unbounded(
        tmp_path, monkeypatch):
    """``max_cloud_bytes=None`` keeps the sidecar read unbounded."""
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = _fixture_or_skip()
    sidecar_src = str(src) + ".ovr"
    shutil.copy(sidecar_src, tmp_path / "u_2121.tif.ovr")
    httpd, port = _start_http_server_2121(tmp_path)
    try:
        url = f"http://127.0.0.1:{port}/u_2121.tif.ovr"
        sidecar = load_sidecar(url, max_cloud_bytes=None)
        assert len(sidecar.ifds) == 2
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------
# End-to-end: read_to_array's cloud_budget propagates into load_sidecar.
# A user-provided ``max_cloud_bytes`` set on the base file flows down to
# the sidecar fetch instead of being silently dropped.
# ---------------------------------------------------------------------------
def test_read_to_array_propagates_max_cloud_bytes_to_sidecar(
        tmp_path, monkeypatch):
    """A fsspec ``file://`` source with a tight budget rejects the sidecar.

    A regression where ``read_to_array(..., max_cloud_bytes=N)`` enforced
    ``N`` only on the base file would silently bypass it for the sidecar
    fetch. This test inflates the sidecar (a normal ``.ovr`` is small
    relative to its base) so the cloud budget that admits the base file
    rejects the sidecar; ``cloud_budget`` routes into ``load_sidecar`` so
    the sidecar fetch trips the same guard.
    """
    pytest.importorskip("fsspec")
    src = _fixture_or_skip()
    base_copy = tmp_path / "rt_2121.tif"
    sidecar_copy = tmp_path / "rt_2121.tif.ovr"
    shutil.copy(src, base_copy)
    shutil.copy(str(src) + ".ovr", sidecar_copy)
    # Pad the sidecar with trailing zeros so its on-disk size exceeds the
    # base. The TIFF parser ignores trailing bytes past the last IFD chain
    # entry, so the inflated sidecar still opens cleanly when the cap
    # admits it. The base file is left untouched.
    base_size = base_copy.stat().st_size
    target_sidecar_size = base_size + 4096
    with sidecar_copy.open("ab") as f:
        f.write(b"\x00" * (target_sidecar_size - sidecar_copy.stat().st_size))
    sidecar_size = sidecar_copy.stat().st_size
    assert sidecar_size > base_size

    # Budget that admits the base file (base_size <= budget) and rejects
    # the sidecar (sidecar_size > budget).
    budget = base_size + 1
    assert base_size <= budget < sidecar_size

    uri = f"file://{base_copy}"
    with pytest.raises(CloudSizeLimitError):
        read_to_array(uri, overview_level=1, max_cloud_bytes=budget)


# ---------------------------------------------------------------------------
# Env-var precedence: XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES caps the sidecar
# when the caller omits the kwarg, mirroring base-file semantics.
# ---------------------------------------------------------------------------
def test_env_var_propagates_to_sidecar(tmp_path, monkeypatch):
    """``XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES`` bounds the sidecar fetch.

    ``read_to_array`` resolves the budget once via
    ``_resolve_max_cloud_bytes`` and passes it into ``load_sidecar``.
    When the caller does not pass the kwarg, the env var should still
    cap the sidecar -- the kwarg path and the env-var path must agree.
    """
    pytest.importorskip("fsspec")
    src = _fixture_or_skip()
    base_copy = tmp_path / "env_2121.tif"
    sidecar_copy = tmp_path / "env_2121.tif.ovr"
    shutil.copy(src, base_copy)
    shutil.copy(str(src) + ".ovr", sidecar_copy)
    base_size = base_copy.stat().st_size
    target_sidecar_size = base_size + 4096
    with sidecar_copy.open("ab") as f:
        f.write(b"\x00" * (target_sidecar_size - sidecar_copy.stat().st_size))
    sidecar_size = sidecar_copy.stat().st_size
    assert sidecar_size > base_size

    # Env-var budget admits the base file but rejects the sidecar. The
    # caller passes no ``max_cloud_bytes``, so the resolver picks up the
    # env var and threads it through to ``load_sidecar``.
    budget = base_size + 1
    assert base_size <= budget < sidecar_size
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_MAX_CLOUD_BYTES", str(budget))

    uri = f"file://{base_copy}"
    with pytest.raises(CloudSizeLimitError):
        read_to_array(uri, overview_level=1)


# ============================================================================
# Section: sidecar_bad_does_not_break_base
#
# Corrupt ``.ovr`` sidecar must not break a base read.
#
# A stale, truncated, or malformed sibling ``.ovr`` file should not take
# the stable base read down: a base read survives garbage sidecar bytes
# (emitting a ``RuntimeWarning``), while an explicit external-overview
# request surfaces the parse error and ``CloudSizeLimitError`` always
# propagates.
# ============================================================================


def _make_base_2416(tmp_path) -> tuple:
    """Write a 4x4 uint16 base TIFF and return (path, expected array)."""
    arr = np.arange(16, dtype=np.uint16).reshape(4, 4)
    xs = np.arange(4) + 0.5
    ys = np.arange(4) + 0.5
    da = xr.DataArray(arr, dims=("y", "x"), coords={"y": ys, "x": xs})
    da.attrs["crs"] = 4326
    path = tmp_path / "base_2416.tif"
    to_geotiff(da, str(path), tiled=False, compression="none")
    return path, arr


def _make_corrupt_sidecar_2416(base_path, payload: bytes) -> None:
    side = str(base_path) + ".ovr"
    with open(side, "wb") as f:
        f.write(payload)


# ---------------------------------------------------------------------------
# Eager CPU path: base read survives a broken sidecar.
# ---------------------------------------------------------------------------
def test_open_geotiff_base_read_survives_unreadable_sidecar(tmp_path):
    path, expected = _make_base_2416(tmp_path)
    _make_corrupt_sidecar_2416(path, b"not a tiff")

    with pytest.warns(RuntimeWarning, match="Ignoring unreadable sidecar"):
        da = open_geotiff(str(path))

    np.testing.assert_array_equal(da.values, expected)


def test_open_geotiff_overview_level_zero_survives_unreadable_sidecar(tmp_path):
    """``overview_level=0`` is also a base read; the sidecar is not needed."""
    path, expected = _make_base_2416(tmp_path)
    _make_corrupt_sidecar_2416(path, b"\x00\x00\x00\x00\x00\x00\x00\x00")

    with pytest.warns(RuntimeWarning, match="Ignoring unreadable sidecar"):
        da = open_geotiff(str(path), overview_level=0)

    np.testing.assert_array_equal(da.values, expected)


def test_read_to_array_base_read_survives_unreadable_sidecar(tmp_path):
    """The lower-level ``read_to_array`` entry point honours the same rule."""
    path, expected = _make_base_2416(tmp_path)
    _make_corrupt_sidecar_2416(path, b"this is not even close to a TIFF header")

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
    path, expected = _make_base_2416(tmp_path)
    _make_corrupt_sidecar_2416(path, payload)

    with pytest.warns(RuntimeWarning, match="Ignoring unreadable sidecar"):
        da = open_geotiff(str(path))

    np.testing.assert_array_equal(da.values, expected)


# ---------------------------------------------------------------------------
# Explicit external-overview requests surface the underlying parse error.
#
# The release contract (``reader.sidecar_ovr`` row in
# ``docs/source/reference/geotiff_release_contract.md``) and the
# silent-fallback warning text both promise that ``overview_level >= 1``
# exposes the actual sidecar parse failure rather than a generic "out of
# range" from ``select_overview_ifd``. Issue #2484.
# ---------------------------------------------------------------------------
def test_open_geotiff_requesting_sidecar_level_surfaces_parse_error(tmp_path):
    """``overview_level=1`` against a corrupt sidecar must surface the
    sidecar parse failure, not a misleading "out of range" raised after
    silent fallback to base-only IFDs (#2484)."""
    path, _ = _make_base_2416(tmp_path)
    _make_corrupt_sidecar_2416(path, b"not a tiff")

    # No silent fallback warning -- the explicit level request short-
    # circuits the warn-and-fall-through branch and raises directly.
    with pytest.raises(ValueError, match="external sidecar") as excinfo:
        open_geotiff(str(path), overview_level=1)

    # The underlying parse exception is chained via ``__cause__`` so the
    # traceback shows what actually went wrong. The outer message must
    # not be the generic "overview_level out of range" string that the
    # pre-fix behaviour produced via ``select_overview_ifd``.
    assert "out of range" not in str(excinfo.value)
    assert excinfo.value.__cause__ is not None
    cause_blob = (
        f"{type(excinfo.value.__cause__).__name__}: "
        f"{excinfo.value.__cause__}"
    )
    # The chained cause type / message also surface on the outer message
    # so callers reading only ``str(exc)`` still see the real reason.
    assert cause_blob in str(excinfo.value)


def test_open_geotiff_overview_level_two_surfaces_parse_error(tmp_path):
    """Any level the base file alone cannot serve triggers the
    surface-the-cause path. With ``_make_base_2416`` writing a single
    IFD, that includes level 2 -- the sidecar would have to supply it,
    so the corrupt-sidecar branch surfaces the parse failure rather
    than letting ``select_overview_ifd`` raise a generic "out of
    range"."""
    path, _ = _make_base_2416(tmp_path)
    _make_corrupt_sidecar_2416(path, b"not a tiff")

    with pytest.raises(ValueError, match="overview_level=2") as excinfo:
        open_geotiff(str(path), overview_level=2)

    assert excinfo.value.__cause__ is not None


def test_open_geotiff_internal_overview_level_silent_on_bad_sidecar(
    tmp_path,
):
    """A TIFF carrying its own internal overview IFDs can satisfy
    ``overview_level=1`` from the base file alone. A sibling corrupt
    sidecar must not turn that valid read into a raise (#2484
    follow-up): the warn-and-fall-back branch handles it, the request
    is reachable from the base IFD chain, and the user gets the
    overview they asked for."""
    # Write a base TIFF with one internal overview level so the IFD
    # chain has length 2 (level 0 + level 1) before any sidecar.
    arr = np.arange(64, dtype=np.uint16).reshape(8, 8)
    xs = np.arange(8) + 0.5
    ys = np.arange(8) + 0.5
    da = xr.DataArray(arr, dims=("y", "x"), coords={"y": ys, "x": xs})
    da.attrs["crs"] = 4326
    path = tmp_path / "internal_ovr_2484.tif"
    to_geotiff(da, str(path), cog=True, compression="none",
               overview_levels=[2])

    _make_corrupt_sidecar_2416(path, b"not a tiff")

    # Level 1 is the internal overview; reachable without the sidecar.
    # Expect a warning (sidecar still tried + fell back), no raise.
    with pytest.warns(RuntimeWarning, match="Ignoring unreadable sidecar"):
        da_lvl1 = open_geotiff(str(path), overview_level=1)
    # Internal overview is a 4x4 reduction of the 8x8 base.
    assert da_lvl1.shape == (4, 4)

    # Level 2 is beyond the base IFD chain -- now the sidecar is
    # required, so the parse error surfaces.
    with pytest.raises(ValueError, match="external sidecar") as excinfo:
        open_geotiff(str(path), overview_level=2)
    assert excinfo.value.__cause__ is not None


def test_open_geotiff_overview_level_none_silently_falls_back(tmp_path):
    """Contrast case: implicit / auto overview discovery
    (``overview_level=None``) keeps the documented silent-fallback
    behaviour. A corrupt sidecar must not break the base read."""
    path, expected = _make_base_2416(tmp_path)
    _make_corrupt_sidecar_2416(path, b"not a tiff")

    with pytest.warns(RuntimeWarning, match="Ignoring unreadable sidecar"):
        da = open_geotiff(str(path), overview_level=None)

    np.testing.assert_array_equal(da.values, expected)


def test_open_geotiff_overview_level_zero_silently_falls_back(tmp_path):
    """``overview_level=0`` is the base IFD, not a sidecar request, so
    it follows the same silent-fallback rule as ``None``."""
    path, expected = _make_base_2416(tmp_path)
    _make_corrupt_sidecar_2416(path, b"not a tiff")

    with pytest.warns(RuntimeWarning, match="Ignoring unreadable sidecar"):
        da = open_geotiff(str(path), overview_level=0)

    np.testing.assert_array_equal(da.values, expected)


# ---------------------------------------------------------------------------
# CloudSizeLimitError still propagates: byte budget is a caller contract.
# ---------------------------------------------------------------------------
def test_cloud_size_limit_error_from_sidecar_is_not_silenced(tmp_path, monkeypatch):
    from xrspatial.geotiff import _sidecar as _sidecar_mod

    path, _ = _make_base_2416(tmp_path)
    _make_corrupt_sidecar_2416(path, b"\x00")  # contents do not matter; we stub

    def _budget_breach(_path, *, max_cloud_bytes=None):
        raise CloudSizeLimitError("sidecar too big for budget")

    monkeypatch.setattr(_sidecar_mod, "load_sidecar", _budget_breach)

    with pytest.raises(CloudSizeLimitError):
        open_geotiff(str(path))


# ---------------------------------------------------------------------------
# Metadata-only path (used by the dask graph builder for local files).
# ---------------------------------------------------------------------------
def test_read_geo_info_base_survives_unreadable_sidecar(tmp_path):
    path, _ = _make_base_2416(tmp_path)
    _make_corrupt_sidecar_2416(path, b"not a tiff")

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
    from xrspatial.geotiff import _sidecar as _sidecar_mod

    path, _ = _make_base_2416(tmp_path)
    _make_corrupt_sidecar_2416(path, b"\x00")

    def _budget_breach(_path, *, max_cloud_bytes=None):
        raise CloudSizeLimitError("sidecar too big for budget")

    monkeypatch.setattr(_sidecar_mod, "load_sidecar", _budget_breach)

    with pytest.raises(CloudSizeLimitError):
        _read_geo_info(str(path))


# ---------------------------------------------------------------------------
# GPU eager path: same contract.
# ---------------------------------------------------------------------------
@requires_gpu
def test_open_geotiff_gpu_base_read_survives_unreadable_sidecar(tmp_path):
    path, expected = _make_base_2416(tmp_path)
    _make_corrupt_sidecar_2416(path, b"not a tiff")

    with pytest.warns(RuntimeWarning, match="Ignoring unreadable sidecar"):
        gpu_da = open_geotiff(str(path), gpu=True)

    np.testing.assert_array_equal(gpu_da.data.get(), expected)


def test_read_geo_info_requesting_sidecar_level_surfaces_parse_error(tmp_path):
    """Metadata-only helper mirrors the eager CPU contract: an explicit
    ``overview_level >= 1`` request against a corrupt sidecar must
    surface the underlying parse error (#2484)."""
    path, _ = _make_base_2416(tmp_path)
    _make_corrupt_sidecar_2416(path, b"not a tiff")

    with pytest.raises(ValueError, match="external sidecar") as excinfo:
        _read_geo_info(str(path), overview_level=1)

    assert "out of range" not in str(excinfo.value)
    assert excinfo.value.__cause__ is not None


@requires_gpu
def test_open_geotiff_gpu_requesting_sidecar_level_surfaces_parse_error(
    tmp_path,
):
    """GPU eager path mirrors the CPU contract for explicit external-
    overview requests (#2484)."""
    path, _ = _make_base_2416(tmp_path)
    _make_corrupt_sidecar_2416(path, b"not a tiff")

    with pytest.raises(ValueError, match="external sidecar") as excinfo:
        open_geotiff(str(path), gpu=True, overview_level=1)

    assert excinfo.value.__cause__ is not None


# ============================================================================
# Section: remote_sidecar_chunked
#
# Chunked remote reads honour external `.ovr` sidecars.
#
# Before this fix the dask HTTP / fsspec paths and ``_read_geo_info``
# skipped the sidecar lookup, so ``open_geotiff(remote, chunks=...,
# overview_level=1)`` raised "overview_level out of range" or resolved to
# a different overview than the eager open of the same URL / URI.
# ============================================================================


def _start_range_http_server_2239(payloads: dict[str, bytes]):
    import http.server
    import socketserver
    import threading

    class _Handler(http.server.BaseHTTPRequestHandler):
        # Class-level mapping so ``do_GET`` resolves the requested path
        # without poking at the server. The handler class is built fresh
        # per call so each server instance has its own payloads dict.
        payloads: dict[str, bytes] = {}

        def log_message(self, *a, **kw):
            return  # silence

        def do_GET(self):  # noqa: N802
            path = self.path
            if path not in self.payloads:
                self.send_response(404)
                self.end_headers()
                return
            body = self.payloads[path]
            rng = self.headers.get('Range')
            if rng and rng.startswith('bytes='):
                spec = rng[len('bytes='):]
                start_s, _, end_s = spec.partition('-')
                start = int(start_s)
                end = int(end_s) if end_s else len(body) - 1
                chunk = body[start:end + 1]
                self.send_response(206)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header(
                    'Content-Range',
                    f'bytes {start}-{start + len(chunk) - 1}/{len(body)}',
                )
                self.send_header('Content-Length', str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    handler_cls = type(
        'RangeHandler2239', (_Handler,), {'payloads': payloads}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


@pytest.fixture
def _http_with_sidecar_2239(monkeypatch):
    """Serve the bundled base + sidecar pair with Range support."""
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    src = _fixture_or_skip()
    base_bytes = src.read_bytes()
    side_bytes = (src.parent / (src.name + ".ovr")).read_bytes()
    payloads = {
        "/x.tif": base_bytes,
        "/x.tif.ovr": side_bytes,
    }
    httpd, port = _start_range_http_server_2239(payloads)
    try:
        yield f"http://127.0.0.1:{port}/x.tif"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def _fsspec_memory_with_sidecar_2239():
    """Stage the bundled base + sidecar pair into an fsspec memory store."""
    fsspec = pytest.importorskip("fsspec")
    src = _fixture_or_skip()
    fs = fsspec.filesystem("memory")
    # ``memory://`` is process-global -- pytest-xdist workers and other
    # tests in the same interpreter all share it. Combine pid + uuid in
    # the key so this fixture cannot collide with a parallel test that
    # happens to stage a different payload at the same path.
    key = f"issue2239-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    base_uri = f"memory://{key}/x.tif"
    side_uri = base_uri + ".ovr"
    with open(src, "rb") as f:
        fs.pipe_file(base_uri.replace("memory://", ""), f.read())
    with open(str(src) + ".ovr", "rb") as f:
        fs.pipe_file(side_uri.replace("memory://", ""), f.read())
    try:
        yield base_uri
    finally:
        # Best-effort cleanup so a long test session doesn't accumulate
        # stale objects in the in-process memory store.
        for p in (base_uri, side_uri):
            try:
                fs.rm_file(p.replace("memory://", ""))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Eager-vs-chunked overview parity for fsspec sources. Before the fix
# the chunked open went through ``_read_geo_info``'s fsspec bypass and
# never saw the sidecar -- so ``overview_level=1`` failed with an
# "overview_level out of range" error while the eager open succeeded.
# Level 0 is included so the base-level regression (no sidecar in play)
# also stays pinned.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("overview_level,expected_shape", [
    (0, (64, 64)),
    (1, (32, 32)),
    (2, (16, 16)),
])
def test_fsspec_chunked_open_resolves_sidecar_overview(
        _fsspec_memory_with_sidecar_2239, overview_level, expected_shape):
    uri = _fsspec_memory_with_sidecar_2239
    eager = open_geotiff(uri, overview_level=overview_level)
    chunked = open_geotiff(uri, chunks=16,
                           overview_level=overview_level)
    assert eager.shape == expected_shape
    assert chunked.shape == eager.shape
    np.testing.assert_array_equal(chunked.values, eager.values)


# ---------------------------------------------------------------------------
# Eager-vs-chunked overview parity for HTTP sources. Same contract as the
# fsspec test, but exercises the HTTP discovery + load + tile-fetch path
# in ``_read_cog_http`` and ``_backends/dask.py``.
# ---------------------------------------------------------------------------
@requires_loopback
@pytest.mark.parametrize("overview_level", [1, 2])
def test_http_chunked_open_resolves_sidecar_overview(
        _http_with_sidecar_2239, overview_level):
    url = _http_with_sidecar_2239
    eager = open_geotiff(url, overview_level=overview_level)
    chunked = open_geotiff(url, chunks=16,
                           overview_level=overview_level)
    assert chunked.shape == eager.shape
    np.testing.assert_array_equal(chunked.values, eager.values)


@requires_loopback
def test_http_eager_reads_sidecar_overview(_http_with_sidecar_2239):
    """The eager HTTP path also needs to honour sidecars."""
    url = _http_with_sidecar_2239
    # The bundled fixture's sidecar carries two overview levels at 32x32
    # and 16x16. A reader that only saw the in-file IFD chain would raise
    # "overview_level out of range" over HTTP for both.
    da32 = open_geotiff(url, overview_level=1)
    da16 = open_geotiff(url, overview_level=2)
    assert da32.shape == (32, 32)
    assert da16.shape == (16, 16)


@requires_loopback
def test_http_eager_vs_local_parity(_http_with_sidecar_2239):
    """Eager HTTP reads should match the eager local read byte-for-byte."""
    url = _http_with_sidecar_2239
    src = _fixture_or_skip()
    for level in (0, 1, 2):
        http_da = open_geotiff(url, overview_level=level)
        local_da = open_geotiff(str(src), overview_level=level)
        assert http_da.shape == local_da.shape
        np.testing.assert_array_equal(http_da.values, local_da.values)


# ---------------------------------------------------------------------------
# ``_read_geo_info`` metadata-only path (the function the dask graph
# builder uses for local sources, exercised here against fsspec to pin
# the sidecar branch). Reports the right IFD dimensions and ``geo_info``
# for the sidecar overview without downloading pixels.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("overview_level,expected", [
    (1, (32, 32)),
    (2, (16, 16)),
])
def test_read_geo_info_fsspec_reports_sidecar_dimensions(
        _fsspec_memory_with_sidecar_2239, overview_level, expected):
    uri = _fsspec_memory_with_sidecar_2239
    geo_info, h, w, dtype, n_bands = _read_geo_info(
        uri, overview_level=overview_level,
    )
    assert (h, w) == expected
    # The base file is single-band uint16; reported dtype / band count
    # should match regardless of which overview we landed on.
    assert dtype == np.dtype('uint16')
    assert n_bands == 0  # single-band convention


# ---------------------------------------------------------------------------
# Out-of-range guard: requesting an overview level beyond what the
# merged pyramid offers should raise with a clear message rather than
# silently swallow the request.
# ---------------------------------------------------------------------------
def test_fsspec_chunked_open_rejects_overview_past_sidecar(
        _fsspec_memory_with_sidecar_2239):
    uri = _fsspec_memory_with_sidecar_2239
    # Base + two sidecar levels => max valid level is 2.
    with pytest.raises(ValueError, match="overview_level"):
        open_geotiff(uri, chunks=16, overview_level=3)


@requires_loopback
def test_http_chunked_open_rejects_overview_past_sidecar(
        _http_with_sidecar_2239):
    url = _http_with_sidecar_2239
    with pytest.raises(ValueError, match="overview_level"):
        open_geotiff(url, chunks=16, overview_level=3)


# ---------------------------------------------------------------------------
# Defensive: ``discover_remote_sidecar`` swallows ``load_sidecar``
# failures (parse error, garbage bytes, transient network) and returns
# the unchanged base IFD list with ``sidecar=None``. The
# ``CloudSizeLimitError`` budget breach is the one exception that
# re-raises so a caller-set ceiling stays observable.
# ---------------------------------------------------------------------------
def test_discover_remote_sidecar_falls_back_when_load_fails(monkeypatch):
    """A probe that succeeds but a load that raises returns base-only."""
    from xrspatial.geotiff import _sidecar
    from xrspatial.geotiff._sidecar import discover_remote_sidecar

    monkeypatch.setattr(
        _sidecar, "find_sidecar", lambda _src: "http://example/x.tif.ovr"
    )

    def _exploding_load(_path, **_kw):
        raise RuntimeError("sidecar bytes did not parse")

    monkeypatch.setattr(_sidecar, "load_sidecar", _exploding_load)

    sentinel_ifds = [object(), object()]
    merged, sidecar, sidecar_ifd_ids = discover_remote_sidecar(
        "http://example/x.tif", sentinel_ifds,
    )
    assert merged == list(sentinel_ifds)
    assert sidecar is None
    assert sidecar_ifd_ids == set()


def test_discover_remote_sidecar_propagates_cloud_size_limit(monkeypatch):
    """The one exception the helper does NOT swallow is the budget breach."""
    from xrspatial.geotiff import _sidecar
    from xrspatial.geotiff._sidecar import discover_remote_sidecar

    monkeypatch.setattr(
        _sidecar, "find_sidecar", lambda _src: "http://example/x.tif.ovr"
    )

    def _budget_breach(_path, **_kw):
        raise CloudSizeLimitError("sidecar exceeds max_cloud_bytes")

    monkeypatch.setattr(_sidecar, "load_sidecar", _budget_breach)

    with pytest.raises(CloudSizeLimitError):
        discover_remote_sidecar(
            "http://example/x.tif", [object()], max_cloud_bytes=1,
        )


def test_parse_cog_http_meta_requires_source_path_when_return_sidecar(
        monkeypatch):
    """Future-proof: the 5-tuple contract needs a non-None source_path."""
    from xrspatial.geotiff._reader import _parse_cog_http_meta

    class _StubSource:
        def read_range(self, _start, _length):
            # Should not be called -- the precondition check fires first.
            raise AssertionError("read_range called despite missing source_path")

    with pytest.raises(TypeError, match="source_path"):
        _parse_cog_http_meta(_StubSource(), return_sidecar=True)


# ---------------------------------------------------------------------------
# Backstop: file-like buffers (``io.BytesIO``) have no sidecar concept;
# the new HTTP/fsspec discovery must not regress the file-like path.
# ---------------------------------------------------------------------------
def test_file_like_chunked_open_unaffected_by_sidecar_discovery():
    src = _fixture_or_skip()
    # ``open_geotiff(io.BytesIO, chunks=...)`` is not supported (the
    # dispatcher rejects file-like + chunks), so verify the eager path
    # still returns the base-level image without trying to probe for
    # a sidecar.
    with open(src, "rb") as f:
        buf = io.BytesIO(f.read())
    da = open_geotiff(buf)
    assert da.shape == (64, 64)


# ============================================================================
# Section: remote_sidecar_byte_order
#
# Mixed-endian remote sidecar reads decode with the sidecar's byte order.
#
# ``_parse_cog_http_meta`` historically returned the base file's
# ``TIFFHeader`` even when the selected IFD came from the sidecar, so a
# big-endian ``.ovr`` paired with a little-endian base (or the reverse)
# scrambled every multi-byte sample on the HTTP and fsspec dask paths.
# ============================================================================


def _mixed_endian_pair_2314(tmp_path, base_byteorder: str,
                            sidecar_byteorder: str, *, tile: int | None = None):
    """Return (base_path, sidecar_path, base_arr, sidecar_arr).

    The two arrays are deliberately different so a wrong-endian decode
    on the sidecar cannot accidentally compare equal to the base array.
    Distinct multipliers also keep the byte patterns visually different
    when a failure dump shows the raw values.

    ``tile`` selects the on-disk layout: ``None`` writes stripped TIFFs
    (tifffile's default), an int writes tiled TIFFs with the given
    square tile size. The stripped layout exercises
    ``_fetch_decode_cog_http_strips`` and the tiled layout exercises
    ``_fetch_decode_cog_http_tiles``; both paths read
    ``header.byte_order`` and benefit from the sidecar header swap.
    """
    tifffile = pytest.importorskip("tifffile")
    base_arr = (np.arange(64 * 64, dtype=np.uint16) * 7).reshape(64, 64)
    sidecar_arr = (np.arange(32 * 32, dtype=np.uint16) * 13).reshape(32, 32)
    suffix = uuid.uuid4().hex[:8]
    base_path = tmp_path / f"mixedendian_2314_{suffix}.tif"
    sidecar_path = tmp_path / f"mixedendian_2314_{suffix}.tif.ovr"
    write_kwargs = {}
    if tile is not None:
        write_kwargs['tile'] = (tile, tile)
    tifffile.imwrite(str(base_path), base_arr,
                     byteorder=base_byteorder, compression=None,
                     **write_kwargs)
    tifffile.imwrite(str(sidecar_path), sidecar_arr,
                     byteorder=sidecar_byteorder, compression=None,
                     **write_kwargs)
    return base_path, sidecar_path, base_arr, sidecar_arr


def _start_range_http_server_2314(payloads: dict[str, bytes]):
    import http.server
    import socketserver
    import threading

    class _Handler(http.server.BaseHTTPRequestHandler):
        payloads: dict[str, bytes] = {}

        def log_message(self, *a, **kw):
            return

        def do_GET(self):  # noqa: N802
            path = self.path
            if path not in self.payloads:
                self.send_response(404)
                self.end_headers()
                return
            body = self.payloads[path]
            rng = self.headers.get('Range')
            if rng and rng.startswith('bytes='):
                spec = rng[len('bytes='):]
                start_s, _, end_s = spec.partition('-')
                start = int(start_s)
                end = int(end_s) if end_s else len(body) - 1
                chunk = body[start:end + 1]
                self.send_response(206)
                self.send_header('Content-Type', 'application/octet-stream')
                self.send_header(
                    'Content-Range',
                    f'bytes {start}-{start + len(chunk) - 1}/{len(body)}',
                )
                self.send_header('Content-Length', str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    handler_cls = type(
        'RangeHandler2314', (_Handler,), {'payloads': payloads}
    )
    httpd = socketserver.TCPServer(('127.0.0.1', 0), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


# ---------------------------------------------------------------------------
# Local sanity check. The local sidecar reader already handles this case
# (``_reader.py:223`` swaps to the sidecar's header). Pin it here so a
# regression on the local path is caught alongside the remote ones.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base_bo,side_bo", [
    ('<', '>'),
    ('>', '<'),
])
def test_local_eager_mixed_endian_sidecar(tmp_path, base_bo, side_bo):
    """The local reader already handles mixed-endian sidecars."""
    base_path, _, base_arr, side_arr = _mixed_endian_pair_2314(
        tmp_path, base_bo, side_bo)
    r0 = open_geotiff(str(base_path), overview_level=0)
    np.testing.assert_array_equal(r0.values, base_arr)
    r1 = open_geotiff(str(base_path), overview_level=1)
    np.testing.assert_array_equal(r1.values, side_arr)


# ---------------------------------------------------------------------------
# Eager HTTP path. Pre-fix this scrambled the sidecar bytes because
# ``_read_cog_http`` passed the base file's header straight into
# ``_fetch_decode_cog_http_tiles`` after swapping the byte source to the
# sidecar URL. The fix returns the sidecar's header from
# ``_parse_cog_http_meta`` when ``used_sidecar=True``.
# ---------------------------------------------------------------------------
@requires_loopback
@pytest.mark.parametrize("base_bo,side_bo", [
    ('<', '>'),
    ('>', '<'),
])
def test_http_eager_mixed_endian_sidecar(tmp_path, monkeypatch,
                                         base_bo, side_bo):
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    base_path, sidecar_path, base_arr, side_arr = _mixed_endian_pair_2314(
        tmp_path, base_bo, side_bo)
    payloads = {
        "/x.tif": base_path.read_bytes(),
        "/x.tif.ovr": sidecar_path.read_bytes(),
    }
    httpd, port = _start_range_http_server_2314(payloads)
    try:
        url = f"http://127.0.0.1:{port}/x.tif"
        r0 = open_geotiff(url, overview_level=0)
        np.testing.assert_array_equal(r0.values, base_arr)
        r1 = open_geotiff(url, overview_level=1)
        np.testing.assert_array_equal(r1.values, side_arr)
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# Dask HTTP path. Pre-fix this set ``http_meta = (http_header, http_ifd)``
# where ``http_header`` was always the base header; per-chunk decoding
# then used the wrong byte order on every chunk. The fix flows through
# ``_parse_cog_http_meta``: when the sidecar is in use, ``http_header``
# is the sidecar's header.
# ---------------------------------------------------------------------------
@requires_loopback
@pytest.mark.parametrize("base_bo,side_bo", [
    ('<', '>'),
    ('>', '<'),
])
def test_http_chunked_mixed_endian_sidecar(tmp_path, monkeypatch,
                                           base_bo, side_bo):
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    base_path, sidecar_path, base_arr, side_arr = _mixed_endian_pair_2314(
        tmp_path, base_bo, side_bo)
    payloads = {
        "/x.tif": base_path.read_bytes(),
        "/x.tif.ovr": sidecar_path.read_bytes(),
    }
    httpd, port = _start_range_http_server_2314(payloads)
    try:
        url = f"http://127.0.0.1:{port}/x.tif"
        # ``chunks`` forces the dask metadata path through
        # ``_backends/dask.py``'s ``_parse_cog_http_meta`` branch and
        # builds a per-chunk graph that hits the per-chunk decode step.
        chunked0 = open_geotiff(url, overview_level=0, chunks=16)
        np.testing.assert_array_equal(chunked0.values, base_arr)
        chunked1 = open_geotiff(url, overview_level=1, chunks=16)
        np.testing.assert_array_equal(chunked1.values, side_arr)
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# fsspec dask path. The fsspec branch of ``_backends/dask.py`` goes
# through the same ``_parse_cog_http_meta`` call, so the same header
# swap covers it. ``memory://`` stages the payload bytes in process and
# bypasses the HTTP server entirely, which makes the fsspec path easy
# to test without DNS / port assumptions.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("base_bo,side_bo", [
    ('<', '>'),
    ('>', '<'),
])
def test_fsspec_chunked_mixed_endian_sidecar(tmp_path, base_bo, side_bo):
    fsspec = pytest.importorskip("fsspec")
    base_path, sidecar_path, base_arr, side_arr = _mixed_endian_pair_2314(
        tmp_path, base_bo, side_bo)
    fs = fsspec.filesystem("memory")
    # memory:// is process-global; collide-proof the key.
    key = f"issue2314-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    base_uri = f"memory://{key}/x.tif"
    side_uri = base_uri + ".ovr"
    fs.pipe_file(base_uri.replace("memory://", ""), base_path.read_bytes())
    fs.pipe_file(side_uri.replace("memory://", ""), sidecar_path.read_bytes())
    try:
        chunked0 = open_geotiff(base_uri, overview_level=0, chunks=16)
        np.testing.assert_array_equal(chunked0.values, base_arr)
        chunked1 = open_geotiff(base_uri, overview_level=1, chunks=16)
        np.testing.assert_array_equal(chunked1.values, side_arr)
    finally:
        for p in (base_uri, side_uri):
            try:
                fs.rm_file(p.replace("memory://", ""))
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Tiled-layout variant. The stripped path above already exercises the
# header swap end to end, but the tiled HTTP path is the more common
# COG shape in the wild and reads ``header.byte_order`` from a different
# call site (``_fetch_decode_cog_http_tiles``'s ``_decode_one`` closure).
# Run one parametrized case here so a future regression that decouples
# the two paths gets caught.
# ---------------------------------------------------------------------------
@requires_loopback
@pytest.mark.parametrize("base_bo,side_bo", [
    ('<', '>'),
    ('>', '<'),
])
def test_http_eager_mixed_endian_sidecar_tiled(tmp_path, monkeypatch,
                                               base_bo, side_bo):
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    base_path, sidecar_path, base_arr, side_arr = _mixed_endian_pair_2314(
        tmp_path, base_bo, side_bo, tile=16)
    payloads = {
        "/x.tif": base_path.read_bytes(),
        "/x.tif.ovr": sidecar_path.read_bytes(),
    }
    httpd, port = _start_range_http_server_2314(payloads)
    try:
        url = f"http://127.0.0.1:{port}/x.tif"
        r0 = open_geotiff(url, overview_level=0)
        np.testing.assert_array_equal(r0.values, base_arr)
        r1 = open_geotiff(url, overview_level=1)
        np.testing.assert_array_equal(r1.values, side_arr)
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# Direct ``_parse_cog_http_meta`` contract test. Asserts the returned
# header's byte order matches the sidecar when ``used_sidecar=True`` and
# the base when ``used_sidecar=False``.
# ---------------------------------------------------------------------------
@requires_loopback
@pytest.mark.parametrize("base_bo,side_bo", [
    ('<', '>'),
    ('>', '<'),
])
def test_parse_cog_http_meta_returns_sidecar_header(tmp_path, monkeypatch,
                                                    base_bo, side_bo):
    monkeypatch.setenv("XRSPATIAL_GEOTIFF_ALLOW_PRIVATE_HOSTS", "1")
    from xrspatial.geotiff._cog_http import _parse_cog_http_meta
    from xrspatial.geotiff._reader import _HTTPSource
    from xrspatial.geotiff._sidecar import close_sidecar
    base_path, sidecar_path, _, _ = _mixed_endian_pair_2314(
        tmp_path, base_bo, side_bo)
    payloads = {
        "/x.tif": base_path.read_bytes(),
        "/x.tif.ovr": sidecar_path.read_bytes(),
    }
    httpd, port = _start_range_http_server_2314(payloads)
    try:
        url = f"http://127.0.0.1:{port}/x.tif"

        # Level 0: base IFD. Header should be the base file's header.
        src = _HTTPSource(url)
        try:
            hdr, ifd, _, _, sidecar_meta = _parse_cog_http_meta(
                src, overview_level=0, source_path=url, return_sidecar=True)
        finally:
            src.close()
        sidecar, _, used = sidecar_meta
        try:
            assert used is False
            assert hdr.byte_order == base_bo
        finally:
            close_sidecar(sidecar)

        # Level 1: sidecar IFD. Header should switch to the sidecar's
        # byte order so downstream decode honours the right endianness.
        src = _HTTPSource(url)
        try:
            hdr, ifd, _, _, sidecar_meta = _parse_cog_http_meta(
                src, overview_level=1, source_path=url, return_sidecar=True)
        finally:
            src.close()
        sidecar, _, used = sidecar_meta
        try:
            assert used is True
            assert hdr.byte_order == side_bo
        finally:
            close_sidecar(sidecar)
    finally:
        httpd.shutdown()
        httpd.server_close()
