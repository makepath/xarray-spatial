"""Shared fixtures for geotiff tests."""
from __future__ import annotations

import importlib.util
import math
import os
import socket
import struct

import numpy as np
import pytest


def gpu_available() -> bool:
    """True iff cupy imports AND a CUDA device is actually usable.

    Some sandboxes ship cupy without a working CUDA runtime. A bare
    ``import cupy`` succeeds there but every device call fails, so test
    files that gate on the import alone show false failures.
    """
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


def loopback_available() -> bool:
    """True iff a loopback TCP bind succeeds on this host.

    Some sandboxed environments deny ``bind(('127.0.0.1', 0))``. HTTP
    tests that stand up a loopback server should skip rather than error
    in that case.
    """
    try:
        s = socket.socket()
        try:
            s.bind(('127.0.0.1', 0))
        finally:
            s.close()
    except OSError:
        return False
    return True


_HAS_GPU = gpu_available()
_HAS_LOOPBACK = loopback_available()

requires_gpu = pytest.mark.skipif(
    not _HAS_GPU, reason="cupy + CUDA required"
)
requires_loopback = pytest.mark.skipif(
    not _HAS_LOOPBACK, reason="loopback bind unavailable in this environment"
)

_RUN_INTEGRATION = os.environ.get("XRSPATIAL_RUN_INTEGRATION", "") not in (
    "", "0", "false", "False"
)
requires_integration = pytest.mark.skipif(
    not _RUN_INTEGRATION,
    reason="integration test; set XRSPATIAL_RUN_INTEGRATION=1 to run locally",
)


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that stand up a loopback HTTP server when the
    sandbox denies socket bind.

    A test needs loopback iff its function body or any fixture in its
    closure references ``socketserver.TCPServer(`` or invokes the
    file-local ``_serve(`` helper. This is finer-grained than skipping
    every test in a module that imports ``socketserver``: mixed files
    (e.g. ``test_miniswhite_backend_parity_1797.py`` has both HTTP and
    a local-file GPU test) keep their non-HTTP coverage in restricted
    sandboxes.
    """
    if _HAS_LOOPBACK:
        return

    import inspect

    def _source_of(obj) -> str:
        try:
            return inspect.getsource(obj)
        except (OSError, TypeError):
            return ''

    def _references_loopback(src: str) -> bool:
        return 'socketserver.TCPServer(' in src or '_serve(' in src

    skip_marker = pytest.mark.skip(
        reason="loopback bind unavailable in this environment"
    )
    for item in items:
        needs_skip = False

        func = getattr(item, 'function', None)
        if func is not None and _references_loopback(_source_of(func)):
            needs_skip = True

        if not needs_skip:
            fixtureinfo = getattr(item, '_fixtureinfo', None)
            if fixtureinfo is not None:
                name2defs = getattr(fixtureinfo, 'name2fixturedefs', {})
                for fname in getattr(fixtureinfo, 'names_closure', ()):
                    for fdef in name2defs.get(fname, ()):
                        ffunc = getattr(fdef, 'func', None)
                        if ffunc is not None and _references_loopback(
                            _source_of(ffunc)
                        ):
                            needs_skip = True
                            break
                    if needs_skip:
                        break

        if needs_skip:
            item.add_marker(skip_marker)


def make_minimal_tiff(
    width: int = 4,
    height: int = 4,
    dtype: np.dtype = np.dtype('float32'),
    pixel_data: np.ndarray | None = None,
    compression: int = 1,
    tiled: bool = False,
    tile_size: int = 4,
    big_endian: bool = False,
    bigtiff: bool = False,
    geo_transform: tuple | None = None,
    epsg: int | None = None,
) -> bytes:
    """Build a minimal valid TIFF file in memory for testing.

    Uses a three-pass approach:
    1. Collect all tags and their raw value data
    2. Compute file layout (IFD size, overflow positions, pixel data offset)
    3. Serialize everything with correct offsets
    """
    bo = '>' if big_endian else '<'
    bom = b'MM' if big_endian else b'II'

    if pixel_data is None:
        pixel_data = np.arange(width * height, dtype=dtype).reshape(height, width)
    else:
        dtype = pixel_data.dtype

    bits_per_sample = dtype.itemsize * 8
    if dtype.kind == 'f':
        sample_format = 3
    elif dtype.kind == 'i':
        sample_format = 2
    else:
        sample_format = 1

    # --- Build pixel data (strips or tiles) ---
    if tiled:
        tiles_across = math.ceil(width / tile_size)
        tiles_down = math.ceil(height / tile_size)
        num_tiles = tiles_across * tiles_down

        tile_blobs = []
        for tr in range(tiles_down):
            for tc in range(tiles_across):
                tile = np.zeros((tile_size, tile_size), dtype=dtype)
                r0, c0 = tr * tile_size, tc * tile_size
                r1 = min(r0 + tile_size, height)
                c1 = min(c0 + tile_size, width)
                tile[:r1 - r0, :c1 - c0] = pixel_data[r0:r1, c0:c1]
                tile_blobs.append(tile.tobytes())

        pixel_bytes = b''.join(tile_blobs)
        tile_byte_counts = [len(b) for b in tile_blobs]
    else:
        if big_endian and pixel_data.dtype.itemsize > 1:
            pixel_bytes = pixel_data.astype(pixel_data.dtype.newbyteorder('>')).tobytes()
        else:
            pixel_bytes = pixel_data.tobytes()

    # --- Collect tags as (tag_id, type_id, value_bytes) ---
    # value_bytes is the serialized value; if len <= 4 it's inline, else overflow.
    tag_list: list[tuple[int, int, int, bytes]] = []  # (tag, type, count, raw_bytes)

    def add_short(tag, val):
        tag_list.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tag_list.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    def add_shorts(tag, vals):
        tag_list.append((tag, 3, len(vals), struct.pack(f'{bo}{len(vals)}H', *vals)))

    def add_longs(tag, vals):
        tag_list.append((tag, 4, len(vals), struct.pack(f'{bo}{len(vals)}I', *vals)))

    def add_doubles(tag, vals):
        tag_list.append((tag, 12, len(vals), struct.pack(f'{bo}{len(vals)}d', *vals)))

    add_short(256, width)            # ImageWidth
    add_short(257, height)           # ImageLength
    add_short(258, bits_per_sample)  # BitsPerSample
    add_short(259, compression)      # Compression
    add_short(262, 1)                # PhotometricInterpretation
    add_short(277, 1)                # SamplesPerPixel
    add_short(339, sample_format)    # SampleFormat

    if tiled:
        add_short(322, tile_size)    # TileWidth
        add_short(323, tile_size)    # TileLength
        # Placeholder offsets -- will be patched after layout is known
        add_longs(324, [0] * num_tiles)            # TileOffsets
        add_longs(325, tile_byte_counts)           # TileByteCounts
    else:
        add_short(278, height)       # RowsPerStrip
        add_long(273, 0)             # StripOffsets (placeholder)
        add_long(279, len(pixel_bytes))  # StripByteCounts

    if geo_transform is not None:
        ox, oy, pw, ph = geo_transform
        add_doubles(33550, [abs(pw), abs(ph), 0.0])     # ModelPixelScale
        add_doubles(33922, [0.0, 0.0, 0.0, ox, oy, 0.0])  # ModelTiepoint

    if epsg is not None:
        if epsg == 4326 or (4000 <= epsg < 5000):
            model_type, key_id = 2, 2048
        else:
            model_type, key_id = 1, 3072
        gkd = [1, 1, 0, 2, 1024, 0, 1, model_type, key_id, 0, 1, epsg]
        add_shorts(34735, gkd)

    # Sort by tag ID (TIFF spec requirement)
    tag_list.sort(key=lambda t: t[0])

    # --- Compute layout ---
    num_entries = len(tag_list)
    ifd_start = 8  # right after header
    ifd_size = 2 + 12 * num_entries + 4  # count + entries + next_ifd_offset
    overflow_start = ifd_start + ifd_size

    # Figure out which tags need overflow (value > 4 bytes)
    overflow_buf = bytearray()
    for _tag, _type, _count, raw in tag_list:
        if len(raw) > 4:
            # This will go to overflow -- just accumulate size for now
            overflow_buf.extend(raw)
            # Word-align
            if len(overflow_buf) % 2:
                overflow_buf.append(0)

    pixel_data_start = overflow_start + len(overflow_buf)

    # --- Patch offset tags ---
    # Now we know where pixel data starts, patch strip/tile offsets
    patched = []
    for tag, typ, count, raw in tag_list:
        if tag == 273:  # StripOffsets
            patched.append((tag, typ, count, struct.pack(f'{bo}I', pixel_data_start)))
        elif tag == 324:  # TileOffsets
            offsets = []
            pos = 0
            for blob in tile_blobs:
                offsets.append(pixel_data_start + pos)
                pos += len(blob)
            patched.append((tag, typ, count, struct.pack(f'{bo}{num_tiles}I', *offsets)))
        else:
            patched.append((tag, typ, count, raw))
    tag_list = patched

    # --- Rebuild overflow with final values ---
    overflow_buf = bytearray()
    tag_offsets = {}  # tag -> offset within overflow_buf (or None if inline)

    for tag, typ, count, raw in tag_list:
        if len(raw) > 4:
            tag_offsets[tag] = len(overflow_buf)
            overflow_buf.extend(raw)
            if len(overflow_buf) % 2:
                overflow_buf.append(0)
        else:
            tag_offsets[tag] = None

    # Recalculate in case overflow size changed from patching
    actual_pixel_start = overflow_start + len(overflow_buf)
    if actual_pixel_start != pixel_data_start:
        # Need another pass to fix offsets
        pixel_data_start = actual_pixel_start
        patched2 = []
        for tag, typ, count, raw in tag_list:
            if tag == 273:
                patched2.append((tag, typ, count, struct.pack(f'{bo}I', pixel_data_start)))
            elif tag == 324:
                offsets = []
                pos = 0
                for blob in tile_blobs:
                    offsets.append(pixel_data_start + pos)
                    pos += len(blob)
                patched2.append((tag, typ, count, struct.pack(f'{bo}{num_tiles}I', *offsets)))
            else:
                patched2.append((tag, typ, count, raw))
        tag_list = patched2

        # Rebuild overflow again
        overflow_buf = bytearray()
        tag_offsets = {}
        for tag, typ, count, raw in tag_list:
            if len(raw) > 4:
                tag_offsets[tag] = len(overflow_buf)
                overflow_buf.extend(raw)
                if len(overflow_buf) % 2:
                    overflow_buf.append(0)
            else:
                tag_offsets[tag] = None

    # --- Serialize ---
    out = bytearray()

    # Header
    out.extend(bom)
    out.extend(struct.pack(f'{bo}H', 42))
    out.extend(struct.pack(f'{bo}I', ifd_start))

    # IFD
    out.extend(struct.pack(f'{bo}H', num_entries))

    for tag, typ, count, raw in tag_list:
        out.extend(struct.pack(f'{bo}HHI', tag, typ, count))
        if len(raw) <= 4:
            # Inline value, padded to 4 bytes
            out.extend(raw.ljust(4, b'\x00'))
        else:
            # Pointer to overflow
            ptr = overflow_start + tag_offsets[tag]
            out.extend(struct.pack(f'{bo}I', ptr))

    # Next IFD offset
    out.extend(struct.pack(f'{bo}I', 0))

    # Overflow
    out.extend(overflow_buf)

    # Pixel data
    out.extend(pixel_bytes)

    return bytes(out)


@pytest.fixture
def simple_float32_tiff():
    """4x4 float32 stripped TIFF with sequential values."""
    return make_minimal_tiff(4, 4, np.dtype('float32'))


@pytest.fixture
def simple_uint16_tiff():
    """4x4 uint16 stripped TIFF."""
    return make_minimal_tiff(4, 4, np.dtype('uint16'))


@pytest.fixture
def geo_tiff_data():
    """4x4 float32 TIFF with geo transform and EPSG 4326."""
    return make_minimal_tiff(
        4, 4, np.dtype('float32'),
        geo_transform=(-120.0, 45.0, 0.001, -0.001),
        epsg=4326,
    )


@pytest.fixture
def tiled_tiff_data():
    """8x8 float32 tiled TIFF with 4x4 tiles."""
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    return make_minimal_tiff(
        8, 8, np.dtype('float32'),
        pixel_data=data,
        tiled=True,
        tile_size=4,
    )
