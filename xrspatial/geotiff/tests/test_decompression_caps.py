"""Tests for decompression-bomb defenses (security finding S1).

Each codec used by the TIFF reader (deflate, zstd, lz4, packbits) accepts an
``expected_size`` argument and refuses to produce more than ~5% above that
size before raising ``ValueError``.  Without these caps a small malicious
TIFF could expand to many GB during decode and OOM the reader before the
post-decode size check ran.

Each end-to-end test here builds a minimal TIFF that declares a 1024x1024
uint8 image (1 MiB of legitimate pixel data) and feeds in a strip whose
decoded size is several MiB. That ratio is enough to trip the cap (~1.05
MiB) without forcing the test process to allocate a multi-gigabyte
payload host-side -- the audit's original 1024:1 framing was symbolic;
what we actually verify is "compressed size << decoded size > cap".
"""
from __future__ import annotations

import importlib.util
import struct

import numpy as np
import pytest
import zlib

from xrspatial.geotiff._compression import (
    deflate_decompress,
    lz4_decompress,
    packbits_decompress,
    zstd_decompress,
)
from xrspatial.geotiff._reader import read_to_array

def _module_available(name: str) -> bool:
    """True iff ``import name`` would succeed.

    ``importlib.util.find_spec`` on a submodule (e.g. ``lz4.frame``) imports
    the parent package to resolve the spec, so it raises ``ModuleNotFoundError``
    rather than returning ``None`` when the parent itself is missing. Wrap
    the lookup in try/except so an absent optional dependency cleanly turns
    into ``False`` regardless of dotted depth.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_HAS_ZSTD = _module_available("zstandard")
_HAS_LZ4 = _module_available("lz4.frame")
_HAS_LERC = _module_available("lerc")
_HAS_GLYMUR = _module_available("glymur")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tiff_with_strip(strip_bytes: bytes, *, compression: int,
                           width: int, height: int) -> bytes:
    """Build a minimal little-endian uint8 TIFF with one strip of opaque bytes.

    The strip is written verbatim (no further encoding); the caller has
    already compressed it.  ``width * height`` is the declared logical
    image size — the reader uses that to compute the expected decompressed
    size and apply the bomb cap.
    """
    bo = '<'

    # Tags: (tag_id, type_id, count, value_or_offset_bytes_4)
    # We keep every tag's value inline so the TIFF body order is:
    #   header(8) | IFD | strip
    tags = []

    def add_short(tag, val):
        tags.append((tag, 3, 1, struct.pack(f'{bo}H', val)))

    def add_long(tag, val):
        tags.append((tag, 4, 1, struct.pack(f'{bo}I', val)))

    add_short(256, width)        # ImageWidth (assume <= 65535)
    add_short(257, height)       # ImageLength
    add_short(258, 8)            # BitsPerSample = 8 (uint8)
    add_short(259, compression)  # Compression
    add_short(262, 1)            # PhotometricInterpretation = BlackIsZero
    add_short(277, 1)            # SamplesPerPixel
    add_short(278, height)       # RowsPerStrip
    add_long(273, 0)             # StripOffsets (placeholder)
    add_long(279, len(strip_bytes))  # StripByteCounts
    add_short(339, 1)            # SampleFormat = unsigned int

    tags.sort(key=lambda t: t[0])
    num_entries = len(tags)
    ifd_size = 2 + 12 * num_entries + 4  # count + entries + next-IFD
    ifd_start = 8
    strip_offset = ifd_start + ifd_size

    # Patch StripOffsets (tag 273) with the real strip location.
    patched = []
    for tag_id, typ, count, raw in tags:
        if tag_id == 273:
            raw = struct.pack(f'{bo}I', strip_offset)
        patched.append((tag_id, typ, count, raw))
    tags = patched

    out = bytearray()
    out += b'II'                                      # little-endian
    out += struct.pack(f'{bo}H', 42)                  # magic
    out += struct.pack(f'{bo}I', ifd_start)           # offset to IFD0
    out += struct.pack(f'{bo}H', num_entries)         # IFD entry count
    for tag_id, typ, count, raw in tags:
        out += struct.pack(f'{bo}HHI', tag_id, typ, count)
        # Pad raw to 4 bytes (all our tags fit inline).
        out += raw.ljust(4, b'\x00')
    out += struct.pack(f'{bo}I', 0)                    # no next IFD
    out += strip_bytes
    return bytes(out)


# ---------------------------------------------------------------------------
# Codec-level direct tests
# ---------------------------------------------------------------------------

# Direct-codec bomb size: 4 MiB of zeros, well above the 1 KiB cap used
# in those tests but small enough to keep CI host allocations under
# control. 4 MiB / 1 KiB ~ 4000:1 is still bomb-shaped territory; the
# cap fires the moment decoded > cap.
_DIRECT_BOMB_BYTES = 4 * 1024 * 1024


class TestCodecDirect:
    """Exercise the codec functions directly with bomb payloads."""

    def test_deflate_bomb_raises(self):
        big = b'\x00' * _DIRECT_BOMB_BYTES
        comp = zlib.compress(big, 9)
        with pytest.raises(ValueError, match="exceed"):
            deflate_decompress(comp, expected_size=1024)

    def test_deflate_legitimate_passes(self):
        data = b'A' * 4096
        comp = zlib.compress(data, 9)
        out = deflate_decompress(comp, expected_size=len(data))
        assert out == data

    def test_deflate_no_cap_when_expected_size_zero(self):
        """Backward-compat: ``expected_size=0`` (default) disables the cap.

        Callers that haven't been updated to supply a size must keep
        getting the unbounded library decode -- otherwise a default
        cap would silently break them. Round-tripping data larger than
        any plausible cap proves the disable path is intact.
        """
        data = b'A' * (256 * 1024)  # 256 KiB, well above any cap default
        comp = zlib.compress(data, 9)
        out = deflate_decompress(comp)  # no expected_size kwarg
        assert out == data

    def test_packbits_bomb_raises(self):
        # 0x81 0x00 = "repeat next byte 128 times".  Repeated 1M times this
        # would decode to 128 MiB of zeros from a 2 MiB input -- the cap
        # stops the decoder long before the full 128 MiB ever materialises.
        bomb = b'\x81\x00' * (1024 * 1024)
        with pytest.raises(ValueError, match="exceed"):
            packbits_decompress(bomb, expected_size=1024)

    def test_packbits_legitimate_passes(self):
        # Literal run: header byte n=3 means "copy next 4 bytes literally".
        data = b'\x03ABCD'
        out = packbits_decompress(data, expected_size=4)
        assert out == b'ABCD'

    def test_packbits_no_cap_when_expected_size_zero(self):
        # Same literal-run pattern, no expected_size argument: the
        # backward-compat path must skip the cap and decode in full.
        data = b'\x03ABCD' * 1024
        out = packbits_decompress(data)
        assert out == b'ABCD' * 1024


@pytest.mark.skipif(not _HAS_ZSTD, reason="zstandard not installed")
class TestZstdDirect:
    def test_zstd_bomb_raises(self):
        import zstandard
        big = b'\x00' * _DIRECT_BOMB_BYTES
        comp = zstandard.ZstdCompressor().compress(big)
        with pytest.raises(ValueError, match="exceed"):
            zstd_decompress(comp, expected_size=1024)

    def test_zstd_legitimate_passes(self):
        import zstandard
        data = b'A' * 4096
        comp = zstandard.ZstdCompressor().compress(data)
        out = zstd_decompress(comp, expected_size=len(data))
        assert out == data

    def test_zstd_no_cap_when_expected_size_zero(self):
        import zstandard
        data = b'A' * (256 * 1024)
        comp = zstandard.ZstdCompressor().compress(data)
        out = zstd_decompress(comp)
        assert out == data


@pytest.mark.skipif(not _HAS_LZ4, reason="lz4 not installed")
class TestLz4Direct:
    def test_lz4_bomb_raises(self):
        import lz4.frame
        big = b'\x00' * _DIRECT_BOMB_BYTES
        comp = lz4.frame.compress(big)
        with pytest.raises(ValueError, match="exceed"):
            lz4_decompress(comp, expected_size=1024)

    def test_lz4_legitimate_passes(self):
        import lz4.frame
        data = b'A' * 4096
        comp = lz4.frame.compress(data)
        out = lz4_decompress(comp, expected_size=len(data))
        assert out == data

    def test_lz4_no_cap_when_expected_size_zero(self):
        import lz4.frame
        data = b'A' * (256 * 1024)
        comp = lz4.frame.compress(data)
        out = lz4_decompress(comp)
        assert out == data


# ---------------------------------------------------------------------------
# End-to-end TIFF tests (audit reproducer shape)
# ---------------------------------------------------------------------------

# 1024 x 1024 uint8 = 1 MiB declared image, cap is ~1.05 MiB. We feed a
# strip whose decoded size is 8 MiB. The cap is exceeded by ~7x, which is
# enough to prove the codec rejects rather than silently truncates, while
# keeping the test's host-side allocation small enough for any CI runner.
# (The audit's original 1024:1 framing was symbolic; the defense fires the
# moment decoded > cap, not at any specific ratio.)
_DECLARED_W = 1024
_DECLARED_H = 1024
_DECLARED_BYTES = _DECLARED_W * _DECLARED_H  # 1 MiB
_BOMB_BYTES = 8 * 1024 * 1024                # 8 MiB


def test_deflate_bomb_rejected(tmp_path):
    """Decoded size > cap must raise rather than allocate the bomb."""
    payload = b'\x00' * _BOMB_BYTES
    strip = zlib.compress(payload, 9)
    assert len(strip) < _BOMB_BYTES // 10
    tiff = _build_tiff_with_strip(strip, compression=8,
                                  width=_DECLARED_W, height=_DECLARED_H)
    path = tmp_path / "deflate_bomb.tif"
    path.write_bytes(tiff)
    with pytest.raises(ValueError, match="exceed"):
        read_to_array(str(path))


@pytest.mark.skipif(not _HAS_ZSTD, reason="zstandard not installed")
def test_zstd_bomb_rejected(tmp_path):
    import zstandard
    payload = b'\x00' * _BOMB_BYTES
    strip = zstandard.ZstdCompressor().compress(payload)
    assert len(strip) < _BOMB_BYTES // 10
    tiff = _build_tiff_with_strip(strip, compression=50000,
                                  width=_DECLARED_W, height=_DECLARED_H)
    path = tmp_path / "zstd_bomb.tif"
    path.write_bytes(tiff)
    with pytest.raises(ValueError, match="exceed"):
        read_to_array(str(path))


@pytest.mark.skipif(not _HAS_LZ4, reason="lz4 not installed")
def test_lz4_bomb_rejected(tmp_path):
    import lz4.frame
    payload = b'\x00' * _BOMB_BYTES
    strip = lz4.frame.compress(payload)
    # LZ4 has a higher floor than deflate/zstd for runs of zeros, but
    # still well below the bomb size.
    assert len(strip) < _BOMB_BYTES // 4
    tiff = _build_tiff_with_strip(strip, compression=50004,
                                  width=_DECLARED_W, height=_DECLARED_H)
    path = tmp_path / "lz4_bomb.tif"
    path.write_bytes(tiff)
    with pytest.raises(ValueError, match="exceed"):
        read_to_array(str(path))


def test_packbits_bomb_rejected(tmp_path):
    # Packbits "repeat next byte 128 times" header is 0x81 0x00 (2 bytes).
    # We declare 1024x1024=1 MiB image but supply a 2 MiB strip that
    # decodes to 128 MiB.  The cap should fire long before allocation.
    strip = b'\x81\x00' * (1024 * 1024)
    tiff = _build_tiff_with_strip(strip, compression=32773,
                                  width=_DECLARED_W, height=_DECLARED_H)
    path = tmp_path / "packbits_bomb.tif"
    path.write_bytes(tiff)
    with pytest.raises(ValueError, match="exceed"):
        read_to_array(str(path))


# ---------------------------------------------------------------------------
# Negative tests: legitimate high-ratio compression must still pass
# ---------------------------------------------------------------------------

def test_legitimate_high_compression_passes(tmp_path):
    """All-zero array compresses to a fraction of declared size — must pass."""
    arr = np.zeros((_DECLARED_H, _DECLARED_W), dtype=np.uint8)
    strip = zlib.compress(arr.tobytes(), 9)
    # Confirm we actually have a high ratio (not a degenerate test).
    assert len(strip) < _DECLARED_BYTES // 50
    tiff = _build_tiff_with_strip(strip, compression=8,
                                  width=_DECLARED_W, height=_DECLARED_H)
    path = tmp_path / "legit.tif"
    path.write_bytes(tiff)
    out, _ = read_to_array(str(path))
    assert out.shape == (_DECLARED_H, _DECLARED_W)
    assert out.dtype == np.uint8
    assert (out == 0).all()


def test_cap_includes_metadata_margin():
    """The cap allows ~5% of legitimate codec metadata above expected size.

    Some encoders emit small framing or trailing bytes; the cap must not
    reject them.  We feed a payload exactly at expected_size + a few bytes
    and confirm it decodes.
    """
    expected = 1000
    # Decompressed size: expected + 30 (3% over).  Within the 5% margin.
    data = b'A' * (expected + 30)
    comp = zlib.compress(data, 9)
    out = deflate_decompress(comp, expected_size=expected)
    assert out == data


# ---------------------------------------------------------------------------
# LERC and JPEG 2000 codec-level bomb tests (issue #1625)
# ---------------------------------------------------------------------------
#
# LERC and JPEG 2000 use external libraries (lerc / glymur) that materialise
# the full decoded buffer before returning, so the existing post-decode
# size check in ``_decode_strip_or_tile`` fires only after the bomb is
# already in memory. The wrappers in ``_compression.py`` now query each
# codestream's declared dimensions (LERC via ``getLercBlobInfo``, JPEG 2000
# via ``Jp2k.shape``) and raise before invoking the underlying decoder.

@pytest.mark.skipif(not _HAS_LERC, reason="lerc not installed")
class TestLercDirect:
    def test_lerc_bomb_raises(self):
        """A LERC blob whose declared dimensions exceed the cap must raise.

        Constant-value rasters compress at >700,000:1 in LERC, so a
        4096x4096 float32 (64 MiB) encodes to ~94 bytes. The cap is set to
        1 KiB, well below the declared 64 MiB, and the wrapper must reject
        the blob before ``lerc.decode`` allocates the output buffer.
        """
        import lerc
        arr = np.zeros((4096, 4096), dtype=np.float32)
        encoded = lerc.encode(arr, 1, False, None, 0.0, 1)
        blob = bytes(encoded[2])
        assert len(blob) < 1024  # confirm this is a real high-ratio blob
        from xrspatial.geotiff._compression import lerc_decompress
        with pytest.raises(ValueError, match="exceed"):
            lerc_decompress(blob, expected_size=1024)

    def test_lerc_legitimate_passes(self):
        """A LERC blob whose declared output matches expected_size passes."""
        import lerc
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        encoded = lerc.encode(arr, 1, False, None, 0.0, 1)
        blob = bytes(encoded[2])
        from xrspatial.geotiff._compression import lerc_decompress
        out = lerc_decompress(blob, expected_size=arr.nbytes)
        decoded = np.frombuffer(out, dtype=np.float32).reshape(8, 8)
        assert np.array_equal(decoded, arr)

    def test_lerc_no_cap_when_expected_size_zero(self):
        """Backward-compat: ``expected_size=0`` disables the cap."""
        import lerc
        arr = np.zeros((128, 128), dtype=np.float32)
        encoded = lerc.encode(arr, 1, False, None, 0.0, 1)
        blob = bytes(encoded[2])
        from xrspatial.geotiff._compression import lerc_decompress
        out = lerc_decompress(blob)  # no expected_size
        decoded = np.frombuffer(out, dtype=np.float32).reshape(128, 128)
        assert decoded.shape == arr.shape


@pytest.mark.skipif(not _HAS_GLYMUR, reason="glymur not installed")
class TestJpeg2000Direct:
    def test_jpeg2000_bomb_raises(self, tmp_path):
        """A JPEG 2000 codestream whose declared shape exceeds the cap raises.

        Glymur reports ``Jp2k(file).shape`` from the SIZ marker without
        triggering pixel decoding, so the wrapper validates the declared
        ``H * W * dtype_bytes`` against the bomb cap before calling
        ``jp2[:]``.
        """
        import glymur
        # Build a real 2000x2000 uint8 codestream (~150 bytes for zeros).
        arr = np.zeros((2000, 2000), dtype=np.uint8)
        tmp = tmp_path / "src.j2k"
        glymur.Jp2k(str(tmp), data=arr)
        blob = tmp.read_bytes()
        assert len(blob) < 10_000  # confirm high ratio
        from xrspatial.geotiff._compression import jpeg2000_decompress
        with pytest.raises(ValueError, match="exceed"):
            # declared output 4 MiB, cap 1 KiB
            jpeg2000_decompress(
                blob, width=2000, height=2000, samples=1,
                expected_size=1024)

    def test_jpeg2000_legitimate_passes(self, tmp_path):
        """A JPEG 2000 blob whose declared output matches expected_size passes."""
        import glymur
        # Use a 64x64 raster: large enough for the default 6-resolution
        # OpenJPEG pyramid without tripping its min-tile-size check.
        arr = (np.arange(64 * 64, dtype=np.uint8) % 200).reshape(64, 64)
        tmp = tmp_path / "legit.j2k"
        glymur.Jp2k(str(tmp), data=arr)
        blob = tmp.read_bytes()
        from xrspatial.geotiff._compression import jpeg2000_decompress
        out = jpeg2000_decompress(
            blob, width=64, height=64, samples=1,
            expected_size=arr.nbytes)
        decoded = np.frombuffer(out, dtype=np.uint8).reshape(64, 64)
        assert np.array_equal(decoded, arr)

    def test_jpeg2000_no_cap_when_expected_size_zero(self, tmp_path):
        """Backward-compat: ``expected_size=0`` disables the cap."""
        import glymur
        arr = np.zeros((128, 128), dtype=np.uint8)
        tmp = tmp_path / "nocap.j2k"
        glymur.Jp2k(str(tmp), data=arr)
        blob = tmp.read_bytes()
        from xrspatial.geotiff._compression import jpeg2000_decompress
        out = jpeg2000_decompress(blob, width=128, height=128, samples=1)
        decoded = np.frombuffer(out, dtype=np.uint8).reshape(128, 128)
        assert decoded.shape == arr.shape

    def test_jpeg2000_unreadable_shape_fails_closed(
            self, tmp_path, monkeypatch):
        """If the SIZ marker is unreadable, refuse to call ``jp2[:]``.

        Earlier the wrapper silently disabled the cap on
        ``Jp2k.shape``/``dtype`` failure, which would let an attacker
        bypass the bomb guard with a malformed-but-decodable
        codestream.  The current behaviour is fail-closed: raise
        ``ValueError`` before any pixel-decoding work runs.
        """
        import glymur
        arr = np.zeros((64, 64), dtype=np.uint8)
        tmp = tmp_path / "broken.j2k"
        glymur.Jp2k(str(tmp), data=arr)
        blob = tmp.read_bytes()

        from xrspatial.geotiff import _compression

        class _BrokenJp2k:
            def __init__(self, *a, **kw):
                pass

            @property
            def shape(self):
                raise RuntimeError("simulated SIZ marker corruption")

            def __getitem__(self, _):
                raise AssertionError(
                    "jp2[:] must not be called when shape is unreadable")

        monkeypatch.setattr(_compression, "_glymur",
                            type("M", (), {"Jp2k": _BrokenJp2k}))
        with pytest.raises(ValueError, match="unable to read declared"):
            _compression.jpeg2000_decompress(
                blob, width=64, height=64, samples=1,
                expected_size=arr.nbytes)
