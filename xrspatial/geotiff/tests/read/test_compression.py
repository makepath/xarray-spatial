"""Reader compression-codec coverage.

Consolidates:

* ``test_compression.py`` -- codec round-trip unit tests for the
  reader's decompression entry points (deflate, LZW, predictor encode /
  decode, the dispatcher).
* ``test_decompression_caps.py`` -- decompression-bomb defenses
  (security finding S1) across deflate, ZSTD, LZ4, PackBits, LERC,
  JPEG 2000, and JPEG.

Several classes here (``TestDeflate``, ``TestLZW``, ``TestPredictor``,
``TestDispatch``, the ``Test*Direct`` codec bomb classes, and the JPEG
SOF cap class) call the codec functions directly rather than going
through ``open_geotiff`` / ``read_to_array``. They live under ``read/``
because the reader is the only consumer of those decode entry points;
the writer side of the same codecs is exercised from PR 7's writer
cluster. Future maintainers scanning ``read/`` should treat these as
reader-internal codec coverage rather than end-to-end read paths.
"""
from __future__ import annotations

import importlib.util
import struct
import zlib

import numpy as np
import pytest

from xrspatial.geotiff._compression import (COMPRESSION_DEFLATE, COMPRESSION_LZW, COMPRESSION_NONE,
                                            compress, decompress, deflate_compress,
                                            deflate_decompress, lz4_decompress, lzw_compress,
                                            lzw_decompress, packbits_decompress, predictor_decode,
                                            predictor_encode, zstd_decompress)
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
_HAS_PILLOW = _module_available("PIL")


# ---------------------------------------------------------------------------
# Codec round-trips (formerly test_compression.py)
# ---------------------------------------------------------------------------


class TestDeflate:
    def test_round_trip(self):
        data = b'hello world! ' * 100
        compressed = deflate_compress(data)
        assert compressed != data
        assert deflate_decompress(compressed) == data

    def test_empty(self):
        compressed = deflate_compress(b'')
        assert deflate_decompress(compressed) == b''

    def test_binary_data(self):
        data = bytes(range(256)) * 10
        compressed = deflate_compress(data)
        assert deflate_decompress(compressed) == data


class TestLZW:
    def test_round_trip_simple(self):
        data = b'ABCABCABCABC'
        compressed = lzw_compress(data)
        decompressed = lzw_decompress(compressed, len(data))
        assert decompressed.tobytes() == data

    def test_round_trip_repetitive(self):
        data = b'\x00' * 1000
        compressed = lzw_compress(data)
        decompressed = lzw_decompress(compressed, len(data))
        assert decompressed.tobytes() == data

    def test_round_trip_sequential(self):
        data = bytes(range(256))
        compressed = lzw_compress(data)
        decompressed = lzw_decompress(compressed, len(data))
        assert decompressed.tobytes() == data

    def test_round_trip_random(self):
        rng = np.random.RandomState(42)
        data = bytes(rng.randint(0, 256, size=500, dtype=np.uint8))
        compressed = lzw_compress(data)
        decompressed = lzw_decompress(compressed, len(data))
        assert decompressed.tobytes() == data

    def test_round_trip_large(self):
        rng = np.random.RandomState(123)
        data = bytes(rng.randint(0, 256, size=10000, dtype=np.uint8))
        compressed = lzw_compress(data)
        decompressed = lzw_decompress(compressed, len(data))
        assert decompressed.tobytes() == data

    def test_empty(self):
        compressed = lzw_compress(b'')
        decompressed = lzw_decompress(compressed, 0)
        assert decompressed.tobytes() == b''


class TestPredictor:
    def test_round_trip_uint8(self):
        # 4x4 image, 1 byte per sample
        data = np.array([10, 20, 30, 40, 50, 60, 70, 80,
                         90, 100, 110, 120, 130, 140, 150, 160],
                        dtype=np.uint8)
        encoded = predictor_encode(data.copy(), 4, 4, 1)
        decoded = predictor_decode(encoded.copy(), 4, 4, 1)
        np.testing.assert_array_equal(decoded, data)

    def test_round_trip_float32(self):
        # 2x3 image, 4 bytes per sample
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        raw = np.frombuffer(arr.tobytes(), dtype=np.uint8).copy()
        encoded = predictor_encode(raw.copy(), 3, 2, 4)
        decoded = predictor_decode(encoded.copy(), 3, 2, 4)
        np.testing.assert_array_equal(decoded, raw)

    def test_predictor_encode_differences(self):
        # First pixel unchanged, rest are differences
        data = np.array([10, 20, 30, 40], dtype=np.uint8)
        encoded = predictor_encode(data.copy(), 4, 1, 1)
        assert encoded[0] == 10
        assert encoded[1] == 10  # 20 - 10
        assert encoded[2] == 10  # 30 - 20
        assert encoded[3] == 10  # 40 - 30


class TestDispatch:
    def test_none(self):
        data = b'hello'
        assert decompress(data, COMPRESSION_NONE).tobytes() == data
        assert compress(data, COMPRESSION_NONE) == data

    def test_deflate(self):
        data = b'test data ' * 50
        compressed = compress(data, COMPRESSION_DEFLATE)
        assert decompress(compressed, COMPRESSION_DEFLATE).tobytes() == data

    def test_lzw(self):
        data = b'ABCABC' * 20
        compressed = compress(data, COMPRESSION_LZW)
        decompressed = decompress(compressed, COMPRESSION_LZW, len(data))
        assert decompressed.tobytes() == data

    def test_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported compression"):
            decompress(b'', 99)
        with pytest.raises(ValueError, match="Unsupported compression"):
            compress(b'', 99)


# ---------------------------------------------------------------------------
# Decompression-bomb caps (formerly test_decompression_caps.py)
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
    ifd_size = 2 + 12 * num_entries + 4
    ifd_start = 8
    strip_offset = ifd_start + ifd_size

    patched = []
    for tag_id, typ, count, raw in tags:
        if tag_id == 273:
            raw = struct.pack(f'{bo}I', strip_offset)
        patched.append((tag_id, typ, count, raw))
    tags = patched

    out = bytearray()
    out += b'II'
    out += struct.pack(f'{bo}H', 42)
    out += struct.pack(f'{bo}I', ifd_start)
    out += struct.pack(f'{bo}H', num_entries)
    for tag_id, typ, count, raw in tags:
        out += struct.pack(f'{bo}HHI', tag_id, typ, count)
        out += raw.ljust(4, b'\x00')
    out += struct.pack(f'{bo}I', 0)
    out += strip_bytes
    return bytes(out)


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
        """Backward-compat: ``expected_size=0`` (default) disables the cap."""
        data = b'A' * (256 * 1024)
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
# End-to-end TIFF bomb tests (audit reproducer shape)
# ---------------------------------------------------------------------------

# 1024 x 1024 uint8 = 1 MiB declared image, cap is ~1.05 MiB. We feed a
# strip whose decoded size is 8 MiB. The cap is exceeded by ~7x, which is
# enough to prove the codec rejects rather than silently truncates, while
# keeping the test's host-side allocation small enough for any CI runner.
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
    assert len(strip) < _BOMB_BYTES // 4
    tiff = _build_tiff_with_strip(strip, compression=50004,
                                  width=_DECLARED_W, height=_DECLARED_H)
    path = tmp_path / "lz4_bomb.tif"
    path.write_bytes(tiff)
    with pytest.raises(ValueError, match="exceed"):
        read_to_array(str(path), allow_experimental_codecs=True)


def test_packbits_bomb_rejected(tmp_path):
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
    """The cap allows ~5% of legitimate codec metadata above expected size."""
    expected = 1000
    data = b'A' * (expected + 30)
    comp = zlib.compress(data, 9)
    out = deflate_decompress(comp, expected_size=expected)
    assert out == data


# ---------------------------------------------------------------------------
# LERC and JPEG 2000 codec-level bomb tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_LERC, reason="lerc not installed")
class TestLercDirect:
    def test_lerc_bomb_raises(self):
        """A LERC blob whose declared dimensions exceed the cap must raise."""
        import lerc
        arr = np.zeros((4096, 4096), dtype=np.float32)
        encoded = lerc.encode(arr, 1, False, None, 0.0, 1)
        blob = bytes(encoded[2])
        assert len(blob) < 1024
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
        out = lerc_decompress(blob)
        decoded = np.frombuffer(out, dtype=np.float32).reshape(128, 128)
        assert decoded.shape == arr.shape


@pytest.mark.skipif(not _HAS_GLYMUR, reason="glymur not installed")
class TestJpeg2000Direct:
    def test_jpeg2000_bomb_raises(self, tmp_path):
        """A JPEG 2000 codestream whose declared shape exceeds the cap raises."""
        import glymur

        arr = np.zeros((2000, 2000), dtype=np.uint8)
        tmp = tmp_path / "src.j2k"
        glymur.Jp2k(str(tmp), data=arr)
        blob = tmp.read_bytes()
        assert len(blob) < 10_000
        from xrspatial.geotiff._compression import jpeg2000_decompress
        with pytest.raises(ValueError, match="exceed"):
            jpeg2000_decompress(
                blob, width=2000, height=2000, samples=1,
                expected_size=1024)

    def test_jpeg2000_legitimate_passes(self, tmp_path):
        """A JPEG 2000 blob whose declared output matches expected_size passes."""
        import glymur

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
        """If the SIZ marker is unreadable, refuse to call ``jp2[:]``."""
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


# ---------------------------------------------------------------------------
# JPEG SOF cap
# ---------------------------------------------------------------------------


def _forge_jpeg_with_sof_dimensions(real_h: int, real_w: int,
                                    real_c: int,
                                    declared_h: int,
                                    declared_w: int) -> bytes:
    """Build a real JPEG and rewrite the SOF marker's H/W fields."""
    import io

    from PIL import Image
    mode = 'RGB' if real_c == 3 else 'L'
    img = Image.new(mode, (real_w, real_h), color=0)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=75)
    data = bytearray(buf.getvalue())
    sof_codes = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    i = 2
    n = len(data)
    while i + 3 < n:
        if data[i] != 0xFF:
            raise AssertionError("forged JPEG lost marker alignment")
        marker = data[i + 1]
        if marker in sof_codes:
            data[i + 5] = (declared_h >> 8) & 0xFF
            data[i + 6] = declared_h & 0xFF
            data[i + 7] = (declared_w >> 8) & 0xFF
            data[i + 8] = declared_w & 0xFF
            return bytes(data)
        if marker == 0xD9 or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            i += 2
            continue
        seg_len = (data[i + 2] << 8) | data[i + 3]
        i += 2 + seg_len
    raise AssertionError("forged JPEG has no SOF marker")


@pytest.mark.skipif(not _HAS_PILLOW, reason="Pillow not installed")
class TestJpegDirect:
    def test_jpeg_bomb_raises(self):
        """A JPEG whose SOF dimensions exceed the per-tile cap must raise."""
        blob = _forge_jpeg_with_sof_dimensions(
            real_h=16, real_w=16, real_c=3,
            declared_h=8000, declared_w=8000,
        )
        from xrspatial.geotiff._compression import jpeg_decompress

        with pytest.raises(
            ValueError,
            match=r"jpeg decode would exceed.*Likely a decompression bomb",
        ):
            jpeg_decompress(blob, width=32, height=32, samples=3)

    def test_jpeg_legitimate_passes(self):
        """A JPEG whose SOF dimensions match the expected tile size passes."""
        import io

        from PIL import Image
        img = Image.new('RGB', (32, 32), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=90)
        from xrspatial.geotiff._compression import jpeg_decompress
        out = jpeg_decompress(buf.getvalue(), width=32, height=32, samples=3)
        arr = np.frombuffer(out, dtype=np.uint8).reshape(32, 32, 3)
        assert arr.shape == (32, 32, 3)

    def test_jpeg_no_cap_when_size_kwargs_default(self):
        """Backward-compat: omitting size kwargs falls back to Pillow's guard."""
        blob = _forge_jpeg_with_sof_dimensions(
            real_h=16, real_w=16, real_c=3,
            declared_h=64, declared_w=64,
        )
        from PIL import Image as _Img  # noqa: F401

        from xrspatial.geotiff._compression import jpeg_decompress
        try:
            jpeg_decompress(blob)
        except ValueError as exc:
            assert "Likely a decompression bomb" not in str(exc)
        except Exception:
            pass

    def test_jpeg_malformed_falls_through_to_pillow(self):
        """A JPEG without a parseable SOF defers to Pillow's own guard."""
        blob = bytes([0xFF, 0xD8, 0xFF, 0xD9])
        from xrspatial.geotiff._compression import jpeg_decompress

        with pytest.raises(Exception):
            jpeg_decompress(blob, width=32, height=32, samples=3)

    def test_jpeg_sof_with_truncated_segment_length_returns_none(self):
        """A SOF segment whose declared length runs past EOF returns None."""
        from xrspatial.geotiff._compression import _read_jpeg_sof

        truncated = bytes([
            0xFF, 0xD8,                  # SOI
            0xFF, 0xC0,                  # SOF0
            0x00, 0x40,                  # seg_len = 64 (lies; buf is short)
            0x08,                        # precision = 8
            0x10, 0x00,                  # height = 4096
            0x10, 0x00,                  # width  = 4096
            0x03,                        # components = 3
        ])
        assert _read_jpeg_sof(truncated) is None

    def test_jpeg_sof_with_segment_length_below_minimum_returns_none(self):
        """A SOF segment whose declared length is < 8 returns None."""
        from xrspatial.geotiff._compression import _read_jpeg_sof

        too_small = bytes([
            0xFF, 0xD8,                  # SOI
            0xFF, 0xC0,                  # SOF0
            0x00, 0x04,                  # seg_len = 4 (too small for SOF)
            0x08,
            0x10, 0x00,
            0x10, 0x00,
            0x03,
        ])
        assert _read_jpeg_sof(too_small) is None
