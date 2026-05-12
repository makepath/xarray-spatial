"""Hypothesis property and fuzz tests for the geotiff module (#1661).

Three property groups:

1. Round-trip: random valid (dtype, compression, tiled, predictor, nodata) ->
   write with ``to_geotiff`` -> read with ``open_geotiff`` -> assert array
   equality and attrs preservation.

2. IFD layout permutations via ``make_minimal_tiff``: assert ``open_geotiff``
   returns a valid array, or raises ``ValueError`` / ``TypeError`` from the
   geotiff module. Never bare ``IndexError`` / ``struct.error`` /
   ``UnicodeDecodeError``.

3. Single-byte mutation: flip one byte in a valid TIFF at a Hypothesis-chosen
   offset. Reader must either parse consistently or raise a typed exception.

The whole file is skipped if ``hypothesis`` is not installed -- it is not a
hard test dep yet (see issue #1661 unresolved questions). Each test bounds
example count and disables Hypothesis's deadline so CI variance doesn't
flake.
"""
from __future__ import annotations

import io
import struct

import numpy as np
import pytest
import xarray as xr

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from xrspatial.geotiff import open_geotiff, to_geotiff  # noqa: E402

from .conftest import make_minimal_tiff  # noqa: E402


# Exception types the geotiff module is allowed to raise on invalid input.
# Any other exception class indicates an undocumented failure mode -- either
# the strategy generated something we should reject explicitly, or there's
# a real bug.
ALLOWED_PARSE_EXCEPTIONS = (ValueError, TypeError)

# Codecs safe for round-trip on every dtype in our strategy. 'jpeg' is
# explicitly rejected on write (see _VALID_COMPRESSIONS docstring); 'lerc' and
# 'jpeg2000' are lossy or dtype-restricted and would need their own narrower
# strategies, so they're omitted here.
LOSSLESS_CODECS = ['none', 'deflate', 'lzw', 'packbits', 'zstd', 'lz4']

# Dtype set kept small to keep CI fast. Float and int, signed and unsigned.
ROUND_TRIP_DTYPES = ['uint8', 'uint16', 'int16', 'int32', 'float32', 'float64']


# --- Strategies ---

@st.composite
def round_trip_inputs(draw):
    """Generate (DataArray, compression, tiled, predictor) for round-trip."""
    width = draw(st.integers(min_value=1, max_value=32))
    height = draw(st.integers(min_value=1, max_value=32))
    dtype = draw(st.sampled_from(ROUND_TRIP_DTYPES))
    compression = draw(st.sampled_from(LOSSLESS_CODECS))
    tiled = draw(st.booleans())

    np_dtype = np.dtype(dtype)
    if np_dtype.kind == 'f':
        # Predictor 3 is for floats only; 0/1 means no predictor.
        predictor = draw(st.sampled_from([False, 3]))
        data = draw(st.integers(min_value=0, max_value=1_000_000))
        rng = np.random.default_rng(data)
        arr = rng.standard_normal((height, width)).astype(np_dtype)
    else:
        # Predictor 2 is horizontal differencing, good for ints.
        predictor = draw(st.sampled_from([False, 2]))
        seed = draw(st.integers(min_value=0, max_value=1_000_000))
        rng = np.random.default_rng(seed)
        info = np.iinfo(np_dtype)
        # Avoid the extreme edge of the type range; some codecs reserve sentinels.
        arr = rng.integers(
            low=info.min // 2 if info.min < 0 else 0,
            high=info.max // 2,
            size=(height, width),
            dtype=np_dtype,
        )

    da = xr.DataArray(arr, dims=('y', 'x'))
    return da, compression, tiled, predictor


@st.composite
def ifd_layout_inputs(draw):
    """Generate a valid (or borderline) make_minimal_tiff invocation."""
    width = draw(st.integers(min_value=1, max_value=16))
    height = draw(st.integers(min_value=1, max_value=16))
    dtype = draw(st.sampled_from(['uint8', 'uint16', 'int16', 'float32']))
    compression = 1  # Uncompressed: make_minimal_tiff only supports type 1.
    tiled = draw(st.booleans())
    tile_size = draw(st.sampled_from([4, 8, 16]))
    big_endian = draw(st.booleans())
    with_geo = draw(st.booleans())

    return dict(
        width=width,
        height=height,
        dtype=np.dtype(dtype),
        compression=compression,
        tiled=tiled,
        tile_size=tile_size,
        big_endian=big_endian,
        with_geo=with_geo,
    )


# --- Group 1: round-trip property ---

@given(inputs=round_trip_inputs())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_round_trip_property(tmp_path_factory, inputs):
    """to_geotiff -> open_geotiff preserves array values bitwise."""
    da, compression, tiled, predictor = inputs

    tmp_dir = tmp_path_factory.mktemp("fuzz_1661_rt")
    path = str(tmp_dir / "rt.tif")

    to_geotiff(
        da,
        path,
        compression=compression,
        tiled=tiled,
        predictor=predictor,
    )

    got = open_geotiff(path, dtype=str(da.dtype))

    # Reader may add a leading band axis; squeeze for the 2D comparison.
    got_arr = got.values
    if got_arr.ndim == 3 and got_arr.shape[0] == 1:
        got_arr = got_arr[0]

    np.testing.assert_array_equal(got_arr, da.values)


# --- Group 2: IFD layout permutations ---

@given(spec=ifd_layout_inputs())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_ifd_layout_typed_errors_only(spec):
    """make_minimal_tiff variations parse cleanly or raise a typed exception.

    The reader is allowed to refuse any specific combination with a
    ValueError/TypeError; what is not allowed is a bare IndexError,
    struct.error, UnicodeDecodeError, or anything else that suggests we
    walked off the end of the byte buffer without checking.
    """
    geo_transform = None
    epsg = None
    if spec['with_geo']:
        geo_transform = (-120.0, 45.0, 0.001, -0.001)
        epsg = 4326

    tiff_bytes = make_minimal_tiff(
        width=spec['width'],
        height=spec['height'],
        dtype=spec['dtype'],
        compression=spec['compression'],
        tiled=spec['tiled'],
        tile_size=spec['tile_size'],
        big_endian=spec['big_endian'],
        geo_transform=geo_transform,
        epsg=epsg,
    )

    try:
        da = open_geotiff(io.BytesIO(tiff_bytes))
    except ALLOWED_PARSE_EXCEPTIONS:
        return  # Typed refusal -- acceptable.
    except Exception as exc:
        pytest.fail(
            f"open_geotiff raised non-typed {type(exc).__name__} on a "
            f"valid-by-construction TIFF: {spec!r} -> {exc!r}"
        )

    # If it parsed, shape should match what we asked for. Reader may add a
    # leading band axis (samples=1), so check the last two dims.
    assert da.shape[-2:] == (spec['height'], spec['width']), (
        f"shape mismatch: got {da.shape}, expected last dims "
        f"({spec['height']}, {spec['width']}) for {spec!r}"
    )


# --- Group 3: byte-level mutation fuzz ---

# Hold a single corpus TIFF and let Hypothesis pick a byte offset + new byte
# value to splice in. Using a fixed corpus keeps the strategy fast (no
# nested TIFF generation per example) and concentrates the search on the
# parser's response to bit-rot.
_CORPUS_SPECS = [
    # (kwargs to make_minimal_tiff, label)
    (dict(width=4, height=4, dtype=np.dtype('float32')), 'le_strip_f32'),
    (dict(width=4, height=4, dtype=np.dtype('uint16'), big_endian=True), 'be_strip_u16'),
    (dict(width=8, height=8, dtype=np.dtype('float32'), tiled=True, tile_size=4),
     'le_tiled_f32'),
    (dict(width=4, height=4, dtype=np.dtype('float32'),
          geo_transform=(-120.0, 45.0, 0.001, -0.001), epsg=4326),
     'le_geo_f32'),
]
_CORPUS = [(label, make_minimal_tiff(**kw)) for kw, label in _CORPUS_SPECS]


@pytest.mark.parametrize("label,base_tiff", _CORPUS, ids=[lab for lab, _ in _CORPUS])
# Regression seeds for bugs surfaced by the initial Hypothesis run on
# the le_strip_f32 corpus member (4x4 float32, 198 bytes total):
#   offset 102, byte 0x00 -> ZeroDivisionError in _read_strips (rps=0)
#   offset 110, byte 0x00 -> IndexError in _read_strips (StripByteCounts trunc)
#   offset 122, byte 0x00 -> IndexError in sample_format (empty tuple)
# These offsets are specific to the le_strip_f32 layout; the other corpus
# entries will exercise the same code with different offsets, and that's
# fine -- the example just guarantees we cover the regression each run.
@example(offset_frac=102 / 198, new_byte=0x00)
@example(offset_frac=110 / 198, new_byte=0x00)
@example(offset_frac=122 / 198, new_byte=0x00)
@given(
    offset_frac=st.floats(min_value=0.0, max_value=0.999),
    new_byte=st.integers(min_value=0, max_value=255),
)
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_single_byte_mutation_typed_errors(label, base_tiff, offset_frac, new_byte):
    """Flip one byte of a valid TIFF; reader must parse or raise typed exc.

    The mutated file might still parse (the byte landed in pixel data, which
    is a valid value for that dtype). What is unacceptable is a bare
    ``IndexError`` / ``struct.error`` from reading past the buffer, or a
    segfault from the GPU/dask paths -- those are kept off this test by
    using the eager numpy path only.
    """
    mutated = bytearray(base_tiff)
    offset = int(offset_frac * len(mutated))
    # Make sure the mutation is actually a flip (not a no-op).
    if mutated[offset] == new_byte:
        new_byte = (new_byte + 1) & 0xFF
    mutated[offset] = new_byte

    try:
        da = open_geotiff(io.BytesIO(bytes(mutated)))
    except ALLOWED_PARSE_EXCEPTIONS:
        return
    except (MemoryError, OverflowError):
        # Header field could decode to an absurd dimension/offset. We treat
        # these as acceptable refusals because the user gets a clear failure
        # rather than wrong data.
        return
    except Exception as exc:
        pytest.fail(
            f"[{label}] single-byte mutation at offset {offset} -> {new_byte:#x} "
            f"raised non-typed {type(exc).__name__}: {exc!r}"
        )

    # If it parsed, the result must at least be a real DataArray with the
    # claimed dtype actually realised. Materialise to catch lazy errors.
    assert isinstance(da, xr.DataArray)
    _ = np.asarray(da.values)


# --- Smoke test that the module wired itself up ---

def test_corpus_baseline_parses():
    """Sanity check: every corpus TIFF parses without mutation."""
    for label, base in _CORPUS:
        da = open_geotiff(io.BytesIO(base))
        assert isinstance(da, xr.DataArray), label
        assert da.size > 0, label


# --- Targeted regressions for bugs found by the property tests above ---
# These three were caught by the byte-mutation property on first run and
# fixed alongside this PR. They live here (not in a separate file) so the
# regression context stays next to the harness that found them.

def test_regression_rows_per_strip_zero_is_typed_error():
    """rps=0 must raise ValueError, not ZeroDivisionError."""
    base = make_minimal_tiff(4, 4, np.dtype('float32'))
    mut = bytearray(base)
    mut[102] = 0  # Zeroes the RowsPerStrip value in this layout.
    with pytest.raises(ValueError):
        open_geotiff(io.BytesIO(bytes(mut)))


def test_regression_strip_table_truncated_is_typed_error():
    """StripByteCounts shorter than strip count must raise ValueError."""
    base = make_minimal_tiff(4, 4, np.dtype('float32'))
    mut = bytearray(base)
    mut[110] = 0  # Truncates the strip table count in this layout.
    with pytest.raises(ValueError):
        open_geotiff(io.BytesIO(bytes(mut)))


def test_regression_empty_sample_format_tuple_does_not_indexerror():
    """SampleFormat tag with count=0 must fall back, not IndexError."""
    base = make_minimal_tiff(4, 4, np.dtype('float32'))
    mut = bytearray(base)
    mut[122] = 0  # Zeroes the SampleFormat count field in this layout.
    # Either parses with the default sample_format (1 = unsigned int) and
    # produces a DataArray, or fails downstream with a typed ValueError --
    # both are acceptable. The non-acceptable outcome is IndexError.
    try:
        da = open_geotiff(io.BytesIO(bytes(mut)))
        assert isinstance(da, xr.DataArray)
    except ValueError:
        pass


# --- Group 4: truncation fuzz on the IFD entry table (#1672) ---
#
# The byte-mutation fuzz in Group 3 flips one byte; it doesn't cover the
# case where the file is truncated before `parse_ifd` has even finished
# reading the entry table. The S2 typed-error work in #1661 fixed the
# *value area* of each entry; the entry table itself (num_entries,
# tag/type/count fields, next-IFD pointer) was still unguarded and
# escaped with `struct.error` on truncation.

from xrspatial.geotiff._header import parse_all_ifds, parse_header  # noqa: E402


# Reuse the same corpus from Group 3; truncating each one covers
# little- and big-endian, strip and tile, with and without geo tags.
@pytest.mark.parametrize(
    "label,base_tiff", _CORPUS, ids=[lab for lab, _ in _CORPUS]
)
@given(offset_frac=st.floats(min_value=0.0, max_value=1.0))
@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_truncation_typed_errors_only(label, base_tiff, offset_frac):
    """Truncating a valid TIFF at any byte must raise a typed error.

    For every truncation point the parser must either succeed (the cut
    landed past the IFD chain), raise ValueError / TypeError, or raise
    one of the documented memory/overflow refusals. `struct.error` from
    `unpack_from` walking off the buffer is the failure mode we are
    guarding against.
    """
    cut = int(offset_frac * len(base_tiff))
    truncated = base_tiff[:cut]

    try:
        header = parse_header(truncated)
        parse_all_ifds(truncated, header)
    except ALLOWED_PARSE_EXCEPTIONS:
        return
    except (MemoryError, OverflowError):
        return
    except struct.error as exc:
        pytest.fail(
            f"[{label}] truncation to {cut} bytes (of {len(base_tiff)}) "
            f"raised struct.error: {exc!r}"
        )
    except Exception as exc:
        pytest.fail(
            f"[{label}] truncation to {cut} bytes (of {len(base_tiff)}) "
            f"raised non-typed {type(exc).__name__}: {exc!r}"
        )


# Targeted examples for each of the three #1672 bounds checks. These
# pin the regression for the exact offsets the fuzz sweeps over and
# give a fast-fail signal if any check is reverted.

@example(byte_count=8)    # Cuts before num_entries.
@example(byte_count=9)    # Splits the 2-byte num_entries field.
@example(byte_count=20)   # Splits the first entry's tag/type/count.
@given(byte_count=st.integers(min_value=0, max_value=400))
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_truncation_in_entry_table_is_valueerror(byte_count):
    """Every truncation inside the IFD entry table raises a typed error."""
    base = make_minimal_tiff(4, 4, np.dtype('float32'))
    truncated = base[:byte_count]
    try:
        header = parse_header(truncated)
        parse_all_ifds(truncated, header)
    except ALLOWED_PARSE_EXCEPTIONS:
        return
    except (MemoryError, OverflowError):
        return
    except struct.error as exc:
        pytest.fail(
            f"truncation to {byte_count} bytes raised struct.error: {exc!r}"
        )
    except Exception as exc:
        pytest.fail(
            f"truncation to {byte_count} bytes raised non-typed "
            f"{type(exc).__name__}: {exc!r}"
        )
