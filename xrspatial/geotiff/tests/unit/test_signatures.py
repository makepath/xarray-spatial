"""Public API signature, annotation, and kwarg-behaviour contract.

Single home for "does the public ``xrspatial.geotiff`` surface still
expose the right kwargs, in the right order, with the right annotations,
and do those kwargs still do what they say." Six sections:

Section 1 -- Parameter annotations
    Reader and writer entry points must annotate ``window``, ``path`` /
    ``source``, ``dtype``, ``on_gpu_failure``, ``nodata``, and
    ``streaming_buffer_bytes`` consistently across siblings. A few
    runtime smoke tests confirm the annotations did not break the call
    semantics they describe.

Section 2 -- Canonical reader kwarg order
    ``open_geotiff`` is the canonical surface; the three backend readers
    list their shared keyword-only params in the same relative order so
    ``inspect.signature``, IDE autocomplete, and Sphinx docs do not
    drift.

Section 3 -- Experimental / internal-only opt-in gates
    Read-side codec gate (LERC / JPEG2000 / LZ4 / JPEG-in-TIFF) and
    writer rich-tag gate (``gdal_metadata_xml`` / ``extra_tags``) each
    require the matching opt-in flag. The flags are pinned on every
    public entry point and the validators are unit-tested directly.

Section 4 -- ``photometric`` kwarg and ``extra_tags`` override
    The writer defaults to MinIsBlack for any band count; RGB / RGBA are
    opt-in. A user-supplied Photometric / ExtraSamples ``extra_tags``
    entry wins over the writer's auto value.

Section 5 -- ``gil_friendly`` deflate kwarg
    The flag forces the deflate path through stdlib ``zlib`` (GIL-
    releasing) instead of the libdeflate binding. Tests cover the codec
    layer, the dispatcher, and every writer call site so a dropped kwarg
    cannot silently regress thread-pool scaling.

Section 6 -- Reader / writer kwarg behaviour
    Override-effect and dtype-cast coverage for kwargs that the
    signature pins above only assert as *accepted*: ``read_geotiff_gpu``
    / ``read_geotiff_dask`` ``name`` and ``max_pixels``, ``write_vrt``
    ``relative`` / ``crs`` / ``nodata``, GPU reader ``dtype``, GPU writer
    ``bigtiff`` / ``predictor``, and ``read_vrt`` ``window``.

The sections share a *concern* (the public API contract) rather than
runtime logic. GPU rows skip when cupy + CUDA are absent via the shared
``requires_gpu`` marker; libdeflate-specific rows skip when the optional
``deflate`` binding is missing.
"""
from __future__ import annotations

import importlib.util
import inspect
import io
import os
import struct
import warnings
import zlib

import numpy as np
import pytest
import xarray as xr

import xrspatial.geotiff as g
import xrspatial.geotiff._compression as comp_mod
from xrspatial.geotiff import (GeoTIFFFallbackWarning, _geotiff_strict_mode, _wkt_to_epsg,
                               open_geotiff, read_geotiff_dask, read_geotiff_gpu, read_vrt,
                               to_geotiff, write_geotiff_gpu, write_vrt)
from xrspatial.geotiff._attrs import (_COMPRESSION_TAG_TO_NAME, _validate_read_codec_optin,
                                      _validate_write_rich_tag_optin)
from xrspatial.geotiff._compression import (_HAVE_LIBDEFLATE, COMPRESSION_DEFLATE, COMPRESSION_LZ4,
                                            COMPRESSION_LZW, COMPRESSION_NONE, COMPRESSION_PACKBITS,
                                            COMPRESSION_ZSTD, LZ4_AVAILABLE, compress,
                                            deflate_compress)
from xrspatial.geotiff._dtypes import SHORT
from xrspatial.geotiff._geotags import _epsg_to_wkt
from xrspatial.geotiff._header import TAG_EXTRA_SAMPLES, TAG_PHOTOMETRIC, parse_header, parse_ifd
from xrspatial.geotiff._reader import read_to_array
from xrspatial.geotiff._vrt import parse_vrt
from xrspatial.geotiff._vrt import write_vrt as _write_vrt_internal
from xrspatial.geotiff._writer import (_PARALLEL_MIN_BYTES, _compress_block, _prepare_strip,
                                       _prepare_tile, _write_stripped, _write_tiled, write)

from .._helpers.markers import requires_gpu

# ===========================================================================
# Section 1 -- Parameter annotations
# ===========================================================================
#
# The public surface had annotation drift: the same parameter was
# annotated on some sibling
# functions but bare ``=None`` on others. Each annotation is pinned here so
# a future signature change cannot silently drop it. ``from __future__
# import annotations`` keeps annotations as strings at runtime, so the
# comparisons match the source literal.


def _annotation(fn, param_name):
    """Return the stringified annotation for ``fn``'s ``param_name``."""
    sig = inspect.signature(fn)
    p = sig.parameters[param_name]
    assert p.annotation is not inspect.Parameter.empty, (
        f"{fn.__name__}({param_name}=...) is missing a type annotation"
    )
    return str(p.annotation)


# --- window: 4-tuple (r0, c0, r1, c1) or None ---


def test_open_geotiff_window_annotated():
    assert _annotation(open_geotiff, 'window') == 'tuple | None'


def test_read_vrt_window_annotated():
    assert _annotation(read_vrt, 'window') == 'tuple | None'


def test_read_geotiff_dask_window_annotated():
    """Pre-existing annotation -- keep it pinned so it does not regress."""
    assert _annotation(read_geotiff_dask, 'window') == 'tuple | None'


def test_read_geotiff_gpu_window_annotated():
    """Pre-existing annotation -- keep it pinned so it does not regress."""
    assert _annotation(read_geotiff_gpu, 'window') == 'tuple | None'


# --- path: str or binary file-like (writer entry points) ---


def test_to_geotiff_path_annotated():
    """``to_geotiff(data, path, ...)`` ``path`` accepts str or BinaryIO."""
    ann = _annotation(to_geotiff, 'path')
    assert 'str' in ann
    assert 'BinaryIO' in ann


def test_write_geotiff_gpu_path_annotated():
    """``write_geotiff_gpu(data, path, ...)`` ``path`` mirrors ``to_geotiff``."""
    ann = _annotation(write_geotiff_gpu, 'path')
    assert 'str' in ann
    assert 'BinaryIO' in ann


def test_write_vrt_path_annotated():
    """``write_vrt(path, ...)`` is str-only (VRT writes are path-only by
    design; no file-like buffer support). The canonical name
    is ``path`` (parity with ``to_geotiff`` / ``write_geotiff_gpu``).
    The annotation is plain ``str``: the default value is a private
    sentinel (not ``None``) so the deprecation shim can distinguish
    ``write_vrt(path=None, ...)`` (rejected with TypeError) from a
    caller who omitted ``path`` entirely (routed through the ``vrt_path``
    alias)."""
    assert _annotation(write_vrt, 'path') == 'str'


def test_write_vrt_vrt_path_annotated():
    """The deprecated ``vrt_path`` alias keeps the same ``str | None``
    annotation as ``path`` (str-only at the type level; ``None`` only
    appears because the sentinel default lets the shim detect omission).
    Pinned so a future re-rename does not silently widen the alias."""
    assert _annotation(write_vrt, 'vrt_path') == 'str | None'


# --- source: str or BinaryIO (open_geotiff is the public dispatch) ---


def test_open_geotiff_source_annotated():
    """``open_geotiff(source, ...)`` accepts ``str | BinaryIO`` to match
    the writer ``path`` annotation and the runtime behaviour the
    docstring documents (BytesIO buffers are routed through the eager
    numpy reader). The dedicated reader entry points stay ``str``-only
    because they reject file-like sources at runtime.
    """
    ann = _annotation(open_geotiff, 'source')
    assert 'str' in ann
    assert 'BinaryIO' in ann


def test_read_geotiff_dask_source_str_only():
    """``read_geotiff_dask(source: str)`` stays str-only: the dask path
    reopens the source by path from each worker task and does not
    support file-like buffers."""
    assert _annotation(read_geotiff_dask, 'source') == 'str'


def test_read_geotiff_gpu_source_str_only():
    """``read_geotiff_gpu(source: str)`` stays str-only: GPU decode
    paths read by path / mmap and do not support file-like buffers."""
    assert _annotation(read_geotiff_gpu, 'source') == 'str'


def test_read_vrt_source_str_only():
    """``read_vrt(source: str)`` stays str-only: the VRT XML references
    its own source files on disk."""
    assert _annotation(read_vrt, 'source') == 'str'


# --- dtype: str | np.dtype | None on every reader entry point ---


def test_open_geotiff_dtype_annotated():
    """``open_geotiff(dtype=...)`` accepts ``str | np.dtype | None``. The
    docstring already documents the accepted-type set; the annotation
    now matches."""
    assert _annotation(open_geotiff, 'dtype') == 'str | np.dtype | None'


def test_read_geotiff_dask_dtype_annotated():
    assert _annotation(read_geotiff_dask, 'dtype') == 'str | np.dtype | None'


def test_read_geotiff_gpu_dtype_annotated():
    assert _annotation(read_geotiff_gpu, 'dtype') == 'str | np.dtype | None'


def test_read_vrt_dtype_annotated():
    assert _annotation(read_vrt, 'dtype') == 'str | np.dtype | None'


# --- on_gpu_failure: 'auto' | 'strict' (GPU failure policy) ---


def test_open_geotiff_on_gpu_failure_annotated():
    assert _annotation(open_geotiff, 'on_gpu_failure') == 'str'


def test_read_geotiff_gpu_on_gpu_failure_annotated():
    assert _annotation(read_geotiff_gpu, 'on_gpu_failure') == 'str'


def test_read_geotiff_gpu_deprecated_gpu_alias_annotated():
    """The deprecated ``gpu=`` alias on ``read_geotiff_gpu`` carries the
    same ``str`` annotation as the new ``on_gpu_failure`` kwarg."""
    assert _annotation(read_geotiff_gpu, 'gpu') == 'str'


# --- nodata: float | int | None on every writer entry point ---


def test_to_geotiff_nodata_annotated():
    assert _annotation(to_geotiff, 'nodata') == 'float | int | None'


def test_write_geotiff_gpu_nodata_annotated():
    assert _annotation(write_geotiff_gpu, 'nodata') == 'float | int | None'


def test_write_vrt_nodata_annotated():
    """Pre-existing annotation -- keep it pinned."""
    assert _annotation(write_vrt, 'nodata') == 'float | int | None'


# --- streaming_buffer_bytes: int on both writer entry points ---


def test_to_geotiff_streaming_buffer_bytes_annotated():
    """Pre-existing -- ``int`` with a 256 MB default."""
    assert _annotation(to_geotiff, 'streaming_buffer_bytes') == 'int'
    assert (
        inspect.signature(to_geotiff)
        .parameters['streaming_buffer_bytes']
        .default
        == 256 * 1024 * 1024
    )


def test_write_geotiff_gpu_streaming_buffer_bytes_annotated():
    """GPU writer must agree with ``to_geotiff`` on type and default so a
    caller forwarding the same kwargs to either entry point sees the same
    hint. The kwarg is a runtime no-op on the GPU writer (deleted on
    entry); the annotation parity is the only consistency dimension."""
    assert _annotation(
        write_geotiff_gpu, 'streaming_buffer_bytes'
    ) == 'int'
    assert (
        inspect.signature(write_geotiff_gpu)
        .parameters['streaming_buffer_bytes']
        .default
        == 256 * 1024 * 1024
    )


# --- Smoke: the annotations did not break runtime call semantics ---


def _annotated_smoke_da():
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    return xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'transform': (1.0, 0, 0.0, 0, -1.0, 8.0)},
    )


def test_open_geotiff_window_kwarg_runtime(tmp_path):
    """The annotated ``window`` kwarg still accepts a 4-tuple and returns
    the requested sub-window. The test does not exercise ``on_gpu_failure``
    because the runtime semantics are GPU-only; the annotation itself is
    pinned by ``test_open_geotiff_on_gpu_failure_annotated``.
    """
    da = _annotated_smoke_da()
    path = str(tmp_path / 'window_kwarg.tif')
    to_geotiff(da, path)
    r = open_geotiff(path, window=(0, 0, 4, 4))
    assert r.shape == (4, 4)


def test_open_geotiff_bytesio_source_runtime(tmp_path):
    """``open_geotiff`` routes a ``BytesIO`` source through the eager
    numpy reader. The annotation pins this contract at the type level;
    this test pins it at the runtime level so a future refactor that
    drops the file-like branch fails CI.
    """
    da = _annotated_smoke_da()
    path = str(tmp_path / 'bytesio_source.tif')
    to_geotiff(da, path)
    with open(path, 'rb') as f:
        buffer = io.BytesIO(f.read())

    r = open_geotiff(buffer)
    assert r.shape == (8, 8)
    assert r.dtype == np.float32


def test_open_geotiff_dtype_kwarg_runtime(tmp_path):
    """``open_geotiff(dtype=...)`` still accepts both a ``str`` token and a
    ``np.dtype`` instance after the annotation tightens to
    ``str | np.dtype | None``. The annotation pins the contract at the
    type level; this test pins it at the runtime level so the contract
    cannot regress without failing CI.
    """
    da = _annotated_smoke_da()
    path = str(tmp_path / 'dtype_kwarg.tif')
    to_geotiff(da, path)

    r_str = open_geotiff(path, dtype='float64')
    assert r_str.dtype == np.float64

    r_dtype = open_geotiff(path, dtype=np.dtype('float64'))
    assert r_dtype.dtype == np.float64

    r_none = open_geotiff(path, dtype=None)
    assert r_none.dtype == np.float32


def test_to_geotiff_nodata_int_runtime(tmp_path):
    """``nodata=<int>`` still round-trips through ``to_geotiff`` and the
    sentinel survives into the read-back attrs."""
    arr = np.full((8, 8), -9999, dtype=np.int32)
    arr[2:6, 2:6] = 42
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'transform': (1.0, 0, 0.0, 0, -1.0, 8.0)},
    )
    path = str(tmp_path / 'nodata_int.tif')
    to_geotiff(da, path, nodata=-9999)
    r = open_geotiff(path)
    assert r.attrs.get('nodata') == -9999


@requires_gpu
def test_write_geotiff_gpu_streaming_buffer_bytes_runtime_noop(tmp_path):
    """Passing an explicit ``streaming_buffer_bytes`` to the GPU writer
    must remain a no-op. The body still does ``del streaming_buffer_bytes``
    so the value has no effect on the produced file."""
    import cupy

    arr_cpu = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    arr_gpu = cupy.asarray(arr_cpu)
    da_gpu = xr.DataArray(
        arr_gpu, dims=['y', 'x'],
        coords={'y': np.arange(64.0, 0, -1), 'x': np.arange(64.0)},
        attrs={'crs': 4326, 'transform': (1.0, 0, 0.0, 0, -1.0, 64.0)},
    )
    p1 = str(tmp_path / 'default.tif')
    p2 = str(tmp_path / 'override.tif')
    write_geotiff_gpu(da_gpu, p1)
    write_geotiff_gpu(da_gpu, p2, streaming_buffer_bytes=8 * 1024 * 1024)
    # Both files have identical sizes -- the buffer kwarg is a no-op.
    assert os.path.getsize(p1) == os.path.getsize(p2)


# ===========================================================================
# Section 2 -- Canonical reader kwarg order
# ===========================================================================
#
# ``open_geotiff`` is the canonical surface. The three backend readers
# (``read_geotiff_gpu``, ``read_geotiff_dask``, ``read_vrt``) must list the
# shared kwargs in the same relative order so ``inspect.signature``, IDE
# autocomplete, and Sphinx-rendered docs do not drift. Each per-backend
# signature carries its own subset of the canonical parameter list;
# backend-specific extras are checked at the tail.

# Canonical order taken from ``open_geotiff``'s public signature.
_CANONICAL_ORDER = (
    "dtype",
    "window",
    # Geographic-space window. Sits immediately after ``window`` so the
    # two sub-region kwargs stay grouped; mutual exclusion is enforced
    # in :func:`open_geotiff`.
    "bbox",
    "overview_level",
    "band",
    "name",
    "chunks",
    "gpu",
    "max_pixels",
    "max_cloud_bytes",
    "on_gpu_failure",
    "missing_sources",
    "allow_rotated",
    "allow_unparseable_crs",
    # Integer-nodata fail-closed opt-out. Sits alongside the other
    # ambiguous-metadata opt-outs so the canonical order keeps the
    # typed-error gates grouped.
    "allow_invalid_nodata",
    # Stable-tier-only read-side gate. Sits alongside the other
    # ambiguous-metadata opt-outs and immediately before the
    # experimental-codec unlock it pairs with in the rejection message,
    # so the canonical order tracks the release-contract grouping.
    "stable_only",
    # Experimental / internal-only codec opt-ins on the read side,
    # mirroring the writer surface. They sit after the other ``allow_*``
    # flags so the canonical order keeps the policy / typed-error gates
    # grouped.
    "allow_experimental_codecs",
    "allow_internal_only_jpeg",
    "band_nodata",
    "mask_nodata",
)


def _kwonly_params(fn):
    """Return the keyword-only parameter names of *fn* in declaration order."""
    sig = inspect.signature(fn)
    return [
        name
        for name, param in sig.parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    ]


def _assert_canonical(fn, allowed_tail=()):
    """Assert *fn*'s kw-only params follow the canonical order.

    Parameters that appear in ``_CANONICAL_ORDER`` must show up in the
    same relative order. Extras (e.g. the deprecated ``gpu`` alias on
    ``read_geotiff_gpu``) are accepted at the tail when listed in
    ``allowed_tail`` and otherwise rejected so new kwargs cannot be
    quietly added in arbitrary positions.
    """
    params = _kwonly_params(fn)
    canonical = [p for p in params if p in _CANONICAL_ORDER]
    expected = [p for p in _CANONICAL_ORDER if p in canonical]
    assert canonical == expected, (
        f"{fn.__name__} kwarg order {canonical!r} does not match the "
        f"canonical subset {expected!r}"
    )
    tail = [p for p in params if p not in _CANONICAL_ORDER]
    unexpected = set(tail) - set(allowed_tail)
    assert not unexpected, (
        f"{fn.__name__} has unexpected kw-only params {sorted(unexpected)!r}; "
        f"add them to _CANONICAL_ORDER or to the test's allowed_tail."
    )


def test_open_geotiff_defines_canonical_order():
    """``open_geotiff``'s signature is the canonical reference."""
    params = _kwonly_params(open_geotiff)
    expected = list(_CANONICAL_ORDER)
    assert params == expected, (
        f"open_geotiff kw-only params {params!r} no longer match the "
        f"canonical order {expected!r}. Update both the function and the "
        f"_CANONICAL_ORDER constant together."
    )


def test_read_geotiff_gpu_matches_canonical_order():
    """``read_geotiff_gpu`` must list shared params in the canonical order."""
    # ``gpu`` here is the deprecated alias for ``on_gpu_failure`` (see
    # ``read_geotiff_gpu``'s docstring). It is not the boolean backend
    # selector that lives on ``open_geotiff`` / ``read_vrt``, so it sits
    # at the tail rather than in its canonical-order slot.
    params = _kwonly_params(read_geotiff_gpu)
    # ``gpu`` is the deprecated alias, intentionally last.
    assert params[-1] == "gpu", (
        f"read_geotiff_gpu must keep the deprecated 'gpu' alias as the last "
        f"kwarg; got {params!r}"
    )
    # Drop the alias and run the canonical-subset check on the rest.
    head = params[:-1]
    canonical_head = [p for p in _CANONICAL_ORDER if p in head]
    assert head == canonical_head, (
        f"read_geotiff_gpu kwarg order {head!r} does not match the canonical "
        f"subset {canonical_head!r}"
    )


def test_read_geotiff_dask_matches_canonical_order():
    """``read_geotiff_dask`` must list shared params in the canonical order."""
    _assert_canonical(read_geotiff_dask)


def test_read_vrt_matches_canonical_order():
    """``read_vrt`` must list shared params in the canonical order.

    ``band_nodata`` is the opt-out for the mixed-band metadata
    check; it is VRT-specific (no analogue on the other readers) and so
    lives in the per-function tail rather than in the shared canonical
    order.
    """
    _assert_canonical(read_vrt, allowed_tail=('band_nodata',))


def test_no_pairwise_order_inversions():
    """For any pair of params shared by two readers, the order is consistent.

    ``read_geotiff_gpu``'s ``gpu`` kwarg is a deprecated alias for
    ``on_gpu_failure`` rather than the boolean backend selector that
    ``open_geotiff`` / ``read_vrt`` expose, so it is excluded from the
    cross-reader pair check.
    """
    readers = (open_geotiff, read_geotiff_gpu, read_geotiff_dask, read_vrt)
    orders = {}
    for fn in readers:
        params = _kwonly_params(fn)
        if fn is read_geotiff_gpu:
            # Drop the deprecated alias before cross-comparing with the other
            # readers' boolean ``gpu`` kwarg (different meaning, same name).
            params = [p for p in params if p != "gpu"]
        orders[fn.__name__] = params
    canonical_pairs = []
    for i, a in enumerate(_CANONICAL_ORDER):
        for b in _CANONICAL_ORDER[i + 1:]:
            canonical_pairs.append((a, b))
    for name, params in orders.items():
        for a, b in canonical_pairs:
            if a in params and b in params:
                assert params.index(a) < params.index(b), (
                    f"{name}: {a!r} must appear before {b!r}; got "
                    f"{params!r}"
                )


# ===========================================================================
# Section 3 -- Experimental / internal-only opt-in gates
# ===========================================================================
#
# The GeoTIFF release contract tiers features into Stable / Advanced /
# Experimental / Internal-only. The writer-side opt-in shape covers every
# Experimental / Internal-only path and mirrors the read-side codec gate.
# Each rejection message names the missing flag, the feature, and the tier
# so a call site can be fixed in one line.


def _make_float32_da(h: int = 32, w: int = 32) -> xr.DataArray:
    """Small float32 raster used for the write-side gate."""
    rng = np.random.RandomState(0)
    arr = rng.standard_normal((h, w)).astype(np.float32)
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={
            "y": np.arange(h, dtype=np.float64),
            "x": np.arange(w, dtype=np.float64),
        },
        attrs={'crs': 4326},
    )


def _write_test_tif(tmp_path, compression: str,
                    *, allow_experimental_codecs=False,
                    allow_internal_only_jpeg=False,
                    dtype=np.float32):
    """Write a small file with the requested codec so the read side has
    a real target. Returns the file path. Skips when the optional
    encoder dependency is missing."""
    h = w = 32
    rng = np.random.RandomState(0)
    if dtype == np.uint8:
        arr = rng.randint(0, 256, size=(h, w), dtype=np.uint8)
    else:
        arr = rng.standard_normal((h, w)).astype(dtype)
    da = xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={
            "y": np.arange(h, dtype=np.float64),
            "x": np.arange(w, dtype=np.float64),
        },
        attrs={'crs': 4326},
    )
    path = os.path.join(str(tmp_path), f'src_{compression}.tif')
    try:
        to_geotiff(
            da, path, compression=compression,
            allow_experimental_codecs=allow_experimental_codecs,
            allow_internal_only_jpeg=allow_internal_only_jpeg,
        )
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"optional encoder missing for {compression}: {e}")
    return path


# --- Signature: every public read entry point exposes the new flags ---


@pytest.mark.parametrize(
    "fn", [open_geotiff, read_geotiff_dask, read_geotiff_gpu])
def test_read_signature_has_codec_optin(fn):
    """``open_geotiff`` / ``read_geotiff_dask`` / ``read_geotiff_gpu``
    expose ``allow_experimental_codecs=False`` and
    ``allow_internal_only_jpeg=False``. The default is ``False`` so
    accidental removal of the gate would surface here.
    """
    params = inspect.signature(fn).parameters
    assert 'allow_experimental_codecs' in params, fn.__name__
    assert params['allow_experimental_codecs'].default is False
    assert 'allow_internal_only_jpeg' in params, fn.__name__
    assert params['allow_internal_only_jpeg'].default is False


# --- Validator unit tests: codec + rich-tag surfaces, no disk IO ---


def test_validate_read_codec_optin_accepts_stable_codecs():
    """A stable codec (deflate / none / lzw / zstd / packbits) does not
    require any opt-in regardless of the flag values.
    """
    for tag in (1, 5, 8, 32773, 50000):  # none, lzw, deflate, packbits, zstd
        _validate_read_codec_optin(
            tag,
            allow_experimental_codecs=False,
            allow_internal_only_jpeg=False,
        )


@pytest.mark.parametrize("codec_name", ['lerc', 'jpeg2000', 'lz4'])
def test_validate_read_codec_optin_rejects_experimental(codec_name):
    """LERC / JPEG2000 / LZ4 raise ``ValueError`` whose message names
    ``allow_experimental_codecs`` so the caller can find the flag from
    the error itself.
    """
    tag = {
        v: k for k, v in _COMPRESSION_TAG_TO_NAME.items()
    }[codec_name]
    with pytest.raises(ValueError, match='allow_experimental_codecs'):
        _validate_read_codec_optin(
            tag,
            allow_experimental_codecs=False,
            allow_internal_only_jpeg=False,
        )


def test_validate_read_codec_optin_rejects_jpeg():
    """JPEG-in-TIFF raises ``ValueError`` whose message names
    ``allow_internal_only_jpeg`` -- the dedicated flag, NOT
    ``allow_experimental_codecs``. The two flags do not collapse.
    """
    with pytest.raises(ValueError, match='allow_internal_only_jpeg'):
        _validate_read_codec_optin(
            7,  # COMPRESSION_JPEG
            allow_experimental_codecs=False,
            allow_internal_only_jpeg=False,
        )
    # ``allow_experimental_codecs=True`` does NOT cover JPEG.
    with pytest.raises(ValueError, match='allow_internal_only_jpeg'):
        _validate_read_codec_optin(
            7,
            allow_experimental_codecs=True,
            allow_internal_only_jpeg=False,
        )


def test_validate_read_codec_optin_accepts_jpeg_with_flag():
    """With ``allow_internal_only_jpeg=True`` the read-side gate lets
    JPEG-in-TIFF through.
    """
    _validate_read_codec_optin(
        7,
        allow_experimental_codecs=False,
        allow_internal_only_jpeg=True,
    )


@pytest.mark.parametrize("codec_name", ['lerc', 'jpeg2000', 'lz4'])
def test_validate_read_codec_optin_accepts_experimental_with_flag(codec_name):
    """With ``allow_experimental_codecs=True`` the read-side gate lets
    LERC / JPEG2000 / LZ4 through.
    """
    tag = {
        v: k for k, v in _COMPRESSION_TAG_TO_NAME.items()
    }[codec_name]
    _validate_read_codec_optin(
        tag,
        allow_experimental_codecs=True,
        allow_internal_only_jpeg=False,
    )


def test_validate_read_codec_optin_message_names_feature_and_tier():
    """The rejection message names the codec, the missing flag, the
    SUPPORTED_FEATURES tier, and the parent epic so a reader can fix
    the call site without grepping the source.
    """
    with pytest.raises(ValueError) as exc:
        _validate_read_codec_optin(
            34887,  # LERC
            allow_experimental_codecs=False,
            allow_internal_only_jpeg=False,
        )
    msg = str(exc.value)
    assert 'lerc' in msg
    assert 'allow_experimental_codecs' in msg
    assert 'experimental' in msg
    assert '#2340' in msg


def test_validate_write_rich_tag_optin_accepts_empty_attrs():
    """No rich-tag attrs and no opt-in: the writer gate is a no-op."""
    _validate_write_rich_tag_optin(
        {}, allow_experimental_codecs=False)


def test_validate_write_rich_tag_optin_rejects_gdal_metadata_xml():
    """``attrs['gdal_metadata_xml']`` triggers the gate; rejection
    message names the attr and the opt-in flag.
    """
    with pytest.raises(ValueError, match='gdal_metadata_xml'):
        _validate_write_rich_tag_optin(
            {'gdal_metadata_xml': '<GDALMetadata/>'},
            allow_experimental_codecs=False,
        )


def test_validate_write_rich_tag_optin_rejects_extra_tags():
    """``attrs['extra_tags']`` triggers the gate; rejection message
    names the attr and the opt-in flag.
    """
    with pytest.raises(ValueError, match='extra_tags'):
        _validate_write_rich_tag_optin(
            {'extra_tags': [(700, 1, 0, b'')]},
            allow_experimental_codecs=False,
        )


def test_validate_write_rich_tag_optin_accepts_with_flag():
    """``allow_experimental_codecs=True`` accepts both rich-tag attrs."""
    _validate_write_rich_tag_optin(
        {'gdal_metadata_xml': '<GDALMetadata/>',
         'extra_tags': [(700, 1, 0, b'')]},
        allow_experimental_codecs=True,
    )


def test_validate_write_rich_tag_optin_exempts_round_trip():
    """An attrs dict carrying the ``_xrspatial_geotiff_contract`` marker
    came from an xrspatial read; round-tripping it back through
    ``to_geotiff`` is the canonical contract and must not
    require a new flag. The marker is the gate's exemption signal.
    """
    _validate_write_rich_tag_optin(
        {'gdal_metadata_xml': '<GDALMetadata/>',
         'extra_tags': [(700, 1, 0, b'')],
         '_xrspatial_geotiff_contract': 2},
        allow_experimental_codecs=False,
    )


# --- Read end-to-end: write an experimental-codec file, then assert the
# read side refuses without the matching opt-in and succeeds with it. ---


@pytest.mark.parametrize("codec", ['lerc', 'lz4'])
def test_open_geotiff_rejects_experimental_codec(tmp_path, codec):
    """A file written with LERC or LZ4 raises ``ValueError`` on read
    by default; the message names ``allow_experimental_codecs``.
    """
    path = _write_test_tif(
        tmp_path, codec, allow_experimental_codecs=True)
    with pytest.raises(ValueError, match='allow_experimental_codecs'):
        open_geotiff(path)


@pytest.mark.parametrize("codec", ['lerc', 'lz4'])
def test_open_geotiff_accepts_experimental_codec_with_flag(tmp_path, codec):
    """``allow_experimental_codecs=True`` lets the read through and
    returns a DataArray with the expected shape.
    """
    path = _write_test_tif(
        tmp_path, codec, allow_experimental_codecs=True)
    try:
        da = open_geotiff(path, allow_experimental_codecs=True)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"optional decoder missing for {codec}: {e}")
    assert da.shape == (32, 32)


def test_open_geotiff_rejects_jpeg2000(tmp_path):
    """JPEG2000 is experimental and requires the same opt-in as LERC /
    LZ4. ``j2k`` is an alias the writer maps to the same codec, so
    only one source file is needed.
    """
    path = _write_test_tif(
        tmp_path, 'jpeg2000', allow_experimental_codecs=True,
        dtype=np.uint8)
    with pytest.raises(ValueError, match='allow_experimental_codecs'):
        open_geotiff(path)


def test_open_geotiff_rejects_jpeg_internal_only(tmp_path):
    """JPEG-in-TIFF is internal-only; the dedicated flag
    ``allow_internal_only_jpeg`` is the gate. Mirrors the writer side
    where ``allow_experimental_codecs`` does NOT cover JPEG.
    """
    path = _write_test_tif(
        tmp_path, 'jpeg', allow_internal_only_jpeg=True,
        dtype=np.uint8)
    with pytest.raises(ValueError, match='allow_internal_only_jpeg'):
        open_geotiff(path)
    # ``allow_experimental_codecs=True`` does NOT unlock JPEG-in-TIFF
    # on the read side either.
    with pytest.raises(ValueError, match='allow_internal_only_jpeg'):
        open_geotiff(path, allow_experimental_codecs=True)


def test_open_geotiff_accepts_jpeg_internal_only_with_flag(tmp_path):
    """``allow_internal_only_jpeg=True`` lets the read through."""
    path = _write_test_tif(
        tmp_path, 'jpeg', allow_internal_only_jpeg=True,
        dtype=np.uint8)
    da = open_geotiff(path, allow_internal_only_jpeg=True)
    assert da.shape == (32, 32)


def test_read_geotiff_dask_rejects_experimental_codec(tmp_path):
    """The dask read path fires the gate at graph build, before any
    chunk task is scheduled.
    """
    path = _write_test_tif(
        tmp_path, 'lz4', allow_experimental_codecs=True)
    with pytest.raises(ValueError, match='allow_experimental_codecs'):
        read_geotiff_dask(path, chunks=16)


def test_read_geotiff_dask_accepts_experimental_codec_with_flag(tmp_path):
    """``allow_experimental_codecs=True`` lets the dask graph build."""
    path = _write_test_tif(
        tmp_path, 'lz4', allow_experimental_codecs=True)
    try:
        da = read_geotiff_dask(
            path, chunks=16, allow_experimental_codecs=True)
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"optional decoder missing: {e}")
    assert da.shape == (32, 32)


# --- Writer rich-tag attrs: gdal_metadata_xml / extra_tags need opt-in ---


def test_to_geotiff_rejects_gdal_metadata_xml_without_flag(tmp_path):
    """A DataArray whose attrs carry ``gdal_metadata_xml`` is rejected
    by ``to_geotiff`` unless the caller passes
    ``allow_experimental_codecs=True``. The message names the attr.
    """
    da = _make_float32_da()
    da.attrs['gdal_metadata_xml'] = (
        '<GDALMetadata><Item name="STATISTICS_MIN">0</Item>'
        '</GDALMetadata>'
    )
    path = os.path.join(str(tmp_path), 'rich_xml.tif')
    with pytest.raises(ValueError, match='gdal_metadata_xml'):
        to_geotiff(da, path)


def test_to_geotiff_rejects_extra_tags_without_flag(tmp_path):
    """Same shape as the ``gdal_metadata_xml`` case but for
    ``attrs['extra_tags']``. Both surfaces feed the same on-disk path
    and ride the same Experimental tier.
    """
    da = _make_float32_da()
    da.attrs['extra_tags'] = [(700, 1, 0, b'')]
    path = os.path.join(str(tmp_path), 'rich_extra.tif')
    with pytest.raises(ValueError, match='extra_tags'):
        to_geotiff(da, path)


def test_to_geotiff_accepts_rich_tags_with_flag(tmp_path):
    """``allow_experimental_codecs=True`` lets both attrs through and
    the write completes.
    """
    da = _make_float32_da()
    da.attrs['gdal_metadata_xml'] = (
        '<GDALMetadata><Item name="STATISTICS_MIN">0</Item>'
        '</GDALMetadata>'
    )
    da.attrs['extra_tags'] = [(700, 1, 0, b'')]
    path = os.path.join(str(tmp_path), 'rich_optin.tif')
    out = to_geotiff(da, path, allow_experimental_codecs=True)
    assert out == path
    assert os.path.exists(path)


def test_write_geotiff_gpu_rejects_rich_tags_without_flag(tmp_path):
    """The GPU writer mirrors ``to_geotiff`` so the two writers expose
    a consistent surface; the rejection fires before any GPU work and
    does not depend on cupy being installed.
    """
    da = _make_float32_da()
    da.attrs['gdal_metadata_xml'] = (
        '<GDALMetadata><Item name="STATISTICS_MIN">0</Item>'
        '</GDALMetadata>'
    )
    path = os.path.join(str(tmp_path), 'rich_gpu.tif')
    with pytest.raises(ValueError, match='gdal_metadata_xml'):
        write_geotiff_gpu(da, path)


# --- Already-gated paths: pin the existing opt-in inventory ---


def test_allow_rotated_default_raises_already_gated():
    """``allow_rotated=False`` (the default) raises on a rotated read.
    Pinned here so the Experimental + Internal-only opt-in inventory
    lives next to the existing ``allow_rotated`` /
    ``allow_unparseable_crs`` gates and a future refactor cannot drop
    one of them without failing this file.

    ``reader.allow_rotated`` is an experimental-tier gate.
    """
    # A signature pin is enough -- the actual rotated-read behaviour is
    # covered by the georef read tests.
    params = inspect.signature(open_geotiff).parameters
    assert 'allow_rotated' in params
    assert params['allow_rotated'].default is False


def test_allow_unparseable_crs_default_raises_already_gated():
    """``allow_unparseable_crs=False`` (the default) raises on an
    unparseable CRS string. ``reader.allow_unparseable_crs`` is an
    experimental-tier gate. Pin the signature here next to the other
    opt-ins so the inventory lives in one file.
    """
    params = inspect.signature(open_geotiff).parameters
    assert 'allow_unparseable_crs' in params
    assert params['allow_unparseable_crs'].default is False


def test_gpu_read_requires_explicit_optin():
    """GPU read is Experimental in ``SUPPORTED_FEATURES`` and the
    opt-in is the boolean ``gpu=True`` kwarg. Pin the default here so
    a future refactor cannot flip GPU read to auto-on.
    """
    params = inspect.signature(open_geotiff).parameters
    assert 'gpu' in params
    assert params['gpu'].default is False


def test_gpu_write_requires_explicit_optin():
    """GPU write is Experimental and gates on ``gpu=True`` /
    ``gpu=None`` (auto-detect from CuPy data). Pin the default here:
    ``None`` is the documented auto-detect sentinel and ``False`` /
    ``True`` are the explicit selectors. A flip to ``True`` default
    would silently route every NumPy write through the GPU pipeline.
    """
    params = inspect.signature(to_geotiff).parameters
    assert 'gpu' in params
    assert params['gpu'].default is None


# ===========================================================================
# Section 4 -- photometric kwarg and extra_tags override
# ===========================================================================
#
# Before this fix the writer silently labelled any 3+ band array as RGB,
# with the 4th band tagged as unassociated alpha; scientific multispectral
# rasters were mis-tagged. The fix adds a ``photometric`` kwarg defaulting
# to ``'auto'`` (MinIsBlack for any band count) and lets a user-supplied
# ``extra_tags`` Photometric / ExtraSamples entry win outright.


def _read_primary_ifd(path: str):
    """Parse the primary IFD of ``path`` and return it."""
    with open(path, 'rb') as f:
        raw = f.read()
    hdr = parse_header(raw[:16])
    return parse_ifd(raw, hdr.first_ifd_offset, hdr)


def _to_da(arr: np.ndarray) -> xr.DataArray:
    if arr.ndim == 3:
        return xr.DataArray(arr, dims=('y', 'x', 'band'))
    return xr.DataArray(arr, dims=('y', 'x'))


def test_four_band_default_is_minisblack_with_unspecified_extras(tmp_path):
    """Default photometric='auto' on a 4-band raster must write
    MinIsBlack + 3 ExtraSamples=unspecified, not RGB+alpha."""
    arr = np.zeros((32, 32, 4), dtype=np.uint16)
    path = str(tmp_path / 'four_band_default_1769.tif')
    to_geotiff(_to_da(arr), path)

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 1  # MinIsBlack
    assert ifd.get_values(TAG_EXTRA_SAMPLES) == (0, 0, 0)


def test_four_band_photometric_rgba_writes_rgb_plus_alpha(tmp_path):
    """photometric='rgba' is the opt-in for the old RGB+alpha behaviour."""
    arr = np.zeros((32, 32, 4), dtype=np.uint16)
    path = str(tmp_path / 'four_band_rgba_1769.tif')
    to_geotiff(_to_da(arr), path, photometric='rgba')

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 2  # RGB
    assert ifd.get_values(TAG_EXTRA_SAMPLES) == (2,)  # unassociated alpha


def test_four_band_photometric_rgb_writes_unspecified_extras(tmp_path):
    """photometric='rgb' on a 4-band emits Photometric=RGB with the
    leftover band tagged as unspecified (not alpha)."""
    arr = np.zeros((32, 32, 4), dtype=np.uint16)
    path = str(tmp_path / 'four_band_rgb_1769.tif')
    to_geotiff(_to_da(arr), path, photometric='rgb')

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 2
    assert ifd.get_values(TAG_EXTRA_SAMPLES) == (0,)


def test_three_band_default_is_minisblack_regression_1769(tmp_path):
    """Default on a 3-band raster must no longer claim RGB.

    The previous default treated samples_per_pixel >= 3 as RGB; the new
    'auto' default writes MinIsBlack regardless of band count so that
    multispectral 3-band rasters (e.g. R, NIR, SWIR) are not silently
    tagged as colour."""
    arr = np.zeros((32, 32, 3), dtype=np.uint16)
    path = str(tmp_path / 'three_band_default_1769.tif')
    to_geotiff(_to_da(arr), path)

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 1
    assert ifd.get_values(TAG_EXTRA_SAMPLES) == (0, 0)


def test_single_band_default_unchanged_1769(tmp_path):
    """1-band rasters stay MinIsBlack with no ExtraSamples tag."""
    arr = np.zeros((16, 16), dtype=np.uint8)
    path = str(tmp_path / 'one_band_default_1769.tif')
    to_geotiff(_to_da(arr), path)

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 1
    # No ExtraSamples tag at all for single-band.
    assert ifd.get_values(TAG_EXTRA_SAMPLES) is None


def test_user_extra_tags_override_extra_samples_1769(tmp_path):
    """A user-supplied (TAG_EXTRA_SAMPLES, ...) entry wins over the
    writer's auto value, even when photometric='rgb' would otherwise
    emit ExtraSamples=[0] for the 4th band."""
    arr = np.zeros((32, 32, 4), dtype=np.uint16)
    da = xr.DataArray(
        arr, dims=('y', 'x', 'band'),
        attrs={'extra_tags': [
            (TAG_EXTRA_SAMPLES, SHORT, 3, [0, 0, 0]),
        ]},
    )
    path = str(tmp_path / 'override_extras_1769.tif')
    # extra_tags is the Experimental write surface, so it needs the opt-in.
    to_geotiff(da, path, photometric='rgb',
               allow_experimental_codecs=True)

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 2  # RGB from kwarg
    # User override gives 3 unspecified entries, not the auto [0] for
    # the single 4th band.
    assert ifd.get_values(TAG_EXTRA_SAMPLES) == (0, 0, 0)


def test_user_extra_tags_override_photometric_1769(tmp_path):
    """A user-supplied (TAG_PHOTOMETRIC, ...) entry wins over the
    photometric kwarg."""
    arr = np.zeros((32, 32, 4), dtype=np.uint16)
    da = xr.DataArray(
        arr, dims=('y', 'x', 'band'),
        attrs={'extra_tags': [
            (TAG_PHOTOMETRIC, SHORT, 1, 0),  # MinIsWhite
        ]},
    )
    path = str(tmp_path / 'override_photometric_1769.tif')
    # photometric='rgb' would otherwise emit Photometric=2.
    # extra_tags is the Experimental write surface, so it needs the opt-in.
    to_geotiff(da, path, photometric='rgb',
               allow_experimental_codecs=True)

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 0  # MinIsWhite from override


def test_explicit_integer_photometric_1769(tmp_path):
    """An int passed as ``photometric`` is written verbatim."""
    arr = np.zeros((32, 32), dtype=np.uint8)
    path = str(tmp_path / 'photometric_int_1769.tif')
    # 0 = MinIsWhite
    to_geotiff(_to_da(arr), path, photometric=0)
    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 0


def test_invalid_photometric_name_raises_1769(tmp_path):
    """An unknown photometric name surfaces a clear ValueError."""
    arr = np.zeros((16, 16), dtype=np.uint8)
    path = str(tmp_path / 'invalid_photo_1769.tif')
    with pytest.raises(ValueError, match='not a valid name'):
        to_geotiff(_to_da(arr), path, photometric='not-a-thing')


def test_rgba_requires_four_bands_1769(tmp_path):
    """photometric='rgba' on a 3-band raster surfaces a clear error."""
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    path = str(tmp_path / 'rgba_three_band_1769.tif')
    with pytest.raises(ValueError, match='at least 4 bands'):
        to_geotiff(_to_da(arr), path, photometric='rgba')


def test_rgb_requires_three_bands_1769(tmp_path):
    """photometric='rgb' on a 2-band raster surfaces a clear error."""
    arr = np.zeros((16, 16, 2), dtype=np.uint8)
    path = str(tmp_path / 'rgb_two_band_1769.tif')
    with pytest.raises(ValueError, match='at least 3 bands'):
        to_geotiff(_to_da(arr), path, photometric='rgb')


def test_explicit_int_rgb_requires_three_bands_1769(tmp_path):
    """photometric=2 (RGB by int) on a 1-band raster also raises."""
    arr = np.zeros((16, 16), dtype=np.uint8)
    path = str(tmp_path / 'rgb_int_one_band_1769.tif')
    with pytest.raises(ValueError, match='at least 3 bands'):
        to_geotiff(_to_da(arr), path, photometric=2)


def test_dask_streaming_default_is_minisblack_1769(tmp_path):
    """The dask streaming write path honours the new default too."""
    dask = pytest.importorskip('dask.array')
    arr = dask.zeros((64, 64, 4), dtype=np.uint16, chunks=(32, 32, 4))
    da = xr.DataArray(arr, dims=('y', 'x', 'band'))
    path = str(tmp_path / 'four_band_dask_1769.tif')
    to_geotiff(da, path)

    ifd = _read_primary_ifd(path)
    assert ifd.get_value(TAG_PHOTOMETRIC) == 1
    assert ifd.get_values(TAG_EXTRA_SAMPLES) == (0, 0, 0)


def test_cog_overviews_carry_same_photometric_1769(tmp_path):
    """COG overviews must share the primary IFD's Photometric so the
    pyramid stays internally consistent."""
    # Use a non-default photometric so we can tell the value propagated
    # rather than matching by chance.
    arr = np.zeros((512, 512, 4), dtype=np.uint8)
    path = str(tmp_path / 'cog_overviews_1769.tif')
    to_geotiff(
        _to_da(arr), path, cog=True, tile_size=128,
        overview_levels=[2, 4], photometric='rgba',
    )

    with open(path, 'rb') as f:
        raw = f.read()
    hdr = parse_header(raw[:16])
    offset = hdr.first_ifd_offset
    seen = []
    while offset:
        ifd = parse_ifd(raw, offset, hdr)
        seen.append(ifd.get_value(TAG_PHOTOMETRIC))
        offset = ifd.next_ifd_offset
    # Primary + two overviews -- all three must be Photometric=RGB.
    assert seen == [2, 2, 2]


# ===========================================================================
# Section 5 -- gil_friendly deflate kwarg
# ===========================================================================
#
# The flag gates a documented optimisation: when ``True`` the deflate path
# is forced through stdlib ``zlib.compress`` (GIL-releasing) even when the
# optional ``deflate`` PyPI binding (which holds the GIL during compress) is
# installed. The parallel writer paths pass ``gil_friendly=True`` so the
# thread pool scales; the sequential paths leave it at the default ``False``
# to pick up libdeflate's per-call speedup. These tests exercise the flag at
# every layer it appears.


def _payload(n: int = 8192) -> bytes:
    """Repeatable payload large enough to exercise real codec branches."""
    rng = np.random.RandomState(1830)
    return (rng.bytes(n))


@pytest.mark.skipif(not _HAVE_LIBDEFLATE,
                    reason='deflate package not installed')
def test_deflate_compress_gil_friendly_true_bypasses_libdeflate(monkeypatch):
    """``gil_friendly=True`` must route through stdlib zlib, not libdeflate.

    A regression dropping the ``and not gil_friendly`` clause would
    silently re-route the parallel writer through the GIL-holding
    libdeflate binding and lose the documented thread-pool scaling
    (5x with zlib vs 1.2x with libdeflate across 8 threads).
    """
    libdeflate_calls = {'n': 0}

    real_zlib_compress = comp_mod._deflate.zlib_compress

    def _spy(data, level):
        libdeflate_calls['n'] += 1
        return real_zlib_compress(data, level)

    monkeypatch.setattr(comp_mod._deflate, 'zlib_compress', _spy)

    raw = _payload()
    # Baseline: gil_friendly omitted defaults to False -> libdeflate fires.
    out_default = deflate_compress(raw, level=6)
    assert libdeflate_calls['n'] == 1, (
        'with libdeflate installed and gil_friendly=False (default), '
        'deflate_compress must call the libdeflate binding'
    )

    # gil_friendly=True must skip libdeflate.
    out_gilfriendly = deflate_compress(raw, level=6, gil_friendly=True)
    assert libdeflate_calls['n'] == 1, (
        'gil_friendly=True must bypass the libdeflate binding even when '
        'it is installed; libdeflate.zlib_compress was called'
    )

    # Both outputs decompress to the original bytes (wire-compatible).
    assert zlib.decompress(out_default) == raw
    assert zlib.decompress(out_gilfriendly) == raw
    # gil_friendly=True output is exactly stdlib zlib.compress at level 6.
    assert out_gilfriendly == zlib.compress(raw, 6)


@pytest.mark.skipif(not _HAVE_LIBDEFLATE,
                    reason='deflate package not installed')
def test_deflate_compress_gil_friendly_false_uses_libdeflate(monkeypatch):
    """Default ``gil_friendly=False`` must call libdeflate when present.

    Pins the sequential-writer fast path: a regression flipping the
    default or always routing to stdlib zlib would silently undo the
    ~3x per-call speedup the libdeflate path delivers.
    """
    calls = {'n': 0}
    real = comp_mod._deflate.zlib_compress

    def _spy(data, level):
        calls['n'] += 1
        return real(data, level)

    monkeypatch.setattr(comp_mod._deflate, 'zlib_compress', _spy)

    raw = _payload()
    out = deflate_compress(raw, level=6)
    assert calls['n'] == 1, (
        'gil_friendly=False (default) must call deflate.zlib_compress'
    )
    out_explicit = deflate_compress(raw, level=6, gil_friendly=False)
    assert calls['n'] == 2
    assert zlib.decompress(out) == raw
    assert zlib.decompress(out_explicit) == raw


def test_deflate_compress_gil_friendly_round_trip_both_directions():
    """Round-trip parity across both flag values, regardless of backend.

    Output bytes may differ (libdeflate is a different encoder), but
    both must zlib-decompress back to the input.
    """
    raw = _payload(16384)
    for gf in (True, False):
        for level in (1, 6, 9):
            blob = deflate_compress(raw, level=level, gil_friendly=gf)
            assert zlib.decompress(blob) == raw, (
                f'gil_friendly={gf}, level={level} did not round-trip'
            )


def test_deflate_compress_fallback_warning_fires_when_libdeflate_missing(
        monkeypatch):
    """One-shot UserWarning must fire when libdeflate is absent.

    A regression removing the warning would let users silently pay the
    3x perf hit on every install missing the optional dep.
    """
    monkeypatch.setattr(comp_mod, '_HAVE_LIBDEFLATE', False)
    monkeypatch.setattr(comp_mod, '_deflate', None)
    monkeypatch.setattr(comp_mod, '_zlib_fallback_warned', False)

    raw = b'1830-warning-fires' * 1024

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = comp_mod.deflate_compress(raw, level=6)

    assert zlib.decompress(out) == raw
    matches = [w for w in caught
               if issubclass(w.category, UserWarning)
               and '`deflate` package is not installed' in str(w.message)]
    assert len(matches) == 1, (
        f'expected exactly one libdeflate-fallback UserWarning, '
        f'got {len(matches)}: {[str(w.message) for w in caught]}'
    )
    # Latch flips after the first call.
    assert comp_mod._zlib_fallback_warned is True


def test_deflate_compress_fallback_warning_is_one_shot(monkeypatch):
    """Subsequent calls after the first must not re-emit the warning.

    The module-global latch ``_zlib_fallback_warned`` is the gate. A
    regression flipping it to per-call would spam every parallel
    writer invocation with the same warning.
    """
    monkeypatch.setattr(comp_mod, '_HAVE_LIBDEFLATE', False)
    monkeypatch.setattr(comp_mod, '_deflate', None)
    monkeypatch.setattr(comp_mod, '_zlib_fallback_warned', False)

    raw = b'1830-one-shot' * 512

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        comp_mod.deflate_compress(raw)
        comp_mod.deflate_compress(raw)
        comp_mod.deflate_compress(raw, level=9)

    matches = [w for w in caught
               if issubclass(w.category, UserWarning)
               and '`deflate` package is not installed' in str(w.message)]
    assert len(matches) == 1, (
        f'fallback warning must fire only on the first call; '
        f'got {len(matches)} emissions'
    )


def test_deflate_compress_fallback_no_warning_when_latch_set(monkeypatch):
    """If the latch is already True, no warning fires (process startup
    typically warms it from the first user write)."""
    monkeypatch.setattr(comp_mod, '_HAVE_LIBDEFLATE', False)
    monkeypatch.setattr(comp_mod, '_deflate', None)
    monkeypatch.setattr(comp_mod, '_zlib_fallback_warned', True)

    raw = b'1830-latch-set' * 256

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = comp_mod.deflate_compress(raw)

    assert zlib.decompress(out) == raw
    assert not [w for w in caught if issubclass(w.category, UserWarning)
                and '`deflate` package' in str(w.message)]


@pytest.mark.skipif(not _HAVE_LIBDEFLATE,
                    reason='deflate package not installed')
def test_compress_forwards_gil_friendly_to_deflate(monkeypatch):
    """``compress(DEFLATE, gil_friendly=True)`` must skip libdeflate.

    Pins the dispatcher in ``_compression.compress``: the kwarg must
    thread through to ``deflate_compress``. A regression dropping the
    forward would silently revert the parallel writer to libdeflate.
    """
    calls = {'n': 0}
    real = comp_mod._deflate.zlib_compress

    def _spy(data, level):
        calls['n'] += 1
        return real(data, level)

    monkeypatch.setattr(comp_mod._deflate, 'zlib_compress', _spy)

    raw = _payload()
    # Default (gil_friendly=False) -> libdeflate fires once.
    compress(raw, COMPRESSION_DEFLATE, level=6)
    assert calls['n'] == 1
    # gil_friendly=True -> libdeflate must NOT fire.
    out = compress(raw, COMPRESSION_DEFLATE, level=6, gil_friendly=True)
    assert calls['n'] == 1
    assert zlib.decompress(out) == raw


def test_compress_gil_friendly_ignored_for_non_deflate_codecs():
    """LZW/PackBits/zstd/lz4/none ignore the flag (their bindings already
    release the GIL). Round-trip results must be identical for both
    flag values; this guards against a future change accidentally
    routing a non-deflate codec through a different code path based on
    the flag.
    """
    from xrspatial.geotiff._compression import decompress

    raw = _payload(4096)

    matrix = [
        (COMPRESSION_NONE, raw),
        (COMPRESSION_PACKBITS, raw),
        (COMPRESSION_LZW, raw),
        (COMPRESSION_ZSTD, raw),
    ]
    # ``lz4`` is an optional dependency. On CI runners that ship without it
    # (some macOS images) the codec dispatch path raises ImportError; skip
    # that row rather than fail the whole non-deflate-codec coverage test.
    if LZ4_AVAILABLE:
        matrix.append((COMPRESSION_LZ4, raw))
    for tag, payload in matrix:
        out_false = compress(payload, tag, gil_friendly=False)
        out_true = compress(payload, tag, gil_friendly=True)
        assert out_false == out_true, (
            f'compression={tag}: gil_friendly must not affect non-deflate '
            f'codec output'
        )
        # Spot-check round-trip on the path that has a public decoder.
        if tag in (COMPRESSION_ZSTD, COMPRESSION_LZW, COMPRESSION_LZ4,
                   COMPRESSION_PACKBITS):
            decoded = decompress(out_true, tag, expected_size=len(payload))
            decoded_bytes = (decoded.tobytes()
                             if hasattr(decoded, 'tobytes') else decoded)
            assert decoded_bytes[:len(payload)] == payload
        elif tag == COMPRESSION_NONE:
            assert out_true == payload


@pytest.mark.skipif(not _HAVE_LIBDEFLATE,
                    reason='deflate package not installed')
def test_compress_default_gil_friendly_is_false(monkeypatch):
    """The dispatcher's default must keep callers on libdeflate.

    A regression flipping the default to True would silently revert
    the documented sequential-path 3x speedup for every read-modify-
    write caller of ``compress`` outside the parallel writer.
    """
    calls = {'n': 0}
    real = comp_mod._deflate.zlib_compress

    def _spy(data, level):
        calls['n'] += 1
        return real(data, level)

    monkeypatch.setattr(comp_mod._deflate, 'zlib_compress', _spy)

    raw = _payload()
    compress(raw, COMPRESSION_DEFLATE, level=6)
    assert calls['n'] == 1, (
        'compress() default must call libdeflate when installed'
    )


class _DeflateCallSpy:
    """Capture every deflate_compress call's gil_friendly value."""

    def __init__(self, monkeypatch):
        self.calls = []  # list of bool
        self._real = comp_mod.deflate_compress
        # Patch at the module that the dispatcher (``compress``) imports
        # from, so all entry points are observed.
        monkeypatch.setattr(comp_mod, 'deflate_compress', self._spy)

    def _spy(self, data, level=6, gil_friendly=False):
        self.calls.append(bool(gil_friendly))
        return self._real(data, level=level, gil_friendly=gil_friendly)


def test_write_stripped_parallel_path_uses_gil_friendly(monkeypatch):
    """The parallel strip writer must call deflate_compress with
    ``gil_friendly=True`` on every strip.

    Pins the writer call site ``_writer.py:764``. A regression dropping
    the kwarg (or passing False) would silently make 8-thread parallel
    deflate writes scale at 1.2x instead of 5x.
    """
    # Large enough payload to take the parallel branch.
    rng = np.random.RandomState(1830)
    arr = rng.rand(2048, 2048).astype(np.float32)
    assert arr.nbytes > _PARALLEL_MIN_BYTES

    spy = _DeflateCallSpy(monkeypatch)
    _write_stripped(arr, COMPRESSION_DEFLATE, predictor=1,
                    rows_per_strip=256)

    assert spy.calls, (
        'expected at least one deflate_compress call from _write_stripped'
    )
    assert all(spy.calls), (
        f'parallel strip writer must pass gil_friendly=True to every '
        f'deflate_compress call; observed flags: {spy.calls}'
    )


def test_write_stripped_sequential_path_uses_default(monkeypatch):
    """The sequential strip writer (small payload) must use
    ``gil_friendly=False`` so the sequential path picks up libdeflate.

    Pins the writer call site ``_writer.py:741``. A regression passing
    True here would silently revert the sequential 3x speedup.
    """
    rng = np.random.RandomState(1830)
    arr = rng.rand(32, 64).astype(np.float32)
    assert arr.nbytes < _PARALLEL_MIN_BYTES

    spy = _DeflateCallSpy(monkeypatch)
    _write_stripped(arr, COMPRESSION_DEFLATE, predictor=1,
                    rows_per_strip=8)

    assert spy.calls, (
        'expected at least one deflate_compress call from _write_stripped'
    )
    assert not any(spy.calls), (
        f'sequential strip writer must use gil_friendly=False; '
        f'observed flags: {spy.calls}'
    )


def test_write_tiled_parallel_path_uses_gil_friendly(monkeypatch):
    """Parallel tile writer must pass ``gil_friendly=True`` to deflate."""
    rng = np.random.RandomState(1830)
    arr = rng.rand(2048, 2048).astype(np.float32)
    assert arr.nbytes > _PARALLEL_MIN_BYTES

    spy = _DeflateCallSpy(monkeypatch)
    _write_tiled(arr, COMPRESSION_DEFLATE, predictor=1, tile_size=512)

    assert spy.calls, (
        'expected at least one deflate_compress call from _write_tiled'
    )
    assert all(spy.calls), (
        f'parallel tile writer must pass gil_friendly=True to every '
        f'deflate_compress call; observed flags: {spy.calls}'
    )


def test_write_tiled_sequential_path_uses_default(monkeypatch):
    """Sequential tile writer (small payload) must keep
    ``gil_friendly=False``."""
    rng = np.random.RandomState(1830)
    arr = rng.rand(128, 128).astype(np.float32)
    assert arr.nbytes < _PARALLEL_MIN_BYTES

    spy = _DeflateCallSpy(monkeypatch)
    _write_tiled(arr, COMPRESSION_DEFLATE, predictor=1, tile_size=32)

    assert spy.calls
    assert not any(spy.calls), (
        f'sequential tile writer must use gil_friendly=False; '
        f'observed flags: {spy.calls}'
    )


def test_prepare_strip_forwards_gil_friendly(monkeypatch):
    """`_prepare_strip` must forward its ``gil_friendly`` kwarg to compress.

    Direct unit pin: walks the writer's per-strip helper for both flag
    values and asserts the deflate call observed the flag.
    """
    rng = np.random.RandomState(1830)
    arr = rng.rand(64, 64).astype(np.float32)

    spy = _DeflateCallSpy(monkeypatch)
    _prepare_strip(arr, 0, 8, 64, 64, 1, np.float32, 4,
                   predictor=1, compression=COMPRESSION_DEFLATE,
                   gil_friendly=True)
    _prepare_strip(arr, 0, 8, 64, 64, 1, np.float32, 4,
                   predictor=1, compression=COMPRESSION_DEFLATE,
                   gil_friendly=False)

    assert spy.calls == [True, False], (
        f'_prepare_strip must forward gil_friendly to deflate_compress; '
        f'observed flags: {spy.calls}'
    )


def test_prepare_tile_forwards_gil_friendly(monkeypatch):
    """`_prepare_tile` must forward its ``gil_friendly`` kwarg to compress."""
    rng = np.random.RandomState(1830)
    arr = rng.rand(64, 64).astype(np.float32)

    spy = _DeflateCallSpy(monkeypatch)
    _prepare_tile(arr, 0, 0, 32, 32, 64, 64, 1, np.float32, 4,
                  predictor=1, compression=COMPRESSION_DEFLATE,
                  gil_friendly=True)
    _prepare_tile(arr, 0, 0, 32, 32, 64, 64, 1, np.float32, 4,
                  predictor=1, compression=COMPRESSION_DEFLATE,
                  gil_friendly=False)

    assert spy.calls == [True, False], (
        f'_prepare_tile must forward gil_friendly to deflate_compress; '
        f'observed flags: {spy.calls}'
    )


def test_write_tiled_parallel_passes_gil_friendly_positionally(monkeypatch):
    """The parallel tile branch passes ``True`` as the *positional*
    ``gil_friendly`` argument to ``_prepare_tile`` (see _writer.py:943).

    Pin the positional contract: if the keyword-order of _prepare_tile
    changes, this test will flag it instead of silently swapping a
    different bool into ``gil_friendly`` and quietly regressing perf.
    """
    captured = []
    real_prepare = _prepare_tile

    def _wrapper(*args, **kwargs):
        # Positional order matches the signature; kwargs holds the rest.
        # gil_friendly is the trailing arg in the call inside _write_tiled.
        captured.append({'args': args, 'kwargs': kwargs})
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(
        'xrspatial.geotiff._writer._prepare_tile', _wrapper)

    rng = np.random.RandomState(1830)
    arr = rng.rand(2048, 2048).astype(np.float32)
    _write_tiled(arr, COMPRESSION_DEFLATE, predictor=1, tile_size=512)

    assert captured, '_prepare_tile must be invoked'
    # The parallel branch invokes _prepare_tile with all 15 positional
    # args from data..gil_friendly. Index 14 is gil_friendly. If a
    # future refactor switches to keywords, the flag must still resolve
    # to True.
    sig = inspect.signature(_prepare_tile)
    param_names = list(sig.parameters.keys())
    gil_idx = param_names.index('gil_friendly')

    for call in captured:
        if len(call['args']) > gil_idx:
            assert call['args'][gil_idx] is True, (
                f'_write_tiled parallel branch must pass True as the '
                f'positional gil_friendly arg (index {gil_idx}); '
                f'got {call["args"][gil_idx]!r}'
            )
        else:
            assert call['kwargs'].get('gil_friendly') is True, (
                f'_write_tiled parallel branch must set gil_friendly=True; '
                f'call args={call["args"]!r} kwargs={call["kwargs"]!r}'
            )


@pytest.mark.skipif(not _HAVE_LIBDEFLATE,
                    reason='deflate package not installed')
def test_compress_block_forwards_gil_friendly_true(monkeypatch):
    """``_compress_block(gil_friendly=True)`` must reach deflate_compress
    with the flag set, so the streaming writer's parallel tile path can
    route every per-tile compress through stdlib zlib.
    """
    spy = _DeflateCallSpy(monkeypatch)
    arr = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    _compress_block(
        np.ascontiguousarray(arr), 64, 64, 1, np.uint8, 1,
        predictor=1, compression=COMPRESSION_DEFLATE,
        gil_friendly=True,
    )
    assert spy.calls == [True], (
        f'_compress_block(gil_friendly=True) must forward to '
        f'deflate_compress; observed flags: {spy.calls}'
    )


@pytest.mark.skipif(not _HAVE_LIBDEFLATE,
                    reason='deflate package not installed')
def test_compress_block_default_gil_friendly_is_false(monkeypatch):
    """Without an explicit kwarg ``_compress_block`` must keep the
    default ``False`` so the serial streaming segment stays on
    libdeflate, matching the eager writer's sequential path.
    """
    spy = _DeflateCallSpy(monkeypatch)
    arr = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    _compress_block(
        np.ascontiguousarray(arr), 64, 64, 1, np.uint8, 1,
        predictor=1, compression=COMPRESSION_DEFLATE,
    )
    assert spy.calls == [False], (
        f'_compress_block default must use gil_friendly=False; '
        f'observed flags: {spy.calls}'
    )


@pytest.mark.skipif(not _HAVE_LIBDEFLATE,
                    reason='deflate package not installed')
def test_write_streaming_parallel_segment_uses_gil_friendly(
    tmp_path, monkeypatch,
):
    """End-to-end pin: ``write_streaming`` on a dask array large enough
    to trigger the parallel tile-segment branch must drive
    ``deflate_compress`` with ``gil_friendly=True`` on every parallel
    call.
    """
    dask_array = pytest.importorskip("dask.array")
    from xrspatial.geotiff._writer import write_streaming

    rng = np.random.RandomState(1830)
    # Two tile rows so the segment loop's parallel branch (n_seg_tiles
    # > 1) actually fires for the first row before the writer drains.
    arr_np = rng.rand(1024, 1024).astype(np.float32)
    dask_arr = dask_array.from_array(arr_np, chunks=(512, 512))

    spy = _DeflateCallSpy(monkeypatch)
    path = str(tmp_path / 'streaming_gil_friendly_1834.tif')
    write_streaming(
        dask_arr, path, compression='deflate', tiled=True, tile_size=512,
    )

    assert spy.calls, 'write_streaming must call deflate_compress'
    # The parallel branch passes gil_friendly=True; the serial branch
    # uses the default False. At this size the parallel branch fires
    # for at least one segment, so True must appear in the observed
    # flags. A regression dropping the kwarg would leave the parallel
    # branch on libdeflate and ``True`` would never appear.
    assert any(spy.calls), (
        f'write_streaming parallel tile-segment branch must call '
        f'deflate_compress with gil_friendly=True; observed flags: '
        f'{spy.calls}'
    )


@pytest.mark.parametrize('size,tiled,tile_size', [
    (2048, False, None),     # large strip parallel path
    (2048, True, 512),       # large tile parallel path
    (32, False, None),       # small strip sequential path
    (128, True, 32),         # small tile sequential path
])
def test_write_deflate_round_trip_across_parallelism_modes(
        tmp_path, size, tiled, tile_size):
    """End-to-end round-trip on both the sequential and parallel paths.

    Whichever ``gil_friendly`` value the writer selects, the bytes must
    decode back to the source exactly.
    """
    rng = np.random.RandomState(1830)
    expected = rng.rand(size, size).astype(np.float32)
    path = str(tmp_path / f'gilfriendly_{size}_{tiled}_{tile_size}.tif')
    kwargs = {'compression': 'deflate', 'tiled': tiled}
    if tile_size is not None:
        kwargs['tile_size'] = tile_size
    write(expected, path, **kwargs)
    arr, _ = read_to_array(path)
    np.testing.assert_array_equal(arr, expected)


# ===========================================================================
# Section 6 -- Reader / writer kwarg behaviour
# ===========================================================================
#
# Override-effect and dtype-cast coverage for kwargs that the signature
# pins in earlier sections assert only as *accepted*. Three groups:
#
#   6a -- ``write_vrt`` ``relative`` / ``crs`` / ``nodata`` override effect,
#         plus the empty-``source_files`` error path.
#   6b -- ``read_geotiff_gpu`` / ``read_geotiff_dask`` ``name`` and
#         ``max_pixels``, ``read_geotiff_gpu`` ``dtype`` cast, GPU writer
#         ``bigtiff``.
#   6c -- GPU writer ``predictor`` encode kernels and ``read_vrt(window=)``
#         windowed-read semantics.


@pytest.fixture
def source_tif(tmp_path):
    """Write a single-band float32 GeoTIFF with EPSG:4326 + nodata."""
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    y = np.linspace(1.0, 0.0, 8)
    x = np.linspace(0.0, 1.0, 8)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'crs': 4326, 'nodata': -1.0},
    )
    p = str(tmp_path / 'src_kwbeh_2026_05_12.tif')
    to_geotiff(da, p, compression='none')
    return p


@pytest.fixture
def float64_tif(tmp_path):
    """Write a float64 GeoTIFF for GPU dtype cast tests."""
    arr = np.random.default_rng(2026_05_12).random((40, 40)).astype(np.float64)
    y = np.linspace(41.0, 40.0, 40)
    x = np.linspace(-105.0, -104.0, 40)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'crs': 4326},
    )
    p = str(tmp_path / 'kwbeh_2026_05_12_f64.tif')
    to_geotiff(da, p, compression='none')
    return p, arr


@pytest.fixture
def uint16_tif(tmp_path):
    """Write a uint16 GeoTIFF for GPU dtype cast tests."""
    arr = np.random.default_rng(2026_05_12).integers(
        0, 10_000, (30, 30), dtype=np.uint16
    )
    y = np.linspace(41.0, 40.0, 30)
    x = np.linspace(-105.0, -104.0, 30)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': y, 'x': x},
        attrs={'crs': 4326},
    )
    p = str(tmp_path / 'kwbeh_2026_05_12_u16.tif')
    to_geotiff(da, p, compression='none')
    return p, arr


@pytest.fixture
def small_tiff_path(tmp_path):
    """Single-band 8x8 float32 GeoTIFF used by the name / max_pixels tests."""
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    p = tmp_path / "small.tif"
    to_geotiff(arr, str(p), tile_size=16)
    return str(p), arr


# --- 6a: write_vrt override effect (relative / crs / nodata) + error path ---


class TestWriteVrtRelativeBehaviour:
    """``relative=`` flips the ``relativeToVRT`` attribute and rewrites the
    source filename. The existing smoke test only asserts both modes are
    *accepted*, not that they actually take effect."""

    def _read_xml(self, path):
        with open(path, 'r') as fh:
            return fh.read()

    def test_relative_true_writes_relative_path(self, source_tif, tmp_path):
        vrt_path = str(tmp_path / 'rel_true.vrt')
        write_vrt(vrt_path, [source_tif], relative=True)

        xml = self._read_xml(vrt_path)
        # The on-disk text must carry the relativeToVRT="1" attribute,
        # not "0", and the SourceFilename text must not contain the
        # absolute path's tmp_path prefix.
        assert 'relativeToVRT="1"' in xml
        assert 'relativeToVRT="0"' not in xml
        # Source path is the bare filename (same directory as the VRT).
        assert os.path.basename(source_tif) in xml
        # The absolute path prefix (the tmp_path directory) is not in
        # the XML; otherwise the writer would have stored the full
        # path despite relative=True.
        assert str(tmp_path) not in xml

    def test_relative_false_writes_absolute_path(self, source_tif, tmp_path):
        vrt_path = str(tmp_path / 'rel_false.vrt')
        write_vrt(vrt_path, [source_tif], relative=False)

        xml = self._read_xml(vrt_path)
        # ``relative=False`` must flip the attribute and emit an absolute
        # path. A regression that ignored ``relative=`` would silently
        # produce the same XML as ``relative=True``.
        assert 'relativeToVRT="0"' in xml
        assert 'relativeToVRT="1"' not in xml
        # Absolute path is in the file's SourceFilename text.
        # Use realpath to handle symlinks tmp_path may carry on macOS.
        abs_src = os.path.realpath(source_tif)
        assert abs_src in xml

    def test_relative_true_parses_back_to_same_source(self, source_tif, tmp_path):
        """relative=True still round-trips: parse_vrt resolves the
        relative path back to the absolute one."""
        vrt_path = str(tmp_path / 'rel_true_rt.vrt')
        write_vrt(vrt_path, [source_tif], relative=True)
        parsed = parse_vrt(self._read_xml(vrt_path), vrt_dir=str(tmp_path))
        assert len(parsed.bands) == 1
        assert len(parsed.bands[0].sources) == 1
        # parse_vrt canonicalises with realpath, so compare against the
        # realpath of the original source.
        assert (
            os.path.realpath(parsed.bands[0].sources[0].filename)
            == os.path.realpath(source_tif)
        )

    def test_relative_false_parses_back_to_same_source(self, source_tif, tmp_path):
        vrt_path = str(tmp_path / 'rel_false_rt.vrt')
        write_vrt(vrt_path, [source_tif], relative=False)
        parsed = parse_vrt(self._read_xml(vrt_path), vrt_dir=str(tmp_path))
        assert len(parsed.bands) == 1
        assert (
            os.path.realpath(parsed.bands[0].sources[0].filename)
            == os.path.realpath(source_tif)
        )


class TestWriteVrtCrsWktBehaviour:
    """``crs=`` overrides the first source's CRS. Without an override,
    the first source's WKT is propagated. With an override, the
    override wins.

    The kwarg was formerly named ``crs_wkt``. The new canonical name
    is ``crs`` (parity with ``to_geotiff`` / ``write_geotiff_gpu``);
    the old name is still accepted with ``DeprecationWarning``. These
    tests exercise the new path; the deprecated path is covered by
    the VRT write tests.
    """

    def _read_parsed(self, vrt_path, tmp_path):
        with open(vrt_path, 'r') as fh:
            return parse_vrt(fh.read(), vrt_dir=str(tmp_path))

    def test_crs_wkt_override_wins(self, source_tif, tmp_path):
        """The supplied WKT must land in <SRS>, not the source's WKT."""
        override = (
            'PROJCS["UnitTest_Override_Sweep_2026_05_12",'
            'GEOGCS["test_datum",DATUM["d",SPHEROID["s",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Transverse_Mercator"],UNIT["metre",1]]'
        )
        vrt_path = str(tmp_path / 'crs_wkt_override.vrt')
        write_vrt(vrt_path, [source_tif], crs=override)
        parsed = self._read_parsed(vrt_path, tmp_path)
        assert parsed.crs_wkt == override

    def test_crs_wkt_none_falls_back_to_first_source(self, source_tif, tmp_path):
        """No override means the first source's WKT is used. Pin the
        contract: the default-VRT's parsed crs_wkt must be present,
        non-empty, and match the source TIF's own crs_wkt (no silent
        substitution, no None on the fall-back path)."""
        vrt_path = str(tmp_path / 'crs_wkt_default.vrt')
        write_vrt(vrt_path, [source_tif])
        parsed = self._read_parsed(vrt_path, tmp_path)

        source_da = open_geotiff(source_tif)
        source_wkt = source_da.attrs.get('crs_wkt')

        assert parsed.crs_wkt is not None
        assert parsed.crs_wkt != ''
        assert parsed.crs_wkt == source_wkt

    def test_crs_wkt_override_distinct_from_default(self, source_tif, tmp_path):
        """The override and default WKT must produce *different* on-disk
        XML. This is the safety-net: even if a future writer change
        normalises the WKT before emitting, the override path must
        still land a distinguishable WKT in the file."""
        marker = "UnitTest_Override_Marker_Sweep_2026_05_12"
        override = (
            f'GEOGCS["{marker}",'
            'DATUM["d",SPHEROID["s",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        )
        # Override path
        vrt_override = str(tmp_path / 'override.vrt')
        write_vrt(vrt_override, [source_tif], crs=override)
        # Default path
        vrt_default = str(tmp_path / 'default.vrt')
        write_vrt(vrt_default, [source_tif])

        with open(vrt_override, 'r') as fh:
            text_override = fh.read()
        with open(vrt_default, 'r') as fh:
            text_default = fh.read()

        assert marker in text_override
        assert marker not in text_default


class TestWriteVrtNodataBehaviour:
    """``nodata=`` overrides the first source's nodata sentinel.
    Source file is written with ``nodata=-1.0``; the override must land
    in every ``<NoDataValue>`` element."""

    def _bands(self, vrt_path, tmp_path):
        with open(vrt_path, 'r') as fh:
            return parse_vrt(fh.read(), vrt_dir=str(tmp_path)).bands

    def test_nodata_override_wins(self, source_tif, tmp_path):
        vrt_path = str(tmp_path / 'nodata_override.vrt')
        write_vrt(vrt_path, [source_tif], nodata=-9999.0)
        bands = self._bands(vrt_path, tmp_path)
        assert len(bands) == 1
        assert bands[0].nodata == -9999.0

    def test_nodata_none_takes_first_source(self, source_tif, tmp_path):
        """No override means the first source's nodata is used. The
        source was written with ``nodata=-1.0`` -- a regression that
        silently dropped the default-from-source code path would land
        ``None`` here."""
        vrt_path = str(tmp_path / 'nodata_default.vrt')
        write_vrt(vrt_path, [source_tif])
        bands = self._bands(vrt_path, tmp_path)
        assert len(bands) == 1
        assert bands[0].nodata == -1.0

    def test_nodata_override_writes_xml_element(self, source_tif, tmp_path):
        """Raw XML check: the override sentinel value lands in a
        <NoDataValue> element."""
        vrt_path = str(tmp_path / 'nodata_xml.vrt')
        write_vrt(vrt_path, [source_tif], nodata=-12345.0)
        with open(vrt_path, 'r') as fh:
            xml = fh.read()
        assert '<NoDataValue>-12345.0</NoDataValue>' in xml


class TestWriteVrtEmptySourceFiles:
    """``write_vrt(source_files=[])`` raises with a clear message.
    The error path is uncovered. A regression dropping the
    pre-validation would surface much further down as an IndexError
    when computing the bounding box of zero sources."""

    def test_empty_list_raises(self, tmp_path):
        vrt_path = str(tmp_path / 'should_not_exist.vrt')
        with pytest.raises(ValueError, match="source_files must not be empty"):
            write_vrt(vrt_path, [])

    def test_empty_list_does_not_create_file(self, tmp_path):
        vrt_path = str(tmp_path / 'should_not_exist_2.vrt')
        try:
            write_vrt(vrt_path, [])
        except ValueError:
            pass
        assert not os.path.exists(vrt_path)


# --- 6b: reader name / max_pixels / dtype coverage + GPU writer bigtiff ---


def test_read_geotiff_dask_name_kwarg_sets_name(small_tiff_path):
    path, arr = small_tiff_path
    da = read_geotiff_dask(path, chunks=4, name="custom_dask")
    assert da.name == "custom_dask"
    np.testing.assert_array_equal(da.values, arr)


def test_read_geotiff_dask_default_name_from_path(small_tiff_path):
    path, _ = small_tiff_path
    da = read_geotiff_dask(path, chunks=4)
    # Default name is filename stem when no override is supplied.
    assert da.name == "small"


@requires_gpu
def test_read_geotiff_gpu_name_kwarg_sets_name(small_tiff_path):
    path, arr = small_tiff_path
    da = read_geotiff_gpu(path, name="custom_gpu")
    assert da.name == "custom_gpu"
    np.testing.assert_array_equal(da.data.get(), arr)


@requires_gpu
def test_read_geotiff_gpu_default_name_from_path(small_tiff_path):
    path, _ = small_tiff_path
    da = read_geotiff_gpu(path)
    assert da.name == "small"


@requires_gpu
def test_read_geotiff_gpu_chunks_name_kwarg_sets_name(small_tiff_path):
    path, arr = small_tiff_path
    da = read_geotiff_gpu(path, chunks=4, name="custom_dask_gpu")
    assert da.name == "custom_dask_gpu"
    np.testing.assert_array_equal(da.data.compute().get(), arr)


@requires_gpu
def test_read_geotiff_gpu_max_pixels_accepts_within_budget(small_tiff_path):
    path, arr = small_tiff_path
    # 8 * 8 = 64 pixels but per-tile dim safety check uses tile_size=16
    # (256 pixels per tile); 300 leaves room. The fixture's tile_size
    # was bumped to 16 to satisfy the TIFF 6 multiple-of-16 rule.
    da = read_geotiff_gpu(path, max_pixels=300)
    np.testing.assert_array_equal(da.data.get(), arr)


@requires_gpu
def test_read_geotiff_gpu_max_pixels_rejects_oversized(small_tiff_path):
    path, _ = small_tiff_path
    with pytest.raises(ValueError, match="safety limit|exceeds max_pixels"):
        read_geotiff_gpu(path, max_pixels=10)


@requires_gpu
def test_read_geotiff_gpu_chunks_max_pixels_rejects_oversized(small_tiff_path):
    """Dask+GPU path enforces ``max_pixels`` per chunk (#2501).

    The cap is now chunk-scoped, so a per-chunk allocation over
    ``max_pixels`` raises either at graph-build (the GDS fast path's
    per-tile and per-chunk checks against the 16x16 TIFF tile dim and
    the 4x4 chunk shape) or on ``.compute()`` (the CPU-fallback path's
    per-chunk ``_check_dimensions`` inside ``read_to_array``). Wrap
    both calls in ``pytest.raises`` so either branch satisfies the
    test depending on whether KvikIO is installed.
    """
    path, _ = small_tiff_path
    with pytest.raises(ValueError, match="safety limit|exceeds max_pixels"):
        da = read_geotiff_gpu(path, chunks=4, max_pixels=10)
        da.compute()


def test_open_geotiff_chunks_name_flows_through(small_tiff_path):
    path, arr = small_tiff_path
    da = open_geotiff(path, chunks=4, name="dispatch_dask")
    assert da.name == "dispatch_dask"
    np.testing.assert_array_equal(da.values, arr)


@requires_gpu
def test_open_geotiff_gpu_name_flows_through(small_tiff_path):
    path, arr = small_tiff_path
    da = open_geotiff(path, gpu=True, name="dispatch_gpu")
    assert da.name == "dispatch_gpu"
    np.testing.assert_array_equal(da.data.get(), arr)


@requires_gpu
def test_open_geotiff_gpu_chunks_name_flows_through(small_tiff_path):
    path, arr = small_tiff_path
    da = open_geotiff(path, gpu=True, chunks=4, name="dispatch_dask_gpu")
    assert da.name == "dispatch_dask_gpu"
    np.testing.assert_array_equal(da.data.compute().get(), arr)


@requires_gpu
def test_open_geotiff_gpu_max_pixels_rejects(small_tiff_path):
    path, _ = small_tiff_path
    with pytest.raises(ValueError, match="safety limit|exceeds max_pixels"):
        open_geotiff(path, gpu=True, max_pixels=10)


@requires_gpu
class TestReadGeotiffGpuDtype:
    """``read_geotiff_gpu(dtype=...)`` casts on device. The eager CPU
    path has TestDtypeEager; the dask path has TestDtypeDask. The GPU
    path had no equivalent."""

    def test_float64_to_float32(self, float64_tif):
        path, orig = float64_tif
        result = read_geotiff_gpu(path, dtype='float32')
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(
            result.data.get(), orig.astype(np.float32), decimal=6)

    def test_float64_to_float16(self, float64_tif):
        path, _ = float64_tif
        result = read_geotiff_gpu(path, dtype=np.float16)
        assert result.dtype == np.float16

    def test_uint16_to_int32(self, uint16_tif):
        path, orig = uint16_tif
        result = read_geotiff_gpu(path, dtype='int32')
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result.data.get(), orig.astype(np.int32))

    def test_uint16_to_uint8(self, uint16_tif):
        path, _ = uint16_tif
        result = read_geotiff_gpu(path, dtype='uint8')
        assert result.dtype == np.uint8

    def test_float_to_int_raises(self, float64_tif):
        path, _ = float64_tif
        # The validator runs before the GPU upload; the error contract is
        # the same as the CPU path (``float`` ... ``int``).
        with pytest.raises(ValueError, match='float.*int'):
            read_geotiff_gpu(path, dtype='int32')

    def test_dtype_none_preserves_native_float64(self, float64_tif):
        path, _ = float64_tif
        result = read_geotiff_gpu(path, dtype=None)
        assert result.dtype == np.float64

    def test_dtype_none_preserves_native_uint16(self, uint16_tif):
        path, _ = uint16_tif
        result = read_geotiff_gpu(path, dtype=None)
        assert result.dtype == np.uint16


@requires_gpu
class TestOpenGeotiffGpuDispatchDtype:
    """``open_geotiff(..., gpu=True, dtype=...)`` forwards through the
    dispatcher into ``read_geotiff_gpu``. Pin the dispatch path so a
    regression dropping ``dtype=`` on the GPU branch surfaces here too."""

    def test_dispatch_float64_to_float32(self, float64_tif):
        path, orig = float64_tif
        result = open_geotiff(path, gpu=True, dtype='float32')
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(
            result.data.get(), orig.astype(np.float32), decimal=6)

    def test_dispatch_float_to_int_raises(self, float64_tif):
        path, _ = float64_tif
        with pytest.raises(ValueError, match='float.*int'):
            open_geotiff(path, gpu=True, dtype='int32')


@requires_gpu
class TestReadGeotiffGpuChunksDtype:
    """``read_geotiff_gpu(chunks=..., dtype=...)`` -- dask + GPU + dtype
    combination is a separate dispatch path through the GPU reader and
    its own ``astype`` step on the cupy array, then a ``chunk`` call.
    Cover the cast for the dask+GPU branch too."""

    def test_chunks_float64_to_float32(self, float64_tif):
        path, orig = float64_tif
        result = read_geotiff_gpu(path, chunks=20, dtype='float32')
        assert result.dtype == np.float32
        # ``.data`` is a dask array of cupy chunks. Compute, then
        # ``.get()`` the resulting cupy host buffer.
        computed = result.data.compute()
        np.testing.assert_array_almost_equal(
            computed.get(), orig.astype(np.float32), decimal=6)


@requires_gpu
class TestWriteGeotiffGpuBigtiff:
    """``write_geotiff_gpu(bigtiff=)`` threads ``force_bigtiff=`` to
    ``_assemble_tiff``. The CPU writer has equivalent header-level
    bigtiff coverage; the GPU writer did not.

    Small arrays are sufficient because the BigTIFF decision is a
    width-of-offset-field switch, not a value-range one -- a forced
    BigTIFF on a 64-pixel array produces the same header magic byte
    pattern that a >4 GB file would."""

    def _read_header_is_bigtiff(self, path):
        with open(path, 'rb') as fh:
            header = parse_header(fh.read(16))
        return header.is_bigtiff

    def test_force_bigtiff_true_writes_bigtiff(self, tmp_path):
        import cupy
        arr = cupy.arange(64, dtype=cupy.float32).reshape(8, 8)
        da = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': np.arange(8, dtype=np.float64),
                    'x': np.arange(8, dtype=np.float64)},
        )
        path = str(tmp_path / 'gpu_bigtiff_true.tif')
        write_geotiff_gpu(da, path, bigtiff=True, tile_size=16)
        assert self._read_header_is_bigtiff(path), (
            "write_geotiff_gpu(bigtiff=True) should emit BigTIFF header "
            "(magic byte 43)."
        )
        # Data round-trips even with the BigTIFF header.
        rd = open_geotiff(path)
        np.testing.assert_array_equal(rd.values, arr.get())

    def test_force_bigtiff_false_writes_classic(self, tmp_path):
        import cupy
        arr = cupy.arange(64, dtype=cupy.float32).reshape(8, 8)
        da = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': np.arange(8, dtype=np.float64),
                    'x': np.arange(8, dtype=np.float64)},
        )
        path = str(tmp_path / 'gpu_bigtiff_false.tif')
        write_geotiff_gpu(da, path, bigtiff=False, tile_size=16)
        assert not self._read_header_is_bigtiff(path), (
            "write_geotiff_gpu(bigtiff=False) should emit classic TIFF."
        )

    def test_bigtiff_none_stays_classic_small_file(self, tmp_path):
        """``bigtiff=None`` (default) is auto: small files should stay
        classic. Without an explicit None test, a regression flipping
        the default to ``True`` would not be caught -- and that would
        break interop with older readers that don't accept BigTIFF."""
        import cupy
        arr = cupy.arange(64, dtype=cupy.float32).reshape(8, 8)
        da = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': np.arange(8, dtype=np.float64),
                    'x': np.arange(8, dtype=np.float64)},
        )
        path = str(tmp_path / 'gpu_bigtiff_default.tif')
        write_geotiff_gpu(da, path, tile_size=16)
        assert not self._read_header_is_bigtiff(path), (
            "write_geotiff_gpu default should auto-pick classic TIFF for "
            "tiny outputs; a default switch to BigTIFF would break "
            "older readers."
        )

    def test_to_geotiff_gpu_bigtiff_threads_through(self, tmp_path):
        """``to_geotiff(..., gpu=True, bigtiff=True)`` dispatches into
        ``write_geotiff_gpu(bigtiff=True)``. Cover the dispatcher's
        thread-through so a regression dropping ``bigtiff=`` on the GPU
        dispatch branch surfaces here too."""
        import cupy
        arr = cupy.arange(64, dtype=cupy.float32).reshape(8, 8)
        da = xr.DataArray(
            arr, dims=['y', 'x'],
            coords={'y': np.arange(8, dtype=np.float64),
                    'x': np.arange(8, dtype=np.float64)},
        )
        path = str(tmp_path / 'to_gpu_bigtiff_true.tif')
        to_geotiff(da, path, gpu=True, bigtiff=True, tile_size=16)
        assert self._read_header_is_bigtiff(path), (
            "to_geotiff(gpu=True, bigtiff=True) should reach the GPU "
            "writer with force_bigtiff=True propagated through."
        )
        rd = open_geotiff(path)
        np.testing.assert_array_equal(rd.values, arr.get())


# --- 6c: GPU writer predictor encode kernels + read_vrt(window=) ---


def _read_predictor_tag(path: str) -> int | None:
    """Read TIFF Predictor tag (id=317). Returns None if absent."""
    with open(path, 'rb') as f:
        header = f.read(8)
    assert header[:2] == b'II', "test fixture writes little-endian"
    magic = struct.unpack('<H', header[2:4])[0]
    assert magic == 42, "classic TIFF expected"
    ifd_offset = struct.unpack('<I', header[4:8])[0]

    with open(path, 'rb') as f:
        f.seek(ifd_offset)
        n_entries = struct.unpack('<H', f.read(2))[0]
        for _ in range(n_entries):
            entry = f.read(12)
            tag, _type_id, _count = struct.unpack('<HHI', entry[:8])
            if tag == 317:
                return struct.unpack('<H', entry[8:10])[0]
    return None  # tag absent => predictor 1 (none)


def _da_with_float_coords(arr) -> xr.DataArray:
    """Wrap a 2D or 3D array of any dtype with float64 y/x coords.

    Accepts numpy or cupy arrays. For 2D inputs returns a (y, x)
    DataArray; for 3D inputs returns a (y, x, band) DataArray with
    an integer band index. The element dtype is preserved from the
    input; only the y/x coordinate arrays are forced to float64 so
    pixel-is-area transforms round-trip cleanly through the
    geotiff/VRT writers.
    """
    h, w = arr.shape[:2]
    coords = {
        'y': np.arange(h, dtype=np.float64),
        'x': np.arange(w, dtype=np.float64),
    }
    if arr.ndim == 2:
        return xr.DataArray(arr, dims=('y', 'x'), coords=coords)
    return xr.DataArray(
        arr, dims=('y', 'x', 'band'),
        coords={**coords, 'band': np.arange(arr.shape[2])},
    )


@requires_gpu
class TestWriteGeotiffGpuPredictor2Uint8:
    """``predictor=True`` / ``predictor=2`` on uint8 data.

    Exercises the ``_predictor_encode_kernel_u8`` CUDA kernel via
    ``_gpu_predictor2_encode`` dispatch.
    """

    def test_predictor_true_uint8_round_trip(self, tmp_path):
        import cupy
        rng = np.random.RandomState(0)
        arr = rng.randint(0, 256, size=(8, 16), dtype=np.uint8)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred2_u8_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=True,
                          tile_size=16)

        # Round-trip through the public reader
        out = open_geotiff(path)
        np.testing.assert_array_equal(out.values, arr)
        # On-disk Predictor tag advertises horizontal differencing
        assert _read_predictor_tag(path) == 2

    def test_predictor_2_uint8_round_trip(self, tmp_path):
        """``predictor=2`` (int form) is equivalent to ``predictor=True``."""
        import cupy
        rng = np.random.RandomState(1)
        arr = rng.randint(0, 256, size=(8, 16), dtype=np.uint8)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred2_int_u8_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=2,
                          tile_size=16)

        out = open_geotiff(path)
        np.testing.assert_array_equal(out.values, arr)
        assert _read_predictor_tag(path) == 2

    def test_predictor_2_uint8_3band_rgb(self, tmp_path):
        """Multi-sample (3-band) uint8 with ``predictor=2``.

        Stride is ``samples_per_pixel`` in the encode kernel, so the
        decode must reverse the same stride. A regression dropping
        ``samples`` from ``_gpu_predictor2_encode`` would write data
        differentiated by 1 byte but advertise multi-sample tiles,
        producing garbled colours on read.
        """
        import cupy
        rng = np.random.RandomState(2)
        arr = rng.randint(0, 256, size=(8, 16, 3), dtype=np.uint8)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred2_u8_3band_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=2,
                          tile_size=16)

        out = open_geotiff(path)
        np.testing.assert_array_equal(out.values, arr)
        assert _read_predictor_tag(path) == 2

    def test_predictor_false_no_predictor_tag(self, tmp_path):
        """``predictor=False`` writes no Predictor tag (default behaviour).

        Pins the contrast with ``predictor=True``: without this test, a
        regression that flipped the default to ``predictor=2`` would
        round-trip but advertise predictor=2 in the output file.
        """
        import cupy
        arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_no_pred_u8_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=False,
                          tile_size=16)

        out = open_geotiff(path)
        np.testing.assert_array_equal(out.values, arr)
        # Predictor tag absent or explicitly 1 (no predictor)
        tag = _read_predictor_tag(path)
        assert tag is None or tag == 1


@requires_gpu
class TestWriteGeotiffGpuPredictor2Uint16:
    """``predictor=2`` on uint16 data.

    Exercises ``_predictor_encode_kernel_u16`` (16-bit sample stride).
    """

    def test_predictor_2_uint16_round_trip(self, tmp_path):
        import cupy
        rng = np.random.RandomState(3)
        arr = rng.randint(0, 60000, size=(8, 16), dtype=np.uint16)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred2_u16_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=2,
                          tile_size=16)

        out = open_geotiff(path)
        np.testing.assert_array_equal(out.values, arr)
        assert _read_predictor_tag(path) == 2


@requires_gpu
class TestWriteGeotiffGpuPredictor2Int32:
    """``predictor=2`` on int32 data.

    Exercises ``_predictor_encode_kernel_u32`` (32-bit sample stride).
    Int32 is viewed as uint32 for differencing semantics; the round
    trip must reproduce the signed values exactly.
    """

    def test_predictor_2_int32_round_trip(self, tmp_path):
        import cupy
        rng = np.random.RandomState(4)
        # Mix of negative and positive to ensure the unsigned-view
        # differencing round-trips through the signed interpretation
        arr = rng.randint(-1_000_000, 1_000_000, size=(8, 16),
                          dtype=np.int32)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred2_i32_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=2,
                          tile_size=16)

        out = open_geotiff(path)
        np.testing.assert_array_equal(out.values, arr)
        assert _read_predictor_tag(path) == 2


@requires_gpu
class TestWriteGeotiffGpuPredictor3Float:
    """``predictor=3`` (floating-point predictor).

    Exercises ``_fp_predictor_encode_kernel`` for both float32 and
    float64 (bps=4 and bps=8). The kernel does a byte-swizzle
    (MSB-first lane layout) followed by horizontal differencing per
    TIFF Technical Note 3; both bps must round-trip exactly.
    """

    def test_predictor_3_float32_round_trip(self, tmp_path):
        import cupy
        rng = np.random.RandomState(5)
        # Smooth-ish values so fp predictor actually compresses
        # (round-trip semantics do not depend on smoothness, but a
        # mix of magnitudes exercises the byte-swizzle on all 4 lanes)
        arr = rng.uniform(-1000.0, 1000.0, size=(8, 16)).astype(np.float32)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred3_f32_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=3,
                          tile_size=16)

        out = open_geotiff(path)
        # FP predictor is lossless: equality, not allclose
        np.testing.assert_array_equal(out.values, arr)
        assert _read_predictor_tag(path) == 3

    def test_predictor_3_float64_round_trip(self, tmp_path):
        import cupy
        rng = np.random.RandomState(6)
        arr = rng.uniform(-1e9, 1e9, size=(8, 16)).astype(np.float64)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred3_f64_2026_05_12_v2.tif')

        write_geotiff_gpu(da, path, compression='deflate', predictor=3,
                          tile_size=16)

        out = open_geotiff(path)
        np.testing.assert_array_equal(out.values, arr)
        assert _read_predictor_tag(path) == 3

    def test_predictor_3_rejects_int_dtype(self, tmp_path):
        """FP predictor refuses non-float dtypes (parity with CPU writer)."""
        import cupy
        arr = np.arange(64, dtype=np.int32).reshape(8, 8)
        da = _da_with_float_coords(cupy.asarray(arr))
        path = str(tmp_path / 'gpu_pred3_reject_2026_05_12_v2.tif')

        with pytest.raises(ValueError,
                           match=r"predictor=3.*requires float"):
            write_geotiff_gpu(da, path, compression='deflate', predictor=3,
                              tile_size=16)


@requires_gpu
class TestWriteGeotiffGpuPredictorCpuParity:
    """Pixel-exact parity between CPU ``to_geotiff(predictor=X)`` and
    GPU ``write_geotiff_gpu(predictor=X)``.

    Predictor encode is a lossless transform: identical inputs must
    produce identical decoded outputs regardless of whether the
    differencing ran on CPU or GPU. The compressed bytes may differ
    (different deflate library calls) but the round-tripped pixels
    must match.
    """

    def test_cpu_gpu_parity_predictor_2_uint16(self, tmp_path):
        import cupy
        rng = np.random.RandomState(7)
        arr = rng.randint(0, 60000, size=(8, 16), dtype=np.uint16)

        cpu_path = str(tmp_path / 'cpu_pred2_u16_v2.tif')
        gpu_path = str(tmp_path / 'gpu_pred2_u16_v2.tif')

        to_geotiff(_da_with_float_coords(arr), cpu_path,
                   compression='deflate', predictor=2, tile_size=16)
        write_geotiff_gpu(_da_with_float_coords(cupy.asarray(arr)), gpu_path,
                          compression='deflate', predictor=2, tile_size=16)

        cpu_out = open_geotiff(cpu_path).values
        gpu_out = open_geotiff(gpu_path).values
        np.testing.assert_array_equal(cpu_out, gpu_out)
        np.testing.assert_array_equal(cpu_out, arr)

    def test_cpu_gpu_parity_predictor_3_float32(self, tmp_path):
        import cupy
        rng = np.random.RandomState(8)
        arr = rng.uniform(-100.0, 100.0, size=(8, 16)).astype(np.float32)

        cpu_path = str(tmp_path / 'cpu_pred3_f32_v2.tif')
        gpu_path = str(tmp_path / 'gpu_pred3_f32_v2.tif')

        to_geotiff(_da_with_float_coords(arr), cpu_path,
                   compression='deflate', predictor=3, tile_size=16)
        write_geotiff_gpu(_da_with_float_coords(cupy.asarray(arr)), gpu_path,
                          compression='deflate', predictor=3, tile_size=16)

        cpu_out = open_geotiff(cpu_path).values
        gpu_out = open_geotiff(gpu_path).values
        np.testing.assert_array_equal(cpu_out, gpu_out)
        np.testing.assert_array_equal(cpu_out, arr)


def _write_tile_to_vrt(tmp_path, name: str, data: np.ndarray) -> str:
    """Write a single-source GeoTIFF tile for VRT inclusion."""
    path = str(tmp_path / name)
    write(data, path, compression='none', tiled=False)
    return path


def _make_single_tile_vrt(tmp_path, arr: np.ndarray) -> str:
    """Create a single-source VRT mosaic.

    Uses ``_vrt.write_vrt`` so source paths land relative to the VRT
    directory; that keeps the path-containment guard happy
    without environment variables.
    """
    tile_path = _write_tile_to_vrt(tmp_path, 'src_tile.tif', arr)
    vrt_path = str(tmp_path / 'single.vrt')
    _write_vrt_internal(vrt_path, [tile_path])
    return vrt_path


def _make_2x1_mosaic_vrt(tmp_path, left: np.ndarray,
                         right: np.ndarray) -> str:
    """Create a 2x1 horizontal mosaic VRT for cross-source window tests.

    Hand-built XML so the dst_rect placements are explicit -- VRT's
    write_vrt helper only handles single-source layouts directly.
    """
    h, lw = left.shape[:2]
    rw = right.shape[1]
    width = lw + rw

    lpath = _write_tile_to_vrt(tmp_path, 'left.tif', left)
    rpath = _write_tile_to_vrt(tmp_path, 'right.tif', right)

    dtype_map = {np.dtype('float32'): 'Float32',
                 np.dtype('float64'): 'Float64',
                 np.dtype('uint8'): 'Byte',
                 np.dtype('int32'): 'Int32',
                 np.dtype('uint16'): 'UInt16'}
    data_type = dtype_map[left.dtype]

    lines = [
        f'<VRTDataset rasterXSize="{width}" rasterYSize="{h}">',
        '  <GeoTransform>0.0, 1.0, 0.0, 0.0, 0.0, -1.0</GeoTransform>',
        f'  <VRTRasterBand dataType="{data_type}" band="1">',
        '    <SimpleSource>',
        f'      <SourceFilename relativeToVRT="1">'
        f'{os.path.basename(lpath)}</SourceFilename>',
        '      <SourceBand>1</SourceBand>',
        f'      <SrcRect xOff="0" yOff="0" xSize="{lw}" ySize="{h}"/>',
        f'      <DstRect xOff="0" yOff="0" xSize="{lw}" ySize="{h}"/>',
        '    </SimpleSource>',
        '    <SimpleSource>',
        f'      <SourceFilename relativeToVRT="1">'
        f'{os.path.basename(rpath)}</SourceFilename>',
        '      <SourceBand>1</SourceBand>',
        f'      <SrcRect xOff="0" yOff="0" xSize="{rw}" ySize="{h}"/>',
        f'      <DstRect xOff="{lw}" yOff="0" xSize="{rw}" ySize="{h}"/>',
        '    </SimpleSource>',
        '  </VRTRasterBand>',
        '</VRTDataset>',
    ]

    vrt_path = str(tmp_path / 'mosaic_2x1.vrt')
    with open(vrt_path, 'w') as f:
        f.write('\n'.join(lines))
    return vrt_path


class TestReadVrtWindowEager:
    """Eager numpy ``read_vrt(window=...)`` slices the assembled raster."""

    def test_window_subregion_of_single_source(self, tmp_path):
        """Window picks a 4x6 sub-block from an 8x16 single-source VRT."""
        arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        # rows 2..6, cols 4..10
        result = read_vrt(vrt, window=(2, 4, 6, 10))

        assert result.shape == (4, 6)
        np.testing.assert_array_equal(result.values, arr[2:6, 4:10])

    def test_window_full_raster_matches_no_window(self, tmp_path):
        """``window=(0, 0, H, W)`` returns the same data as no window."""
        arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        full = read_vrt(vrt).values
        windowed = read_vrt(vrt, window=(0, 0, 8, 16)).values

        np.testing.assert_array_equal(windowed, full)

    def test_window_outside_raster_bounds_rejected(self, tmp_path):
        """Window extending past raster bounds raises ``ValueError``.

        ``read_vrt`` used to silently clamp out-of-bounds windows. That
        masked caller bugs (typo'd coords, off-by-one extents) and made
        the returned shape disagree with the caller's coord arrays. The
        validator now rejects such windows up front with a typed
        ``ValueError`` instead.
        """
        arr = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        with pytest.raises(ValueError, match="outside the VRT extent"):
            read_vrt(vrt, window=(0, 0, 100, 100))

    def test_window_negative_offsets_rejected(self, tmp_path):
        """Negative start offsets raise ``ValueError``.

        ``read_vrt`` validates the window
        against the VRT extent. Negative offsets are rejected the same
        way an over-large window is, rather than being silently clamped
        to zero.
        """
        arr = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        with pytest.raises(ValueError, match="outside the VRT extent"):
            read_vrt(vrt, window=(-1, -2, 3, 4))

    def test_window_across_mosaic_seam(self, tmp_path):
        """Window straddling a multi-source seam reads both sources.

        2x1 mosaic of two 4x4 tiles laid out side-by-side (total 4x8).
        A window from col 0 to col 6 covers cols 0-3 of left and cols
        0-1 of right (the seam sits at col 4). The src_rect coordinate
        mapping inside ``_vrt.read_vrt`` must clip each source's
        source-coords correctly; a regression to the dst-to-src
        translation would return mis-aligned columns.
        """
        left = np.arange(16, dtype=np.float32).reshape(4, 4)
        right = (np.arange(16, dtype=np.float32) + 100).reshape(4, 4)

        vrt = _make_2x1_mosaic_vrt(tmp_path, left, right)

        # Window rows 0..4, cols 0..6 (cuts across seam at col 4)
        result = read_vrt(vrt, window=(0, 0, 4, 6))

        assert result.shape == (4, 6)
        # cols 0-3 of window are cols 0-3 of left
        np.testing.assert_array_equal(result.values[:, :4], left[:, :4])
        # cols 4-5 of window are cols 0-1 of right (after seam)
        np.testing.assert_array_equal(result.values[:, 4:6], right[:, :2])

    def test_window_offset_into_mosaic(self, tmp_path):
        """Window starting past the seam reads only the right source."""
        left = np.arange(16, dtype=np.float32).reshape(4, 4)
        right = (np.arange(16, dtype=np.float32) + 100).reshape(4, 4)

        vrt = _make_2x1_mosaic_vrt(tmp_path, left, right)

        # Window cols 5..8 -> right cols 1..4
        result = read_vrt(vrt, window=(0, 5, 4, 8))

        assert result.shape == (4, 3)
        np.testing.assert_array_equal(result.values, right[:, 1:4])

    def test_window_transform_origin_shift(self, tmp_path):
        """``attrs['transform']`` reflects the window origin.

        With GeoTransform ``(origin_x=0, res=1, origin_y=0, res=-1)``
        and a window ``(r0=2, c0=3, ...)``, the output's transform
        must advertise the shifted origin ``origin_x' = origin_x +
        c0*res_x`` and ``origin_y' = origin_y + r0*res_y``. This is
        the metadata-propagation contract that ``open_geotiff
        (window=)`` already honours; ``read_vrt(window=)`` must
        agree.
        """
        arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        result = read_vrt(vrt, window=(2, 3, 6, 10))

        # GeoTransform from _vrt.write_vrt default: pixel-is-area,
        # res_x=1.0, res_y=-1.0, origin (0, 0).
        # Expected: origin shifts by (3 * 1.0, 2 * -1.0) = (3.0, -2.0)
        assert 'transform' in result.attrs
        pw, _, ox, _, ph, oy = result.attrs['transform']
        assert pw == pytest.approx(1.0)
        assert ph == pytest.approx(-1.0)
        assert ox == pytest.approx(3.0)
        assert oy == pytest.approx(-2.0)

    def test_window_coords_match_transform_shift(self, tmp_path):
        """y/x coords reflect the window's origin shift.

        Pixel-is-area convention: coord(0, 0) sits at the *center* of
        the windowed pixel (0, 0). With res_x=1.0, res_y=-1.0,
        origin (0, 0), and window starting at (r0=2, c0=3), the
        first x coord must be ``0 + (3 + 0.5) * 1.0 = 3.5`` and the
        first y coord must be ``0 + (2 + 0.5) * -1.0 = -2.5``.
        """
        arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        result = read_vrt(vrt, window=(2, 3, 6, 10))

        assert float(result.x[0]) == pytest.approx(3.5)
        assert float(result.y[0]) == pytest.approx(-2.5)


class TestReadVrtWindowWithBand:
    """``read_vrt(window=, band=)`` combinations.

    A regression in either kwarg's interaction with the other (band
    selection after window slicing, nodata sentinel resolved per
    band) would mis-mask the windowed region.
    """

    def _make_multiband_vrt(self, tmp_path) -> tuple[str, np.ndarray]:
        """Two-band VRT with distinct values per band."""
        h, w = 4, 8
        band0 = np.arange(h * w, dtype=np.float32).reshape(h, w)
        band1 = (band0 * -1.0).astype(np.float32)
        # Stack into 3D so write_vrt produces a multi-band TIFF source
        full = np.stack([band0, band1], axis=-1)

        tile_path = str(tmp_path / 'multi.tif')
        to_geotiff(_da_with_float_coords(full), tile_path, compression='none')

        vrt_path = str(tmp_path / 'multi_band.vrt')
        _write_vrt_internal(vrt_path, [tile_path])
        return vrt_path, full

    def test_window_plus_band_selection(self, tmp_path):
        vrt, full = self._make_multiband_vrt(tmp_path)

        # window rows 1..3, cols 2..6, band 1
        result = read_vrt(vrt, window=(1, 2, 3, 6), band=1)

        assert result.ndim == 2  # band selection yields 2D
        assert result.shape == (2, 4)
        np.testing.assert_array_equal(
            result.values, full[1:3, 2:6, 1]
        )


class TestReadVrtWindowDask:
    """``read_vrt(window=, chunks=)`` returns a dask-chunked DataArray.

    The chunk size must apply to the windowed shape, not the full
    VRT extent. A regression that dropped the window before chunking
    would over-allocate the dask graph.
    """

    def test_window_chunks_returns_dask(self, tmp_path):
        import dask.array as da_mod

        arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        result = read_vrt(vrt, window=(2, 4, 6, 10), chunks=2)

        assert isinstance(result.data, da_mod.Array)
        assert result.shape == (4, 6)
        np.testing.assert_array_equal(
            result.values, arr[2:6, 4:10]
        )


@requires_gpu
class TestReadVrtWindowGpu:
    """``read_vrt(window=, gpu=True)`` returns a CuPy-backed DataArray.

    The eager VRT decode happens on CPU (the internal reader walks
    SimpleSources and assembles); the final ``if gpu: cupy.asarray``
    block uploads the windowed result. Window slicing must happen
    *before* the upload so the GPU array carries only the requested
    pixels.
    """

    def test_window_gpu_returns_cupy(self, tmp_path):
        import cupy

        arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        result = read_vrt(vrt, window=(2, 4, 6, 10), gpu=True)

        assert isinstance(result.data, cupy.ndarray)
        assert result.shape == (4, 6)
        np.testing.assert_array_equal(
            result.data.get(), arr[2:6, 4:10]
        )

    def test_window_gpu_chunks_returns_dask_cupy(self, tmp_path):
        """``window + gpu + chunks`` -> Dask+CuPy with window-sized data."""
        import cupy
        import dask.array as da_mod

        arr = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
        vrt = _make_single_tile_vrt(tmp_path, arr)

        result = read_vrt(vrt, window=(2, 4, 6, 10), gpu=True, chunks=2)

        assert isinstance(result.data, da_mod.Array)
        assert isinstance(result.data._meta, cupy.ndarray)
        assert result.shape == (4, 6)
        np.testing.assert_array_equal(
            result.compute().data.get(), arr[2:6, 4:10]
        )

# ===========================================================================
# XRSPATIAL_GEOTIFF_STRICT typed-warning gate (#1662)
# Source: test_strict_mode_1662.py
# ===========================================================================


@pytest.fixture
def clear_strict_env(monkeypatch):
    """Ensure the strict-mode env var is unset for the default-mode test."""
    monkeypatch.delenv('XRSPATIAL_GEOTIFF_STRICT', raising=False)


@pytest.fixture
def set_strict_env(monkeypatch):
    """Set ``XRSPATIAL_GEOTIFF_STRICT=1`` for strict-mode tests."""
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', '1')


# ---------------------------------------------------------------------------
# _geotiff_strict_mode() helper
# ---------------------------------------------------------------------------

def test_strict_mode_default_false(clear_strict_env):
    assert _geotiff_strict_mode() is False


@pytest.mark.parametrize('value', ['1', 'true', 'True', 'yes', 'YES'])
def test_strict_mode_truthy_values(monkeypatch, value):
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', value)
    assert _geotiff_strict_mode() is True


@pytest.mark.parametrize('value', ['0', 'false', 'no', '', 'maybe'])
def test_strict_mode_falsy_values(monkeypatch, value):
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', value)
    assert _geotiff_strict_mode() is False


# ---------------------------------------------------------------------------
# _wkt_to_epsg
# ---------------------------------------------------------------------------

def test_wkt_to_epsg_default_warns_returns_none(clear_strict_env):
    """In default mode a broken WKT input warns and returns None."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = _wkt_to_epsg('not-a-valid-wkt-string-1662')

    assert result is None
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    assert '_wkt_to_epsg failed' in str(fallback_warnings[0].message)


def test_wkt_to_epsg_strict_reraises(set_strict_env):
    """Under XRSPATIAL_GEOTIFF_STRICT=1, _wkt_to_epsg re-raises."""
    with pytest.raises(Exception):
        _wkt_to_epsg('not-a-valid-wkt-string-1662')


def test_wkt_to_epsg_valid_input_no_warning(clear_strict_env):
    """A real WKT string returns its EPSG without any warning."""
    pyproj = pytest.importorskip('pyproj')
    wkt = pyproj.CRS.from_epsg(4326).to_wkt()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = _wkt_to_epsg(wkt)

    assert result == 4326
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert fallback_warnings == []


# ---------------------------------------------------------------------------
# _epsg_to_wkt
# ---------------------------------------------------------------------------

def test_epsg_to_wkt_default_warns_returns_none(clear_strict_env):
    """An unknown EPSG warns and returns None in default mode."""
    pytest.importorskip('pyproj')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # EPSG 999999 is unassigned; pyproj raises CRSError.
        result = _epsg_to_wkt(999999)

    assert result is None
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    assert '_epsg_to_wkt' in str(fallback_warnings[0].message)


def test_epsg_to_wkt_strict_reraises(set_strict_env):
    """Strict mode re-raises rather than warning."""
    pytest.importorskip('pyproj')
    with pytest.raises(Exception):
        _epsg_to_wkt(999999)


# ---------------------------------------------------------------------------
# VRT source skip
# ---------------------------------------------------------------------------

def test_vrt_missing_source_default_warns_then_continues(
    clear_strict_env, tmp_path,
):
    """A VRT referencing a missing source file warns once and skips it."""
    from xrspatial.geotiff import read_vrt

    vrt_path = tmp_path / 'mosaic_1662_missing.vrt'
    missing_src = f'{tmp_path}/does_not_exist_1662.tif'
    vrt_path.write_text(
        '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        '  <SRS></SRS>\n'
        '  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <NoDataValue>-9999</NoDataValue>\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{missing_src}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # Public ``read_vrt`` defaults to ``missing_sources='raise'``
        # since #1860. Opt back into the lenient warn-then-continue
        # behaviour to keep exercising the warning path.
        da = read_vrt(str(vrt_path), missing_sources='warn')

    # The mosaic should still load (with a hole) and one warning should
    # describe the skipped source.
    assert da.shape == (4, 4)
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) >= 1
    msgs = ' '.join(str(x.message) for x in fallback_warnings)
    assert 'VRT source' in msgs
    assert 'does_not_exist_1662' in msgs


def test_vrt_missing_source_strict_raises(set_strict_env, tmp_path):
    """In strict mode the missing source surfaces as an exception."""
    from xrspatial.geotiff import read_vrt

    vrt_path = tmp_path / 'mosaic_1662_missing_strict.vrt'
    missing_src = f'{tmp_path}/does_not_exist_1662_strict.tif'
    vrt_path.write_text(
        '<VRTDataset rasterXSize="4" rasterYSize="4">\n'
        '  <SRS></SRS>\n'
        '  <GeoTransform>0, 1, 0, 0, 0, -1</GeoTransform>\n'
        '  <VRTRasterBand dataType="Float32" band="1">\n'
        '    <NoDataValue>-9999</NoDataValue>\n'
        '    <SimpleSource>\n'
        f'      <SourceFilename relativeToVRT="0">{missing_src}'
        '</SourceFilename>\n'
        '      <SourceBand>1</SourceBand>\n'
        '      <SrcRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '      <DstRect xOff="0" yOff="0" xSize="4" ySize="4"/>\n'
        '    </SimpleSource>\n'
        '  </VRTRasterBand>\n'
        '</VRTDataset>\n'
    )

    with pytest.raises(Exception):
        read_vrt(str(vrt_path))


# ---------------------------------------------------------------------------
# _warn_or_raise_gpu_fallback pins the warn-vs-raise contract for every
# GPU helper site flagged in #1662. Exercised directly so the test does
# not depend on having a working CUDA/GDS/nvCOMP stack.
# ---------------------------------------------------------------------------


def test_warn_or_raise_gpu_fallback_default_warns(clear_strict_env):
    """Default mode emits one GeoTIFFFallbackWarning carrying type + msg."""
    from xrspatial.geotiff._gpu_decode import _warn_or_raise_gpu_fallback

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = _warn_or_raise_gpu_fallback(
            "_try_nvjpeg_batch_decode", RuntimeError("bogus 1662"))

    # Default mode returns False so the caller falls back to None.
    assert result is False
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert len(fallback_warnings) == 1
    msg = str(fallback_warnings[0].message)
    assert '_try_nvjpeg_batch_decode' in msg
    assert 'RuntimeError' in msg
    assert 'bogus 1662' in msg


def test_warn_or_raise_gpu_fallback_strict_returns_true(set_strict_env):
    """Strict mode returns True so the caller can ``raise`` itself.

    The helper deliberately does not raise here -- re-raising the
    captured exception from inside this frame would clobber the
    original traceback. Returning True lets each call site bubble the
    live exception up via a bare ``raise`` from its own ``except``
    block.
    """
    from xrspatial.geotiff._gpu_decode import _warn_or_raise_gpu_fallback

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = _warn_or_raise_gpu_fallback(
            "_try_nvjpeg_batch_decode", RuntimeError("bogus 1662 strict"))

    assert result is True
    # No warning should be emitted in strict mode.
    fallback_warnings = [
        x for x in w if issubclass(x.category, GeoTIFFFallbackWarning)
    ]
    assert fallback_warnings == []


def test_warn_or_raise_gpu_fallback_preserves_traceback(set_strict_env):
    """The call-site pattern preserves the original exception traceback.

    The helper returns True in strict mode; the caller re-raises with a
    bare ``raise``. The resulting traceback should point at the original
    failure (``raise RuntimeError(...)`` below), not at the helper.
    """
    from xrspatial.geotiff._gpu_decode import _warn_or_raise_gpu_fallback

    def site():
        try:
            raise RuntimeError("bogus 1662 traceback")
        except Exception as e:
            if _warn_or_raise_gpu_fallback("_dummy_stage", e):
                raise
            return None

    with pytest.raises(RuntimeError, match='bogus 1662 traceback') as excinfo:
        site()
    # The deepest traceback frame should be ``site``'s ``raise
    # RuntimeError`` line, not ``_warn_or_raise_gpu_fallback``.
    tb = excinfo.tb
    while tb.tb_next is not None:
        tb = tb.tb_next
    assert tb.tb_frame.f_code.co_name == 'site'


# ---------------------------------------------------------------------------
# read_geotiff_gpu on_gpu_failure='auto' + env var integration
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy
        return bool(cupy.cuda.is_available())
    except Exception:
        return False


_HAS_GPU = _gpu_available()


@pytest.mark.skipif(
    not _HAS_GPU,
    reason="cupy + CUDA required for read_geotiff_gpu fallback test",
)
def test_read_geotiff_gpu_env_var_promotes_to_strict(monkeypatch, tmp_path):
    """With on_gpu_failure='auto' but XRSPATIAL_GEOTIFF_STRICT=1, a GPU
    decode failure surfaces instead of falling back to CPU.

    The seam: ``read_geotiff_gpu`` does a local
    ``from ._gpu_decode import gpu_decode_tiles_from_file`` inside the
    function body, so rebinding the attribute on the
    ``xrspatial.geotiff._gpu_decode`` module is picked up on the next
    call. Stubbing it to raise lets us exercise both branches against
    a real on-disk TIF without needing a broken GPU stack.
    """
    import cupy as cp
    import numpy as np
    import xarray as xr

    from xrspatial.geotiff import _gpu_decode, read_geotiff_gpu, to_geotiff

    # 1. Write a small valid TIF so the metadata parse succeeds and we
    # reach the GPU decode stage.
    h, w = 16, 16
    arr = np.arange(h * w, dtype=np.float32).reshape(h, w)
    y = np.arange(h, dtype=np.float64) * -1.0 + 100.0
    x = np.arange(w, dtype=np.float64) * 1.0 + 0.0
    da = xr.DataArray(arr, dims=['y', 'x'], coords={'y': y, 'x': x})
    p = str(tmp_path / 'strict_promote_1662.tif')
    to_geotiff(da, p, crs=4326)

    sentinel = "bogus 1662 promote"

    def _boom(*args, **kwargs):
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles_from_file', _boom)
    monkeypatch.setattr(
        _gpu_decode, 'gpu_decode_tiles', _boom)

    # 2. Default mode: XRSPATIAL_GEOTIFF_STRICT unset, the failure
    # should be absorbed and the CPU fallback should return a
    # CuPy-backed DataArray.
    monkeypatch.delenv('XRSPATIAL_GEOTIFF_STRICT', raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = read_geotiff_gpu(p)
    assert isinstance(result, xr.DataArray)
    assert isinstance(result.data, cp.ndarray)

    # 3. With XRSPATIAL_GEOTIFF_STRICT=1, the same call should re-raise
    # the patched RuntimeError instead of falling back.
    monkeypatch.setenv('XRSPATIAL_GEOTIFF_STRICT', '1')
    with pytest.raises(RuntimeError, match=sentinel):
        read_geotiff_gpu(p)

# ===========================================================================
# Public namespace no-leak regression (#1708)
# Source: test_namespace_no_leak.py
# ===========================================================================


def test_read_to_array_not_in_module_namespace():
    """``read_to_array`` is internal and must not be a public attribute
    of ``xrspatial.geotiff``. Tests and internal callers go through
    ``xrspatial.geotiff._reader`` directly."""
    assert not hasattr(g, 'read_to_array'), (
        "read_to_array leaked into xrspatial.geotiff's public namespace. "
        "It should be bound as _read_to_array (underscore-prefix) so it "
        "does not appear in `from xrspatial.geotiff import *` or tab "
        "completion. See issue #1708."
    )


def test_read_to_array_not_in_all():
    """``__all__`` does not advertise ``read_to_array``. (Pre-existing
    state, pinned so a future change can't quietly promote it without
    also adding it to the module-level Public API docstring.)"""
    assert 'read_to_array' not in g.__all__


def test_read_to_array_still_importable_from_reader_submodule():
    """The function itself is still reachable via the canonical private
    path so tests and internal code keep working."""
    from xrspatial.geotiff._reader import read_to_array

    assert callable(read_to_array)


def test_import_from_top_level_now_fails():
    """Direct top-level import is the regression we are guarding
    against. Before #1708 this succeeded; afterwards it must raise
    ImportError."""
    with pytest.raises(ImportError):
        from xrspatial.geotiff import read_to_array  # noqa: F401
