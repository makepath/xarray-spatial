"""Tier inventory + opt-in gates for the geotiff public surface.

Background
----------
Issue #2137 tiers the geotiff public surface into Stable / Advanced /
Experimental / Internal-only. The :data:`SUPPORTED_FEATURES` constant
enumerates every feature with its tier and is the single source of
truth that the docstrings, writer gates, and user-guide notebook all
read from. The writer adds an ``allow_experimental_codecs=True``
opt-in for Tier 3 codecs (``lerc``, ``jpeg2000`` / ``j2k``, ``lz4``)
modelled on the existing ``allow_internal_only_jpeg`` flag for the
Tier 4 ``jpeg`` codec (#1845).

What this test pins
-------------------
* The mapping covers every codec name listed in ``_VALID_COMPRESSIONS``
  -- callers cannot ship a codec the tiering does not classify.
* Stable codecs accept a default ``to_geotiff`` call.
* Every Tier 3 codec is rejected by default and accepted when the
  caller passes ``allow_experimental_codecs=True``; the rejection
  message names the flag and the opt-in emits
  ``GeoTIFFFallbackWarning`` once per call.
* The Tier 4 ``jpeg`` codec rejects without ``allow_internal_only_jpeg``
  and is NOT covered by ``allow_experimental_codecs`` -- the two flags
  do not collapse into one switch.
* The signature of ``to_geotiff`` and ``write_geotiff_gpu`` carries
  the new kwarg with a ``False`` default.
"""
from __future__ import annotations

import inspect
import os
import warnings

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (SUPPORTED_FEATURES, GeoTIFFFallbackWarning, to_geotiff,
                               write_geotiff_gpu)
from xrspatial.geotiff._attrs import _VALID_COMPRESSIONS

_TIER_VALUES = {'stable', 'advanced', 'experimental', 'internal_only'}


def _make_float32_da(h: int = 32, w: int = 32) -> xr.DataArray:
    """Small float32 raster with axis-aligned coords; round-trips
    through every Tier 1 codec and exercises the experimental codec
    gate without exhausting CI time.
    """
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


def _make_uint8_da(h: int = 32, w: int = 32) -> xr.DataArray:
    """uint8 raster for codecs (jpeg2000 / j2k via glymur) that only
    accept integer input.
    """
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, size=(h, w), dtype=np.uint8)
    return xr.DataArray(
        arr,
        dims=("y", "x"),
        coords={
            "y": np.arange(h, dtype=np.float64),
            "x": np.arange(w, dtype=np.float64),
        },
        attrs={'crs': 4326},
    )


# Some Tier 3 codecs constrain the supported input dtype (glymur's
# JPEG2000 encoder accepts only uint8/uint16). Pick the dtype that
# exercises the actual encode without re-litigating per-codec limits.
_EXPERIMENTAL_CODEC_INPUT = {
    'jpeg2000': _make_uint8_da,
    'j2k': _make_uint8_da,
    'lerc': _make_float32_da,
    'lz4': _make_float32_da,
}


def test_supported_features_is_a_mapping():
    """``SUPPORTED_FEATURES`` is a non-empty mapping from feature name
    to tier label. The notebook and the test suite both iterate it, so
    accidental removal would break the documentation generator and the
    parity matrix's tier-aware selection.
    """
    assert isinstance(SUPPORTED_FEATURES, dict)
    assert len(SUPPORTED_FEATURES) > 0
    for name, tier in SUPPORTED_FEATURES.items():
        assert isinstance(name, str) and '.' in name, name
        assert tier in _TIER_VALUES, (name, tier)


def test_supported_features_has_split_cog_keys():
    """The COG entry is split into three keys (issue #2291) so the
    writer, local reader, and HTTP reader can promote between tiers on
    independent tracks. All three keys must resolve in
    ``SUPPORTED_FEATURES`` with a known tier label.

    Pinned here so a future refactor that folds the keys back together
    has to update the docs, the notebook, and this test in one commit.
    """
    for key in ('writer.cog', 'reader.local_cog', 'reader.http_cog'):
        assert key in SUPPORTED_FEATURES, (
            f"{key!r} missing from SUPPORTED_FEATURES; the split was "
            "introduced in #2291 and must stay surfaced so the writer / "
            "local reader / HTTP reader tracks stay independent."
        )
        assert SUPPORTED_FEATURES[key] in _TIER_VALUES, (
            key, SUPPORTED_FEATURES[key])


def test_supported_features_covers_every_valid_codec():
    """Every codec name in ``_VALID_COMPRESSIONS`` carries a tier in
    ``SUPPORTED_FEATURES``. The gate cannot silently miss a codec.
    """
    classified = {
        name.split('.', 1)[1].lower()
        for name in SUPPORTED_FEATURES
        if name.startswith('codec.')
    }
    for codec in _VALID_COMPRESSIONS:
        assert codec.lower() in classified, (
            f"codec {codec!r} is in _VALID_COMPRESSIONS but missing from "
            "SUPPORTED_FEATURES; add a 'codec.<name>' entry classified "
            "into one of stable / experimental / internal_only.")


def test_to_geotiff_signature_has_allow_experimental_codecs():
    """``to_geotiff`` exposes ``allow_experimental_codecs=False``.

    Pinning the signature catches accidental removal during future
    refactors: if the kwarg disappears, the writer silently drops back
    to the unconditional acceptance of Tier 3 codecs and the issue
    regresses.
    """
    params = inspect.signature(to_geotiff).parameters
    assert 'allow_experimental_codecs' in params
    assert params['allow_experimental_codecs'].default is False


def test_write_geotiff_gpu_signature_has_allow_experimental_codecs():
    """``write_geotiff_gpu`` carries the same kwarg with the same
    default, so the two writers expose a consistent surface and the
    auto-dispatch path forwards a single value to either.
    """
    params = inspect.signature(write_geotiff_gpu).parameters
    assert 'allow_experimental_codecs' in params
    assert params['allow_experimental_codecs'].default is False


@pytest.mark.parametrize(
    "codec",
    sorted(
        name.split('.', 1)[1]
        for name, tier in SUPPORTED_FEATURES.items()
        if name.startswith('codec.') and tier == 'stable'
    ),
)
def test_stable_codecs_accept_default_call(tmp_path, codec):
    """Tier 1 codecs round-trip a small float32 raster with no flags.
    A regression that accidentally gates a stable codec behind the new
    flag would surface here.
    """
    da = _make_float32_da()
    path = os.path.join(str(tmp_path), f'stable_{codec}_2137.tif')
    out = to_geotiff(da, path, compression=codec)
    assert out == path
    assert os.path.exists(path)


@pytest.mark.parametrize(
    "codec",
    sorted(
        name.split('.', 1)[1]
        for name, tier in SUPPORTED_FEATURES.items()
        if name.startswith('codec.') and tier == 'experimental'
    ),
)
def test_experimental_codec_rejected_by_default(tmp_path, codec):
    """Tier 3 codecs raise ``ValueError`` whose message names the
    ``allow_experimental_codecs`` flag so the caller learns the
    opt-in name from the rejection itself.
    """
    da = _make_float32_da()
    path = os.path.join(str(tmp_path), f'reject_{codec}_2137.tif')
    with pytest.raises(ValueError, match='allow_experimental_codecs'):
        to_geotiff(da, path, compression=codec)


@pytest.mark.parametrize(
    "codec",
    sorted(
        name.split('.', 1)[1]
        for name, tier in SUPPORTED_FEATURES.items()
        if name.startswith('codec.') and tier == 'experimental'
    ),
)
def test_experimental_codec_opt_in_emits_warning(tmp_path, codec):
    """``allow_experimental_codecs=True`` lets the codec through and
    emits ``GeoTIFFFallbackWarning`` once per call. The warning shape
    matches the existing ``allow_internal_only_jpeg`` opt-in so docs
    and downstream warning filters can target a single class.
    """
    da = _EXPERIMENTAL_CODEC_INPUT.get(codec, _make_float32_da)()
    path = os.path.join(str(tmp_path), f'optin_{codec}_2137.tif')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            to_geotiff(da, path, compression=codec,
                       allow_experimental_codecs=True)
        except (ImportError, ModuleNotFoundError) as e:
            # ``jpeg2000`` / ``j2k`` need glymur; ``lerc`` needs a
            # codec backend. The opt-in warning still fires before the
            # encode runs, so the warning assertion below holds even
            # when the optional dependency is missing on the runner.
            pytest.skip(f"optional dependency missing for {codec}: {e}")
    fallback = [w for w in caught
                if issubclass(w.category, GeoTIFFFallbackWarning)]
    assert fallback, (
        f"to_geotiff(compression={codec!r}, allow_experimental_codecs="
        "True) must emit GeoTIFFFallbackWarning so the caller knows "
        "the codec carries no cross-backend parity claim.")
    # Exactly one warning per call. Pinning the count catches the
    # double-warn regression where the CPU dispatcher fires the
    # warning and then ``write_geotiff_gpu`` fires it again on the GPU
    # dispatch path; the CPU dispatcher gates its warning on
    # ``not use_gpu`` to keep this invariant on the GPU path too.
    assert len(fallback) == 1, (
        f"expected exactly one GeoTIFFFallbackWarning for "
        f"to_geotiff(compression={codec!r}, allow_experimental_codecs="
        f"True); got {len(fallback)}: "
        f"{[str(w.message) for w in fallback]}")
    # Warning text names both the codec and the opt-in flag so logs
    # are self-describing rather than pointing to a docs URL.
    msg = str(fallback[0].message)
    assert 'allow_experimental_codecs' in msg
    assert codec in msg


def test_jpeg_internal_only_not_covered_by_experimental_flag(tmp_path):
    """``allow_experimental_codecs=True`` does NOT unlock
    ``compression='jpeg'`` -- internal-only is the strictest tier and
    keeps its own dedicated flag (``allow_internal_only_jpeg``). The
    two flags do not collapse into one switch.
    """
    da = _make_float32_da().astype(np.uint8)
    path = os.path.join(str(tmp_path), 'jpeg_only_experimental_2137.tif')
    with pytest.raises(ValueError, match='allow_internal_only_jpeg'):
        to_geotiff(
            da, path, compression='jpeg',
            allow_experimental_codecs=True,
        )


def test_jpeg_rejected_without_its_own_flag(tmp_path):
    """``compression='jpeg'`` without ``allow_internal_only_jpeg=True``
    raises ``ValueError`` whose message names the dedicated flag.
    Pinned here so the Tier 4 contract sits alongside the Tier 3
    contract in one file.
    """
    da = _make_float32_da().astype(np.uint8)
    path = os.path.join(str(tmp_path), 'jpeg_no_flag_2137.tif')
    with pytest.raises(ValueError, match='allow_internal_only_jpeg'):
        to_geotiff(da, path, compression='jpeg')
