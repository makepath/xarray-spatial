"""Cross-backend signature, docstring, and release-contract parity.

Sibling to ``parity/test_backend_matrix.py``. These tests pin the
*advertised* contract of the public GeoTIFF surface so the writer trio and
the reader quartet stay consistent with each other and with the docs.
Three sections, each a former top-level file:

Section 1 -- Writer signature / docstring parity (#1631)
    ``write_vrt`` exposes its documented kwargs through an explicit
    signature (no ``**kwargs`` catch-all), ``write_geotiff_gpu`` lists
    ``'cubic'`` in its ``overview_resampling`` docstring, and its
    ``data`` parameter carries the same type hint as ``to_geotiff``.

Section 2 -- Read entry-point docstring/param parity (#2274)
    Every signature kwarg on ``open_geotiff`` / ``read_geotiff_dask`` /
    ``read_geotiff_gpu`` / ``read_vrt`` has a matching numpy-style
    Parameters entry, and every Parameters entry maps to a real kwarg.

Section 3 -- Release-contract tier parity (#2389)
    ``docs/source/reference/geotiff_release_contract.md`` promises its
    tier strings match ``SUPPORTED_FEATURES`` at runtime. This section
    parses the contract table and asserts every key/tier pair agrees.

GPU rows skip when cupy + CUDA are absent via the shared ``requires_gpu``
marker from ``_helpers/markers.py``.
"""
from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import (SUPPORTED_FEATURES, open_geotiff, read_geotiff_dask,
                               read_geotiff_gpu, read_vrt, to_geotiff, write_geotiff_gpu, write_vrt)

from .._helpers.markers import requires_gpu

# ===========================================================================
# Section 1 -- Writer signature / docstring parity (#1631)
# ===========================================================================
#
# Three drifts flagged by the api-consistency sweep on 2026-05-11:
# ``write_vrt`` swallowed every kwarg into ``**kwargs`` so the documented
# ``relative`` / ``crs`` / ``nodata`` were invisible to ``inspect.signature``;
# ``write_geotiff_gpu``'s ``overview_resampling`` docstring omitted
# ``'cubic'``; and ``write_geotiff_gpu(data, ...)`` lacked the type hint
# ``to_geotiff(data, ...)`` carries.


def test_write_vrt_signature_exposes_documented_kwargs():
    """``inspect.signature(write_vrt)`` reports the four accepted kwargs.

    Prior to #1631 the public wrapper used ``**kwargs``, so
    ``inspect.signature`` only saw ``vrt_path`` and ``source_files``.
    Issue #1715 added ``crs`` for parity with ``to_geotiff`` /
    ``write_geotiff_gpu`` while keeping the historic ``crs_wkt`` as a
    deprecated alias (sentinel default so the deprecation shim can
    tell "user passed nothing" from "user passed crs_wkt=None").
    """
    sig = inspect.signature(write_vrt)
    params = sig.parameters
    assert 'relative' in params
    assert 'crs' in params  # added in #1715
    assert 'crs_wkt' in params  # deprecated alias
    assert 'nodata' in params
    assert params['relative'].default is True
    # ``crs`` is the new canonical kwarg; default None means "pick from
    # the first source", matching to_geotiff / write_geotiff_gpu.
    assert params['crs'].default is None
    # ``crs_wkt`` carries a sentinel default so the deprecation shim
    # can distinguish "user passed nothing" (no warning) from "user
    # passed crs_wkt=None" (deprecated-but-explicit, warn). The
    # sentinel itself is private; check that it is NOT None so a
    # future maintainer cannot accidentally drop the sentinel logic.
    assert params['crs_wkt'].default is not None
    assert params['crs_wkt'].default is not inspect.Parameter.empty
    assert params['nodata'].default is None
    # No catch-all VAR_KEYWORD
    kinds = {p.kind for p in params.values()}
    assert inspect.Parameter.VAR_KEYWORD not in kinds


def test_write_vrt_unknown_kwarg_rejected_at_public_level(tmp_path):
    """A typo'd kwarg now raises ``TypeError`` from the public function
    rather than from deep inside ``_vrt.write_vrt``.
    """
    arr = np.zeros((8, 8), dtype=np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'transform': (1.0, 0, 0.0, 0, -1.0, 8.0)},
    )
    tif_path = str(tmp_path / 't.tif')
    to_geotiff(da, tif_path)

    with pytest.raises(TypeError, match='typo_kwarg'):
        write_vrt(str(tmp_path / 't.vrt'), [tif_path], typo_kwarg=1)


def test_write_vrt_accepts_documented_kwargs(tmp_path):
    """Each documented kwarg round-trips through the explicit signature.

    Uses the new ``crs=None`` kwarg form (issue #1715). The deprecated
    ``crs_wkt`` alias is exercised separately in
    ``test_write_vrt_crs_1715.py``.
    """
    arr = np.zeros((8, 8), dtype=np.float32)
    da = xr.DataArray(
        arr, dims=['y', 'x'],
        coords={'y': np.arange(8.0, 0, -1), 'x': np.arange(8.0)},
        attrs={'crs': 4326, 'transform': (1.0, 0, 0.0, 0, -1.0, 8.0)},
    )
    tif_path = str(tmp_path / 't.tif')
    to_geotiff(da, tif_path)

    vrt_path = str(tmp_path / 't.vrt')
    out = write_vrt(
        vrt_path, [tif_path],
        relative=False, crs=None, nodata=-9999.0,
    )
    assert out == vrt_path
    assert os.path.exists(vrt_path)


def test_write_geotiff_gpu_docstring_lists_cubic():
    """``overview_resampling`` docstring includes ``'cubic'`` so it
    matches ``to_geotiff`` and the underlying ``make_overview_gpu``.
    """
    doc = write_geotiff_gpu.__doc__
    assert doc is not None
    # Find the overview_resampling block
    assert 'overview_resampling' in doc
    # The block must mention cubic
    block_start = doc.index('overview_resampling')
    block_end = doc.index('bigtiff', block_start)
    block = doc[block_start:block_end]
    assert 'cubic' in block


def test_write_geotiff_gpu_data_has_type_hint():
    """``data`` parameter is annotated, matching ``to_geotiff(data, ...)``.

    The annotation also covers ``np.ndarray`` because the implementation
    accepts numpy inputs (uploaded via ``cupy.asarray(np.asarray(data))``)
    and the test suite exercises that path (e.g.
    ``test_backend_kwarg_parity_1561.py`` passes a numpy ``dummy``).
    """
    sig = inspect.signature(write_geotiff_gpu)
    data_param = sig.parameters['data']
    assert data_param.annotation is not inspect.Parameter.empty
    # The annotation is a forward reference under ``from __future__ import
    # annotations``; just confirm it mentions the documented types.
    ann_str = str(data_param.annotation)
    assert 'DataArray' in ann_str
    assert 'cupy' in ann_str
    assert 'ndarray' in ann_str  # numpy parity vs to_geotiff


@requires_gpu
def test_write_geotiff_gpu_cubic_overview_round_trip(tmp_path):
    """``overview_resampling='cubic'`` works on the GPU writer.

    Sanity check that the docstring update is not advertising an
    unsupported codec. ``make_overview_gpu`` falls back to the CPU
    cubic implementation for parity with the CPU writer.
    """
    import cupy

    arr_cpu = np.random.RandomState(0).rand(256, 256).astype(np.float32)
    arr_gpu = cupy.asarray(arr_cpu)
    da_gpu = xr.DataArray(
        arr_gpu, dims=['y', 'x'],
        coords={'y': np.arange(256.0, 0, -1), 'x': np.arange(256.0)},
    )
    path = str(tmp_path / 'cog.tif')
    write_geotiff_gpu(
        da_gpu, path,
        cog=True, tile_size=64, overview_resampling='cubic',
    )
    # Overview level 1 = 1/2 resolution
    ov = open_geotiff(path, overview_level=1)
    assert ov.shape == (128, 128)


# ===========================================================================
# Section 2 -- Read entry-point docstring / param parity (#2274)
# ===========================================================================
#
# The four read entry points accept ``allow_rotated`` and
# ``allow_unparseable_crs`` plus several gated kwargs whose only purpose is
# to raise ``ValueError`` on the wrong backend so all four readers stay
# error-symmetric. Those kwargs were missing Parameters-section entries on
# the backends that reject them.

READ_ENTRY_POINTS = (
    open_geotiff,
    read_geotiff_dask,
    read_geotiff_gpu,
    read_vrt,
)


# Numpy-style docstring parameter heading pattern. Matches lines like
# ``    name : type`` after ``inspect.getdoc`` has normalised the
# leading indentation to column zero.
_PARAM_HEADING = re.compile(r"^(\w+) : ", flags=re.MULTILINE)


def _signature_params(fn):
    return set(inspect.signature(fn).parameters)


def _documented_params(fn):
    doc = inspect.getdoc(fn) or ""
    return set(_PARAM_HEADING.findall(doc))


@pytest.mark.parametrize("fn", READ_ENTRY_POINTS, ids=lambda f: f.__name__)
def test_read_entry_point_kwargs_have_docstring_entries(fn):
    """Every signature kwarg appears in the Parameters section."""
    params = _signature_params(fn)
    documented = _documented_params(fn)
    missing = sorted(params - documented)
    assert missing == [], (
        f"{fn.__name__} has kwargs without Parameters-section entries: "
        f"{missing}. Add a numpy-style ``name : type`` heading for each "
        f"so the docstring agrees with the signature. The kwargs may be "
        f"gated (raise ValueError on the wrong backend) but they are "
        f"still on the public surface, and tools that read the "
        f"docstring (Sphinx, IDE help) cannot tell the kwarg exists "
        f"without an entry. See #2274."
    )


@pytest.mark.parametrize("fn", READ_ENTRY_POINTS, ids=lambda f: f.__name__)
def test_read_entry_point_docstring_does_not_invent_params(fn):
    """Every Parameters entry maps to a real signature kwarg.

    Catches the inverse drift: a kwarg removed from the signature but
    still listed in the Parameters section.
    """
    params = _signature_params(fn)
    documented = _documented_params(fn)
    extra = sorted(documented - params)
    assert extra == [], (
        f"{fn.__name__} has Parameters-section entries that do not "
        f"appear in the signature: {extra}. Either remove the entry "
        f"or restore the kwarg."
    )


@pytest.mark.parametrize("fn", READ_ENTRY_POINTS, ids=lambda f: f.__name__)
def test_allow_rotated_documented(fn):
    """``allow_rotated`` was the load-bearing #2274 gap on the backends.

    Pin it explicitly so a future commit that strips the Parameters
    entry while keeping the signature kwarg fails loudly.
    """
    assert "allow_rotated" in _signature_params(fn), (
        f"{fn.__name__} unexpectedly dropped allow_rotated from its "
        f"signature"
    )
    assert "allow_rotated" in _documented_params(fn), (
        f"{fn.__name__} accepts allow_rotated but does not document it "
        f"in its Parameters section (#2274)."
    )


@pytest.mark.parametrize("fn", READ_ENTRY_POINTS, ids=lambda f: f.__name__)
def test_allow_unparseable_crs_documented(fn):
    """``allow_unparseable_crs`` was the other shared #2274 gap.

    ``open_geotiff`` had the kwarg only in the Tier prose paragraph;
    the three backends did not mention it at all.
    """
    assert "allow_unparseable_crs" in _signature_params(fn), (
        f"{fn.__name__} unexpectedly dropped allow_unparseable_crs from "
        f"its signature"
    )
    assert "allow_unparseable_crs" in _documented_params(fn), (
        f"{fn.__name__} accepts allow_unparseable_crs but does not "
        f"document it in its Parameters section (#2274)."
    )


# ===========================================================================
# Section 3 -- Release-contract tier parity (#2389)
# ===========================================================================
#
# ``docs/source/reference/geotiff_release_contract.md`` lists every public
# GeoTIFF feature with its tier and promises the tier strings match
# ``SUPPORTED_FEATURES`` at runtime. Nothing in CI checked that claim before
# this gate, so the contract drifted twice in two releases (#2381, #2389).

_HERE = Path(__file__).resolve()
# tests/parity/test_signature_contract.py -> parents: parity, tests,
# geotiff, xrspatial, REPO_ROOT. Matches the depth that
# release_gates/test_stable_features.py uses (parents[4]).
_REPO_ROOT = _HERE.parents[4]
_CONTRACT = (
    _REPO_ROOT / "docs" / "source" / "reference" / "geotiff_release_contract.md"
)

# Match table rows of the form:
#   | `codec.none` | stable | Uncompressed... |
# The key column is always in backticks; the tier column is the bare
# tier label that should appear verbatim in SUPPORTED_FEATURES.
_ROW_RE = re.compile(
    r"^\|\s*`([a-z_]+\.[a-z0-9_]+)`\s*\|\s*([a-z_]+)\s*\|",
    re.MULTILINE,
)


def _contract_rows() -> list[tuple[str, tuple[str, str]]]:
    """Return ``(line_number_hint, (key, tier))`` for every table row.

    The line-number hint is the 1-based offset of the match inside the
    file so assertion failures can point a maintainer at the exact row.
    """
    text = _CONTRACT.read_text(encoding="utf-8")
    rows: list[tuple[str, tuple[str, str]]] = []
    for match in _ROW_RE.finditer(text):
        line_no = text.count("\n", 0, match.start()) + 1
        rows.append((f"{_CONTRACT.name}:{line_no}", (match.group(1), match.group(2))))
    return rows


def test_contract_table_parses_into_rows() -> None:
    """The regex catches the table rows. If a future doc rewrite breaks
    the row shape, fail loudly here instead of silently passing the
    tier check on zero rows.
    """
    rows = _contract_rows()
    assert rows, (
        f"no contract rows parsed from {_CONTRACT}; the markdown table "
        "shape may have changed and this test's regex needs to follow."
    )
    # Sanity floor: the contract today lists roughly 28 keys. Use a
    # conservative lower bound so a sweeping accidental table truncation
    # fails the gate. The exact count is not pinned; tiers move.
    assert len(rows) >= 20, (
        f"only {len(rows)} contract rows parsed; the table may have been "
        "truncated or the row format changed."
    )


def test_contract_keys_are_real_supported_features() -> None:
    """Every key in the contract table exists in ``SUPPORTED_FEATURES``.
    A stray row left behind after a key is removed from ``_attrs.py``
    fails here.
    """
    bad: list[tuple[str, str]] = []
    for where, (key, _tier) in _contract_rows():
        if key not in SUPPORTED_FEATURES:
            bad.append((where, key))
    assert not bad, (
        "contract table lists keys that are not in SUPPORTED_FEATURES; "
        "either the key was removed from _attrs.py and the doc row was "
        "left behind, or the row's backticked text is wrong: "
        f"{bad}"
    )


def test_contract_tiers_match_supported_features() -> None:
    """Every row's tier column matches ``SUPPORTED_FEATURES[key]``.
    This is the gate that would have caught the #2381 / #2389 drift.
    """
    mismatches: list[tuple[str, str, str, str]] = []
    for where, (key, tier) in _contract_rows():
        if key not in SUPPORTED_FEATURES:
            # Reported by ``test_contract_keys_are_real_supported_features``;
            # skip here to keep this failure focused on tier drift.
            continue
        expected = SUPPORTED_FEATURES[key]
        if tier != expected:
            mismatches.append((where, key, tier, expected))
    assert not mismatches, (
        "contract page tier strings disagree with SUPPORTED_FEATURES; "
        "the contract page promises the two match verbatim. Update the "
        "tier column in geotiff_release_contract.md to the runtime tier "
        "(format: (where, key, doc_tier, runtime_tier)): "
        f"{mismatches}"
    )
