"""Tests for xrspatial.corridor.least_cost_corridor."""

try:
    import dask.array as da
except ImportError:
    da = None

import numpy as np
import pytest
import xarray as xr

from xrspatial.corridor import least_cost_corridor
from xrspatial.cost_distance import cost_distance
from xrspatial.utils import has_cuda_and_cupy


def _make_raster(data, backend="numpy", chunks=(3, 3)):
    """Build a DataArray with y/x coords, optionally dask/cupy-backed."""
    h, w = data.shape
    raster = xr.DataArray(
        data.astype(np.float64),
        dims=["y", "x"],
        attrs={"res": (1.0, 1.0)},
    )
    raster["y"] = np.arange(h, dtype=np.float64)
    raster["x"] = np.arange(w, dtype=np.float64)
    if "dask" in backend and da is not None:
        raster.data = da.from_array(raster.data, chunks=chunks)
    if "cupy" in backend and has_cuda_and_cupy():
        import cupy

        if isinstance(raster.data, da.Array):
            raster.data = raster.data.map_blocks(cupy.asarray)
        else:
            raster.data = cupy.asarray(raster.data)
    return raster


def _compute(arr):
    """Extract numpy data from DataArray (works for numpy, dask, or cupy)."""
    if da is not None and isinstance(arr.data, da.Array):
        val = arr.data.compute()
        if hasattr(val, "get"):
            return val.get()
        return val
    if hasattr(arr.data, "get"):
        return arr.data.get()
    return arr.data


# -----------------------------------------------------------------------
# Basic corridor correctness
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend", ["numpy", "dask+numpy", "cupy", "dask+cupy"]
)
def test_basic_corridor_symmetry(backend):
    """Corridor between two sources on uniform friction is symmetric."""
    n = 7
    friction_data = np.ones((n, n))

    src_a = np.zeros((n, n))
    src_a[3, 0] = 1.0  # left edge

    src_b = np.zeros((n, n))
    src_b[3, 6] = 1.0  # right edge

    friction = _make_raster(friction_data, backend=backend, chunks=(7, 7))
    sa = _make_raster(src_a, backend=backend, chunks=(7, 7))
    sb = _make_raster(src_b, backend=backend, chunks=(7, 7))

    result = least_cost_corridor(friction, sa, sb)
    out = _compute(result)

    # Minimum corridor cost should be 0 (after normalization)
    assert np.nanmin(out) == pytest.approx(0.0, abs=1e-5)

    # Corridor should be symmetric about the vertical midline
    np.testing.assert_allclose(out[:, :3], out[:, -1:-4:-1], atol=1e-5)


@pytest.mark.parametrize(
    "backend", ["numpy", "dask+numpy", "cupy", "dask+cupy"]
)
def test_corridor_minimum_on_optimal_path(backend):
    """Cells on the optimal path between sources have corridor value 0."""
    n = 5
    friction_data = np.ones((n, n))

    src_a = np.zeros((n, n))
    src_a[2, 0] = 1.0

    src_b = np.zeros((n, n))
    src_b[2, 4] = 1.0

    friction = _make_raster(friction_data, backend=backend, chunks=(5, 5))
    sa = _make_raster(src_a, backend=backend, chunks=(5, 5))
    sb = _make_raster(src_b, backend=backend, chunks=(5, 5))

    result = least_cost_corridor(friction, sa, sb)
    out = _compute(result)

    # The middle row (row 2) should be the optimal path on uniform friction.
    # All cells on row 2 should have the minimum corridor value (0).
    for col in range(n):
        assert out[2, col] == pytest.approx(0.0, abs=1e-5)


# -----------------------------------------------------------------------
# Threshold tests
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend", ["numpy", "dask+numpy", "cupy", "dask+cupy"]
)
def test_absolute_threshold(backend):
    """Absolute threshold masks cells with normalized cost > threshold."""
    n = 7
    friction_data = np.ones((n, n))

    src_a = np.zeros((n, n))
    src_a[3, 0] = 1.0

    src_b = np.zeros((n, n))
    src_b[3, 6] = 1.0

    friction = _make_raster(friction_data, backend=backend, chunks=(7, 7))
    sa = _make_raster(src_a, backend=backend, chunks=(7, 7))
    sb = _make_raster(src_b, backend=backend, chunks=(7, 7))

    result = least_cost_corridor(friction, sa, sb, threshold=0.5)
    out = _compute(result)

    # Cells with normalized cost > 0.5 should be NaN
    assert np.all(np.isnan(out) | (out <= 0.5 + 1e-5))

    # The optimal path (row 3) should not be masked
    for col in range(n):
        assert np.isfinite(out[3, col])


@pytest.mark.parametrize(
    "backend", ["numpy", "dask+numpy", "cupy", "dask+cupy"]
)
def test_relative_threshold(backend):
    """Relative threshold uses fraction of minimum corridor cost."""
    n = 7
    friction_data = np.ones((n, n))

    src_a = np.zeros((n, n))
    src_a[3, 0] = 1.0

    src_b = np.zeros((n, n))
    src_b[3, 6] = 1.0

    friction = _make_raster(friction_data, backend=backend, chunks=(7, 7))
    sa = _make_raster(src_a, backend=backend, chunks=(7, 7))
    sb = _make_raster(src_b, backend=backend, chunks=(7, 7))

    # No threshold -- get full corridor
    full = least_cost_corridor(friction, sa, sb)
    full_out = _compute(full)

    # Relative threshold of 50%
    result = least_cost_corridor(
        friction, sa, sb, threshold=0.5, relative=True
    )
    out = _compute(result)

    # Count finite cells -- threshold version should have fewer
    assert np.sum(np.isfinite(out)) < np.sum(np.isfinite(full_out))

    # Optimal path cells should survive
    for col in range(n):
        assert np.isfinite(out[3, col])


# -----------------------------------------------------------------------
# Precomputed cost-distance surfaces
# -----------------------------------------------------------------------


def test_precomputed_matches_regular():
    """Precomputed=True with manual cost_distance matches default path."""
    n = 7
    friction_data = np.ones((n, n))

    src_a = np.zeros((n, n))
    src_a[3, 0] = 1.0

    src_b = np.zeros((n, n))
    src_b[3, 6] = 1.0

    friction = _make_raster(friction_data)
    sa = _make_raster(src_a)
    sb = _make_raster(src_b)

    # Regular path
    result_regular = least_cost_corridor(friction, sa, sb)

    # Precomputed path
    cd_a = cost_distance(sa, friction)
    cd_b = cost_distance(sb, friction)
    result_precomputed = least_cost_corridor(
        friction, cd_a, cd_b, precomputed=True
    )

    np.testing.assert_allclose(
        _compute(result_regular),
        _compute(result_precomputed),
        atol=1e-5,
    )


# -----------------------------------------------------------------------
# Multi-source pairwise
# -----------------------------------------------------------------------


def test_pairwise_corridor():
    """Pairwise mode with 3 sources returns Dataset with 3 corridors."""
    n = 7
    friction_data = np.ones((n, n))

    sources = []
    for r, c in [(0, 0), (0, 6), (6, 3)]:
        s = np.zeros((n, n))
        s[r, c] = 1.0
        sources.append(_make_raster(s))

    friction = _make_raster(friction_data)

    result = least_cost_corridor(
        friction, sources=sources, pairwise=True
    )

    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {
        "corridor_0_1",
        "corridor_0_2",
        "corridor_1_2",
    }

    # Each corridor should have minimum 0
    for name in result.data_vars:
        out = _compute(result[name])
        assert np.nanmin(out) == pytest.approx(0.0, abs=1e-5)


def test_pairwise_two_sources_returns_dataset():
    """Pairwise=True with exactly 2 sources still returns a Dataset."""
    n = 5
    friction_data = np.ones((n, n))

    s0 = np.zeros((n, n))
    s0[0, 0] = 1.0
    s1 = np.zeros((n, n))
    s1[4, 4] = 1.0

    friction = _make_raster(friction_data)
    result = least_cost_corridor(
        friction,
        sources=[_make_raster(s0), _make_raster(s1)],
        pairwise=True,
    )

    assert isinstance(result, xr.Dataset)
    assert "corridor_0_1" in result.data_vars


# -----------------------------------------------------------------------
# NaN / barrier handling
# -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "backend", ["numpy", "dask+numpy", "cupy", "dask+cupy"]
)
def test_barrier_blocks_corridor(backend):
    """NaN barrier between sources makes certain cells unreachable."""
    n = 7
    friction_data = np.ones((n, n))
    # Wall of NaN except a gap at row 3
    friction_data[:3, 3] = np.nan
    friction_data[4:, 3] = np.nan

    src_a = np.zeros((n, n))
    src_a[3, 0] = 1.0

    src_b = np.zeros((n, n))
    src_b[3, 6] = 1.0

    friction = _make_raster(friction_data, backend=backend, chunks=(7, 7))
    sa = _make_raster(src_a, backend=backend, chunks=(7, 7))
    sb = _make_raster(src_b, backend=backend, chunks=(7, 7))

    result = least_cost_corridor(friction, sa, sb)
    out = _compute(result)

    # The gap row should still be reachable
    assert np.isfinite(out[3, 3])


@pytest.mark.parametrize(
    "backend", ["numpy", "dask+numpy", "cupy", "dask+cupy"]
)
def test_unreachable_sources(backend):
    """Full barrier between sources produces all-NaN corridor."""
    n = 5
    friction_data = np.ones((n, n))
    friction_data[:, 2] = np.nan  # impenetrable wall

    src_a = np.zeros((n, n))
    src_a[2, 0] = 1.0

    src_b = np.zeros((n, n))
    src_b[2, 4] = 1.0

    friction = _make_raster(friction_data, backend=backend, chunks=(5, 5))
    sa = _make_raster(src_a, backend=backend, chunks=(5, 5))
    sb = _make_raster(src_b, backend=backend, chunks=(5, 5))

    result = least_cost_corridor(friction, sa, sb)
    out = _compute(result)

    assert np.all(np.isnan(out))


# -----------------------------------------------------------------------
# Edge cases and validation
# -----------------------------------------------------------------------


def test_single_cell_raster():
    """1x1 raster where both sources are the same cell."""
    friction = _make_raster(np.ones((1, 1)))
    src = _make_raster(np.ones((1, 1)))

    result = least_cost_corridor(friction, src, src)
    out = _compute(result)

    assert out[0, 0] == pytest.approx(0.0, abs=1e-5)


def test_missing_sources_raises():
    """Omitting both source_a/source_b and sources raises ValueError."""
    friction = _make_raster(np.ones((3, 3)))
    with pytest.raises(ValueError, match="source_a and source_b are required"):
        least_cost_corridor(friction)


def test_both_source_modes_raises():
    """Providing source_a/source_b AND sources raises ValueError."""
    friction = _make_raster(np.ones((3, 3)))
    src = _make_raster(np.ones((3, 3)))
    with pytest.raises(ValueError, match="not both"):
        least_cost_corridor(friction, src, src, sources=[src, src])


def test_negative_threshold_raises():
    """Negative threshold raises ValueError."""
    friction = _make_raster(np.ones((3, 3)))
    src = _make_raster(np.ones((3, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        least_cost_corridor(friction, src, src, threshold=-1.0)


def test_single_source_in_list_raises():
    """sources with fewer than 2 entries raises ValueError."""
    friction = _make_raster(np.ones((3, 3)))
    src = _make_raster(np.ones((3, 3)))
    with pytest.raises(ValueError, match="at least 2"):
        least_cost_corridor(friction, sources=[src])


def test_precomputed_mismatched_shape_raises():
    """Precomputed surfaces of differing shape raise instead of aligning.

    Without the shape check, xarray silently aligns the two surfaces on the
    intersection of their coordinates and returns a truncated corridor with
    wrong values (e.g. 4x4 + 3x3 -> an all-zero 3x3 result).
    """
    friction = _make_raster(np.ones((4, 4)))
    cd_a = _make_raster(np.ones((4, 4)))
    cd_b = _make_raster(np.ones((3, 3)))
    with pytest.raises(ValueError, match="does not match"):
        least_cost_corridor(friction, cd_a, cd_b, precomputed=True)


def test_precomputed_mismatched_shape_pairwise_raises():
    """Pairwise precomputed surfaces of differing shape raise."""
    friction = _make_raster(np.ones((4, 4)))
    sources = [
        _make_raster(np.ones((4, 4))),
        _make_raster(np.ones((3, 3))),
    ]
    with pytest.raises(ValueError, match="does not match"):
        least_cost_corridor(
            friction, sources=sources, precomputed=True, pairwise=True
        )


@pytest.mark.skipif(da is None, reason="dask not installed")
def test_precomputed_mismatched_shape_dask_raises():
    """The shape check fires on dask surfaces without triggering a compute."""
    friction = _make_raster(np.ones((4, 4)), backend="dask+numpy", chunks=(4, 4))
    cd_a = _make_raster(np.ones((4, 4)), backend="dask+numpy", chunks=(4, 4))
    cd_b = _make_raster(np.ones((3, 3)), backend="dask+numpy", chunks=(3, 3))
    with pytest.raises(ValueError, match="does not match"):
        least_cost_corridor(friction, cd_a, cd_b, precomputed=True)


# -----------------------------------------------------------------------
# Degenerate strip shapes (Nx1 / 1xN)
# -----------------------------------------------------------------------


def test_1xn_strip():
    """1xN single-row raster with sources at each end."""
    n = 5
    friction = _make_raster(np.ones((1, n)))

    src_a = np.zeros((1, n))
    src_a[0, 0] = 1.0
    src_b = np.zeros((1, n))
    src_b[0, n - 1] = 1.0

    result = least_cost_corridor(friction, _make_raster(src_a),
                                 _make_raster(src_b))
    out = _compute(result)

    assert out.shape == (1, n)
    # Every cell lies on the only path, so all are optimal (cost 0).
    np.testing.assert_allclose(out[0], 0.0, atol=1e-5)


def test_nx1_strip():
    """Nx1 single-column raster with sources at each end."""
    n = 5
    friction = _make_raster(np.ones((n, 1)))

    src_a = np.zeros((n, 1))
    src_a[0, 0] = 1.0
    src_b = np.zeros((n, 1))
    src_b[n - 1, 0] = 1.0

    result = least_cost_corridor(friction, _make_raster(src_a),
                                 _make_raster(src_b))
    out = _compute(result)

    assert out.shape == (n, 1)
    np.testing.assert_allclose(out[:, 0], 0.0, atol=1e-5)


# -----------------------------------------------------------------------
# Cross-backend equivalence
# -----------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["dask+numpy", "cupy", "dask+cupy"])
def test_numpy_matches_other_backends(backend):
    """Each non-numpy backend produces the same corridor as numpy."""
    if "cupy" in backend and not has_cuda_and_cupy():
        pytest.skip("Requires CUDA and CuPy")

    n = 7
    friction_data = np.ones((n, n))
    src_a = np.zeros((n, n))
    src_a[3, 0] = 1.0
    src_b = np.zeros((n, n))
    src_b[3, 6] = 1.0

    numpy_out = _compute(
        least_cost_corridor(
            _make_raster(friction_data),
            _make_raster(src_a),
            _make_raster(src_b),
        )
    )

    other_out = _compute(
        least_cost_corridor(
            _make_raster(friction_data, backend=backend, chunks=(7, 7)),
            _make_raster(src_a, backend=backend, chunks=(7, 7)),
            _make_raster(src_b, backend=backend, chunks=(7, 7)),
        )
    )

    np.testing.assert_allclose(other_out, numpy_out, equal_nan=True, atol=1e-5)


# -----------------------------------------------------------------------
# Forwarded cost_distance parameters (connectivity, max_cost)
# -----------------------------------------------------------------------


def test_connectivity_4():
    """connectivity=4 is forwarded to cost_distance and yields a corridor."""
    n = 5
    friction = _make_raster(np.ones((n, n)))

    src_a = np.zeros((n, n))
    src_a[2, 0] = 1.0
    src_b = np.zeros((n, n))
    src_b[2, 4] = 1.0

    result = least_cost_corridor(
        friction, _make_raster(src_a), _make_raster(src_b), connectivity=4
    )
    out = _compute(result)

    assert out.shape == (n, n)
    # Optimal corridor minimum is 0 after normalization.
    assert np.nanmin(out) == pytest.approx(0.0, abs=1e-5)
    # Row 2 is the straight cardinal-only path; all cells optimal.
    for col in range(n):
        assert out[2, col] == pytest.approx(0.0, abs=1e-5)


def test_max_cost_forwarded():
    """A finite max_cost reaches cost_distance; optimal path still resolves."""
    n = 7
    friction = _make_raster(np.ones((n, n)))

    src_a = np.zeros((n, n))
    src_a[3, 0] = 1.0
    src_b = np.zeros((n, n))
    src_b[3, 6] = 1.0

    result = least_cost_corridor(
        friction, _make_raster(src_a), _make_raster(src_b), max_cost=20.0
    )
    out = _compute(result)

    assert out.shape == (n, n)
    # The straight middle row stays within budget and normalizes to 0.
    for col in range(n):
        assert out[3, col] == pytest.approx(0.0, abs=1e-5)


# -----------------------------------------------------------------------
# Metadata / coordinate / dim-name preservation
# -----------------------------------------------------------------------


def test_attrs_coords_preserved():
    """Output preserves input attrs, dims, and coordinates."""
    n = 5
    friction = _make_raster(np.ones((n, n)))

    src_a = np.zeros((n, n))
    src_a[2, 0] = 1.0
    src_b = np.zeros((n, n))
    src_b[2, 4] = 1.0
    sa = _make_raster(src_a)
    sb = _make_raster(src_b)

    result = least_cost_corridor(friction, sa, sb)

    assert result.dims == friction.dims
    assert result.attrs == friction.attrs
    np.testing.assert_array_equal(result["y"].data, friction["y"].data)
    np.testing.assert_array_equal(result["x"].data, friction["x"].data)


def test_custom_dim_names_preserved():
    """Non-default lat/lon dim names propagate through to the output."""
    n = 5
    friction = xr.DataArray(
        np.ones((n, n), dtype=np.float64),
        dims=["lat", "lon"],
        attrs={"res": (1.0, 1.0)},
    )
    friction["lat"] = np.arange(n, dtype=np.float64)
    friction["lon"] = np.arange(n, dtype=np.float64)

    def _src(r, c):
        s = friction.copy(data=np.zeros((n, n), dtype=np.float64))
        s.data[r, c] = 1.0
        return s

    result = least_cost_corridor(
        friction, _src(2, 0), _src(2, 4), x="lon", y="lat"
    )

    assert result.dims == ("lat", "lon")
    np.testing.assert_array_equal(result["lat"].data, friction["lat"].data)
    np.testing.assert_array_equal(result["lon"].data, friction["lon"].data)


# -----------------------------------------------------------------------
# Metadata propagation (issue #3446)
# -----------------------------------------------------------------------


_GEO_ATTRS = {
    "res": (1.0, 1.0),
    "crs": "EPSG:4326",
    "transform": (1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    "nodatavals": (-9999.0,),
}


def _make_geo_raster(data, attrs, name, backend="numpy"):
    """Build a raster with explicit attrs/name on top of ``_make_raster``."""
    raster = _make_raster(data, backend=backend, chunks=data.shape)
    raster.attrs = dict(attrs)
    raster.name = name
    return raster


@pytest.mark.parametrize(
    "backend", ["numpy", "dask+numpy", "cupy", "dask+cupy"]
)
def test_corridor_inherits_friction_geo_attrs(backend):
    """Corridor carries friction's geo-attrs/name even when sources have none."""
    n = 7
    friction = _make_geo_raster(
        np.ones((n, n)), _GEO_ATTRS, "friction", backend=backend
    )
    sa_d = np.zeros((n, n))
    sa_d[3, 0] = 1.0
    sb_d = np.zeros((n, n))
    sb_d[3, 6] = 1.0
    # Source masks deliberately carry no attrs and no name.
    sa = _make_geo_raster(sa_d, {}, None, backend=backend)
    sb = _make_geo_raster(sb_d, {}, None, backend=backend)

    result = least_cost_corridor(friction, sa, sb)

    assert dict(result.attrs) == _GEO_ATTRS
    assert result.name == "friction"
    assert result.dims == ("y", "x")
    assert list(result.coords) == ["y", "x"]


@pytest.mark.parametrize(
    "backend", ["numpy", "dask+numpy", "cupy", "dask+cupy"]
)
def test_corridor_threshold_keeps_geo_attrs(backend):
    """Thresholded corridor still carries friction's geo-attrs."""
    n = 7
    friction = _make_geo_raster(
        np.ones((n, n)), _GEO_ATTRS, "friction", backend=backend
    )
    sa_d = np.zeros((n, n))
    sa_d[3, 0] = 1.0
    sb_d = np.zeros((n, n))
    sb_d[3, 6] = 1.0
    sa = _make_geo_raster(sa_d, {}, None, backend=backend)
    sb = _make_geo_raster(sb_d, {}, None, backend=backend)

    result = least_cost_corridor(friction, sa, sb, threshold=0.5)

    assert dict(result.attrs) == _GEO_ATTRS
    assert result.name == "friction"


def test_corridor_unreachable_keeps_geo_attrs():
    """All-NaN corridor (unreachable sources) still carries geo-attrs."""
    n = 5
    fr_d = np.ones((n, n))
    fr_d[:, 2] = np.nan  # impenetrable wall
    friction = _make_geo_raster(fr_d, _GEO_ATTRS, "friction")
    sa_d = np.zeros((n, n))
    sa_d[2, 0] = 1.0
    sb_d = np.zeros((n, n))
    sb_d[2, 4] = 1.0
    sa = _make_geo_raster(sa_d, {}, None)
    sb = _make_geo_raster(sb_d, {}, None)

    result = least_cost_corridor(friction, sa, sb)

    assert np.all(np.isnan(_compute(result)))
    assert dict(result.attrs) == _GEO_ATTRS


def test_pairwise_inherits_friction_geo_attrs():
    """Every variable in a pairwise Dataset carries friction's geo-attrs."""
    n = 7
    friction = _make_geo_raster(np.ones((n, n)), _GEO_ATTRS, "friction")
    sources = []
    for r, c in [(0, 0), (0, 6), (6, 3)]:
        s = np.zeros((n, n))
        s[r, c] = 1.0
        sources.append(_make_geo_raster(s, {}, None))

    result = least_cost_corridor(friction, sources=sources, pairwise=True)

    assert isinstance(result, xr.Dataset)
    for name in result.data_vars:
        assert dict(result[name].attrs) == _GEO_ATTRS


def test_precomputed_keeps_source_attrs_not_friction():
    """Precomputed path leaves the source-derived attrs alone (no friction)."""
    n = 7
    friction = _make_geo_raster(np.ones((n, n)), _GEO_ATTRS, "friction")
    sa_d = np.zeros((n, n))
    sa_d[3, 0] = 1.0
    sb_d = np.zeros((n, n))
    sb_d[3, 6] = 1.0
    src_attrs = {"res": (1.0, 1.0), "crs": "EPSG:3857"}
    cd_a = _make_geo_raster(sa_d, src_attrs, "cd")
    cd_b = _make_geo_raster(sb_d, src_attrs, "cd")

    result = least_cost_corridor(friction, cd_a, cd_b, precomputed=True)

    # Friction's attrs must NOT leak into a precomputed corridor; the
    # matching source attrs survive xarray's binary-op intersection.
    assert dict(result.attrs) == src_attrs
    assert "EPSG:4326" not in result.attrs.get("crs", "")
