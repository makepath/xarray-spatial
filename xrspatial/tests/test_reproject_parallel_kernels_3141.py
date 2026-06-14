"""Concurrent launches of numba parallel=True kernels (issue #3141).

The reproject package compiles its coordinate-transform, geoid, and ITRF
kernels with ``@njit(parallel=True)``. Launching such a kernel from two
host threads at once is unsupported on numba's workqueue threading layer
(the macOS arm64 default): numba detects the concurrent access and calls
``abort()``, killing the interpreter. Every public entry point that
launches one of these kernels therefore serializes the launch behind
``_projections._PARALLEL_KERNEL_LOCK``.

These tests hammer each entry point from a thread pool and compare
against single-threaded results. On a workqueue platform without the
lock, they hard-crash the test process; everywhere else they still
verify that concurrent callers get correct, independent results.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

try:
    import pyproj  # noqa: F401
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

pytestmark = pytest.mark.skipif(
    not HAS_PYPROJ, reason="pyproj required for reproject tests"
)

N_THREADS = 8
N_CALLS = 32


def _hammer(fn, n_calls=N_CALLS, n_threads=N_THREADS):
    """Run *fn* (index -> result) concurrently and return all results."""
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        return list(pool.map(fn, range(n_calls)))


class TestTransformConcurrency:
    """try_numba_transform / transform_points from multiple threads."""

    def test_try_numba_transform_concurrent(self):
        from xrspatial.reproject._crs_utils import _resolve_crs
        from xrspatial.reproject._projections import try_numba_transform

        src_crs = _resolve_crs('EPSG:4326')
        tgt_crs = _resolve_crs('EPSG:32633')
        bounds = (400000.0, 5000000.0, 500000.0, 5100000.0)
        shape = (64, 64)

        expected = try_numba_transform(src_crs, tgt_crs, bounds, shape)
        assert expected is not None  # the fast path must engage

        def one(_):
            return try_numba_transform(src_crs, tgt_crs, bounds, shape)

        for got in _hammer(one):
            assert got is not None
            np.testing.assert_array_equal(got[0], expected[0])
            np.testing.assert_array_equal(got[1], expected[1])

    def test_transform_points_concurrent(self):
        from xrspatial.reproject._crs_utils import _resolve_crs
        from xrspatial.reproject._projections import transform_points

        src_crs = _resolve_crs('EPSG:4326')
        tgt_crs = _resolve_crs('EPSG:3857')
        xs = np.linspace(-170.0, 170.0, 512)
        ys = np.linspace(-80.0, 80.0, 512)

        expected = transform_points(src_crs, tgt_crs, xs, ys)
        assert expected is not None

        def one(_):
            return transform_points(src_crs, tgt_crs, xs, ys)

        for got in _hammer(one):
            assert got is not None
            np.testing.assert_array_equal(got[0], expected[0])
            np.testing.assert_array_equal(got[1], expected[1])


class TestGeoidConcurrency:
    """geoid_height launches _interp_geoid_batch (parallel=True)."""

    def test_geoid_height_concurrent(self):
        from xrspatial.reproject import geoid_height

        lons = np.linspace(-120.0, 30.0, 256)
        lats = np.linspace(-60.0, 60.0, 256)
        expected = geoid_height(lons, lats)

        def one(_):
            return geoid_height(lons, lats)

        for got in _hammer(one):
            np.testing.assert_array_equal(got, expected)


class TestItrfConcurrency:
    """itrf_transform launches _itrf_batch (parallel=True)."""

    def test_itrf_transform_concurrent(self):
        from xrspatial.reproject import itrf_frames, itrf_transform

        frames = itrf_frames()
        if not ('ITRF2014' in frames and 'ITRF2008' in frames):
            pytest.skip("PROJ ITRF parameter files not available")

        lons = np.linspace(-120.0, 30.0, 256)
        lats = np.linspace(-60.0, 60.0, 256)
        kwargs = dict(src='ITRF2014', tgt='ITRF2008', epoch=2020.0)
        expected = itrf_transform(lons, lats, 100.0, **kwargs)

        def one(_):
            return itrf_transform(lons, lats, 100.0, **kwargs)

        for got in _hammer(one):
            np.testing.assert_array_equal(got[0], expected[0])
            np.testing.assert_array_equal(got[1], expected[1])
            np.testing.assert_array_equal(got[2], expected[2])
