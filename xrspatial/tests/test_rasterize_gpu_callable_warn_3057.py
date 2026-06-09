"""Issue #3057: a custom callable merge on the GPU backend uses a
non-atomic read-modify-write, so overlap pixels are nondeterministic.

``rasterize`` must warn the caller when a callable ``merge`` is paired
with ``use_cuda=True``.  The warning fires up front, before any geometry
burning or the CuPy availability check, so these tests do not need a GPU:
they record warnings manually and ignore whatever the (GPU-less) call
raises afterwards.
"""
import re
import warnings

import pytest

try:
    from shapely.geometry import box
    has_shapely = True
except ImportError:
    has_shapely = False

if has_shapely:
    from xrspatial.rasterize import rasterize

from xrspatial.utils import ngjit

pytestmark = pytest.mark.skipif(
    not has_shapely, reason="shapely not installed"
)

_WARN_RE = re.compile("non-atomic", re.IGNORECASE)


@ngjit
def _my_sum(pixel, props, is_first):
    if is_first:
        return props[0]
    return pixel + props[0]


def _overlap_pairs():
    return [(box(0, 0, 6, 6), 1.0), (box(4, 4, 10, 10), 2.0)]


def _overlap_warnings(**kwargs):
    """Run rasterize, returning the recorded warnings.  Any exception
    raised after the warning (e.g. ImportError when CuPy is missing, or a
    numba/CUDA error when there is no GPU device) is swallowed -- the
    warning is emitted before any of that, so it is already recorded.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        try:
            rasterize(_overlap_pairs(), width=10, height=10,
                      bounds=(0, 0, 10, 10), fill=0, **kwargs)
        except Exception:
            pass
    return [w for w in record if _WARN_RE.search(str(w.message))]


def test_callable_gpu_merge_warns():
    """Callable merge + use_cuda=True emits the overlap UserWarning."""
    matched = _overlap_warnings(merge=_my_sum, use_cuda=True)
    assert len(matched) == 1
    assert matched[0].category is UserWarning


def test_callable_gpu_merge_chunks_warns():
    """The warning also fires for the dask+cupy path (chunks + use_cuda)."""
    matched = _overlap_warnings(merge=_my_sum, use_cuda=True, chunks=(5, 5))
    assert len(matched) == 1
    assert matched[0].category is UserWarning


def test_callable_cpu_merge_does_not_warn():
    """A callable merge on the CPU backend must not emit the GPU warning."""
    assert _overlap_warnings(merge=_my_sum) == []


def test_builtin_gpu_merge_does_not_warn():
    """A built-in string merge on the GPU backend stays silent -- it uses
    atomics and is deterministic over overlap."""
    assert _overlap_warnings(merge='sum', use_cuda=True) == []
