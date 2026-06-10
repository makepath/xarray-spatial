"""Issue #3057: a custom callable merge on the GPU backend uses a
non-atomic read-modify-write, so overlap pixels are nondeterministic.

``rasterize`` must warn the caller when a callable ``merge`` is paired
with ``gpu=True``.  The warning fires after the CuPy import check
but before the GPU kernel launch, so these tests need CuPy importable but
not an actual GPU device: they record warnings manually and ignore the
numba/CUDA error the (device-less) launch raises afterwards.
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

try:
    import cupy  # noqa: F401
    has_cupy = True
except ImportError:
    has_cupy = False

pytestmark = pytest.mark.skipif(
    not has_shapely, reason="shapely not installed"
)

skip_no_cupy = pytest.mark.skipif(
    not has_cupy, reason="CuPy not importable (GPU warning fires after the "
                         "CuPy check)")

_WARN_RE = re.compile("non-atomic", re.IGNORECASE)


@ngjit
def _my_sum(pixel, props, is_first):
    if is_first:
        return props[0]
    return pixel + props[0]


def _overlap_pairs():
    return [(box(0, 0, 6, 6), 1.0), (box(4, 4, 10, 10), 2.0)]


def _overlap_warnings(**kwargs):
    """Run rasterize, returning the overlap warnings it recorded.

    The GPU paths reach the kernel launch, which raises on a device-less
    box; that error comes after the warn line, so swallowing it does not
    hide a missing warning -- the callers below assert the exact warning
    count, which fails loudly if rasterize raised before warning.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        try:
            rasterize(_overlap_pairs(), width=10, height=10,
                      bounds=(0, 0, 10, 10), fill=0, **kwargs)
        except Exception:
            pass
    return [w for w in record if _WARN_RE.search(str(w.message))]


@skip_no_cupy
def test_callable_gpu_merge_warns():
    """Callable merge + gpu=True emits the overlap UserWarning."""
    matched = _overlap_warnings(merge=_my_sum, gpu=True)
    assert len(matched) == 1
    assert matched[0].category is UserWarning


@skip_no_cupy
def test_callable_gpu_merge_chunks_warns():
    """The warning also fires for the dask+cupy path (chunks + gpu)."""
    matched = _overlap_warnings(merge=_my_sum, gpu=True, chunks=(5, 5))
    assert len(matched) == 1
    assert matched[0].category is UserWarning


def test_callable_cpu_merge_does_not_warn():
    """A callable merge on the CPU backend must not emit the GPU warning.

    This path completes without a GPU, so no exception is swallowed.
    """
    assert _overlap_warnings(merge=_my_sum) == []


@skip_no_cupy
def test_builtin_gpu_merge_does_not_warn():
    """A built-in string merge on the GPU backend stays silent -- it uses
    atomics and is deterministic over overlap."""
    assert _overlap_warnings(merge='sum', gpu=True) == []
