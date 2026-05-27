"""ngjit 2x2 reduction kernels for COG overview generation.

These kernels replace the generic numpy ``reshape + np.nanmean /
np.nanmin / np.nanmax / np.nanmedian`` pipeline that :func:`_block_reduce_2d`
in :mod:`._overview` uses for the float mean / min / max / median methods.

Each kernel walks output pixels directly, fuses the sentinel comparison
into the read (so no intermediate ``np.where`` buffer is allocated), and
bounds-checks the right / bottom edges instead of pre-padding the input
to even dimensions. All-NaN / all-sentinel blocks return NaN so the
caller's existing post-reduction rewrite (integer NaN -> sentinel,
float NaN passthrough) still applies unchanged.

Scope: float32 / float64 mean, min, max, median. Integer dtypes still
flow through the numpy nan-aware path in :func:`_block_reduce_2d`
because the existing implementation already computes the sentinel mask
at native integer width before promoting to float64, which is required
for 64-bit sentinels that round when cast to float64 (``INT64_MAX``,
``UINT64_MAX``). A native-integer kernel is a possible follow-up.

The kernels rely on numba specializing on the input array's dtype, so
the same source covers both float32 and float64 inputs. The mean
accumulator is a Python float, which numba unifies to float64; the
final value is downcast to the input dtype on store. That keeps the
float32 mean's accumulation precision at float64 (matching the prior
``np.nanmean`` path on this code's typical input sizes) at the cost of
a couple of extra promote/downcast instructions per output pixel. The
min / max ``best`` locals follow the same pattern; the result is
identical to the input value either way since both reductions are
exact.

The four kernels share a near-identical 4-cell fetch + NaN/sentinel
filter. Factoring it into a helper would interfere with ngjit
inlining, so the duplication is deliberate: keep the inner loop body
visible to the JIT for each method.
"""
from __future__ import annotations

import math

import numpy as np

from xrspatial.utils import ngjit


@ngjit
def _reduce2x2_mean(arr, out, sentinel, has_sentinel):
    """Mean of each 2x2 source block, NaN- and sentinel-aware.

    Output shape is ``out.shape == ((h+1)//2, (w+1)//2)``. When the
    input is odd-sized, the trailing residual blocks contain fewer
    than 4 source cells; only the in-bounds cells contribute and the
    block reduces to NaN when none are valid.
    """
    h, w = arr.shape
    oh, ow = out.shape
    for i in range(oh):
        r0 = 2 * i
        for j in range(ow):
            c0 = 2 * j
            total = 0.0
            n = 0
            v = arr[r0, c0]
            if not math.isnan(v) and (not has_sentinel or v != sentinel):
                total += v
                n += 1
            if c0 + 1 < w:
                v = arr[r0, c0 + 1]
                if not math.isnan(v) and (not has_sentinel or v != sentinel):
                    total += v
                    n += 1
            if r0 + 1 < h:
                v = arr[r0 + 1, c0]
                if not math.isnan(v) and (not has_sentinel or v != sentinel):
                    total += v
                    n += 1
                if c0 + 1 < w:
                    v = arr[r0 + 1, c0 + 1]
                    if not math.isnan(v) and (not has_sentinel or v != sentinel):
                        total += v
                        n += 1
            if n == 0:
                out[i, j] = np.nan
            else:
                out[i, j] = total / n


@ngjit
def _reduce2x2_min(arr, out, sentinel, has_sentinel):
    """Min of each 2x2 source block, NaN- and sentinel-aware."""
    h, w = arr.shape
    oh, ow = out.shape
    for i in range(oh):
        r0 = 2 * i
        for j in range(ow):
            c0 = 2 * j
            best = 0.0
            found = False
            v = arr[r0, c0]
            if not math.isnan(v) and (not has_sentinel or v != sentinel):
                best = v
                found = True
            if c0 + 1 < w:
                v = arr[r0, c0 + 1]
                if not math.isnan(v) and (not has_sentinel or v != sentinel):
                    if not found or v < best:
                        best = v
                    found = True
            if r0 + 1 < h:
                v = arr[r0 + 1, c0]
                if not math.isnan(v) and (not has_sentinel or v != sentinel):
                    if not found or v < best:
                        best = v
                    found = True
                if c0 + 1 < w:
                    v = arr[r0 + 1, c0 + 1]
                    if not math.isnan(v) and (not has_sentinel or v != sentinel):
                        if not found or v < best:
                            best = v
                        found = True
            if not found:
                out[i, j] = np.nan
            else:
                out[i, j] = best


@ngjit
def _reduce2x2_max(arr, out, sentinel, has_sentinel):
    """Max of each 2x2 source block, NaN- and sentinel-aware."""
    h, w = arr.shape
    oh, ow = out.shape
    for i in range(oh):
        r0 = 2 * i
        for j in range(ow):
            c0 = 2 * j
            best = 0.0
            found = False
            v = arr[r0, c0]
            if not math.isnan(v) and (not has_sentinel or v != sentinel):
                best = v
                found = True
            if c0 + 1 < w:
                v = arr[r0, c0 + 1]
                if not math.isnan(v) and (not has_sentinel or v != sentinel):
                    if not found or v > best:
                        best = v
                    found = True
            if r0 + 1 < h:
                v = arr[r0 + 1, c0]
                if not math.isnan(v) and (not has_sentinel or v != sentinel):
                    if not found or v > best:
                        best = v
                    found = True
                if c0 + 1 < w:
                    v = arr[r0 + 1, c0 + 1]
                    if not math.isnan(v) and (not has_sentinel or v != sentinel):
                        if not found or v > best:
                            best = v
                        found = True
            if not found:
                out[i, j] = np.nan
            else:
                out[i, j] = best


@ngjit
def _reduce2x2_median(arr, out, sentinel, has_sentinel):
    """Median of each 2x2 source block, NaN- and sentinel-aware.

    Up to four valid values per output pixel are collected into local
    scalars (no per-iteration heap allocation), sorted in place via a
    small insertion-style network, and the median read off by count:
    one value -> itself, two -> their mean, three -> the middle, four
    -> mean of the two middles. Matches ``np.nanmedian`` semantics over
    the same input.
    """
    h, w = arr.shape
    oh, ow = out.shape
    for i in range(oh):
        r0 = 2 * i
        for j in range(ow):
            c0 = 2 * j
            v0 = 0.0
            v1 = 0.0
            v2 = 0.0
            v3 = 0.0
            n = 0
            v = arr[r0, c0]
            if not math.isnan(v) and (not has_sentinel or v != sentinel):
                v0 = v
                n = 1
            if c0 + 1 < w:
                v = arr[r0, c0 + 1]
                if not math.isnan(v) and (not has_sentinel or v != sentinel):
                    if n == 0:
                        v0 = v
                    else:
                        v1 = v
                    n += 1
            if r0 + 1 < h:
                v = arr[r0 + 1, c0]
                if not math.isnan(v) and (not has_sentinel or v != sentinel):
                    if n == 0:
                        v0 = v
                    elif n == 1:
                        v1 = v
                    else:
                        v2 = v
                    n += 1
                if c0 + 1 < w:
                    v = arr[r0 + 1, c0 + 1]
                    if not math.isnan(v) and (not has_sentinel or v != sentinel):
                        if n == 0:
                            v0 = v
                        elif n == 1:
                            v1 = v
                        elif n == 2:
                            v2 = v
                        else:
                            v3 = v
                        n += 1

            # Sort the first ``n`` slots ascending. The compare-swap
            # ladder below sorts up to four floats in place; comparisons
            # past ``n`` are gated so unused slots (still 0.0) do not
            # leak into the result. Trace for n=4, (v0,v1,v2,v3) =
            # (4,2,3,1):
            #   after n>=2 swap : (2,4,3,1)
            #   after n>=3 pass : (2,3,4,1)  # v1>v2 swap, then v0,v1
            #   after n>=4 pass : (1,2,3,4)  # v2>v3, v1>v2, v0>v1
            if n >= 2 and v0 > v1:
                v0, v1 = v1, v0
            if n >= 3:
                if v1 > v2:
                    v1, v2 = v2, v1
                if v0 > v1:
                    v0, v1 = v1, v0
            if n >= 4:
                if v2 > v3:
                    v2, v3 = v3, v2
                if v1 > v2:
                    v1, v2 = v2, v1
                if v0 > v1:
                    v0, v1 = v1, v0

            if n == 0:
                out[i, j] = np.nan
            elif n == 1:
                out[i, j] = v0
            elif n == 2:
                out[i, j] = 0.5 * (v0 + v1)
            elif n == 3:
                out[i, j] = v1
            else:
                out[i, j] = 0.5 * (v1 + v2)


# Dispatch table used by :func:`_block_reduce_2d` when ``method`` is one
# of the four nan-aware reductions and ``arr.dtype.kind == 'f'``. Other
# methods (nearest, cubic, mode) and integer dtypes keep their existing
# numpy paths.
KERNELS = {
    'mean': _reduce2x2_mean,
    'min': _reduce2x2_min,
    'max': _reduce2x2_max,
    'median': _reduce2x2_median,
}
