"""Regression test for #1922: write_geotiff_gpu kwarg order matches
to_geotiff (with the documented exception for ``gpu``).

The two writers are advertised as parity siblings. The GPU writer's
own docstring says "Accepted at the signature level for API parity with
``to_geotiff``" for ``max_z_error`` and ``streaming_buffer_bytes``, but
the two kwargs were in opposite order across the two signatures:

    to_geotiff:          ..., bigtiff, gpu, streaming_buffer_bytes,
                              max_z_error, photometric, ...
    write_geotiff_gpu:   ..., bigtiff,      max_z_error,
                              streaming_buffer_bytes, photometric, ...

Both are keyword-only so calling code did not break, but
``inspect.signature()``, IDE autocomplete, and Sphinx-rendered docs all
exposed the drift. Detected by deep-sweep-api-consistency on 2026-05-15.
"""
from __future__ import annotations

import inspect

from xrspatial.geotiff import to_geotiff, write_geotiff_gpu


def test_writer_kwarg_order_matches_to_geotiff():
    """``write_geotiff_gpu`` lists its kwargs in the same order as
    ``to_geotiff``, modulo the ``gpu`` kwarg the GPU writer omits.

    Both signatures use keyword-only kwargs so positional callers are
    unaffected. The order still matters for IDE autocomplete, generated
    docs, and any caller that inspects ``inspect.signature``.
    """
    eager_params = list(inspect.signature(to_geotiff).parameters)
    gpu_params = list(inspect.signature(write_geotiff_gpu).parameters)

    # to_geotiff has ``gpu`` (auto-dispatch flag); write_geotiff_gpu does
    # not. Drop it from the comparison instead of asserting on the
    # missing kwarg directly, so unrelated future additions to either
    # signature still surface here.
    assert 'gpu' in eager_params
    assert 'gpu' not in gpu_params
    eager_params_no_gpu = [p for p in eager_params if p != 'gpu']

    assert gpu_params == eager_params_no_gpu, (
        "write_geotiff_gpu and to_geotiff kwarg order diverged.\n"
        f"  to_geotiff (with 'gpu' removed): {eager_params_no_gpu}\n"
        f"  write_geotiff_gpu:               {gpu_params}\n"
        "Reorder write_geotiff_gpu to match to_geotiff (see #1922)."
    )


def test_writer_kwarg_defaults_match_to_geotiff():
    """The kwargs both writers share also have identical defaults.

    A surprise-free dispatch ``to_geotiff(..., gpu=True)`` requires
    ``write_geotiff_gpu`` to default the same way for every kwarg the
    auto-dispatch entry point forwards (issue #1916 added
    ``allow_internal_only_jpeg`` to satisfy that contract; this test
    pins the broader parity).
    """
    eager_sig = inspect.signature(to_geotiff)
    gpu_sig = inspect.signature(write_geotiff_gpu)

    shared = set(eager_sig.parameters) & set(gpu_sig.parameters)
    # ``data`` and ``path`` are required positionals with no default;
    # comparing inspect.Parameter.empty against itself is fine.
    mismatches = []
    for name in sorted(shared):
        ed = eager_sig.parameters[name].default
        gd = gpu_sig.parameters[name].default
        if ed != gd:
            mismatches.append((name, ed, gd))
    assert not mismatches, (
        "write_geotiff_gpu and to_geotiff disagree on defaults: "
        f"{mismatches}"
    )
