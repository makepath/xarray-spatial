"""Regression test for issue #2208.

After #2207 routed all three GPU eager sites through
``_finalize_eager_read``, the sibling helper
``_apply_nodata_mask_gpu_with_presence`` had no remaining callers. The
helper was removed in this PR. This test pins the removal so a future
PR cannot quietly re-introduce a dead callable.

``_apply_nodata_mask_gpu`` is still alive on the chunked GPU dask path
(``_backends/gpu.py`` ``_chunk_task``), so this test also asserts that
helper is still importable as a sanity check that the removal was
surgical.
"""
import pytest

from xrspatial.geotiff._backends import _gpu_helpers


def test_apply_nodata_mask_gpu_with_presence_is_gone_2208():
    assert not hasattr(
        _gpu_helpers, '_apply_nodata_mask_gpu_with_presence'
    ), (
        "_apply_nodata_mask_gpu_with_presence was removed in #2208 after "
        "#2207 routed all GPU eager sites through _finalize_eager_read; "
        "the helper had zero remaining callers."
    )


def test_apply_nodata_mask_gpu_with_presence_not_importable_2208():
    with pytest.raises(ImportError):
        from xrspatial.geotiff._backends._gpu_helpers import (  # noqa: F401
            _apply_nodata_mask_gpu_with_presence,
        )


def test_apply_nodata_mask_gpu_still_present_2208():
    # _apply_nodata_mask_gpu is still on the chunked GPU dask path
    # (_chunk_task in _backends/gpu.py); removal in #2208 was scoped
    # to the dead sibling only.
    assert hasattr(_gpu_helpers, '_apply_nodata_mask_gpu')
    assert callable(_gpu_helpers._apply_nodata_mask_gpu)
