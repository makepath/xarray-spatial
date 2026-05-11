"""Regression tests for issue #1560.

``read_geotiff_gpu`` previously took a ``gpu={'auto','strict'}`` kwarg
that controlled GPU-failure policy, sharing a name with the boolean
``gpu=`` kwarg on ``open_geotiff`` / ``to_geotiff`` / ``read_vrt``.
Calling ``read_geotiff_gpu(path, gpu=True)`` -- the mental model after
using ``open_geotiff(path, gpu=True)`` -- raised the unhelpful
``ValueError: gpu must be 'auto' or 'strict', got True``.

The fix renames the kwarg to ``on_gpu_failure`` and keeps ``gpu=`` as a
deprecation shim:

* ``on_gpu_failure`` alone behaves like the old ``gpu`` kwarg.
* ``gpu`` alone still works, but emits ``DeprecationWarning``.
* Passing both raises ``TypeError``.

These tests exercise the validation path only, which fires before the
``cupy`` import inside ``read_geotiff_gpu``. No GPU runtime needed.
"""
from __future__ import annotations

import warnings

import pytest


def test_on_gpu_failure_invalid_value_raises_value_error():
    """Bad ``on_gpu_failure`` value still raises ``ValueError``."""
    from xrspatial.geotiff import read_geotiff_gpu

    with pytest.raises(ValueError, match="on_gpu_failure must be"):
        read_geotiff_gpu("/nonexistent.tif", on_gpu_failure='loose')


def test_gpu_alias_emits_deprecation_warning():
    """Old ``gpu=`` kwarg still routes through, with a DeprecationWarning."""
    from xrspatial.geotiff import read_geotiff_gpu

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        # Pass an invalid sentinel so we don't have to mock the full GPU
        # pipeline; ValueError fires after the deprecation handler runs.
        with pytest.raises(ValueError, match="on_gpu_failure must be"):
            read_geotiff_gpu("/nonexistent.tif", gpu='loose')

    deprecations = [r for r in records if issubclass(r.category, DeprecationWarning)]
    assert deprecations, "expected DeprecationWarning when gpu= is used"
    assert "on_gpu_failure" in str(deprecations[0].message)


def test_gpu_alias_accepts_old_values_without_validation_error():
    """``gpu='strict'`` was the legacy spelling; should still validate."""
    from xrspatial.geotiff import read_geotiff_gpu

    with warnings.catch_warnings():
        # Suppress the deprecation noise; we only care that the value
        # passes validation and the call proceeds past the value check.
        # In CPU-only CI the next step is ``import cupy`` which raises
        # ``ImportError`` (cupy is an optional extra); on a GPU host it
        # gets to the file-read stage and raises ``FileNotFoundError``.
        # Either is fine: both mean validation passed.
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(
                (FileNotFoundError, OSError, ValueError, ImportError)
        ) as exc_info:
            read_geotiff_gpu("/nonexistent.tif", gpu='strict')

    # The validation ValueError carries our exact message; a generic
    # file-read or cupy-import failure is fine because it means
    # validation passed.
    if isinstance(exc_info.value, ValueError):
        assert "on_gpu_failure must be" not in str(exc_info.value)


def test_passing_both_raises_type_error():
    """Mixing the new and deprecated names is ambiguous; refuse."""
    from xrspatial.geotiff import read_geotiff_gpu

    with pytest.raises(TypeError, match="pass either 'on_gpu_failure' or"):
        read_geotiff_gpu(
            "/nonexistent.tif",
            on_gpu_failure='strict',
            gpu='auto',
        )


@pytest.mark.parametrize("on_gpu_failure_val,gpu_val", [
    ('auto', 'strict'),
    ('auto', 'auto'),
    ('strict', 'strict'),
])
def test_passing_both_raises_regardless_of_values(on_gpu_failure_val, gpu_val):
    """Both-supplied is rejected even when ``on_gpu_failure='auto'``.

    A sentinel-based detection (rather than ``!= 'auto'``) catches the
    case where the caller passes the default value explicitly alongside
    the deprecated alias.
    """
    from xrspatial.geotiff import read_geotiff_gpu

    with pytest.raises(TypeError, match="pass either 'on_gpu_failure' or"):
        read_geotiff_gpu(
            "/nonexistent.tif",
            on_gpu_failure=on_gpu_failure_val,
            gpu=gpu_val,
        )


def test_gpu_alias_bool_no_longer_misleading_value_error():
    """Calling with ``gpu=True`` -- the documented bool from the dispatchers --
    used to raise ``ValueError: gpu must be 'auto' or 'strict', got True``.
    The new error explicitly names ``on_gpu_failure`` so the rename is
    discoverable from the traceback.
    """
    from xrspatial.geotiff import read_geotiff_gpu

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ValueError, match="on_gpu_failure must be"):
            read_geotiff_gpu("/nonexistent.tif", gpu=True)
