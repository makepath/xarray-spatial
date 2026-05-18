"""Tests for the ambiguous-metadata validator framework (issue #1987 PR 0).

PR 0 lands the error class hierarchy in ``_errors.py`` and the
register / dispatch framework in ``_validation.py``. No raises yet;
each per-case PR (#1987 PRs 2-7) registers its own check.

These tests cover:

- the error class hierarchy is what the per-case PRs expect to subclass
- the hooks are no-ops when no checks are registered (so PR 0 cannot
  regress any existing entry point)
- registration is idempotent and ordered
- unregistration is tolerant of unknown callables
- a registered check that raises propagates through the hook
- a context mapping is forwarded verbatim to each check
"""
from __future__ import annotations

import pytest

from xrspatial.geotiff._errors import (
    ConflictingCRSError,
    ConflictingNodataError,
    GeoTIFFAmbiguousMetadataError,
    InvalidCRSCodeError,
    MixedBandMetadataError,
    NonUniformCoordsError,
    RotatedTransformError,
    UnparseableCRSError,
)
from xrspatial.geotiff import _validation as _validation_mod
from xrspatial.geotiff._validation import (
    _registered_read_metadata_checks,
    _registered_write_metadata_checks,
    register_read_metadata_check,
    register_write_metadata_check,
    unregister_read_metadata_check,
    unregister_write_metadata_check,
    validate_read_metadata,
    validate_write_metadata,
)


@pytest.fixture(autouse=True)
def _reset_metadata_check_registries():
    """Snapshot and restore the process-global check registries (#1987).

    The registries are module-global lists. A test that registers a
    check and crashes before its ``try/finally unregister`` would
    leave a stale callable in place and pollute later tests. This
    autouse fixture snapshots both registries before the test and
    restores them after, regardless of whether the test passed,
    failed, or raised, so the file is robust to per-test mistakes.
    """
    read_snapshot = list(_validation_mod._READ_METADATA_CHECKS)
    write_snapshot = list(_validation_mod._WRITE_METADATA_CHECKS)
    try:
        yield
    finally:
        _validation_mod._READ_METADATA_CHECKS[:] = read_snapshot
        _validation_mod._WRITE_METADATA_CHECKS[:] = write_snapshot


# ----------------------------------------------------------------------
# Error class hierarchy
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "subclass",
    [
        InvalidCRSCodeError,
        UnparseableCRSError,
        RotatedTransformError,
        NonUniformCoordsError,
        MixedBandMetadataError,
        ConflictingCRSError,
        ConflictingNodataError,
    ],
)
def test_subclass_inherits_from_base(subclass):
    """Each per-case error is catchable via the family base class."""
    assert issubclass(subclass, GeoTIFFAmbiguousMetadataError)


def test_base_is_value_error_subclass():
    """Existing ``except ValueError`` callers keep catching the family."""
    assert issubclass(GeoTIFFAmbiguousMetadataError, ValueError)


def test_error_classes_reexported_from_public_namespace():
    """User-facing ambiguity errors must be importable from ``xrspatial.geotiff``.

    ``UnsafeURLError`` and ``GeoTIFFFallbackWarning`` follow the same
    convention. Without the re-export, callers would have to dig into
    the private ``_errors`` module to write ``except`` clauses, which
    couples them to an implementation-detail import path.
    """
    import xrspatial.geotiff as geotiff_pkg

    for name in (
        "GeoTIFFAmbiguousMetadataError",
        "InvalidCRSCodeError",
        "UnparseableCRSError",
        "RotatedTransformError",
        "NonUniformCoordsError",
        "MixedBandMetadataError",
        "ConflictingCRSError",
        "ConflictingNodataError",
    ):
        assert hasattr(geotiff_pkg, name), (
            f"{name} not exposed on xrspatial.geotiff")
        assert name in geotiff_pkg.__all__, (
            f"{name} not listed in xrspatial.geotiff.__all__")
    # The re-exported class is the same object as the one in ``_errors``.
    assert (geotiff_pkg.GeoTIFFAmbiguousMetadataError
            is GeoTIFFAmbiguousMetadataError)


def test_subclass_catch_does_not_catch_siblings():
    """``except UnparseableCRSError`` must not catch ``RotatedTransformError``."""
    with pytest.raises(RotatedTransformError):
        try:
            raise RotatedTransformError("rotated")
        except UnparseableCRSError:
            pytest.fail("sibling subclass should not catch")


# ----------------------------------------------------------------------
# Hook no-op behaviour
# ----------------------------------------------------------------------


def test_read_hook_is_noop_when_no_checks_registered():
    """PR 0 must not change behaviour at any read entry point."""
    # Even with no context, an empty registry must return cleanly.
    validate_read_metadata()
    validate_read_metadata({})
    validate_read_metadata({"unused": object()})


def test_write_hook_is_noop_when_no_checks_registered():
    """PR 0 must not change behaviour at any write entry point."""
    validate_write_metadata()
    validate_write_metadata({})
    validate_write_metadata({"unused": object()})


# ----------------------------------------------------------------------
# Registration / dispatch
# ----------------------------------------------------------------------


def test_register_and_dispatch_read_check():
    # Use an opaque payload key that no registered #1987 check inspects,
    # so this test stays scoped to the dispatch mechanism rather than
    # the semantics of any one check.
    seen: list[dict] = []

    def check(ctx):
        seen.append(dict(ctx))

    register_read_metadata_check(check)
    try:
        validate_read_metadata({"_dispatch_probe": "value"})
        assert seen == [{"_dispatch_probe": "value"}]
    finally:
        unregister_read_metadata_check(check)


def test_register_and_dispatch_write_check():
    # Same isolation as the read-side test above.
    seen: list[dict] = []

    def check(ctx):
        seen.append(dict(ctx))

    register_write_metadata_check(check)
    try:
        validate_write_metadata({"_dispatch_probe": "value"})
        assert seen == [{"_dispatch_probe": "value"}]
    finally:
        unregister_write_metadata_check(check)


def test_register_is_idempotent_read():
    def check(ctx):
        return None

    register_read_metadata_check(check)
    register_read_metadata_check(check)
    try:
        # Same callable registered twice still appears once.
        assert _registered_read_metadata_checks().count(check) == 1
    finally:
        unregister_read_metadata_check(check)


def test_register_is_idempotent_write():
    def check(ctx):
        return None

    register_write_metadata_check(check)
    register_write_metadata_check(check)
    try:
        assert _registered_write_metadata_checks().count(check) == 1
    finally:
        unregister_write_metadata_check(check)


def test_dispatch_preserves_registration_order():
    order: list[str] = []

    def first(ctx):
        order.append("first")

    def second(ctx):
        order.append("second")

    register_read_metadata_check(first)
    register_read_metadata_check(second)
    try:
        validate_read_metadata({})
        assert order == ["first", "second"]
    finally:
        unregister_read_metadata_check(first)
        unregister_read_metadata_check(second)


def test_write_dispatch_preserves_registration_order():
    """Symmetry: write-side has its own registry, must order the same."""
    order: list[str] = []

    def first(ctx):
        order.append("first")

    def second(ctx):
        order.append("second")

    register_write_metadata_check(first)
    register_write_metadata_check(second)
    try:
        validate_write_metadata({})
        assert order == ["first", "second"]
    finally:
        unregister_write_metadata_check(first)
        unregister_write_metadata_check(second)


def test_write_first_raising_check_short_circuits():
    """Symmetry: write-side short-circuits on first raise like read-side."""
    later_called = {"flag": False}

    def deny(ctx):
        raise ConflictingCRSError("crs mismatch")

    def later(ctx):
        later_called["flag"] = True

    register_write_metadata_check(deny)
    register_write_metadata_check(later)
    try:
        with pytest.raises(ConflictingCRSError):
            validate_write_metadata({})
        assert later_called["flag"] is False
    finally:
        unregister_write_metadata_check(deny)
        unregister_write_metadata_check(later)


def test_read_and_write_registries_are_independent():
    """A read-side check must not fire from the write hook (and vice versa)."""
    read_calls = {"count": 0}
    write_calls = {"count": 0}

    def read_check(ctx):
        read_calls["count"] += 1

    def write_check(ctx):
        write_calls["count"] += 1

    register_read_metadata_check(read_check)
    register_write_metadata_check(write_check)
    try:
        validate_read_metadata({})
        assert read_calls["count"] == 1
        assert write_calls["count"] == 0

        validate_write_metadata({})
        assert read_calls["count"] == 1
        assert write_calls["count"] == 1
    finally:
        unregister_read_metadata_check(read_check)
        unregister_write_metadata_check(write_check)


def test_unregister_unknown_check_is_safe():
    """Test teardown must tolerate double-unregister."""

    def never_registered(ctx):
        return None

    # Must not raise.
    unregister_read_metadata_check(never_registered)
    unregister_write_metadata_check(never_registered)


def test_check_can_raise_typed_error():
    def deny(ctx):
        raise UnparseableCRSError("bad WKT")

    register_read_metadata_check(deny)
    try:
        with pytest.raises(UnparseableCRSError, match="bad WKT"):
            # Opaque key so the test does not collide with the registered
            # ``_check_read_unparseable_crs`` check's behaviour on real
            # ``crs_wkt`` payloads.
            validate_read_metadata({"_dispatch_probe": "value"})
    finally:
        unregister_read_metadata_check(deny)


def test_first_raising_check_short_circuits():
    """Once a check raises, later checks must not see the call."""
    later_called = {"flag": False}

    def deny(ctx):
        raise RotatedTransformError("rotated")

    def later(ctx):
        later_called["flag"] = True

    register_read_metadata_check(deny)
    register_read_metadata_check(later)
    try:
        with pytest.raises(RotatedTransformError):
            validate_read_metadata({})
        assert later_called["flag"] is False
    finally:
        unregister_read_metadata_check(deny)
        unregister_read_metadata_check(later)


def test_dispatch_is_safe_against_registry_mutation_read():
    """A check that mutates the read registry must not skip later checks.

    The dispatcher iterates over a snapshot of the registry, so a check
    that registers or unregisters another callable (whether on purpose
    or via an import side effect) cannot reshape the live list mid-loop
    and skip a sibling check or visit one twice.
    """
    seen: list[str] = []

    def mutating(ctx):
        seen.append("mutating")
        # Self-unregister mid-dispatch; the live list would otherwise
        # shift left and the iterator would skip ``second``.
        unregister_read_metadata_check(mutating)

    def second(ctx):
        seen.append("second")

    register_read_metadata_check(mutating)
    register_read_metadata_check(second)
    try:
        validate_read_metadata({})
        assert seen == ["mutating", "second"]
    finally:
        unregister_read_metadata_check(mutating)
        unregister_read_metadata_check(second)


def test_dispatch_is_safe_against_registry_mutation_write():
    """Write hook mirrors the read-side snapshot guarantee."""
    seen: list[str] = []

    def mutating(ctx):
        seen.append("mutating")
        unregister_write_metadata_check(mutating)

    def second(ctx):
        seen.append("second")

    register_write_metadata_check(mutating)
    register_write_metadata_check(second)
    try:
        validate_write_metadata({})
        assert seen == ["mutating", "second"]
    finally:
        unregister_write_metadata_check(mutating)
        unregister_write_metadata_check(second)


def test_none_context_is_treated_as_empty_mapping():
    """Calling ``validate_*_metadata()`` with no args must not crash a check."""
    seen: list[object] = []

    def check(ctx):
        # The contract: ctx is a mapping, never None.
        seen.append(dict(ctx))

    register_read_metadata_check(check)
    try:
        validate_read_metadata()
        assert seen == [{}]
    finally:
        unregister_read_metadata_check(check)
