"""Nodata lifecycle helper.

Centralizes the decisions previously scattered across the geotiff
read / write backends:

* whether a raw sentinel is usable against an integer dtype (finite,
  representable, integer-valued)
* what the *effective* sentinel is after MinIsWhite photometric
  inversion (the value the in-memory buffer actually carries)
* whether the read path's nodata-to-NaN promotion will or did fire
  (``masking_occurred``)
* whether an explicit ``dtype=`` cast on read promoted the buffer
  (``dtype_cast_occurred``) and what value to record in
  ``attrs['nodata_dtype_cast']``
* whether the writer's NaN-to-sentinel restore is appropriate for the
  current `(nodata, buffer dtype, masked_nodata attr)` triple
  (``writer_restore_sentinel``)

The backend kernels keep their array op (masking, photometric
inversion, dtype cast, NaN-to-sentinel rewrite); this helper owns the
policy questions so the inline ``if nodata is not None`` branches in
each backend collapse to a single ``NodataLifecycle(...)`` construction
plus attribute access.

PR-C of issue #2211 (sub-issue #2226).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


def _is_finite_integer(value) -> bool:
    """Return True iff *value* is a finite, integer-valued scalar.

    Used to gate integer-sentinel masking and integer-sentinel writer
    restore. Non-finite (NaN/Inf) and fractional sentinels return
    False so the caller skips the mask / restore step entirely.
    Bools are treated as integers by Python's ``float``/``int``; the
    writer's bool-rejection (issue #1973) already runs upstream.
    """
    if value is None:
        return False
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(as_float):
        return False
    return as_float.is_integer()


def _sentinel_fits_dtype(sentinel, dtype: np.dtype) -> bool:
    """Return True iff *sentinel* can be matched against pixels of *dtype*.

    Float dtypes always fit (NaN is special-cased by the caller); integer
    dtypes require the sentinel be a finite integer-valued scalar
    inside ``np.iinfo(dtype)``. Mirrors the eager / dask / GPU mask
    gates so the lifecycle helper can answer "would masking even do
    anything?" without duplicating the gate logic.
    """
    if sentinel is None:
        return False
    if dtype.kind == 'f':
        # NaN is a no-op on a float buffer (every NaN already matches),
        # but the comparison still "fits" the dtype, so return True and
        # let the caller's ``np.isnan(sentinel)`` branch short-circuit.
        return True
    if dtype.kind in ('u', 'i'):
        if not _is_finite_integer(sentinel):
            return False
        nodata_int = int(sentinel)
        info = np.iinfo(dtype)
        return info.min <= nodata_int <= info.max
    return False


def _invert_for_miniswhite(sentinel, dtype: np.dtype):
    """Return the sentinel value after MinIsWhite inversion.

    Mirrors ``_reader._miniswhite_inverted_nodata`` exactly so the
    lifecycle helper does not depend on the reader module (which would
    create a circular import via ``_reader -> _nodata -> _reader``).

    The caller has already decided that MinIsWhite applies
    (``photometric == 0`` and single-band); this helper just performs
    the value mapping. Non-finite or out-of-range integer sentinels
    are returned unchanged so the downstream mask gate still skips
    them.
    """
    if sentinel is None:
        return sentinel
    if dtype.kind == 'u':
        if not np.isfinite(sentinel):
            return sentinel
        if not float(sentinel).is_integer():
            return sentinel
        vi = int(sentinel)
        info = np.iinfo(dtype)
        if not (info.min <= vi <= info.max):
            return sentinel
        return info.max - vi
    if dtype.kind == 'f':
        if np.isnan(sentinel):
            return sentinel
        return -float(sentinel)
    return sentinel


@dataclass
class NodataLifecycle:
    """Value object describing a single read or write's nodata lifecycle.

    Parameters
    ----------
    declared : scalar or None
        The raw sentinel as declared on the source (e.g. parsed from
        ``GDAL_NODATA``, ``attrs['nodata']``, or the writer kwarg).
        ``None`` means "the source did not declare a sentinel".
    photometric : int or None
        TIFF Photometric Interpretation value (TIFF 6 tag 262). ``0``
        is MinIsWhite -- the reader inverts pixel values, and the
        effective sentinel post-inversion is what masking compares
        against. Any other value (1 MinIsBlack, 2 RGB, ...) leaves the
        sentinel unchanged. ``None`` indicates the caller has no
        photometric tag to apply (writer-side, VRT, raw arrays).
    dtype_in : numpy.dtype or None
        The dtype of the in-memory buffer the sentinel is being compared
        against. For reads this is the file's source dtype; for writes
        it is the in-memory buffer's dtype. Drives the
        ``_sentinel_fits_dtype`` and ``writer_restore_sentinel`` gates.
    dtype_request : numpy.dtype-like or None
        The caller's explicit ``dtype=`` kwarg (or ``None`` if the
        caller did not request a cast). Drives ``dtype_cast_occurred``
        / ``cast_dtype_name``.
    samples_per_pixel : int, default 1
        Used together with ``photometric`` to decide whether MinIsWhite
        inversion applies. The reader's gate is
        ``photometric == 0 and samples_per_pixel == 1`` (multi-band
        photometric=0 is not the same MinIsWhite case). Defaulting to
        1 means single-band call sites can omit it.

    Attributes
    ----------
    raw_sentinel
        Alias for ``declared``. Exposed for clarity at backend call
        sites that want "the raw, pre-inversion sentinel for
        ``attrs['nodata']``".
    effective_sentinel
        The sentinel value to compare against the in-memory buffer.
        Equal to ``raw_sentinel`` when MinIsWhite does not apply;
        otherwise the inverted value (per
        ``_reader._miniswhite_inverted_nodata`` semantics).
    pixels_present : bool or None
        Set by the execution path after the masking scan runs (the
        lifecycle helper itself does not own array data, only
        decisions). ``None`` means "no scan happened yet". The reader
        / writer assigns this attribute on the lifecycle instance
        after the per-buffer scan completes; downstream attrs
        propagation reads it back.
    """

    declared: Any
    photometric: Optional[int] = None
    dtype_in: Optional[np.dtype] = None
    dtype_request: Any = None
    samples_per_pixel: int = 1

    # Slot for execution to record the per-buffer scan result. Not part
    # of the constructor signature -- the helper's job is decisions,
    # not array operations. Backend kernels set this after their scan.
    pixels_present: Optional[bool] = field(default=None, init=False)

    # ------------------------------------------------------------------
    # Sentinel accessors
    # ------------------------------------------------------------------
    @property
    def raw_sentinel(self):
        """The declared sentinel, pre-photometric. Carried on ``attrs['nodata']``.

        Read paths put this value on ``attrs['nodata']`` so the on-disk
        sentinel is preserved across a round-trip even when the
        in-memory buffer carries the inverted (MinIsWhite) sentinel.
        """
        return self.declared

    @property
    def _applies_miniswhite(self) -> bool:
        """True iff MinIsWhite inversion applies to the current state."""
        return (
            self.photometric == 0
            and self.samples_per_pixel == 1
            and self.declared is not None
            and self.dtype_in is not None
        )

    @property
    def effective_sentinel(self):
        """The sentinel to compare against the in-memory buffer.

        Equals ``raw_sentinel`` when MinIsWhite does not apply;
        otherwise the inverted value. The reader's mask path and the
        GPU helper's ``mask_sentinel`` argument both want this value.
        """
        if not self._applies_miniswhite:
            return self.declared
        return _invert_for_miniswhite(self.declared, self.dtype_in)

    # ------------------------------------------------------------------
    # Mask / cast decisions
    # ------------------------------------------------------------------
    @property
    def sentinel_fits_buffer(self) -> bool:
        """True iff comparing pixels against ``effective_sentinel`` is meaningful.

        False when the source declared no sentinel, the integer dtype
        cannot represent it (out-of-range, fractional, non-finite), or
        the dtype is something the masking step does not support
        (complex, structured, ...). Backends gate their masking step
        on this property so the inline finite/integer/in-range checks
        collapse to one helper call.
        """
        if self.declared is None or self.dtype_in is None:
            return False
        return _sentinel_fits_dtype(self.effective_sentinel, self.dtype_in)

    def masking_occurred(self, *, mask_nodata: bool) -> bool:
        """Return the value the read path should put on ``attrs['masked_nodata']``.

        Matches the call sites' current contract:

        * ``True`` iff the caller opted into masking AND the final
          buffer is float (the integer-promotion branch only runs
          when the sentinel actually matches a pixel; an integer
          buffer surviving means "no pixels were masked").
        * ``False`` when the caller passed ``mask_nodata=False``
          (buffer keeps the literal sentinel) or when there is no
          declared sentinel.

        The ``final_dtype`` is taken from ``dtype_request`` when a
        caller cast was requested (the cast lands AFTER masking, so
        the final dtype is the request), otherwise from
        ``dtype_in``. This mirrors the eager finalizer ordering in
        ``_finalize_eager_read``.
        """
        if self.declared is None or not mask_nodata:
            return False
        final_dtype = self._final_dtype()
        if final_dtype is None:
            return False
        return final_dtype.kind == 'f'

    def _final_dtype(self) -> Optional[np.dtype]:
        """The dtype the buffer will have at the end of read finalization.

        Caller cast wins (it runs after masking). Otherwise the masking
        step may promote integer -> float64 when the sentinel matches a
        pixel; we cannot answer that without scanning, so the helper
        approximates with the input dtype. ``masking_occurred`` checks
        for ``kind == 'f'`` which is True iff (a) caller cast to a float
        dtype, (b) the source was already float, or (c) the integer
        promotion path actually fired. Returning the input dtype keeps
        the gate consistent with the existing call sites.
        """
        if self.dtype_request is not None:
            return np.dtype(self.dtype_request)
        if self.dtype_in is None:
            return None
        return np.dtype(self.dtype_in)

    @property
    def dtype_cast_occurred(self) -> bool:
        """True iff the caller requested an explicit dtype cast on read.

        Used to populate ``attrs['nodata_dtype_cast']`` (#2135). The
        attr is omitted when False.
        """
        return self.dtype_request is not None

    @property
    def cast_dtype_name(self) -> Optional[str]:
        """The string dtype name to record on ``attrs['nodata_dtype_cast']``.

        ``None`` when no caller cast happened; otherwise the canonical
        dtype name (e.g. ``"float64"``). Pairs with
        ``dtype_cast_occurred`` -- when that is False, this is None.
        """
        if self.dtype_request is None:
            return None
        return np.dtype(self.dtype_request).name

    # ------------------------------------------------------------------
    # Writer-side decision
    # ------------------------------------------------------------------
    def writer_restore_sentinel(
        self,
        *,
        buffer_dtype: np.dtype,
        masked_nodata_attr: Optional[bool] = None,
    ) -> bool:
        """Return True iff the writer should NaN-to-sentinel rewrite.

        Mirrors ``_attrs._should_restore_nan_sentinel`` plus the
        per-writer dtype/finite gates so the inline check in the
        writer collapses to one helper call:

        * No declared sentinel -> False (nothing to restore).
        * Buffer dtype not float -> False (no NaN to convert).
        * Sentinel itself is NaN -> False (NaN -> NaN is a no-op
          and the dtype.type(NaN) cast on integer fallbacks is
          undefined; the current writer's ``not np.isnan(nodata)``
          gate matches this).
        * ``masked_nodata`` attr explicitly ``False`` -> False
          (issue #1988: the read path did NOT mask, so any NaN in
          the buffer was put there by the user for other reasons;
          rewriting would corrupt their data).
        * Otherwise True (pre-#1988 default).

        ``masked_nodata_attr`` is the value read off
        ``attrs['masked_nodata']`` by the caller. ``None`` means the
        attr was absent (the pre-#1988 default applies). Pass ``False``
        explicitly only when the attr was literal ``False`` -- truthy
        values and absence both collapse to the True default.
        """
        if self.declared is None:
            return False
        if buffer_dtype is None or np.dtype(buffer_dtype).kind != 'f':
            return False
        # NaN sentinel: no rewrite needed (NaN already matches NaN).
        try:
            if np.isnan(self.declared):
                return False
        except TypeError:
            # Non-numeric sentinel (defensive; validators reject this
            # upstream). Treat as "do not restore" rather than crash.
            return False
        return masked_nodata_attr is not False
