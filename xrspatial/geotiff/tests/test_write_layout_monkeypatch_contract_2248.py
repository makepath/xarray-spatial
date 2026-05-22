"""Lock down the ``_writer.*`` monkeypatch contract for ``_assemble_tiff``.

When the IFD-assembly and layout helpers were extracted from
``_writer.py`` into ``_write_layout.py`` (issue #2248), the eager
writer's ``_assemble_tiff`` retained the pre-extraction property that
patching ``_writer._compute_classic_ifd_overhead`` (and the other
helpers ``_assemble_tiff`` dispatches to) flows through into
``_assemble_tiff``'s execution. The single existing regression at
``test_eager_bigtiff_overhead_exact_1905`` covers the
``_compute_classic_ifd_overhead`` indirection only. These tests cover
the remaining indirected names so a future refactor that inlines any
of them at the call site is caught immediately.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrspatial.geotiff import _writer as writer_mod
from xrspatial.geotiff import to_geotiff


def _make_float32(h: int = 8, w: int = 8) -> xr.DataArray:
    arr = np.arange(h * w, dtype=np.float32).reshape(h, w)
    return xr.DataArray(
        arr,
        dims=["y", "x"],
        coords={
            "x": np.arange(w, dtype=np.float64),
            "y": np.arange(h, dtype=np.float64),
        },
        attrs={"crs": 4326},
    )


@pytest.mark.parametrize(
    "helper_name",
    [
        "_promote_offsets_to_long8",
        "_assemble_standard_layout",
        "_assemble_cog_layout",
        "_resolve_photometric",
    ],
)
def test_assemble_tiff_resolves_helper_through_writer_module(
    monkeypatch, tmp_path, helper_name,
):
    """``_assemble_tiff`` must look up ``helper_name`` via ``_writer``.

    Replace the helper on the ``_writer`` module with a sentinel that
    records the call and delegates to the real implementation. If
    ``_assemble_tiff`` were to bind the helper at import time (rather
    than resolving it through ``_writer`` on each call), the sentinel
    would never fire and the assertion would fail.
    """
    real = getattr(writer_mod, helper_name)
    calls: list[tuple] = []

    def _wrapped(*args, **kwargs):
        calls.append((args, tuple(sorted(kwargs.items()))))
        return real(*args, **kwargs)

    monkeypatch.setattr(writer_mod, helper_name, _wrapped)

    da = _make_float32(8, 8)
    path = str(tmp_path / f"monkeypatch_{helper_name}_2248.tif")

    # ``_assemble_cog_layout`` only fires when at least one overview
    # is written; ``_promote_offsets_to_long8`` only fires when the
    # writer chooses BigTIFF. Pass the right kwargs per helper so each
    # one is exercised by ``_assemble_tiff`` on this call.
    if helper_name == "_assemble_cog_layout":
        to_geotiff(da, path, cog=True, overview_levels=[2])
    elif helper_name == "_promote_offsets_to_long8":
        to_geotiff(da, path, bigtiff=True)
    else:
        to_geotiff(da, path)

    assert calls, (
        f"_assemble_tiff did not call _writer.{helper_name}; the "
        f"monkeypatch on the _writer namespace was bypassed."
    )
