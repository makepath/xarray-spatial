"""Public export of read-path exceptions (#3265).

``VRTUnsupportedError``, ``CloudSizeLimitError``, and
``PixelSafetyLimitError`` are raised on public ``open_geotiff`` paths
(VRT capability validation, the ``max_cloud_bytes`` budget, and the
``max_pixels`` cap) but used to require private-module imports to catch
specifically, while the other 17 error classes were exported. These
tests pin the additive export: the names resolve on the package, sit in
``__all__``, and are the same objects the private modules define, so
existing private import paths keep working.
"""
import numpy as np
import pytest
import xarray as xr

import xrspatial.geotiff as g
from xrspatial.geotiff import open_geotiff, to_geotiff


@pytest.mark.parametrize("name", [
    "VRTUnsupportedError",
    "CloudSizeLimitError",
    "PixelSafetyLimitError",
])
def test_exception_is_public_3265(name):
    assert hasattr(g, name), f"{name} not exposed on xrspatial.geotiff"
    assert name in g.__all__, f"{name} not listed in xrspatial.geotiff.__all__"
    assert issubclass(getattr(g, name), ValueError)


def test_public_names_are_the_private_definitions_3265():
    """Same objects as the defining modules: private imports keep working."""
    from xrspatial.geotiff._errors import VRTUnsupportedError
    from xrspatial.geotiff._layout import PixelSafetyLimitError
    from xrspatial.geotiff._sources import CloudSizeLimitError

    assert g.VRTUnsupportedError is VRTUnsupportedError
    assert g.PixelSafetyLimitError is PixelSafetyLimitError
    assert g.CloudSizeLimitError is CloudSizeLimitError
    # The historical _reader re-export surface is unchanged too.
    from xrspatial.geotiff._reader import \
        PixelSafetyLimitError as reader_pixel_err
    assert g.PixelSafetyLimitError is reader_pixel_err


def test_vrt_unsupported_subclasses_exported_base_3265():
    """Broad catches via the exported base class still work."""
    assert issubclass(g.VRTUnsupportedError, g.GeoTIFFAmbiguousMetadataError)


def test_max_pixels_raises_publicly_catchable_error_3265(tmp_path):
    """The [stable] max_pixels cap raises the now-public class."""
    path = str(tmp_path / "t3265_cap.tif")
    da = xr.DataArray(
        np.arange(64, dtype=np.uint8).reshape(8, 8), dims=("y", "x"),
        coords={"y": np.arange(8, 0, -1) - 0.5, "x": np.arange(8) + 0.5},
        attrs={"crs": 4326})
    to_geotiff(da, path)

    with pytest.raises(g.PixelSafetyLimitError):
        open_geotiff(path, max_pixels=4)
