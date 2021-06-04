try:
    from . import (  # noqa: F401
        reprojection,
        reprojection_rasterio,
        reprojection_rio
        )
except ImportError as e:
    msg = (
        "gdal module not installed.\n"
        "Please use pip install xarray-spatial[gdal]\n"
        "to install this module"
    )
    raise ImportError(str(e) + "\n" + msg) from e
