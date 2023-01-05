from xrspatial.aspect import aspect  # noqa
from xrspatial.bump import bump  # noqa
from xrspatial.classify import binary  # noqa
from xrspatial.classify import equal_interval  # noqa
from xrspatial.classify import natural_breaks  # noqa
from xrspatial.classify import quantile  # noqa
from xrspatial.classify import reclassify  # noqa
from xrspatial.curvature import curvature  # noqa
from xrspatial.focal import mean  # noqa
from xrspatial.hillshade import hillshade  # noqa
from xrspatial.multispectral import arvi  # noqa
from xrspatial.multispectral import evi  # noqa
from xrspatial.multispectral import nbr  # noqa
from xrspatial.multispectral import ndvi  # noqa
from xrspatial.multispectral import savi  # noqa
from xrspatial.multispectral import sipi  # noqa
from xrspatial.pathfinding import a_star_search  # noqa
from xrspatial.perlin import perlin  # noqa
from xrspatial.proximity import allocation  # noqa
from xrspatial.proximity import direction  # noqa
from xrspatial.proximity import euclidean_distance  # noqa
from xrspatial.proximity import great_circle_distance  # noqa
from xrspatial.proximity import manhattan_distance  # noqa
from xrspatial.proximity import proximity  # noqa
from xrspatial.slope import slope  # noqa
from xrspatial.terrain import generate_terrain  # noqa
from xrspatial.viewshed import viewshed  # noqa
from xrspatial.zonal import apply as zonal_apply  # noqa
from xrspatial.zonal import crop  # noqa
from xrspatial.zonal import trim  # noqa
from xrspatial.zonal import crosstab as zonal_crosstab  # noqa
from xrspatial.zonal import regions as regions  # noqa
from xrspatial.zonal import stats as zonal_stats  # noqa
from xrspatial.zonal import suggest_zonal_canvas as suggest_zonal_canvas  # noqa


try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


def test():
    """Run the xarray-spatial test suite."""
    import os
    try:
        import pytest
    except ImportError:
        import sys
        sys.stderr.write("You need to install py.test to run tests.\n\n")
        raise
    pytest.main([os.path.dirname(__file__)])
