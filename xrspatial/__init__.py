from xrspatial.aspect import aspect  # noqa
from xrspatial.balanced_allocation import balanced_allocation  # noqa
from xrspatial.bump import bump  # noqa
from xrspatial.cost_distance import cost_distance  # noqa
from xrspatial.dasymetric import disaggregate  # noqa
from xrspatial.dasymetric import pycnophylactic  # noqa
from xrspatial.dasymetric import validate_disaggregation  # noqa
from xrspatial.classify import binary  # noqa
from xrspatial.classify import box_plot  # noqa
from xrspatial.classify import head_tail_breaks  # noqa
from xrspatial.classify import maximum_breaks  # noqa
from xrspatial.classify import percentiles  # noqa
from xrspatial.classify import std_mean  # noqa
from xrspatial.diagnostics import diagnose  # noqa
from xrspatial.diffusion import diffuse  # noqa
from xrspatial.classify import equal_interval  # noqa
from xrspatial.classify import natural_breaks  # noqa
from xrspatial.classify import quantile  # noqa
from xrspatial.classify import reclassify  # noqa
from xrspatial.curvature import curvature  # noqa
from xrspatial.emerging_hotspots import emerging_hotspots  # noqa
from xrspatial.erosion import erode  # noqa
from xrspatial.fill import fill  # noqa
from xrspatial.interpolate import idw  # noqa
from xrspatial.interpolate import kriging  # noqa
from xrspatial.interpolate import spline  # noqa
from xrspatial.fire import burn_severity_class  # noqa
from xrspatial.fire import dnbr  # noqa
from xrspatial.fire import fireline_intensity  # noqa
from xrspatial.fire import flame_length  # noqa
from xrspatial.fire import kbdi  # noqa
from xrspatial.fire import rate_of_spread  # noqa
from xrspatial.fire import rdnbr  # noqa
from xrspatial.flood import curve_number_runoff  # noqa
from xrspatial.flood import flood_depth  # noqa
from xrspatial.flood import inundation  # noqa
from xrspatial.flood import travel_time  # noqa
from xrspatial.flow_accumulation import flow_accumulation  # noqa
from xrspatial.flow_accumulation_mfd import flow_accumulation_mfd  # noqa
from xrspatial.flow_direction import flow_direction  # noqa
from xrspatial.flow_direction_dinf import flow_direction_dinf  # noqa
from xrspatial.flow_direction_mfd import flow_direction_mfd  # noqa
from xrspatial.flow_length import flow_length  # noqa
from xrspatial.flow_path import flow_path  # noqa
from xrspatial.focal import mean  # noqa
from xrspatial.morphology import morph_closing  # noqa
from xrspatial.morphology import morph_dilate  # noqa
from xrspatial.morphology import morph_erode  # noqa
from xrspatial.morphology import morph_opening  # noqa
from xrspatial.hand import hand  # noqa
from xrspatial.hillshade import hillshade  # noqa
from xrspatial.mahalanobis import mahalanobis  # noqa
from xrspatial.multispectral import arvi  # noqa
from xrspatial.multispectral import evi  # noqa
from xrspatial.multispectral import nbr  # noqa
from xrspatial.multispectral import ndvi  # noqa
from xrspatial.multispectral import savi  # noqa
from xrspatial.multispectral import sipi  # noqa
from xrspatial.pathfinding import a_star_search  # noqa
from xrspatial.pathfinding import multi_stop_search  # noqa
from xrspatial.perlin import perlin  # noqa
from xrspatial.proximity import allocation  # noqa
from xrspatial.proximity import direction  # noqa
from xrspatial.proximity import euclidean_distance  # noqa
from xrspatial.proximity import great_circle_distance  # noqa
from xrspatial.proximity import manhattan_distance  # noqa
from xrspatial.proximity import proximity  # noqa
from xrspatial.sink import sink  # noqa
from xrspatial.snap_pour_point import snap_pour_point  # noqa
from xrspatial.stream_link import stream_link  # noqa
from xrspatial.stream_order import stream_order  # noqa
from xrspatial.slope import slope  # noqa
from xrspatial.surface_distance import surface_allocation  # noqa
from xrspatial.surface_distance import surface_direction  # noqa
from xrspatial.surface_distance import surface_distance  # noqa
from xrspatial.terrain import generate_terrain  # noqa
from xrspatial.terrain_metrics import landforms  # noqa
from xrspatial.terrain_metrics import LANDFORM_CLASSES  # noqa
from xrspatial.terrain_metrics import roughness  # noqa
from xrspatial.terrain_metrics import tpi  # noqa
from xrspatial.terrain_metrics import tri  # noqa
from xrspatial.twi import twi  # noqa
from xrspatial.polygonize import polygonize  # noqa
from xrspatial.viewshed import viewshed  # noqa
from xrspatial.basin import basin  # noqa
from xrspatial.watershed import basins  # noqa
from xrspatial.watershed import watershed  # noqa
from xrspatial.zonal import apply as zonal_apply  # noqa
from xrspatial.zonal import crop  # noqa
from xrspatial.zonal import trim  # noqa
from xrspatial.zonal import crosstab as zonal_crosstab  # noqa
from xrspatial.zonal import regions as regions  # noqa
from xrspatial.zonal import stats as zonal_stats  # noqa
from xrspatial.zonal import suggest_zonal_canvas as suggest_zonal_canvas  # noqa

import xrspatial.accessor  # noqa: F401  — registers .xrs accessors


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
