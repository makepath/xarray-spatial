from xrspatial.aspect import aspect  # noqa
from xrspatial.aspect import eastness  # noqa
from xrspatial.aspect import northness  # noqa
from xrspatial.balanced_allocation import balanced_allocation  # noqa
from xrspatial.bilateral import bilateral  # noqa
from xrspatial.contour import contours  # noqa
from xrspatial.bump import bump  # noqa
from xrspatial.corridor import least_cost_corridor  # noqa
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
from xrspatial.edge_detection import laplacian  # noqa
from xrspatial.edge_detection import prewitt_x  # noqa
from xrspatial.edge_detection import prewitt_y  # noqa
from xrspatial.edge_detection import sobel_x  # noqa
from xrspatial.edge_detection import sobel_y  # noqa
from xrspatial.classify import equal_interval  # noqa
from xrspatial.classify import natural_breaks  # noqa
from xrspatial.classify import quantile  # noqa
from xrspatial.classify import reclassify  # noqa
from xrspatial.curvature import curvature  # noqa
from xrspatial.emerging_hotspots import emerging_hotspots  # noqa
from xrspatial.erosion import erode  # noqa
from xrspatial.hydro import fill  # noqa: unified wrapper
from xrspatial.hydro import fill_d8  # noqa
from xrspatial.interpolate import idw  # noqa
from xrspatial.interpolate import kriging  # noqa
from xrspatial.interpolate import spline  # noqa
from xrspatial.kde import kde  # noqa
from xrspatial.kde import line_density  # noqa
from xrspatial.fire import burn_severity_class  # noqa
from xrspatial.fire import dnbr  # noqa
from xrspatial.fire import fireline_intensity  # noqa
from xrspatial.fire import flame_length  # noqa
from xrspatial.fire import kbdi  # noqa
from xrspatial.fire import rate_of_spread  # noqa
from xrspatial.fire import rdnbr  # noqa
from xrspatial.flood import curve_number_runoff  # noqa
from xrspatial.flood import flood_depth  # noqa
from xrspatial.flood import flood_depth_vegetation  # noqa
from xrspatial.flood import inundation  # noqa
from xrspatial.flood import travel_time  # noqa
from xrspatial.flood import vegetation_curve_number  # noqa
from xrspatial.flood import vegetation_roughness  # noqa
from xrspatial.hydro import flow_accumulation  # noqa: unified wrapper
from xrspatial.hydro import flow_accumulation_d8, flow_accumulation_dinf, flow_accumulation_mfd  # noqa
from xrspatial.hydro import flow_direction  # noqa: unified wrapper
from xrspatial.hydro import flow_direction_d8, flow_direction_dinf, flow_direction_mfd  # noqa
from xrspatial.hydro import flow_length  # noqa: unified wrapper
from xrspatial.hydro import flow_length_d8, flow_length_dinf, flow_length_mfd  # noqa
from xrspatial.hydro import flow_path  # noqa: unified wrapper
from xrspatial.hydro import flow_path_d8, flow_path_dinf, flow_path_mfd  # noqa
from xrspatial.focal import mean  # noqa
from xrspatial.glcm import glcm_texture  # noqa
from xrspatial.morphology import morph_black_tophat  # noqa
from xrspatial.morphology import morph_closing  # noqa
from xrspatial.morphology import morph_dilate  # noqa
from xrspatial.morphology import morph_erode  # noqa
from xrspatial.morphology import morph_gradient  # noqa
from xrspatial.morphology import morph_opening  # noqa
from xrspatial.morphology import morph_white_tophat  # noqa
from xrspatial.hydro import hand  # noqa: unified wrapper
from xrspatial.hydro import hand_d8, hand_dinf, hand_mfd  # noqa
from xrspatial.hillshade import hillshade  # noqa
from xrspatial.mahalanobis import mahalanobis  # noqa
from xrspatial.multispectral import arvi  # noqa
from xrspatial.multispectral import bai  # noqa
from xrspatial.multispectral import evi  # noqa
from xrspatial.multispectral import mndwi  # noqa
from xrspatial.multispectral import msavi2  # noqa
from xrspatial.multispectral import nbr  # noqa
from xrspatial.multispectral import ndbi  # noqa
from xrspatial.multispectral import ndsi  # noqa
from xrspatial.multispectral import ndvi  # noqa
from xrspatial.multispectral import ndwi  # noqa
from xrspatial.multispectral import osavi  # noqa
from xrspatial.multispectral import savi  # noqa
from xrspatial.multispectral import sipi  # noqa
from xrspatial.pathfinding import a_star_search  # noqa
from xrspatial.pathfinding import multi_stop_search  # noqa
from xrspatial.normalize import rescale  # noqa
from xrspatial.normalize import standardize  # noqa
from xrspatial.perlin import perlin  # noqa
from xrspatial.preview import preview  # noqa
from xrspatial.proximity import allocation  # noqa
from xrspatial.rasterize import rasterize  # noqa
from xrspatial.resample import resample  # noqa
from xrspatial.proximity import direction  # noqa
from xrspatial.proximity import euclidean_distance  # noqa
from xrspatial.proximity import great_circle_distance  # noqa
from xrspatial.proximity import manhattan_distance  # noqa
from xrspatial.proximity import proximity  # noqa
from xrspatial.hydro import sink  # noqa: unified wrapper
from xrspatial.hydro import sink_d8  # noqa
from xrspatial.hydro import snap_pour_point  # noqa: unified wrapper
from xrspatial.hydro import snap_pour_point_d8  # noqa
from xrspatial.hydro import stream_link  # noqa: unified wrapper
from xrspatial.hydro import stream_link_d8, stream_link_dinf, stream_link_mfd  # noqa
from xrspatial.hydro import stream_order  # noqa: unified wrapper
from xrspatial.hydro import stream_order_d8, stream_order_dinf, stream_order_mfd  # noqa
from xrspatial.sieve import sieve  # noqa
from xrspatial.sky_view_factor import sky_view_factor  # noqa
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
from xrspatial.hydro import twi  # noqa: unified wrapper
from xrspatial.hydro import twi_d8  # noqa
from xrspatial.polygonize import polygonize  # noqa
from xrspatial.viewshed import viewshed  # noqa
from xrspatial.visibility import cumulative_viewshed  # noqa
from xrspatial.visibility import line_of_sight  # noqa
from xrspatial.visibility import visibility_frequency  # noqa
from xrspatial.hydro import basin  # noqa: unified wrapper
from xrspatial.hydro import basin_d8  # noqa
from xrspatial.hydro import basins  # noqa: backward-compat alias
from xrspatial.hydro import basins_d8  # noqa
from xrspatial.hydro import watershed  # noqa: unified wrapper
from xrspatial.hydro import watershed_d8, watershed_dinf, watershed_mfd  # noqa
from xrspatial.zonal import apply as zonal_apply  # noqa
from xrspatial.zonal import crop  # noqa
from xrspatial.zonal import trim  # noqa
from xrspatial.zonal import crosstab as zonal_crosstab  # noqa
from xrspatial.zonal import regions as regions  # noqa
from xrspatial.zonal import stats as zonal_stats  # noqa
from xrspatial.zonal import suggest_zonal_canvas as suggest_zonal_canvas  # noqa
from xrspatial.reproject import merge  # noqa
from xrspatial.reproject import reproject  # noqa
from xrspatial.utils import rechunk_no_shuffle  # noqa
from xrspatial.utils import fused_overlap  # noqa
from xrspatial.utils import multi_overlap  # noqa

import xrspatial.mcda  # noqa: F401  — exposes xrspatial.mcda subpackage
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
