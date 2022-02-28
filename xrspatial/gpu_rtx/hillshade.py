# Only call functions in this file if has_rtx() returns True as this checks
# that the required dependent libraries are installed.

import cupy
import numba as nb
import numpy as np
import xarray as xr
from rtxpy import RTX
from scipy.spatial.transform import Rotation as R

from ..utils import calc_cuda_dims
from .cuda_utils import add, dot, invert, make_float3, mul
from .mesh_utils import create_triangulation


@nb.cuda.jit
def _generate_primary_rays_kernel(data, H, W):
    """
    A GPU kernel that given a set of x and y discrete coordinates on a raster
    terrain generates in @data a list of parallel rays that represent camera
    rays generated from an orthographic camera that is looking straight down
    at the surface from an origin height 10000.
    """
    i, j = nb.cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        if (j == W-1):
            data[i, j, 0] = j - 1e-3
        else:
            data[i, j, 0] = j + 1e-3

        if (i == H-1):
            data[i, j, 1] = i - 1e-3
        else:
            data[i, j, 1] = i + 1e-3

        data[i, j, 2] = 10000  # Location of the camera (height)
        data[i, j, 3] = 1e-3
        data[i, j, 4] = 0
        data[i, j, 5] = 0
        data[i, j, 6] = -1
        data[i, j, 7] = np.inf


def _generate_primary_rays(rays, H, W):
    griddim, blockdim = calc_cuda_dims((H, W))
    _generate_primary_rays_kernel[griddim, blockdim](rays, H, W)
    return 0


@nb.cuda.jit
def _generate_shadow_rays_kernel(rays, hits, normals, H, W, sun_dir):
    """
    A GPU kernel that takes a set rays and their intersection points with the
    triangulated surface, and calculates a set of shadow rays (overwriting the
    original rays) that have their origins at the intersection points and
    directions towards the sun.
    The normals vectors at the point of intersection of the original rays are
    cached in normals, thus we can later use them to do Lambertian shading,
    after the shadow rays have been traced.
    """
    i, j = nb.cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        dist = hits[i, j, 0]
        norm = make_float3(hits[i, j], 1)
        if (norm[2] < 0):
            norm = invert(norm)
        ray = rays[i, j]
        ray_origin = make_float3(ray, 0)
        ray_dir = make_float3(ray, 4)
        p = add(ray_origin, mul(ray_dir, dist))

        new_origin = add(p, mul(norm, 1e-3))
        ray[0] = new_origin[0]
        ray[1] = new_origin[1]
        ray[2] = new_origin[2]
        ray[3] = 1e-3
        ray[4] = sun_dir[0]
        ray[5] = sun_dir[1]
        ray[6] = sun_dir[2]
        ray[7] = np.inf if dist > 0 else 0

        normals[i, j, 0] = norm[0]
        normals[i, j, 1] = norm[1]
        normals[i, j, 2] = norm[2]


def _generate_shadow_rays(rays, hits, normals, H, W, sunDir):
    griddim, blockdim = calc_cuda_dims((H, W))
    _generate_shadow_rays_kernel[griddim, blockdim](
        rays, hits, normals, H, W, sunDir)
    return 0


@nb.cuda.jit
def _shade_lambert_kernel(hits, normals, output, H, W, sun_dir, cast_shadows):
    """
    This kernel does a simple Lambertian shading.
    The hits array contains the results of tracing the shadow rays through the
    scene. If the value in hits[x, y, 0] is > 0 then a valid intersection
    occurred and that means that the point at location x, y is in shadow.
    The normals array stores the normal at the intersecion point of each
    camera ray. We then use the information for light visibility and normal to
    apply Lambert's cosine law.
    """
    i, j = nb.cuda.grid(2)
    if i >= 0 and i < H and j >= 0 and j < W:
        # Normal at the intersection of camera ray (i,j) with the scene
        norm = make_float3(normals[i, j], 0)

        light_dir = make_float3(sun_dir, 0)
        cos_theta = dot(light_dir, norm)  # light_dir and norm are normalised.

        temp = (cos_theta + 1) / 2

        if cast_shadows and hits[i, j, 0] >= 0:
            temp = temp / 2

        if temp > 1:
            temp = 1
        elif temp < 0:
            temp = 0

        output[i, j] = temp


def _shade_lambert(hits, normals, output, H, W, sun_dir, cast_shadows):
    griddim, blockdim = calc_cuda_dims((H, W))
    _shade_lambert_kernel[griddim, blockdim](
        hits, normals, output, H, W, sun_dir, cast_shadows)
    return 0


def _get_sun_dir(angle_altitude, azimuth):
    """
    Calculate the vector towards the sun based on sun altitude angle and
    azimuth.
    """
    north = (0, 1, 0)
    rx = R.from_euler('x', angle_altitude, degrees=True)
    rz = R.from_euler('z', azimuth+180, degrees=True)
    sun_dir = rx.apply(north)
    sun_dir = rz.apply(sun_dir)
    return sun_dir


def _hillshade_rt(raster: xr.DataArray,
                  optix: RTX,
                  azimuth: int,
                  angle_altitude: int,
                  shadows: bool) -> xr.DataArray:
    H, W = raster.shape
    sun_dir = cupy.array(_get_sun_dir(angle_altitude, azimuth))

    # Device buffers
    d_rays = cupy.empty((H, W, 8), np.float32)
    d_hits = cupy.empty((H, W, 4), np.float32)
    d_aux = cupy.empty((H, W, 3), np.float32)
    d_output = cupy.empty((H, W), np.float32)

    _generate_primary_rays(d_rays, H, W)
    device = cupy.cuda.Device(0)
    device.synchronize()
    res = optix.trace(d_rays, d_hits, W*H)
    if res:
        raise RuntimeError(f"Failed trace 1, error code: {res}")

    _generate_shadow_rays(d_rays, d_hits, d_aux, H, W, sun_dir)
    if shadows:
        device.synchronize()
        res = optix.trace(d_rays, d_hits, W*H)
        if res:
            raise RuntimeError(f"Failed trace 2, error code: {res}")

    _shade_lambert(d_hits, d_aux, d_output, H, W, sun_dir, shadows)

    d_output[0, :] = cupy.nan
    d_output[-1, :] = cupy.nan
    d_output[:, 0] = cupy.nan
    d_output[:, -1] = cupy.nan

    return d_output


def hillshade_rtx(raster: xr.DataArray,
                  azimuth: int,
                  angle_altitude: int,
                  shadows: bool) -> xr.DataArray:
    if not isinstance(raster.data, cupy.ndarray):
        raise TypeError("raster.data must be a cupy array")

    optix = RTX()
    create_triangulation(raster, optix)

    return _hillshade_rt(
        raster, optix, azimuth=azimuth, angle_altitude=angle_altitude,
        shadows=shadows)
