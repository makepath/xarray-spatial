import cupy
import numba as nb
import numpy as np


def _axis_resolution(index, n):
    """Cell size along one axis, or 1.0 when it cannot be determined.

    Falls back to unit spacing (the old integer-grid behaviour) when the
    raster has no coordinate index on that axis or the axis is a single
    cell, so a coordinate-less raster still builds a sensible mesh.
    """
    if index is None or n <= 1:
        return 1.0
    coords = index.values
    return abs(float((coords[-1] - coords[0]) / (n - 1)))


def _cell_resolution(raster):
    """Return the (ew_res, ns_res) cell sizes from the raster x/y coords."""
    H, W = raster.shape
    ew_res = _axis_resolution(raster.indexes.get('x'), W)
    ns_res = _axis_resolution(raster.indexes.get('y'), H)
    return ew_res, ns_res


def create_triangulation(raster, optix):
    """Build a triangulated mesh on the GPU from a 2D elevation raster.

    Mesh vertex x/y coordinates use the real cell resolution
    (``ew_res``, ``ns_res``) read from the raster's x/y coordinates instead
    of integer grid indices, so the mesh has the same shape the CPU sweep
    works on (issue #2861).  The z-coordinate is scaled by the
    resolution-independent factor ``max(H, W) / maxH`` so the terrain ratio
    stays suitable for ray tracing; this factor must NOT depend on
    ``ew_res``/``ns_res`` (a resolution-dependent z-scale would cancel the
    x/y stretch).

    Note on viewshed parity: ray-triangle occlusion is invariant under a
    per-axis (linear) scaling of the mesh, and the viewshed output angle is
    computed from the real ``ew_res``/``ns_res`` downstream, so this change
    does not by itself alter the viewshed result -- it keeps the GPU mesh
    geometry consistent with the CPU coordinate convention rather than
    fixing a result divergence.  See the PR for issue #2861.

    A positive, finite maximum elevation is required: an all-zero or
    all-NaN raster has no elevation variance and yields ``inf`` or ``NaN``
    mesh vertices that the OptiX raytracer cannot use sensibly.

    Returns
    -------
    (scale, ew_res, ns_res)
        The z scale factor and the cell resolution used to place the mesh
        vertices.  Callers need the resolution to cast camera rays at the
        matching real-world x/y positions.

    Raises
    ------
    ValueError
        If ``cupy.amax(raster.data)`` is non-positive or non-finite, i.e.
        the raster has no positive elevation variance to scale by.
    """
    # Calculate a scale factor for the height that maintains the ratio
    # width/height
    H, W = raster.shape

    ew_res, ns_res = _cell_resolution(raster)

    # Scale the terrain so that the width is proportional to the height
    # Thus the terrain would be neither too flat nor too steep and
    # raytracing will give best accuracy.  maxDim stays in grid cells so the
    # z-scale is a single constant, independent of the anisotropic x/y
    # resolution applied to the mesh vertices below (see docstring).
    maxH = float(cupy.amax(raster.data))
    maxDim = max(H, W)

    # Guard against a divide-by-zero / divide-by-NaN that would propagate
    # inf or NaN into every vertex z-coordinate (issue #1378).  An all-zero
    # raster gives maxH == 0.0, an all-NaN raster gives maxH == nan, and a
    # raster with only non-positive values would invert the mesh.  All of
    # these cases produce garbage geometry downstream, so fail fast before
    # any hash or device-buffer work.
    if not np.isfinite(maxH) or maxH <= 0.0:
        raise ValueError(
            "raster has no positive elevation variance "
            f"(max={maxH}); cannot build mesh for hillshade/viewshed"
        )
    scale = maxDim / maxH

    datahash = np.uint64(hash(str(raster.data.get())) % (1 << 64))
    optixhash = np.uint64(optix.getHash())

    if optixhash != datahash:
        num_tris = (H - 1) * (W - 1) * 2
        verts = cupy.empty(H * W * 3, np.float32)
        triangles = cupy.empty(num_tris * 3, np.int32)
        # Generate a mesh from the terrain (buffers are on the GPU, so
        # generation happens also on GPU)
        res = _triangulate_terrain(verts, triangles, raster, scale,
                                   ew_res, ns_res)
        if res:
            raise RuntimeError(
                f"Failed to generate mesh from terrain, error code: {res}")

        res = optix.build(datahash, verts, triangles)
        if res:
            raise RuntimeError(
                f"OptiX failed to build GAS, error code: {res}")

        # Enable for debug purposes
        if False:
            write("mesh.stl", verts, triangles)
        # Clear some GPU memory that we no longer need
        verts = None
        triangles = None
        cupy.get_default_memory_pool().free_all_blocks()
    return scale, ew_res, ns_res


@nb.cuda.jit
def _triangulate_terrain_kernel(verts, triangles, data, H, W, scale,
                                ew_res, ns_res, stride):
    global_id = stride + nb.cuda.grid(1)
    if global_id < W*H:
        h = global_id // W
        w = global_id % W
        mesh_map_index = h * W + w

        val = data[h, w]

        offset = 3*mesh_map_index
        verts[offset] = w * ew_res
        verts[offset+1] = h * ns_res
        verts[offset+2] = val * scale

        if w != W - 1 and h != H - 1:
            offset = 6*(h * (W-1) + w)
            triangles[offset+0] = np.int32(mesh_map_index + W)
            triangles[offset+1] = np.int32(mesh_map_index + W + 1)
            triangles[offset+2] = np.int32(mesh_map_index)
            triangles[offset+3] = np.int32(mesh_map_index + W + 1)
            triangles[offset+4] = np.int32(mesh_map_index + 1)
            triangles[offset+5] = np.int32(mesh_map_index)


@nb.njit(parallel=True)
def _triangulate_cpu(verts, triangles, data, H, W, scale, ew_res, ns_res):
    for h in nb.prange(H):
        for w in range(W):
            mesh_map_index = h * W + w

            val = data[h, w]

            offset = 3*mesh_map_index
            verts[offset] = w * ew_res
            verts[offset+1] = h * ns_res
            verts[offset+2] = val * scale

            if w != W - 1 and h != H - 1:
                offset = 6*(h*(W-1) + w)
                triangles[offset+0] = np.int32(mesh_map_index + W)
                triangles[offset+1] = np.int32(mesh_map_index + W+1)
                triangles[offset+2] = np.int32(mesh_map_index)
                triangles[offset+3] = np.int32(mesh_map_index + W+1)
                triangles[offset+4] = np.int32(mesh_map_index + 1)
                triangles[offset+5] = np.int32(mesh_map_index)


def _triangulate_terrain(verts, triangles, terrain, scale=1,
                         ew_res=1.0, ns_res=1.0):
    H, W = terrain.shape
    if isinstance(terrain.data, np.ndarray):
        _triangulate_cpu(verts, triangles, terrain.data, H, W, scale,
                         ew_res, ns_res)
    if isinstance(terrain.data, cupy.ndarray):
        job_size = H*W
        blockdim = 1024
        griddim = (job_size + blockdim - 1) // 1024
        d = 100
        offset = 0
        while job_size > 0:
            batch = min(d, griddim)
            _triangulate_terrain_kernel[batch, blockdim](
                verts, triangles, terrain.data, H, W, scale,
                ew_res, ns_res, offset)
            offset += batch*blockdim
            job_size -= batch*blockdim
    return 0


@nb.jit(nopython=True)
def _fill_contents(content, verts, triangles, num_tris):
    v = np.empty(12, np.float32)
    pad = np.zeros(2, np.int8)
    offset = 0
    for i in range(num_tris):
        t0 = triangles[3*i+0]
        t1 = triangles[3*i+1]
        t2 = triangles[3*i+2]
        v[3*0+0] = 0
        v[3*0+1] = 0
        v[3*0+2] = 0
        v[3*1+0] = verts[3*t0+0]
        v[3*1+1] = verts[3*t0+1]
        v[3*1+2] = verts[3*t0+2]
        v[3*2+0] = verts[3*t1+0]
        v[3*2+1] = verts[3*t1+1]
        v[3*2+2] = verts[3*t1+2]
        v[3*3+0] = verts[3*t2+0]
        v[3*3+1] = verts[3*t2+1]
        v[3*3+2] = verts[3*t2+2]

        offset = 50*i
        content[offset:offset+48] = v.view(np.uint8)
        content[offset+48:offset+50] = pad


def write(name, verts, triangles):
    """
    Save a triangulated raster to a standard STL file.
    Windows has a default STL viewer and probably all 3D viewers have native
    support for it because of its simplicity. Can be used to verify the
    correctness of the algorithm or to visualize the mesh to get a notion of
    the size/complexity etc.
    @param name - The name of the mesh file we're going to save.
                  Should end in .stl
    @param verts - A numpy array containing all the vertices of the mesh.
                   Format is 3 float32 per vertex (vertex buffer)
    @param triangles - A numpy array containing all the triangles of the mesh.
                       Format is 3 int32 per triangle (index buffer)
    """
    ib = triangles
    vb = verts
    if isinstance(ib, cupy.ndarray):
        ib = cupy.asnumpy(ib)
    if isinstance(vb, cupy.ndarray):
        vb = cupy.asnumpy(vb)

    header = np.zeros(80, np.uint8)
    nf = np.empty(1, np.uint32)
    num_tris = triangles.shape[0] // 3
    nf[0] = num_tris
    f = open(name, 'wb')
    f.write(header)
    f.write(nf)

    # size of 1 triangle in STL is 50 bytes
    # 12 floats (each 4 bytes) for a total of 48
    # And additional 2 bytes for padding
    content = np.empty(num_tris*(50), np.uint8)
    _fill_contents(content, vb, ib, num_tris)
    f.write(content)
    f.close()
