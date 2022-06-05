import cupy
import numba as nb
import numpy as np


def create_triangulation(raster, optix):
    datahash = np.uint64(hash(str(raster.data.get())))
    optixhash = np.uint64(optix.getHash())

    # Calculate a scale factor for the height that maintains the ratio
    # width/height
    H, W = raster.shape

    # Scale the terrain so that the width is proportional to the height
    # Thus the terrain would be neither too flat nor too steep and
    # raytracing will give best accuracy
    maxH = float(cupy.amax(raster.data))
    maxDim = max(H, W)
    scale = maxDim / maxH

    if optixhash != datahash:
        num_tris = (H - 1) * (W - 1) * 2
        verts = cupy.empty(H * W * 3, np.float32)
        triangles = cupy.empty(num_tris * 3, np.int32)
        # Generate a mesh from the terrain (buffers are on the GPU, so
        # generation happens also on GPU)
        res = _triangulate_terrain(verts, triangles, raster, scale)
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
    return scale


@nb.cuda.jit
def _triangulate_terrain_kernel(verts, triangles, data, H, W, scale, stride):
    global_id = stride + nb.cuda.grid(1)
    if global_id < W*H:
        h = global_id // W
        w = global_id % W
        mesh_map_index = h * W + w

        val = data[h, w]

        offset = 3*mesh_map_index
        verts[offset] = w
        verts[offset+1] = h
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
def _triangulate_cpu(verts, triangles, data, H, W, scale):
    for h in nb.prange(H):
        for w in range(W):
            mesh_map_index = h * W + w

            val = data[h, w]

            offset = 3*mesh_map_index
            verts[offset] = w
            verts[offset+1] = h
            verts[offset+2] = val * scale

            if w != W - 1 and h != H - 1:
                offset = 6*(h*(W-1) + w)
                triangles[offset+0] = np.int32(mesh_map_index + W)
                triangles[offset+1] = np.int32(mesh_map_index + W+1)
                triangles[offset+2] = np.int32(mesh_map_index)
                triangles[offset+3] = np.int32(mesh_map_index + W+1)
                triangles[offset+4] = np.int32(mesh_map_index + 1)
                triangles[offset+5] = np.int32(mesh_map_index)


def _triangulate_terrain(verts, triangles, terrain, scale=1):
    H, W = terrain.shape
    if isinstance(terrain.data, np.ndarray):
        _triangulate_cpu(verts, triangles, terrain.data, H, W, scale)
    if isinstance(terrain.data, cupy.ndarray):
        job_size = H*W
        blockdim = 1024
        griddim = (job_size + blockdim - 1) // 1024
        d = 100
        offset = 0
        while job_size > 0:
            batch = min(d, griddim)
            _triangulate_terrain_kernel[batch, blockdim](
                verts, triangles, terrain.data, H, W, scale, offset)
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
