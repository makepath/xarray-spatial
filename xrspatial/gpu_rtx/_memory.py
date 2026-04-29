# Memory guards shared by the gpu_rtx subpackage.
#
# The OptiX/RTX entry points allocate several cupy device buffers sized by the
# input raster shape -- a triangulated mesh (verts + triangles), per-pixel ray
# buffers, hit buffers, and an output buffer.  We bound those allocations with
# a single up-front check so the user gets a clean MemoryError that names the
# raster shape rather than an opaque cupy allocator failure deep inside trace().
#
# Pattern mirrors cost_distance._available_gpu_memory_bytes / _check_gpu_memory.

# Worst-case bytes/pixel across the gpu_rtx entry points:
#   create_triangulation: verts (H*W*3 float32 = 12 B/px)
#                       + triangles (~6 int32/px = 24 B/px)
#   hillshade_rtx       : d_rays  (H*W*8 float32 = 32 B/px)
#                       + d_hits  (H*W*4 float32 = 16 B/px)
#                       + d_aux   (H*W*3 float32 = 12 B/px)
#                       + d_output(H*W   float32 =  4 B/px)
#                       = 64 B/px on top of mesh -> 100 B/px total
#   viewshed_gpu        : d_rays  (32 B/px) + d_hits (16 B/px)
#                       + d_visgrid (4 B/px) + d_vsrays (32 B/px)
#                       = 84 B/px on top of mesh -> 120 B/px total
#
# We use 120 B/px as a single budget covering the worst case.  The 50%-of-free
# threshold leaves headroom for the OptiX BVH that the RTX backend builds
# internally from the verts/triangles buffers.
_GPU_BYTES_PER_PIXEL = 120


def _available_gpu_memory_bytes():
    """Best-effort estimate of free GPU memory in bytes.

    Returns 0 if CuPy / CUDA is unavailable or the query fails -- callers
    treat that as a sentinel meaning "no GPU info, skip the guard".
    """
    try:
        import cupy as _cp
        free, _total = _cp.cuda.runtime.memGetInfo()
        return int(free)
    except Exception:
        return 0


def _check_gpu_memory(func_name, height, width):
    """Raise MemoryError if a gpu_rtx call would exceed 50% of free GPU RAM.

    Skips the check (returns silently) when ``_available_gpu_memory_bytes``
    cannot determine the free memory -- e.g. when cupy isn't installed, where
    the call will fail at the cupy.empty boundary with a clearer error anyway.
    """
    available = _available_gpu_memory_bytes()
    if available <= 0:
        return
    required = int(height) * int(width) * _GPU_BYTES_PER_PIXEL
    if required > 0.5 * available:
        raise MemoryError(
            f"{func_name} on a {height}x{width} raster requires "
            f"~{required / 1e9:.1f} GB of GPU working memory but only "
            f"~{available / 1e9:.1f} GB is free on the active device.  "
            f"Use a smaller raster or fall back to the CPU path."
        )
