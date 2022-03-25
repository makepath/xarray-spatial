from ..utils import has_cuda_and_cupy

try:
    from rtxpy import RTX
except ImportError:
    RTX = None


def has_rtx():
    return has_cuda_and_cupy and RTX is not None
