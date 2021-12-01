from ..utils import has_cuda, has_cupy


try:
    from rtxpy import RTX
except ImportError:
    RTX = None


def has_rtx():
    return has_cupy() and has_cuda() and RTX is not None
