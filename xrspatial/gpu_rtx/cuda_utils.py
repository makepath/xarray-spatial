import numba as nb
import numpy as np


@nb.cuda.jit(device=True)
def add(a, b):
    return float3(a[0]+b[0], a[1]+b[1], a[2]+b[2])


@nb.cuda.jit(device=True)
def diff(a, b):
    return float3(a[0]-b[0], a[1]-b[1], a[2]-b[2])


@nb.cuda.jit(device=True)
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@nb.cuda.jit(device=True)
def float3(a, b, c):
    return (np.float32(a), np.float32(b), np.float32(c))


@nb.cuda.jit(device=True)
def invert(a):
    return float3(-a[0], -a[1], -a[2])


@nb.cuda.jit(device=True)
def mix(a, b, k):
    return add(mul(a, k), mul(b, 1-k))


@nb.cuda.jit(device=True)
def make_float3(a, offset):
    return float3(a[offset], a[offset+1], a[offset+2])


@nb.cuda.jit(device=True)
def mul(a, b):
    return float3(a[0]*b, a[1]*b, a[2]*b)


@nb.cuda.jit(device=True)
def mult_color(a, b):
    return float3(a[0]*b[0], a[1]*b[1], a[2]*b[2])
