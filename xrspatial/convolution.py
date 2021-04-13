from numba import cuda, float32, prange, jit

from xrspatial.utils import has_cuda
from xrspatial.utils import cuda_args

import numpy as np
import xarray as xr


def convolve_2d(image: xr.DataArray,
                kernel,
                pad=True,
                use_cuda=True) -> xr.DataArray:
    """
    Calculates, for all inner cells of an array, the 2D convolution of
    each cell via Numba. To account for edge cells, a pad can be added
    to the image array. Convolution is frequently used for image
    processing, such as smoothing, sharpening, and edge detection of
    images by elimatig spurious data or enhancing features in the data.

    Parameters:
    ----------
    image: xarray.DataArray
        2D array of values to processed and padded.
    kernel: array-like object
        Impulse kernel, determines area to apply
        impulse function for each cell.
    pad: Boolean
        To compute edges set to True.
    use-cuda: Boolean
        For parallel computing set to True.

    Returns:
    ----------
    convolve_agg: xarray.DataArray
        2D array representation of the impulse function.
        All other input attributes are preserverd.

    Examples:
    ----------
    Imports
    >>> import numpy as np
    >>> import xarray as xr
    >>> from xrspatial import convolution, focal

    Create Data Array
    >>> agg = xr.DataArray(np.array([[0, 0, 0, 0, 0, 0, 0],
    >>>                              [0, 0, 2, 4, 0, 8, 0],
    >>>                              [0, 2, 2, 4, 6, 8, 0],
    >>>                              [0, 4, 4, 4, 6, 8, 0],
    >>>                              [0, 6, 6, 6, 6, 8, 0],
    >>>                              [0, 8, 8, 8, 8, 8, 0],
    >>>                              [0, 0, 0, 0, 0, 0, 0]]),
    >>>                     dims = ["lat", "lon"],
    >>>                     attrs = dict(res = 1))
    >>> height, width = agg.shape
    >>> _lon = np.linspace(0, width - 1, width)
    >>> _lat = np.linspace(0, height - 1, height)
    >>> agg["lon"] = _lon
    >>> agg["lat"] = _lat

        Create Kernel
    >>> kernel = focal.circle_kernel(1, 1, 1)

        Create Convolution Data Array
    >>> print(convolution.convolve_2d(agg, kernel))
    [[ 0.  0.  4.  8.  0. 16.  0.]
     [ 0.  4.  8. 10. 18. 16. 16.]
     [ 4.  8. 14. 20. 24. 30. 16.]
     [ 8. 16. 20. 24. 30. 30. 16.]
     [12. 24. 30. 30. 34. 30. 16.]
     [16. 22. 30. 30. 30. 24. 16.]
     [ 0. 16. 16. 16. 16. 16.  0.]]
    """
    # Don't allow padding on (1, 1) kernel
    if (kernel.shape[0] == 1 and kernel.shape[1] == 1):
        pad = False

    if pad:
        pad_rows = kernel.shape[0] // 2
        pad_cols = kernel.shape[1] // 2
        pad_width = ((pad_rows, pad_rows),
                     (pad_cols, pad_cols))
    else:
        # If padding is not desired, set pads to 0
        pad_rows = 0
        pad_cols = 0
        pad_width = 0

    padded_image = np.pad(image, pad_width=pad_width, mode="reflect")
    result = np.empty_like(padded_image)

    if has_cuda() and use_cuda:
        griddim, blockdim = cuda_args(padded_image.shape)
        _convolve_2d_cuda[griddim, blockdim](result, kernel, padded_image)
    else:
        result = _convolve_2d(kernel, padded_image)

    if pad:
        result = result[pad_rows:-pad_rows, pad_cols:-pad_cols]

    if result.shape != image.shape:
        raise ValueError("Output and input rasters are not the same shape.")

    return result


@jit(nopython=True, nogil=True, parallel=True)
def _convolve_2d(kernel, image):
    """Apply kernel to image."""

    nx = image.shape[0]
    ny = image.shape[1]
    nkx = kernel.shape[0]
    nky = kernel.shape[1]
    wkx = nkx // 2
    wky = nky // 2

    result = np.zeros(image.shape, dtype=float32)

    for i in prange(0, nx, 1):
        iimin = max(i - wkx, 0)
        iimax = min(i + wkx + 1, nx)
        for j in prange(0, ny, 1):
            jjmin = max(j - wky, 0)
            jjmax = min(j + wky + 1, ny)
            num = 0.0
            for ii in range(iimin, iimax, 1):
                iii = wkx + ii - i
                for jj in range(jjmin, jjmax, 1):
                    jjj = wky + jj - j
                    num += kernel[iii, jjj] * image[ii, jj]
            result[i, j] = num

    return result


# https://www.vincent-lunot.com/post/an-introduction-to-cuda-in-python-part-3/
@cuda.jit
def _convolve_2d_cuda(result, kernel, image):
    # expects a 2D grid and 2D blocks,
    # a kernel with odd numbers of rows and columns, (-1-)
    # a grayscale image

    # (-2-) 2D coordinates of the current thread:
    i, j = cuda.grid(2)

    # (-3-) if the thread coordinates are outside of the image,
    # we ignore the thread:
    image_rows, image_cols = image.shape
    if (i >= image_rows) or (j >= image_cols):
        return

    # To compute the result at coordinates (i, j), we need to
    # use delta_rows rows of the image before and after the
    # i_th row, as well as delta_cols columns of the image
    # before and after the j_th column:
    delta_rows = kernel.shape[0] // 2
    delta_cols = kernel.shape[1] // 2

    # The result at coordinates (i, j) is equal to
    # sum_{k, l} kernel[k, l] * image[i - k + delta_rows, j - l + delta_cols]
    # with k and l going through the whole kernel array:
    s = 0
    for k in range(kernel.shape[0]):
        for l in range(kernel.shape[1]): # noqa
            i_k = i - k + delta_rows
            j_l = j - l + delta_cols
            # (-4-) Check if (i_k, j_k)
            # coordinates are inside the image:
            if (i_k >= 0) and (i_k < image_rows) and \
                    (j_l >= 0) and (j_l < image_cols):
                s += kernel[k, l] * image[i_k, j_l]
    result[i, j] = s
