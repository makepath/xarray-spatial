import numpy as np
import xarray as xr

import time
import argparse
from xrspatial.zonal import stats
from xrspatial.utils import has_cuda
from xrspatial.utils import doesnt_have_cuda


parser = argparse.ArgumentParser(description='Simple zonal-stats use example, to be used a benchmarking template.',
                                 usage='python test-zonal.py -width width -height height -i iteration')

parser.add_argument('-width', '--width', type=int, default=100,
                    help='Canvas width.')

parser.add_argument('-height', '--height', type=int, default=100,
                    help='Canvas height.')

parser.add_argument('-zh', '--zone-height', type=int, default=2,
                    help='Zones Height.')

parser.add_argument('-zw', '--zone-width', type=int, default=2,
                    help='Zones width.')

parser.add_argument('-i', '--iterations', type=int, default=1,
                    help='Number of times to repeat the function call.')

parser.add_argument('-b', '--backend', type=str, choices=['numpy', 'cupy', 'dask'],
                    default='numpy',
                    help='Computational backend to use.')

parser.add_argument('-p', '--profile', action='store_true',
                    help='Turns on the profiling mode.')


def create_arr(data=None, H=10, W=10, backend='numpy'):
    assert(backend in ['numpy', 'cupy', 'dask'])
    if data is None:
        data = np.zeros((H, W), dtype=np.float32)
    raster = xr.DataArray(data, dims=['y', 'x'])

    if has_cuda() and 'cupy' in backend:
        import cupy
        raster.data = cupy.asarray(raster.data)

    if 'dask' in backend:
        import dask.array as da
        raster.data = da.from_array(raster.data, chunks=(10, 10))

    return raster


if __name__ == '__main__':
    args = parser.parse_args()

    W = args.width
    H = args.height
    zH = args.zone_height
    zW = args.zone_width
    assert(W/zW == W//zW)
    assert(H/zH == H//zH)

    # Values raster
    values = xr.DataArray(np.arange(H * W).reshape(H, W))
    values = create_arr(values, backend=args.backend)
    # Zones raster
    zones = xr.DataArray(np.zeros(H * W).reshape(H, W))
    hstep = H//zH
    wstep = W//zW
    for i in range(zH):
        for j in range(zW):
            zones[i * hstep: (i+1)*hstep, j*wstep: (j+1)*wstep] = i*zW + j

    zones = create_arr(zones, backend=args.backend)
    # Profiling region
    start = time.time()
    stats_df = stats(zones=zones, values=values)
    warm_up_sec = time.time() - start

    if args.profile:
        from pyprof import timing
        timing.mode = 'timing'
        timing.reset()

    elapsed_sec = 0
    for i in range(args.iterations):
        start = time.time()
        stats_df = stats(zones=zones, values=values)
        # if args.backend == 'dask':
        #     stats_df = stats_df.compute()
        elapsed_sec += time.time() - start
    print('HxW,Runs,total_time(sec),time_per_run(sec),warm_up_time(sec)')
    print('{}x{},{},{:.4f},{:.4f},{:.4f}'.format(
        H, W,
        args.iterations, elapsed_sec,
        elapsed_sec/args.iterations, warm_up_sec))
    print('Result: ', stats_df)

    if args.profile:
        from datetime import datetime
        now = datetime.now()
        now = now.strftime("%H:%M:%S-%d-%m-%Y")
        timing.report()
        timing.report(out_dir='./',
                      out_file=f'timing-{args.backend}-h{H}-w{W}-zh{zH}-zw{zW}-i{args.iterations}-{now}.csv')
    # timing.report()

    #     from dask.distributed import Client
    #     import dask.array as da
    #     client = Client(processes=False, threads_per_worker=1,
    #                     n_workers=4, memory_limit='2GB')
    #     data = xr.DataArray(da.zeros((H, W), dtype=np.float32, chunks=2048),
    #                         name='dask_numpy_terrain')
    #     data.persist()

    # if args.backend is 'dask':
    #     client.close()
