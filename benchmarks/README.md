Benchmarking
============

`xarray-spatial` uses ASV (https://asv.readthedocs.io) for benchmarking.

Installing ASV
--------------

ASV creates virtualenvs to run benchmarks in. Before using it you need to

```
pip install asv virtualenv
```
or the `conda` equivalent.

Running benchmarks
------------------

ASV configuration information is stored in `benchmarks/asv.conf.json`. This includes a `matrix` section that lists the dependencies to install in the virtual environments in addition to those installed by default. You always need `pyct` as `setup.py` uses it. There are also some other optional dependencies that are commented out in the `matrix` section.

If you want to benchmark `cupy`-backed `DataArray`s and have the hardware and drivers to support this then uncomment the `cupy-cuda101` line in `asv.conf.json` and change the `101` version number part of this to match the version of your CUDA setup. This can by determined by the last line of the output of `nvcc --version`.

If you want to benchmark algorithms that use the ray-tracing code in `rtxpy`, then uncomment the `rtxpy` line in `asv.conf.json` as well as the `cupy` line.

To run all benchmarks against the default `master` branch:
```
cd benchmarks
asv run
```

The first time this is run it will create a machine file to store information about your machine. Then a virtual environment will be created and each benchmark will be run multiple times to obtain a statistically valid benchmark time.

To list the benchmark timings stored for the `master` branch use:
```
asv show master
```

ASV ships with its own simple webserver to interactively display the results in a webbrowser. To use this:
```
asv publish
asv preview
```
and then open a web browser at the URL specified.

If you want to quickly run all benchmarks once only to check for errors, etc, use:
```
asv dev
```
instead of `asv run`.


Adding new benchmarks
---------------------

Add new benchmarks to existing or new classes in the `benchmarks/benchmarks` directory. Any class member function with a name that starts with `time` will be identified as a timing benchmark when `asv` is run.

Data that is required to run benchmarks is usually created in the `setup()` member function. This ensures that the time taken to setup the data is not included in the benchmark time. The `setup()` function is called once for each invocation of each benchmark, the data are not cached.

At the top of each benchmark class there are lists of parameter names and values. Each benchmark is repeated for each unique combination of these parameters.

If you wish to benchmark `cupy` and/or `rtxpy` functionality, ensure that you test for the availability of the correct libraries and hardware first. This is illustrated in the `get_xr_dataarray()` function.

If you only want to run a subset of benchmarks, use syntax like:
```
asv run -b Slope
```
where the text after the `-b` flag is used as a regex to match benchmark file, class and function names.


Benchmarking code changes
-------------------------

You can compare the performance of code on different branches and in different commits. Usually if you want to determine how much faster a new algorithm is, the old code will be in the `master` branch and the new code will be in a new feature branch. Because ASV uses virtual environments and checks out the `xarray-spatial` source code into these virtual environments, your new code must be committed into the new feature branch.

To benchmark the latest commits on `master` and your new feature branch, edit `asv.conf.json` to change the line
```
"branches": ["master"],
```
into
```
"branches": ["master", "new_feature_branch"],
```
or similar.

Now when you `asv run` the benchmarks will be run against both branches in turn.

Then use
```
asv show
```
to list the commits that have been benchmarked, and
```
asv compare commit1 commit2
```
to give you a side-by-side comparison of the two commits.
