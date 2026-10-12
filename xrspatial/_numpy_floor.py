"""Import-time NumPy floor. Imported first by xrspatial/__init__.py.

numba and pyarrow both reject numpy 1.x in their own ways (numba at
import, pyarrow 26 at import with no metadata pin that pip could act
on), which surfaces deep inside a submodule import as an unrelated
error. Checking here, before any submodule loads, turns that into one
plain sentence that names the version found.
"""
import numpy as _np


def _check_numpy_version(version):
    """Raise ImportError if ``version`` is a NumPy 1.x release string."""
    try:
        major = int(version.split(".")[0])
    except ValueError:
        return
    if major < 2:
        raise ImportError(
            f"xarray-spatial requires NumPy 2.0 or newer, found {version}. "
            "Upgrade with: pip install 'numpy>=2'"
        )


_check_numpy_version(_np.__version__)
