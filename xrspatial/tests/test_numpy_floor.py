"""The import-time NumPy floor check in xrspatial/__init__.py.

NumPy 1.x is not supported. Nothing in the dependency metadata stops a
user from ending up with numpy 1.26 next to a numba or pyarrow build
that rejects it (pyarrow 26 declares no numpy requirement at all and
raises at import), so the package checks once, up front, and says
plainly what it found.
"""
import pytest

from xrspatial import _check_numpy_version


@pytest.mark.parametrize("version", ["1.26.0", "1.26.4", "1.26.0.dev0+abc", "1.99.9"])
def test_numpy_1x_is_rejected(version):
    with pytest.raises(ImportError) as excinfo:
        _check_numpy_version(version)
    msg = str(excinfo.value)
    assert "NumPy 2.0 or newer" in msg
    assert version in msg


@pytest.mark.parametrize("version", ["2.0.0", "2.0.2", "2.5.4", "2.6.0.dev0+git", "3.0.0"])
def test_numpy_2_and_later_pass(version):
    _check_numpy_version(version)


def test_installed_numpy_passes_the_check():
    import numpy
    _check_numpy_version(numpy.__version__)
