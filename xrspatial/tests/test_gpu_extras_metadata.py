"""Guards on the GPU extras declared in setup.cfg (issue #3699).

``pip install xarray-spatial[gpu]`` used to fail on every machine. The extra
named two packages that pip cannot install:

* ``cuspatial`` -- the PyPI project of that name is a 2020 name-holder whose
  ``setup.py`` raises "Please install cuspatial via the rapidsai conda
  channel". Real cuSpatial is conda/RAPIDS only, or ``cuspatial-cu12``.
* ``cupy`` -- sdist-only at every release, so pip compiles CuPy from source
  and needs a full nvcc toolchain. The wheels live under the CUDA-suffixed
  names ``cupy-cuda12x`` / ``cupy-cuda13x``.

The workflow job ``gpu-extras-resolve`` catches a regression by actually
resolving the extras against PyPI. These tests are the offline half: they read
the declared requirements and never touch the network.
"""
import configparser
import re
from pathlib import Path

import pytest

import xrspatial

SETUP_CFG = Path(xrspatial.__file__).resolve().parent.parent / "setup.cfg"

pytestmark = pytest.mark.skipif(
    not SETUP_CFG.is_file(),
    reason="setup.cfg is only on disk for a source checkout / editable install",
)


def _extras():
    """Map extra name -> list of requirement strings, comments stripped."""
    # interpolation=None: a stray `%` anywhere in setup.cfg would otherwise
    # raise InterpolationSyntaxError and fail these tests for an unrelated
    # reason.
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(SETUP_CFG)
    out = {}
    for name, block in cfg["options.extras_require"].items():
        reqs = []
        for line in block.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                reqs.append(line)
        out[name] = reqs
    return out


def _project_name(req):
    """Leading project name of a requirement string, PEP 503 normalized.

    Normalizing means `cupy_cuda12x` and `cupy.cuda12x` compare equal to
    `cupy-cuda12x`, the way pip treats them.
    """
    name = re.split(r"[\s<>=!~;\[]", req, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def test_no_extra_declares_cuspatial():
    """cuspatial is not pip-installable and nothing in xrspatial imports it."""
    offenders = {
        name: reqs for name, reqs in _extras().items()
        if any(_project_name(r) == "cuspatial" for r in reqs)
    }
    assert offenders == {}, (
        "cuspatial on PyPI is a name-holder whose install raises; use the "
        f"rapidsai conda channel instead. Declared in: {sorted(offenders)}"
    )


def test_no_extra_declares_bare_cupy():
    """Bare `cupy` is sdist-only, so pip would build CuPy from source."""
    offenders = {
        name: reqs for name, reqs in _extras().items()
        if any(_project_name(r) == "cupy" for r in reqs)
    }
    assert offenders == {}, (
        "the `cupy` project on PyPI ships no wheels; name a CUDA-suffixed "
        f"wheel such as cupy-cuda12x instead. Declared in: {sorted(offenders)}"
    )


@pytest.mark.parametrize(
    "extra, project, floor",
    [
        ("gpu", "cupy-cuda12x", "12.3"),
        ("gpu-cuda13", "cupy-cuda13x", "13.6"),
    ],
)
def test_gpu_extra_pins_a_cuda_wheel_with_a_floor(extra, project, floor):
    """Each GPU extra names one prebuilt cupy wheel and declares a floor.

    The floors come from PyPI release metadata: setup.cfg sets
    ``python_requires = >=3.12``, cupy-cuda12x 12.3.0 is the first release
    with cp312 wheels, and cupy-cuda13x starts at 13.6.0.
    """
    reqs = _extras()[extra]
    matching = [r for r in reqs if _project_name(r) == project]
    assert matching, f"the `{extra}` extra must declare {project}, got {reqs}"
    assert f">={floor}" in matching[0].replace(" ", ""), (
        f"the `{extra}` extra must floor {project} at {floor}, got "
        f"{matching[0]!r}"
    )


def test_rasterize_does_not_import_cuspatial():
    """The dead `cuspatial` import in rasterize.py stays gone.

    ``xrspatial.rasterize`` the name resolves to the function, so reach the
    module through sys.modules.
    """
    import sys

    import xrspatial.rasterize  # noqa: F401

    module = sys.modules["xrspatial.rasterize"]
    assert not hasattr(module, "cuspatial")
    assert "cuspatial" not in Path(module.__file__).read_text()
