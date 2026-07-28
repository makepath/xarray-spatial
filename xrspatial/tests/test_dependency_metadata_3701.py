"""Guards on the dependency metadata declared in setup.cfg (issue #3701).

These read setup.cfg from the source checkout rather than the installed
distribution metadata, so they report on the working tree instead of on
whatever was last `pip install`ed.
"""
import configparser
from functools import lru_cache
from pathlib import Path

import pytest
from packaging.requirements import Requirement

SETUP_CFG = Path(__file__).resolve().parents[2] / "setup.cfg"

pytestmark = pytest.mark.skipif(
    not SETUP_CFG.exists(), reason="setup.cfg is only present in a source checkout")


@lru_cache(maxsize=1)
def _config():
    parser = configparser.ConfigParser()
    parser.read(SETUP_CFG)
    return parser


def _parse(section, key):
    raw = _config().get(section, key)
    out = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        req = Requirement(line)
        out[req.name] = req.specifier
    return out


def _install_requires():
    return _parse("options", "install_requires")


def _extra(name):
    return _parse("options.extras_require", name)


def test_pandas_is_declared():
    """pandas is imported at module scope in zonal, dasymetric and focal.

    It also leaks into the public API: zonal_stats and crosstab return
    pandas DataFrames. Relying on xarray to drag it in transitively is
    how it went undeclared for so long.
    """
    assert "pandas" in _install_requires()


def test_zstandard_is_optional():
    """zstandard belongs to the geotiff extra, not install_requires.

    xrspatial/geotiff/_compression.py guards the import behind
    ZSTD_AVAILABLE and the codec functions raise a pip-install hint when
    it is missing, so the package imports fine without it.
    """
    assert "zstandard" not in _install_requires()
    assert "zstandard" in _extra("geotiff")
    # Keeps the ZSTD codec tests from silently skipping in CI.
    assert "zstandard" in _extra("tests")


def test_every_required_dependency_has_a_lower_bound():
    for name, spec in _install_requires().items():
        assert any(s.operator in (">=", "==", "~=") for s in spec), \
            f"{name} is declared without a lower bound"


def test_numpy_has_an_upper_bound():
    """numba caps numpy and re-checks the cap at import time.

    numba._ensure_critical_deps raises ImportError on an unsupported
    numpy, which takes `import xrspatial` down with it. A bare `numpy`
    here lets pip install exactly that combination, so the declaration
    needs a ceiling of its own. The value tracks numba and is reviewed
    when numba widens its own bound; this only checks one is present.
    """
    spec = _install_requires()["numpy"]
    assert any(s.operator in ("<", "<=") for s in spec), \
        "numpy is declared without an upper bound; see issue #3701"


def test_installed_numpy_satisfies_both_declarations():
    """The environment actually running the suite must be self-consistent.

    Catches a floor pin (the min-deps job) or a fresh resolve that lands
    on a numpy which either setup.cfg or numba rejects.
    """
    numpy = pytest.importorskip("numpy")
    import importlib.metadata as md

    ours = _install_requires()["numpy"]
    assert ours.contains(numpy.__version__, prereleases=True), \
        f"installed numpy {numpy.__version__} is outside setup.cfg's numpy{ours}"

    for dist in md.distributions():
        if dist.metadata.get("Name", "").lower() != "numba":
            continue
        for raw in dist.requires or []:
            req = Requirement(raw)
            if req.name == "numpy":
                assert req.specifier.contains(numpy.__version__, prereleases=True), \
                    (f"installed numpy {numpy.__version__} is outside numba's "
                     f"numpy{req.specifier}")
