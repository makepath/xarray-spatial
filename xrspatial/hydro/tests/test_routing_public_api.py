"""Public hydrology API surface: routing wrappers only (#3528).

The three routing flavors (d8, dinf, mfd) are selected with the ``routing``
keyword on a single public wrapper per family. The suffixed implementations
(``flow_direction_d8`` etc.) are internal to ``xrspatial.hydro`` and are not
exported at the top level.
"""

import numpy as np
import pytest

import xrspatial
from xrspatial import hydro
from xrspatial.tests.general_checks import create_test_raster

# One public wrapper per family.
WRAPPERS = [
    'flow_direction', 'flow_accumulation', 'flow_length', 'flow_path',
    'watershed', 'hand', 'stream_link', 'stream_order',
    'basin', 'basins', 'sink', 'snap_pour_point', 'fill', 'twi',
]

# Representative suffixed implementations across every family.
SUFFIXED = [
    'flow_direction_d8', 'flow_direction_dinf', 'flow_direction_mfd',
    'flow_accumulation_d8', 'flow_accumulation_dinf', 'flow_accumulation_mfd',
    'flow_length_d8', 'flow_path_mfd', 'watershed_dinf', 'hand_mfd',
    'stream_link_d8', 'stream_order_dinf',
    'basin_d8', 'basins_d8', 'sink_d8', 'snap_pour_point_d8', 'fill_d8',
    'twi_d8',
]


def _ramp():
    """A plane z = i + j so dinf/mfd produce real (non-pit) flow."""
    data = np.add.outer(np.arange(6.0), np.arange(6.0))
    return create_test_raster(data)


@pytest.mark.parametrize('name', WRAPPERS)
def test_wrapper_is_public(name):
    assert hasattr(xrspatial, name)
    # the top-level name is the same object exported from xrspatial.hydro
    assert getattr(xrspatial, name) is getattr(hydro, name)


@pytest.mark.parametrize('name', SUFFIXED)
def test_suffixed_not_public_top_level(name):
    assert not hasattr(xrspatial, name)


@pytest.mark.parametrize('name', SUFFIXED)
def test_suffixed_importable_from_hydro(name):
    # still reachable internally, where the wrappers dispatch to them
    assert hasattr(hydro, name)


@pytest.mark.parametrize('routing', ['d8', 'dinf', 'mfd'])
def test_dispatch_matches_direct_impl(routing):
    agg = _ramp()
    direct = getattr(hydro, f'flow_direction_{routing}')
    expected = direct(agg)
    result = xrspatial.flow_direction(agg, routing=routing)
    np.testing.assert_array_equal(
        np.asarray(result.data), np.asarray(expected.data)
    )


def test_default_routing_is_d8():
    agg = _ramp()
    default = xrspatial.flow_direction(agg)
    d8 = xrspatial.flow_direction(agg, routing='d8')
    np.testing.assert_array_equal(
        np.asarray(default.data), np.asarray(d8.data)
    )


def test_unknown_routing_raises():
    agg = _ramp()
    with pytest.raises(ValueError, match="Unknown routing"):
        xrspatial.flow_direction(agg, routing='bogus')


def test_stream_order_threads_ordering():
    # the d8 dispatch must forward `ordering` to stream_order_d8
    agg = _ramp()
    fdir = xrspatial.flow_direction(agg, routing='d8')
    accum = xrspatial.flow_accumulation(fdir, routing='d8')
    via_wrapper = xrspatial.stream_order(
        fdir, accum, threshold=1, ordering='shreve'
    )
    direct = hydro.stream_order_d8(fdir, accum, threshold=1, ordering='shreve')
    np.testing.assert_array_equal(
        np.asarray(via_wrapper.data), np.asarray(direct.data)
    )


def test_wrapper_is_self_describing():
    assert xrspatial.flow_direction.__name__ == 'flow_direction'
    assert 'routing' in xrspatial.flow_direction.__doc__
