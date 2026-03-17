"""Hydrology analysis modules for xarray-spatial.

Includes flow direction, flow accumulation, flow length, flow path,
watershed delineation, basin labeling, HAND, stream ordering, and
related utilities for D8, D-infinity, and MFD routing.

Each function family provides a unified wrapper that accepts a
``routing`` parameter ('d8', 'dinf', or 'mfd') and dispatches to
the corresponding implementation.  The suffixed variants
(e.g. ``flow_direction_d8``) are also importable directly.
"""

# -- concrete D8 implementations ------------------------------------------
from xrspatial.hydro.basin_d8 import basin_d8  # noqa
from xrspatial.hydro.fill_d8 import fill_d8  # noqa
from xrspatial.hydro.flow_accumulation_d8 import flow_accumulation_d8  # noqa
from xrspatial.hydro.flow_direction_d8 import flow_direction_d8  # noqa
from xrspatial.hydro.flow_length_d8 import flow_length_d8  # noqa
from xrspatial.hydro.flow_path_d8 import flow_path_d8  # noqa
from xrspatial.hydro.hand_d8 import hand_d8  # noqa
from xrspatial.hydro.sink_d8 import sink_d8  # noqa
from xrspatial.hydro.snap_pour_point_d8 import snap_pour_point_d8  # noqa
from xrspatial.hydro.stream_link_d8 import stream_link_d8  # noqa
from xrspatial.hydro.stream_order_d8 import stream_order_d8  # noqa
from xrspatial.hydro.twi_d8 import twi_d8  # noqa
from xrspatial.hydro.watershed_d8 import basins_d8  # noqa
from xrspatial.hydro.watershed_d8 import watershed_d8  # noqa

# -- concrete D-infinity implementations -----------------------------------
from xrspatial.hydro.flow_accumulation_dinf import flow_accumulation_dinf  # noqa
from xrspatial.hydro.flow_direction_dinf import flow_direction_dinf  # noqa
from xrspatial.hydro.flow_length_dinf import flow_length_dinf  # noqa
from xrspatial.hydro.flow_path_dinf import flow_path_dinf  # noqa
from xrspatial.hydro.hand_dinf import hand_dinf  # noqa
from xrspatial.hydro.stream_link_dinf import stream_link_dinf  # noqa
from xrspatial.hydro.stream_order_dinf import stream_order_dinf  # noqa
from xrspatial.hydro.watershed_dinf import watershed_dinf  # noqa

# -- concrete MFD implementations -----------------------------------------
from xrspatial.hydro.flow_accumulation_mfd import flow_accumulation_mfd  # noqa
from xrspatial.hydro.flow_direction_mfd import flow_direction_mfd  # noqa
from xrspatial.hydro.flow_length_mfd import flow_length_mfd  # noqa
from xrspatial.hydro.flow_path_mfd import flow_path_mfd  # noqa
from xrspatial.hydro.hand_mfd import hand_mfd  # noqa
from xrspatial.hydro.stream_link_mfd import stream_link_mfd  # noqa
from xrspatial.hydro.stream_order_mfd import stream_order_mfd  # noqa
from xrspatial.hydro.watershed_mfd import watershed_mfd  # noqa


# =========================================================================
# Routing dispatch
# =========================================================================

class _RoutingDispatch:
    """Map routing algorithm names to concrete implementations.

    Inspired by ArrayTypeFunctionMapping but keyed on a string
    (``'d8'``, ``'dinf'``, ``'mfd'``) rather than array type.
    """

    __slots__ = ('_name', '_impls')

    def __init__(self, name, **impls):
        self._name = name
        self._impls = impls

    def __call__(self, *args, routing='d8', **kwargs):
        try:
            fn = self._impls[routing]
        except KeyError:
            opts = ', '.join(repr(k) for k in self._impls)
            raise ValueError(
                f"Unknown routing {routing!r} for {self._name}; "
                f"expected one of {opts}"
            ) from None
        return fn(*args, **kwargs)


# -- 8 families with d8 / dinf / mfd variants ----------------------------

flow_direction = _RoutingDispatch(
    'flow_direction',
    d8=flow_direction_d8, dinf=flow_direction_dinf, mfd=flow_direction_mfd,
)

flow_accumulation = _RoutingDispatch(
    'flow_accumulation',
    d8=flow_accumulation_d8, dinf=flow_accumulation_dinf,
    mfd=flow_accumulation_mfd,
)

flow_length = _RoutingDispatch(
    'flow_length',
    d8=flow_length_d8, dinf=flow_length_dinf, mfd=flow_length_mfd,
)

flow_path = _RoutingDispatch(
    'flow_path',
    d8=flow_path_d8, dinf=flow_path_dinf, mfd=flow_path_mfd,
)

watershed = _RoutingDispatch(
    'watershed',
    d8=watershed_d8, dinf=watershed_dinf, mfd=watershed_mfd,
)

hand = _RoutingDispatch(
    'hand',
    d8=hand_d8, dinf=hand_dinf, mfd=hand_mfd,
)

stream_link = _RoutingDispatch(
    'stream_link',
    d8=stream_link_d8, dinf=stream_link_dinf, mfd=stream_link_mfd,
)


# stream_order needs special handling: the ordering param (strahler/shreve)
# is called `ordering` in d8 and `method` in dinf/mfd.

class _StreamOrderDispatch(_RoutingDispatch):
    def __call__(self, *args, routing='d8', ordering='strahler', **kwargs):
        try:
            fn = self._impls[routing]
        except KeyError:
            opts = ', '.join(repr(k) for k in self._impls)
            raise ValueError(
                f"Unknown routing {routing!r} for {self._name}; "
                f"expected one of {opts}"
            ) from None
        if routing == 'd8':
            return fn(*args, ordering=ordering, **kwargs)
        return fn(*args, method=ordering, **kwargs)


stream_order = _StreamOrderDispatch(
    'stream_order',
    d8=stream_order_d8, dinf=stream_order_dinf, mfd=stream_order_mfd,
)


# -- 5 D8-only functions (future-proofed with routing param) --------------

basin = _RoutingDispatch('basin', d8=basin_d8)
basins = _RoutingDispatch('basins', d8=basins_d8)
sink = _RoutingDispatch('sink', d8=sink_d8)
snap_pour_point = _RoutingDispatch('snap_pour_point', d8=snap_pour_point_d8)
fill = _RoutingDispatch('fill', d8=fill_d8)
twi = _RoutingDispatch('twi', d8=twi_d8)
