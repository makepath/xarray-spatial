..  _reference.hydrology:

*********
Hydrology
*********

.. warning::

   NaN cells act as **impassable barriers** in all hydrology functions.
   Flow cannot cross them.  If your DEM has NaN holes (e.g. water bodies
   masked out), fill or interpolate them first, or expect disconnected
   drainage networks.

Each family exposes a single public function. The routing algorithm
(``'d8'``, ``'dinf'``, or ``'mfd'``) is chosen with the ``routing`` keyword,
which defaults to ``'d8'``::

    import xrspatial
    fdir = xrspatial.flow_direction(dem, routing='dinf')
    acc = xrspatial.flow_accumulation(fdir, routing='dinf')

The wrapper dispatches to the per-routing implementations listed under each
family. Those implementations live in ``xrspatial.hydro`` and carry the full
parameter list and algorithm references.

Flow Direction
==============
.. py:function:: xrspatial.flow_direction(agg, *, routing='d8', **kwargs)

   Direction of steepest descent out of each cell.  ``routing`` selects
   ``'d8'``, ``'dinf'``, or ``'mfd'`` (default ``'d8'``).

Routing variants:

.. autosummary::
    :toctree: _autosummary

    xrspatial.hydro.flow_direction_d8.flow_direction_d8
    xrspatial.hydro.flow_direction_dinf.flow_direction_dinf
    xrspatial.hydro.flow_direction_mfd.flow_direction_mfd

Flow Accumulation
=================
.. py:function:: xrspatial.flow_accumulation(flow_dir, *, routing='d8', **kwargs)

   Upstream cells or area draining through each cell.  ``routing`` selects
   ``'d8'``, ``'dinf'``, or ``'mfd'`` (default ``'d8'``).

Routing variants:

.. autosummary::
    :toctree: _autosummary

    xrspatial.hydro.flow_accumulation_d8.flow_accumulation_d8
    xrspatial.hydro.flow_accumulation_dinf.flow_accumulation_dinf
    xrspatial.hydro.flow_accumulation_mfd.flow_accumulation_mfd

Flow Length
===========
.. py:function:: xrspatial.flow_length(flow_dir, *, routing='d8', **kwargs)

   Distance along the flow path to the outlet or from the divide.
   ``routing`` selects ``'d8'``, ``'dinf'``, or ``'mfd'`` (default ``'d8'``).

Routing variants:

.. autosummary::
    :toctree: _autosummary

    xrspatial.hydro.flow_length_d8.flow_length_d8
    xrspatial.hydro.flow_length_dinf.flow_length_dinf
    xrspatial.hydro.flow_length_mfd.flow_length_mfd

Flow Path
=========
.. py:function:: xrspatial.flow_path(flow_dir, start_points, *, routing='d8', **kwargs)

   Trace downstream flow paths from a set of start points.  ``routing``
   selects ``'d8'``, ``'dinf'``, or ``'mfd'`` (default ``'d8'``).

Routing variants:

.. autosummary::
    :toctree: _autosummary

    xrspatial.hydro.flow_path_d8.flow_path_d8
    xrspatial.hydro.flow_path_dinf.flow_path_dinf
    xrspatial.hydro.flow_path_mfd.flow_path_mfd

Watershed
=========
.. py:function:: xrspatial.watershed(flow_dir, pour_points, *, routing='d8', **kwargs)

   Label each cell with the pour point it drains to.  ``routing`` selects
   ``'d8'``, ``'dinf'``, or ``'mfd'`` (default ``'d8'``).

Routing variants:

.. autosummary::
    :toctree: _autosummary

    xrspatial.hydro.watershed_d8.watershed_d8
    xrspatial.hydro.watershed_dinf.watershed_dinf
    xrspatial.hydro.watershed_mfd.watershed_mfd
    xrspatial.hydro.watershed_d8.basins_d8

Stream Link
===========
.. py:function:: xrspatial.stream_link(flow_dir, flow_accum, *, routing='d8', threshold=100, **kwargs)

   Assign unique IDs to stream segments above a flow-accumulation threshold.
   ``routing`` selects ``'d8'``, ``'dinf'``, or ``'mfd'`` (default ``'d8'``).

Routing variants:

.. autosummary::
    :toctree: _autosummary

    xrspatial.hydro.stream_link_d8.stream_link_d8
    xrspatial.hydro.stream_link_dinf.stream_link_dinf
    xrspatial.hydro.stream_link_mfd.stream_link_mfd

Stream Order
============
.. py:function:: xrspatial.stream_order(flow_dir, flow_accum, *, routing='d8', ordering='strahler', threshold=100, **kwargs)

   Strahler or Shreve stream ordering of the stream network.  ``routing``
   selects ``'d8'``, ``'dinf'``, or ``'mfd'`` (default ``'d8'``); ``ordering``
   selects ``'strahler'`` or ``'shreve'``.

Routing variants:

.. autosummary::
    :toctree: _autosummary

    xrspatial.hydro.stream_order_d8.stream_order_d8
    xrspatial.hydro.stream_order_dinf.stream_order_dinf
    xrspatial.hydro.stream_order_mfd.stream_order_mfd

Height Above Nearest Drainage (HAND)
====================================
.. py:function:: xrspatial.hand(flow_dir, flow_accum, elevation, *, routing='d8', threshold=100, **kwargs)

   Height above the nearest drainage.  ``routing`` selects ``'d8'``,
   ``'dinf'``, or ``'mfd'`` (default ``'d8'``).

Routing variants:

.. autosummary::
    :toctree: _autosummary

    xrspatial.hydro.hand_d8.hand_d8
    xrspatial.hydro.hand_dinf.hand_dinf
    xrspatial.hydro.hand_mfd.hand_mfd

D8-only functions
=================

These families currently implement D8 routing only. They take the same
``routing`` keyword for forward compatibility, where ``'d8'`` is the only
accepted value.

.. autosummary::
    :toctree: _autosummary

    xrspatial.hydro.fill_d8.fill_d8
    xrspatial.hydro.sink_d8.sink_d8
    xrspatial.hydro.basin_d8.basin_d8
    xrspatial.hydro.snap_pour_point_d8.snap_pour_point_d8
    xrspatial.hydro.twi_d8.twi_d8
