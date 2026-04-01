..  _reference.classification:

**************
Classification
**************

.. warning::

   Classification functions silently set NaN and infinite input values
   to NaN in the output.  Clean infinities before classifying if you
   want every cell assigned to a bin.

Equal Interval
==============
.. autosummary::
    :toctree: _autosummary

    xrspatial.classify.equal_interval

Natural Breaks
==============
.. autosummary::
    :toctree: _autosummary

    xrspatial.classify.natural_breaks

Reclassify
==========
.. autosummary::
    :toctree: _autosummary

    xrspatial.classify.reclassify

Quantile
========
.. autosummary::
    :toctree: _autosummary

    xrspatial.classify.quantile

Binary
======
.. autosummary::
    :toctree: _autosummary

    xrspatial.classify.binary

Box Plot
========
.. autosummary::
    :toctree: _autosummary

    xrspatial.classify.box_plot

Head/Tail Breaks
================
.. autosummary::
    :toctree: _autosummary

    xrspatial.classify.head_tail_breaks

Maximum Breaks
==============
.. autosummary::
    :toctree: _autosummary

    xrspatial.classify.maximum_breaks

Percentiles
===========
.. autosummary::
    :toctree: _autosummary

    xrspatial.classify.percentiles

Standard Mean
=============
.. autosummary::
    :toctree: _autosummary

    xrspatial.classify.std_mean
