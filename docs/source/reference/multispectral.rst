..  _reference.multispectral:

*************
Multispectral
*************

.. note::

   All spectral indices output **float32**.  Division by zero (e.g.
   NDVI where NIR + Red = 0) produces NaN or inf silently.  Clean
   the result with ``xr.where(np.isfinite(result), result, np.nan)``
   if needed.

Atmospherically Resistant Vegetation Index (ARVI)
=================================================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.arvi

Burn Area Index (BAI)
=====================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.bai

Enhanced Built-Up and Bareness Index (EBBI)
============================================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.ebbi

Enhanced Vegetation Index (EVI)
===============================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.evi

Green Chlorophyll Index (GCI)
=============================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.gci

Modified Soil Adjusted Vegetation Index (MSAVI2)
=================================================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.msavi2

Normalized Burn Ratio (NBR)
===========================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.nbr

Normalized Burn Ratio 2 (NBR2)
==============================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.nbr2

Normalized Difference Built-up Index (NDBI)
===========================================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.ndbi

Normalized Difference Moisture Index (NDMI)
===========================================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.ndmi

Normalized Difference Snow Index (NDSI)
=======================================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.ndsi

Normalized Difference Water Index (NDWI)
========================================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.ndwi

Modified Normalized Difference Water Index (MNDWI)
==================================================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.mndwi

Normalized Difference Vegetation Index (NDVI)
=============================================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.ndvi

Optimized Soil Adjusted Vegetation Index (OSAVI)
=================================================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.osavi

Soil Adjusted Vegetation Index (SAVI)
=====================================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.savi

Structure Insensitive Pigment Index (SIPI)
==========================================
.. autosummary::
    :toctree: _autosummary

    xrspatial.multispectral.sipi
    xrspatial.multispectral.true_color
