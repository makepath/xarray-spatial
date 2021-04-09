..  _index:

****************************
Xarray-spatial documentation
****************************

**Xarray-Spatial implements common raster analysis functions using Numba and provides an easy-to-install, easy-to-extend codebase for raster analysis.**

xarray-spatial grew out of the `Datashader project <https://datashader.org/>`_, which provides fast rasterization of vector data (points, lines, polygons, meshes, and rasters) for use with xarray-spatial.

xarray-spatial does not depend on GDAL / GEOS, which makes it fully extensible in Python but does limit the breadth of operations that can be covered.  xarray-spatial is meant to include the core raster-a

-------

.. raw:: html
   :file: _templates/description_band.html

-------

.. panels::
   :body: examples-item
   :container: container-fluid examples-container
   :column: examples-card

   ---
   .. image:: _static/img/0-0.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples
   ---
   .. image:: _static/img/0-1.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/2_Proximity.ipynb
   ---
   .. image:: _static/img/0-2.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/2_Proximity.ipynb
   ---
   .. image:: _static/img/0-3.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/2_Proximity.ipynb
   ---
   .. image:: _static/img/0-4.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/pharmacy-deserts.ipynb
   ---
   .. image:: _static/img/1-0.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/1_Surface.ipynb
   ---
   .. image:: _static/img/1-1.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/1_Surface.ipynb
   ---
   .. image:: _static/img/1-2.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/1_Surface.ipynb
   ---
   .. image:: _static/img/1-3.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/1_Surface.ipynb
   ---
   .. image:: _static/img/1-4.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/pharmacy-deserts.ipynb
   ---
   .. image:: _static/img/2-0.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/1_Surface.ipynb
   ---
   .. image:: _static/img/2-1.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/8_Remote_Sensing.ipynb
   ---
   .. image:: _static/img/2-2.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/8_Remote_Sensing.ipynb
   ---
   .. image:: _static/img/2-3.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/5_Classification.ipynb
   ---
   .. image:: _static/img/2-4.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/pharmacy-deserts.ipynb
   ---
   .. image:: _static/img/3-0.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples
   ---
   .. image:: _static/img/3-1.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples
   ---
   .. image:: _static/img/3-2.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/5_Classification.ipynb
   ---
   .. image:: _static/img/3-3.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/pharmacy-deserts.ipynb
   ---
   .. image:: _static/img/3-4.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples
   ---
   .. image:: _static/img/4-0.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/Path-finding_City-of-Austin-Road-Network.ipynb
   ---
   .. image:: _static/img/4-1.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/1_Surface.ipynb#Hillshade
   ---
   .. image:: _static/img/4-2.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/1_Surface.ipynb#Hillshade
   ---
   .. image:: _static/img/4-3.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/user_guide/1_Surface.ipynb#Slope
   ---
   .. image:: _static/img/4-4.png
      :target: https://github.com/makepath/xarray-spatial/blob/master/examples/pharmacy-deserts.ipynb#Create-a-%22Distance-to-Nearest-Pharmacy%22-Layer-&-Classify-into-5-Groups

.. toctree::
   :maxdepth: 3
   :hidden:
   :titlesonly:

   getting_started/index
   reference/index
