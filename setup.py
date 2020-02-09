from setuptools import find_packages, setup

import versioneer

setup(name='xarray-spatial',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='xarray-based spatial analysis tools',
      packages=find_packages(),
      install_requires=['xarray', 'numba', 'pytest'],
      zip_safe=False,
      include_package_data=True)
