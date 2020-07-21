from setuptools import setup

import versioneer

setup(name='xarray-spatial',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='xarray-based spatial analysis tools',
      packages=['xrspatial',
                'xrspatial.tests'],
      install_requires=['datashader', 'pytest'],
      zip_safe=False,
      classifiers=["Programming Language :: Python :: 3",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent"],
      include_package_data=True)
