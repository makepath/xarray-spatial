from setuptools import setup


setup(name='xarray-spatial',
      use_scm_version={
        "write_to": "xrspatial/_version.py",
        "write_to_template": '__version__ = "{version}"',
        "tag_regex": r"^(?P<prefix>v)?(?P<version>[^\+]+)(?P<suffix>.*)?$",
      },
      description='xarray-based spatial analysis tools',
      packages=['xrspatial',
                'xrspatial.tests'],
      install_requires=['datashader'],
      zip_safe=False,
      classifiers=["Programming Language :: Python :: 3",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent"],
      include_package_data=True)
