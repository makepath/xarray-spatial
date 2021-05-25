import os
import sys
import shutil
from setuptools import setup


# build dependencies
import pyct.build
import param

# dependencies

# datashader first, then pyct unless pyct version compatible with ds
# is specified
# spatialpandas may not be required in final pharmacy_desert version
# pyct may not be required after pyctdev is released
install_requires = [
    'dask',
    'datashader',
    'numba',
    'numpy',
    'pandas',
    'pillow',
    'requests',
    'scipy',
    'xarray',
    'pyct <=0.4.6',
    'param >=1.6.1',
    'distributed >=2021.03.0',
    'spatialpandas',
]

examples = [
]

# Additional tests dependencies and examples_extra may be needed in the future
extras_require = {
    'tests': [
        'pytest',
        'noise >=1.2.2',
    ],
    'examples': examples,
}

# additional doc dependencies may be needed
extras_require['doc'] = extras_require['examples'] + ['numpydoc']

extras_require['all'] = sorted(set(sum(extras_require.values(), [])))

version = param.version.get_setup_version(__file__, 'xarray-spatial',
                                          pkgname='xrspatial',
                                          archive_commit="$Format:%h$")

if 'sdist' in sys.argv and 'bdist_wheel' in sys.argv:
    try:
        version_test = version.split('.post')[1]
        version = version.split('.post')[0]
    except IndexError:
        version = version.split('+')[0]
    if version is None:
        sys.exit('invalid version')


# metadata for setuptools

setup_args = dict(
    name='xarray-spatial',
    version=version,
    description='xarray-based spatial analysis tools',
    install_requires=install_requires,
    extras_require=extras_require,
    tests_require=extras_require['tests'],
    zip_safe=False,
    classifiers=['Programming Language :: Python :: 3',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: OS Independent'],
    packages=['xrspatial',
              'xrspatial.tests'
              ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'xrspatial = xrspatial.__main__:main'
        ]
    },
)


if __name__ == '__main__':
    example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'xrspatial', 'examples')
    if 'develop' not in sys.argv:
        pyct.build.examples(example_path, __file__, force=True)
    setup(**setup_args)

    if os.path.isdir(example_path):
        shutil.rmtree(example_path)
