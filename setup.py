import os
import shutil
import sys

import param
import pyct.build
from setuptools import setup

version = param.version.get_setup_version(
    __file__,
    'xarray-spatial',
    pkgname='xrspatial',
    archive_commit="$Format:%h$",
)

if 'sdist' in sys.argv and 'bdist_wheel' in sys.argv:
    try:
        version_test = version.split('.post')[1]
        version = version.split('.post')[0]
    except IndexError:
        version = version.split('+')[0]
    if version is None:
        sys.exit('invalid version')


if __name__ == '__main__':
    example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'xrspatial', 'examples')
    if 'develop' not in sys.argv:
        pyct.build.examples(example_path, __file__, force=True)
    setup(version=version)

    if os.path.isdir(example_path):
        shutil.rmtree(example_path)
