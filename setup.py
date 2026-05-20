import os
import shutil
import sys

import pyct.build
from setuptools import setup


if __name__ == '__main__':
    _package_dir = os.path.dirname(os.path.abspath(__file__))
    example_path = os.path.join(_package_dir, 'xrspatial', 'examples')
    _is_develop = any(arg == 'develop' or arg.endswith('develop') for arg in sys.argv)
    if not _is_develop:
        pyct.build.examples(example_path, __file__, force=True)

    use_scm = {
        "write_to": "xrspatial/_version.py"
    }

    setup(use_scm_version=use_scm)

    if os.path.isdir(example_path):
        _real_example_path = os.path.realpath(example_path)
        _real_package_dir = os.path.realpath(_package_dir)
        if _real_example_path.startswith(_real_package_dir + os.sep):
            shutil.rmtree(example_path)
