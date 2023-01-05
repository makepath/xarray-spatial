import os
import shutil
import sys

import pyct.build
from setuptools import setup


if __name__ == '__main__':
    example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'xrspatial', 'examples')
    if 'develop' not in sys.argv:
        pyct.build.examples(example_path, __file__, force=True)

    use_scm = {
        "write_to": "xrspatial/_version.py"
    }

    setup(use_scm_version=use_scm)

    if os.path.isdir(example_path):
        shutil.rmtree(example_path)
