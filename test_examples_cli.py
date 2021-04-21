import sys
from xrspatial import __main__ as m
from unittest.mock import patch


# test_args include copy-examples, fetch-data, or examples (does both)
@pytest.mark.skip(reason="meant only for internal use")
def run_examples_cmds(*cli_cmds):
    """
    Run conda package cli commands to download examples and fetch data
    for notebooks.

    Parameters: 'copy-examples', 'fetch-data', 'examples'
    Returns: downloads examples and data to new xrspatial-examples
    directory in xarray-spatialx
    """

    for arg in cli_cmds:
        with patch.object(sys, 'argv', ['xrspatial', arg]):
            m.main()


run_examples_cmds('examples')
