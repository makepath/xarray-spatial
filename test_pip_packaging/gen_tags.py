import os
import param


version_before = param.version.get_setup_version(
    os.path.dirname(__file__),
    'xarray-spatial',
    pkgname='xrspatial',
    archive_commit="$Format:%h$")

try:
    version_tag = '.'.join((version_before.split('.post')[0],
                            str(int(version_before.split('.post')[1]
                                    .split('+')[0]) + 1)))
except IndexError:
    version_tag = '.'.join((version_before.split('+')[0].rsplit('.', 1)[0],
                            str(int(version_before.split('+')[0]
                                    .rsplit('.', 1)[1]) + 1)))
print(f'{version_before}')
