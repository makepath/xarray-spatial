import os
import sys
import param


version_before = param.version.get_setup_version(
    os.path.dirname(__file__),
    'xarray-spatial',
    pkgname='xrspatial',
    archive_commit="$Format:%h$")

version_digits = None

try:
    version_digits_test = version_before.split('.post')[1]
    version_digits = version_before.split('.post')[0]
except IndexError:
    version_digits = version_before.rsplit('+')[0]

if version_digits is None:
    sys.exit('version digits not found; '
             'check your git tags with cmd: git tag -l')

digits_list = version_digits.split('.')

if len(digits_list) < 3:
    sys.exit('less than 3 digits in version')
elif len(digits_list) == 4:
    last_digit = digits_list[-1]
    first_digits = '.'.join(digits_list[0], digits_list[1], digits_list[2])
elif len(digits_list) == 3:
    last_digit = '0'
    first_digits = version_digits

next_version = '.'.join([first_digits, str(int(last_digit) + 1)])

if (len(next_version.split('.')) < 4 or
        len(next_version.rsplit('.', 1)[0].split('.')) != 3):
    sys.exit(f"next version not formatted correctly {next_version}")

print(f'v{next_version}')
