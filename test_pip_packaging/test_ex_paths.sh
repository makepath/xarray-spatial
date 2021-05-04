#!/bin/bash

# generates new package version, builds pip package,
# uploads to pypi, creates new env and new dir,
# installs xarray-spatial, runs examples, and ls
# to see if correct files are in folder

# clear previous build detritus
rm -rf build/
rm -rf dist/
rm -rf xarray_spatial_*
rm -f xrspatial/.version


# change name of package


# add tag
version="$(python test_pip_packaging/gen_tags.py)"
echo "$version"
git tag -a "$version" -m new
git push origin "$version"


# clear previous tag
#last_version="$(git describe --abbrev=0 --tags "$(git rev-list --tags --skip=1 --max-count=1)")"
#echo "$last_version"
#git tag -d "$last_version"
#git push --delete origin "$last_version"


# run sdist and upload twine
python setup.py sdist bdist_wheel
twine upload --username="$username" --password="$password" dist/*


# test out in new env
# conda deactivate
# cd ..
# mkdir test_cli || cd .
# cd test_cli || exit
# rm -rf xrspatial-examples
# conda deactivate
# conda create -n test_cli -y
# conda activate test_cli
# conda install -c anaconda pip -y
# version_arr=(${version//v/ })
# version_num=${version_arr[1]}
# sleep 10s
# pip install xarray-spatial=="$version_num"
# xrspatial examples
# pip uninstall xarray-spatial -y
# ls xrspatial-examples
# conda deactivate
# cd ../xarray-spatial || exit
