### Release Prep
- [ ] Update CHANGELOG.md
- [ ] Increment Version in `xrspatial/__init__.py`
- [ ] Increment Version in `conda.recipe/meta.yaml`
- [ ] Commit changes and push

### Tag Release
- [ ] Create New Tag
```bash
git tag -a v0.0.1 -m "Version 0.0.1"
git push --tags
git checkout v0.0.1
```

### Build / Upload package for pypi
- [ ] Build Pip Package
```bash
python setup.py sdist bdist_wheel
```

- [ ] Upload Pip Package
```bash
python -m twine upload dist/*
```

- [ ] Build Conda Packages
```bash
conda build conda.recipe --python 3.6 -c conda-forge
conda build conda.recipe --python 3.7 -c conda-forge
conda build conda.recipe --python 3.8 -c conda-forge
```

- [ ] Create Packages for different platforms
```bash
VERSION=0.0.6
cd /Users/<user>/miniconda3/conda-bld/
cd osx-64
conda convert --platform win-64 xarray-spatial-$VERSION*.tar.bz2 -o ../
conda convert --platform linux-64 xarray-spatial-$VERSION*.tar.bz2 -o ../
```

### Build upload to Anaconda.org
```bash
anaconda upload xarray-spatial-$VERSION*.tar.bz2
anaconda upload ../linux-64/xarray-spatial-$VERSION*.tar.bz2
anaconda upload ../win-64/xarray-spatial-$VERSION*.tar.bz2
```
