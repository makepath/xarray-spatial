### Tag Release
```bash
git tag -a v0.0.1 -m "Version 0.0.1"
git push --tags
git checkout v0.0.1
```

### Build / Upload package for pypi
```bash
python setup.py sdist bdist_wheel
python -m twine upload dist/*
```

### Build upload to Anaconda.org
```bash
conda build conda.recipe --python 3.6 -c conda-forge
conda build conda.recipe --python 3.7 -c conda-forge
conda build conda.recipe --python 3.8 -c conda-forge
```
