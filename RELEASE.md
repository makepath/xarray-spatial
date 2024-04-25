## Release process

### Preparation
- Create a new branch containing the following changes:
  - Update CHANGELOG.md with new version number and list of changes extracted from `git log --pretty=oneline --pretty=format:"- %s" <lastest_release_tag>..HEAD`.
- Commit changes and submit them as a PR to the `master` branch.
- If the CI passes OK, merge the PR.

### Tag release
- To sign the release you need a GPG key registered with your github account. See https://docs.github.com/en/authentication/managing-commit-signature-verification
- Create new tag, with the correct version number, using:
```bash
git tag -a v0.1.2 -s -m "Version 0.1.2"
git push --tags
```

### PyPI packages
- These are automatically built and uploaded to PyPI via a github action when a new tag is pushed to the github repo.
- Check that both an sdist (`.tar.gz` file) and wheel (`.whl` file) are available on PyPI.
- Check you can install the new version in a new virtual environment using `pip install xarray-spatial`.

### github release notes
- Convert the tag into a release on github:
  - On the right-hand side of the github repo, click on `Releases`.
  - Click on `Draft a new release`.
  - Select the correct tag, and enter the title and description by copying and pasting from the CHANGELOG.md.
  - Click `Publish release`.

### Documentation

- When the github release is created, a github action automatically builds the documentation and uploads it to https://xarray-spatial.readthedocs.io/.

### conda-forge packages
- A bot in https://github.com/conda-forge/xarray-spatial-feedstock runs periodically to identify the new PyPI release and update the conda recipe appropriately. This should create a new PR, run tests to check that the conda build works, and automatically upload the packages to conda-forge if everything is OK. Check this works, a few hours after the PyPI release.
