# Contributing to Xarray-Spatial

As stated in [Xarray Spatial code of conduct](https://github.com/makepath/xarray-spatial/blob/master/CODE_OF_CONDUCT.md), a primary goal of Xarray Spatial is to be inclusive to the largest number of contributors. However, we do have some requests for how contributions should be made. Please read these guidelines before contributing to have a most positive experience with Xarray Spatial.

### Getting Started

Information about installation and setting up a development environment can be found at the [Getting Started page] https://xarray-spatial.org/getting_started/index.html.

### Choosing something to work on 

The issue tracker has a list of items that you can start working on. 
In order to avoid duplication of effort, it's always a good idea to comment on an issue and let everybody know that you intend to work on it.

### Opening a new issue

1. Avoid duplicate reports. Search GitHub for similar or identical issues. Keyword searches for your error messages are usually effective.

2. The issue may already be resolved. Always try to reproduce the issue in the latest stable release.

3. Always include a minimal, self-contained, reproducible test case or example. It is not possible to investigate issues that cannot be reproduced.

4. Include relevant system information.

5. State the expected behavior.


### Creating a pull request (PR)

1. Make sure that there is a corresponding issue for your change first. If there isn't yet, create one.

2. Create a fork of the Xarray Spatial repository on GitHub (this is only done before *first*) contribution).

3. Create a branch off the `master` branch with a meaningful name. Preferably include issue number and a few keywords, so that we will have a rough idea what the branch refers to, without looking up the issue. 

4. Commit your changes and push them to GitHub.

5. Create a pull request against the default base branch. The PR must have a meaningful title and a message explaining what was achieved, what remains to be done, maybe an example, etc.

6. Use the Create draft pull request option as you first open a pull request, so everyone knows it's a work in progress. Once you finish the work on the pull request you can convert it to Ready for review. In addition to this, please use the labels WIP and ready to merge.

7. We don't accept code contributions without tests. If there are valid reasons for not including a test, please discuss this in the issue.

8. We will review your PR as time permits. Reviewers may comment on your contributions, ask you questions regarding the implementation or request changes. If changes are requested, push new commits to the existing branch. Do *NOT* rebase, amend, or cherry-pick published commits. Any of those actions will make us start the review from scratch. If you need updates from `master`, just merge it into your branch.


### Attribution

Portions of text derived from [Bokeh CONTRIBUTING file]: (https://github.com/bokeh/bokeh/blob/branch-2.4/.github/CONTRIBUTING.md)
