#!/bin/bash

set -e
set -x

BLD_DIR='pwd'
DIRTY=1
SRC_DIR=$RECIPE_DIR/..
pushd $SRC_DIR

$PYTHON setup.py --quiet install --single-version-externally-managed --record=record.txt
cp -r $SRC_DIR/examples $PREFIX/share/xarray-spatial-examples

popd
