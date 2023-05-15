#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates.

# Run this script at project root by "./dev/linter.sh" before you commit

set -v

echo "Running isort ..."
isort --atomic .

echo "Running black ..."
black -l 180 .

echo "Running flake8 ..."
if [ -x "$(command -v flake8-3)" ]; then
  flake8-3 .
else
  python3 -m flake8 .
fi
