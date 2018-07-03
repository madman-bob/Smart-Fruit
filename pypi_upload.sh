#!/usr/bin/env bash

# Build distribution
python3 setup.py sdist bdist_wheel

# Upload distribution to PyPI
twine upload --config-file .pypirc dist/*
# For testing, add `--repository testpypi`

# Tidy up
rm -rf build
rm -rf dist
rm -rf smart_fruit.egg-info
