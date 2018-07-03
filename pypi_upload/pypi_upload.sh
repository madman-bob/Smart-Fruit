#!/usr/bin/env bash

# Build distribution
python3 pypi_upload/setup.py sdist bdist_wheel

# Upload distribution to PyPI
twine upload --config-file pypi_upload/.pypirc dist/*
# For testing, add `--repository testpypi`

# Tidy up
rm -rf build
rm -rf dist
rm -rf smart_fruit.egg-info
