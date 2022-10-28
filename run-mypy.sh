#!/usr/bin/env bash
# from: https://jaredkhan.com/blog/mypy-pre-commit#step-2-creating-our-own-pre-commit-hook
# to use local mypy instead of pre-commit's mypy

# set -o errexit

# Change directory to the project root directory.
cd "$(dirname "$0")"

# Because I'm using namespace packages,
# I have used --package acme rather than using 
# the path 'src/acme', which would correctly
# collect my files but erroneously add 
# 'src/acme' to the Mypy search path.
# We only want 'src' in the path so that Mypy
# knows our modules by their fully qualified names.
mypy main.py --namespace-packages
mypy autorl_landscape --namespace-packages
mypy tests --namespace-packages
