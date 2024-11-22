#!/bin/bash

# Ensure Sphinx is installed
pip install sphinx sphinx-autobuild

# Create Sphinx project
sphinx-quickstart docs --quiet --sep --project "Portfolio Optimization" \
  --author "Moein" --release "1.0" \
  --extensions "sphinx.ext.autodoc,sphinx.ext.napoleon,sphinx.ext.viewcode"


# Update conf.py to include project path
echo "import os, sys" >> docs/source/conf.py
echo "sys.path.insert(0, os.path.abspath('../'))" >> docs/source/conf.py

# Generate RST files
sphinx-apidoc -o docs/source/ portfawn

# Build HTML documentation
make -C docs html
