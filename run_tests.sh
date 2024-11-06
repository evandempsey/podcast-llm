#!/bin/bash
python -m pytest --cov --cov-report=xml

# Build documentation
echo "Building documentation..."
cd docs && make clean && make html
cd ..
