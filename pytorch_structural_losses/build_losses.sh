#!/bin/bash

readonly PYTHON_VERSION=$(python -c 'import sys; print("%d.%d" % (sys.version_info[0], sys.version_info[1]))')

cd pytorch_structural_losses || exit
rm StructuralLossesBackend.cpython* || true
rm -rf build || true

python setup.py build

cp `find build/lib* -name "StructuralLossesBackend*"` .
