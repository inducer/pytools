#! /bin/bash

set -ex

mypy --show-error-codes pytools

mypy --strict pytools/datatable.py pytools/graph.py pytools/mpi.py pytools/__init__.py
