#! /bin/bash

set -ex

mypy --show-error-codes pytools

mypy --strict --follow-imports=skip pytools/datatable.py pytools/graph.py
