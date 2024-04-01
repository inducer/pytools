#! /bin/bash

set -ex

mypy pytools

mypy --strict --follow-imports=skip pytools/datatable.py
