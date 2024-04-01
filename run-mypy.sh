#! /bin/bash

set -ex

mypy pytools

mypy --strict pytools/datatable.py
