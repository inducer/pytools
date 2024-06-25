#! /bin/bash

set -ex

mypy --show-error-codes pytools

mypy --strict --follow-imports=silent pytools/datatable.py pytools/persistent_dict.py
