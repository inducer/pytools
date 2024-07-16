#! /bin/bash

set -ex

mypy --show-error-codes pytools

mypy --strict --follow-imports=silent \
    pytools/tag.py \
    pytools/datatable.py \
    pytools/persistent_dict.py
