#! /bin/bash

set -ex

mypy pytools

mypy --strict --follow-imports=silent \
    pytools/tag.py \
    pytools/graph.py \
    pytools/datatable.py \
    pytools/persistent_dict.py
