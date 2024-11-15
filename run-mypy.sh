#! /bin/bash

set -ex

python -m mypy pytools

python -m mypy --strict --follow-imports=silent \
    pytools/datatable.py \
    pytools/graph.py \
    pytools/persistent_dict.py \
    pytools/prefork.py \
    pytools/tag.py \
