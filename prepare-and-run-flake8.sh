#! /bin/bash

curl -L -O -k https://gitlab.tiker.net/inducer/ci-support/raw/master/ci-support.sh
source ci-support.sh

print_status_message
clean_up_repo_and_working_env
create_and_set_up_virtualenv
install_and_run_flake8 "$@"
