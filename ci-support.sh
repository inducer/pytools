#! /bin/bash
# ^^ (not a script: only here for shellcheck's benefit)

set -e
set -o pipefail

ci_support="https://gitlab.tiker.net/inducer/ci-support/raw/master"

if [ "$PY_EXE" == "" ]; then
  if [ "$py_version" == "" ]; then
    PY_EXE=python3
  else
    PY_EXE=python${py_version}
  fi
fi


if [ "$(uname)" = "Darwin" ]; then
  PLATFORM=MacOSX
else
  PLATFORM=Linux
fi

if test "$CI_SERVER_NAME" = "GitLab" && test -d ~/.local/lib; then
  echo "ERROR: $HOME/.local/lib exists. It really shouldn't. Here's what it contains:"
  find ~/.local/lib
  exit 1
fi


# {{{ utilities

function with_echo()
{
  echo "+++" "$@"
  "$@"
}

function get_proj_name()
{
  if [ -n "$CI_PROJECT_NAME" ]; then
    echo "$CI_PROJECT_NAME"
  else
    basename "$GITHUB_REPOSITORY"
  fi
}

print_status_message()
{
  echo "-----------------------------------------------"
  echo "Current directory: $(pwd)"
  echo "Python executable: ${PY_EXE}"
  echo "PYOPENCL_TEST: ${PYOPENCL_TEST}"
  echo "PYTEST_ADDOPTS: ${PYTEST_ADDOPTS}"
  echo "PROJECT_INSTALL_FLAGS: ${PROJECT_INSTALL_FLAGS}"
  echo "git revision: $(git rev-parse --short HEAD)"
  echo "git status:"
  git status -s
  echo "-----------------------------------------------"
}


create_and_set_up_virtualenv()
{
  ${PY_EXE} -m venv .env
  . .env/bin/activate

  # https://github.com/pypa/pip/issues/5345#issuecomment-386443351
  export XDG_CACHE_HOME=$HOME/.cache/$CI_RUNNER_ID

  if [[ "${PY_EXE}" == pypy3* ]]; then
    PY_EXE=pypy3-c
  fi

  RESOLVED_PY_EXE=$(which ${PY_EXE})
  case "$RESOLVED_PY_EXE" in
    $PWD/.env/*) ;;
    *)
      echo "Python executable $PY_EXE not in virtualenv"
      exit 1
      ;;
  esac


  # https://github.com/pypa/pip/issues/8667 -AK, 2020-08-02
  $PY_EXE -m pip install --upgrade "pip<20.2"
  $PY_EXE -m pip install setuptools
}


install_miniforge()
{
  MINIFORGE_VERSION=3
  MINIFORGE_INSTALL_DIR=.miniforge${MINIFORGE_VERSION}

  MINIFORGE_INSTALL_SH=Miniforge3-$PLATFORM-x86_64.sh
  curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/$MINIFORGE_INSTALL_SH"

  rm -Rf "$MINIFORGE_INSTALL_DIR"

  bash "$MINIFORGE_INSTALL_SH" -b -p "$MINIFORGE_INSTALL_DIR"
}


handle_extra_install()
{
  if test "$EXTRA_INSTALL" != ""; then
    for i in $EXTRA_INSTALL ; do
      if [[ "$i" = *pybind11* ]] && [[ "${PY_EXE}" == pypy* ]]; then
         # context:
         # https://github.com/conda-forge/pyopencl-feedstock/pull/45
         # https://github.com/pybind/pybind11/pull/2146
         with_echo "$PY_EXE" -m pip install git+https://github.com/isuruf/pybind11@pypy3
      else
        with_echo "$PY_EXE" -m pip install "$i"
      fi
    done
  fi
}


pip_install_project()
{
  handle_extra_install

  if test "$REQUIREMENTS_TXT" == ""; then
    REQUIREMENTS_TXT="requirements.txt"
  fi

  if test -f "$REQUIREMENTS_TXT"; then
    with_echo pip install -r "$REQUIREMENTS_TXT"
  fi

  if test -f .conda-ci-build-configure.sh; then
    with_echo source .conda-ci-build-configure.sh
  fi

  if test -f .ci-build-configure.sh; then
    with_echo source .ci-build-configure.sh
  fi

  # Append --editable to PROJECT_INSTALL_FLAGS, if not there already.
  # See: https://gitlab.tiker.net/inducer/ci-support/-/issues/3
  # Can be removed after https://github.com/pypa/pip/issues/2195 is resolved.
  if [[ ! $PROJECT_INSTALL_FLAGS =~ (^|[[:space:]]*)(--editable|-e)[[:space:]]*$ ]]; then
      PROJECT_INSTALL_FLAGS="$PROJECT_INSTALL_FLAGS --editable"
  fi

  with_echo "$PY_EXE" -m pip install $PROJECT_INSTALL_FLAGS .
}


# }}}


# {{{ cleanup

clean_up_repo_and_working_env()
{
  rm -Rf .env
  rm -Rf build
  find . -name '*.pyc' -delete

  rm -Rf env
  git clean -fdx \
    -e siteconf.py \
    -e boost-numeric-bindings \
    -e '.pylintrc.yml' \
    -e 'prepare-and-run-*.sh' \
    -e 'ci-support.sh' \
    -e 'run-*.py' \
    -e '.test-*.yml' \
    $GIT_CLEAN_EXCLUDE


  if test `find "siteconf.py" -mmin +1`; then
    echo "siteconf.py older than a minute, assumed stale, deleted"
    rm -f siteconf.py
  fi

  if [[ "$NO_SUBMODULES" = "" ]]; then
    git submodule update --init --recursive
  fi
}

# }}}


# {{{ virtualenv build

build_py_project_in_venv()
{
  print_status_message
  clean_up_repo_and_working_env
  create_and_set_up_virtualenv

  pip_install_project
}

# }}}


# {{{ miniconda build

build_py_project_in_conda_env()
{
  print_status_message
  clean_up_repo_and_working_env
  install_miniforge

  PATH="$MINIFORGE_INSTALL_DIR/bin/:$PATH" with_echo conda update conda --yes --quiet

  PATH="$MINIFORGE_INSTALL_DIR/bin/:$PATH" with_echo conda update --all --yes --quiet

  PATH="$MINIFORGE_INSTALL_DIR/bin:$PATH" with_echo conda env create --file "$CONDA_ENVIRONMENT" --name testing

  source "$MINIFORGE_INSTALL_DIR/bin/activate" testing

  # https://github.com/conda-forge/ocl-icd-feedstock/issues/11#issuecomment-456270634
  rm -f $MINIFORGE_INSTALL_DIR/envs/testing/etc/OpenCL/vendors/system-*.icd
  # https://gitlab.tiker.net/inducer/pytential/issues/112
  rm -f $MINIFORGE_INSTALL_DIR/envs/testing/etc/OpenCL/vendors/apple.icd

  # https://github.com/pypa/pip/issues/5345#issuecomment-386443351
  export XDG_CACHE_HOME=$HOME/.cache/$CI_RUNNER_ID

  # https://github.com/pypa/pip/issues/8667 -AK, 2020-08-02
  with_echo conda install --quiet --yes "pip<20.2"
  with_echo conda list

  # Using pip instead of conda to install pytest (see test_py_project) avoids
  # ridiculous uninstall chains like these:
  # https://gitlab.tiker.net/inducer/pyopencl/-/jobs/61543

  pip_install_project
}

# }}}


# {{{ generic build

build_py_project()
{
  if test "$USE_CONDA_BUILD" == "1"; then
    build_py_project_in_conda_env
  else
    build_py_project_in_venv
  fi
}

# }}}


# {{{ test

test_py_project()
{
  $PY_EXE -m pip install pytest

  # pytest-xdist fails on pypy with: ImportError: cannot import name '_psutil_linux'
  # AK, 2020-08-20
  if [[ "${PY_EXE}" == pypy* ]]; then
    CISUPPORT_PARALLEL_PYTEST=no
  else
    $PY_EXE -m pip install pytest-xdist
  fi

  AK_PROJ_NAME="$(get_proj_name)"

  TESTABLES=""
  if [ -d test ]; then
    cd test

    if ! [ -f .not-actually-ci-tests ]; then
      TESTABLES="$TESTABLES ."
    fi

    if [ -z "$NO_DOCTESTS" ]; then
      RST_FILES=(../doc/*.rst)

      for f in "${RST_FILES[@]}"; do
        if [ -e "$f}" ]; then
          if ! grep -q no-doctest "$f"; then
            TESTABLES="$TESTABLES $f"
          fi
        fi
      done

      # macOS bash is too old for mapfile: Oh well, no doctests on mac.
      if [ "$(uname)" != "Darwin" ]; then
        mapfile -t DOCTEST_MODULES < <( git grep -l doctest -- ":(glob,top)$AK_PROJ_NAME/**/*.py" )
        TESTABLES="$TESTABLES ${DOCTEST_MODULES[*]}"
      fi
    fi

    if [[ -n "$TESTABLES" ]]; then
      # Core dumps? Sure, take them.
      ulimit -c unlimited

      # 10 GiB should be enough for just about anyone :)
      ulimit -m "$(python -c 'print(1024*1024*10)')"

     if [[ $CISUPPORT_PARALLEL_PYTEST == "" || $CISUPPORT_PARALLEL_PYTEST == "xdist" ]]; then
       # Default: parallel if Not (Gitlab and GPU CI)?
       PYTEST_PARALLEL_FLAGS=""

       # CI_RUNNER_DESCRIPTION is set by Gitlab
       if [[ $CI_RUNNER_DESCRIPTION != *-gpu ]]; then
         if [[ $CISUPPORT_PYTEST_NRUNNERS == "" ]]; then
           PYTEST_PARALLEL_FLAGS="-n 4"
         else
           PYTEST_PARALLEL_FLAGS="-n $CISUPPORT_PYTEST_NRUNNERS"
         fi
       fi

     elif [[ $CISUPPORT_PARALLEL_PYTEST == "no" ]]; then
         PYTEST_PARALLEL_FLAGS=""
     else
       echo "unrecognized scheme in CISUPPORT_PARALLEL_PYTEST"
     fi

     # It... somehow... (?) seems to cause crashes for pytential.
     # https://gitlab.tiker.net/inducer/pytential/-/issues/146
     if [[ $CISUPPORT_PYTEST_NO_DOCTEST_MODULES == "" ]]; then
       DOCTEST_MODULES_FLAG="--doctest-modules"
     else
       DOCTEST_MODULES_FLAG=""
     fi

      with_echo "${PY_EXE}" -m pytest \
        --durations=10 \
        --tb=native  \
        --junitxml=pytest.xml \
        $DOCTEST_MODULES_FLAG \
        -rxsw \
        $PYTEST_FLAGS $PYTEST_PARALLEL_FLAGS $TESTABLES
    fi
  fi
}

# }}}


# {{{ run examples

run_examples()
{
  if ! test -d examples; then
    echo "!!! No 'examples' directory found"
  else
    cd examples
    for i in $(find . -name '*.py' -exec grep -q __main__ '{}' \; -print ); do
      echo "-----------------------------------------------------------------------"
      echo "RUNNING $i"
      echo "-----------------------------------------------------------------------"
      dn=$(dirname "$i")
      bn=$(basename "$i")
      (cd "$dn"; time ${PY_EXE} "$bn")
    done
  fi
}

# }}}


# {{{ docs

build_docs()
{
  # >=3.2.1 for https://github.com/sphinx-doc/sphinx/issues/8084
  with_echo $PY_EXE -m pip install "sphinx>=3.2.1"

  cd doc

  if test "$1" = "--no-check"; then
    with_echo make html
  else
    with_echo make html SPHINXOPTS="-W --keep-going -n"
  fi
}

maybe_upload_docs()
{
  if test -n "${DOC_UPLOAD_KEY}" && test "$CI_COMMIT_REF_NAME" = "master"; then
    cat > doc_upload_ssh_config <<END
Host doc-upload
   User doc
   IdentityFile doc_upload_key
   IdentitiesOnly yes
   Hostname marten.tiker.net
   StrictHostKeyChecking false
END

    echo "${DOC_UPLOAD_KEY}" > doc_upload_key
    chmod 0600 doc_upload_key
    RSYNC_RSH="ssh -F doc_upload_ssh_config" ./upload-docs.sh || { rm doc_upload_key; exit 1; }
    rm doc_upload_key
  else
    echo "Not uploading docs. No DOC_UPLOAD_KEY or not on master on Gitlab."
  fi
}

# }}}


# {{{ flake8

install_and_run_flake8()
{
  FLAKE8_PACKAGES=(flake8 pep8-naming)
  if grep -q quotes setup.cfg; then
    true
    FLAKE8_PACKAGES+=(flake8-quotes)
  else
    echo "-----------------------------------------------------------------"
    echo "Consider enabling quote checking for this package by configuring"
    echo "https://github.com/zheller/flake8-quotes"
    echo "in setup.cfg. Follow this example:"
    echo "https://github.com/illinois-ceesd/mirgecom/blob/45457596cac2eeb4a0e38bf6845fe4b7c323f6f5/setup.cfg#L5-L7"
    echo "-----------------------------------------------------------------"
  fi

  ${PY_EXE} -m pip install "${FLAKE8_PACKAGES[@]}"
  ${PY_EXE} -m flake8 "$@"
}

# }}}

# vim: foldmethod=marker:sw=2
