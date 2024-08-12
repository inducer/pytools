Pytools: Lots of Little Utilities
=================================

.. image:: https://gitlab.tiker.net/inducer/pytools/badges/main/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/pytools/commits/main
.. image:: https://github.com/inducer/pytools/workflows/CI/badge.svg?branch=main
    :alt: Github Build Status
    :target: https://github.com/inducer/pytools/actions?query=branch%3Amain+workflow%3ACI
.. image:: https://badge.fury.io/py/pytools.png
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/pytools/
.. image:: https://zenodo.org/badge/1575270.svg
    :alt: Zenodo DOI for latest release
    :target: https://zenodo.org/badge/latestdoi/1575270

Pytools is a big bag of things that are "missing" from the Python standard
library. This is mainly a dependency of my other software packages, and is
probably of little interest to you unless you use those. If you're curious
nonetheless, here's what's on offer:

* A ton of small tool functions such as `len_iterable`, `argmin`,
  tuple generation, permutation generation, ASCII table pretty printing,
  GvR's monkeypatch_xxx() hack, the elusive `flatten`, and much more.
* Batch job submission, `pytools.batchjob`.
* A lexer, `pytools.lex`.
* A persistent key-value store, `pytools.persistent_dict`.

Links:

* `Documentation <https://documen.tician.de/pytools>`_

* `Github <https://github.com/inducer/pytools>`_

* ``pytools.log`` has been spun out into a separate project,
  `logpyle <https://github.com/illinois-ceesd/logpyle>`__.
