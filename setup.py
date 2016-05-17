#! /usr/bin/env python
# -*- coding: latin1 -*-

from setuptools import setup

setup(name="pytools",
      version="2016.2.1",
      description="A collection of tools for Python",
      long_description="""
      Pytools is a big bag of things that are "missing" from the Python standard
      library. This is mainly a dependency of my other software packages, and is
      probably of little interest to you unless you use those. If you're curious
      nonetheless, here's what's on offer:

      * A ton of small tool functions such as `len_iterable`, `argmin`,
        tuple generation, permutation generation, ASCII table pretty printing,
        GvR's mokeypatch_xxx() hack, the elusive `flatten`, and much more.
      * Michele Simionato's decorator module
      * A time-series logging module, `pytools.log`.
      * Batch job submission, `pytools.batchjob`.
      * A lexer, `pytools.lex`.
      """,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Other Audience',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Software Development :: Libraries',
          'Topic :: Utilities',
          ],

      install_requires=[
          "decorator>=3.2.0",
          "appdirs>=1.4.0",
          "six>=1.8.0",
          "numpy>=1.6.0",
          ],

      author="Andreas Kloeckner",
      url="http://pypi.python.org/pypi/pytools",
      author_email="inform@tiker.net",
      license="MIT",
      packages=["pytools"])
