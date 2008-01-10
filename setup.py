#!/usr/bin/env python
# -*- coding: latin1 -*-

from distutils.core import setup,Extension
import glob
import os
import os.path

setup(name="pytools",
      version="0.10",
      description="A collection of tools for Python",
      author=u"Andreas Kloeckner",
      scripts=["bin/logtool"],
      author_email="inform@tiker.net",
      license = "BSD, like Python",
      url="http://news.tiker.net/software/pytools",
      packages=["pytools"],
      package_dir={"pytools": "src"}
     )
