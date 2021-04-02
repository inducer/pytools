#! /usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

ver_dic = {}
version_file = open("pytools/version.py")
try:
    version_file_contents = version_file.read()
finally:
    version_file.close()

exec(compile(version_file_contents, "pytools/version.py", "exec"), ver_dic)

setup(name="pytools",
      version=ver_dic["VERSION_TEXT"],
      description="A collection of tools for Python",
      long_description=open("README.rst", "r").read(),
      classifiers=[
          "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          "Intended Audience :: Other Audience",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: MIT License",
          "Natural Language :: English",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Information Analysis",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Visualization",
          "Topic :: Software Development :: Libraries",
          "Topic :: Utilities",
          ],

      python_requires="~=3.6",

      install_requires=[
          "appdirs>=1.4.0",
          "numpy>=1.6.0",
          "dataclasses>=0.7;python_version<='3.6'"
          ],

      package_data={"pytools": ["py.typed"]},

      author="Andreas Kloeckner",
      url="http://pypi.python.org/pypi/pytools",
      author_email="inform@tiker.net",
      license="MIT",
      packages=["pytools"])
