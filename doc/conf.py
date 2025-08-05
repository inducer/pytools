from __future__ import annotations

from importlib import metadata
from urllib.request import urlopen


_conf_url = "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2009-21, Andreas Kloeckner"
author = "Andreas Kloeckner"
release = metadata.version("pytools")
version = ".".join(release.split(".")[:2])

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "loopy": ("https://documen.tician.de/loopy", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "platformdirs": ("https://platformdirs.readthedocs.io/en/latest", None),
    "pymbolic": ("https://documen.tician.de/pymbolic", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
    "python": ("https://docs.python.org/3", None),
    "setuptools": ("https://setuptools.pypa.io/en/latest", None),
}

nitpicky = True
autodoc_type_aliases = {
    "GraphT": "pytools.graph.GraphT",
    "NodeT": "pytools.graph.NodeT",
}

sphinxconfig_missing_reference_aliases = {
    # numpy typing
    "NDArray": "obj:numpy.typing.NDArray",
    "np.dtype": "class:numpy.dtype",
    "np.ndarray": "class:numpy.ndarray",
    "np.floating": "class:numpy.floating",
    # pytools typing
    "BoundingBox": "obj:pytools.spatial_btree.BoundingBox",
    "Element": "obj:pytools.spatial_btree.Element",
    "ObjectArray1D": "obj:pytools.obj_array.ObjectArray1D",
    "Point": "obj:pytools.spatial_btree.Point",
    "ReadableBuffer": "data:pytools.ReadableBuffer",
}


def setup(app):
    app.connect("missing-reference", process_autodoc_missing_reference)  # noqa: F821
