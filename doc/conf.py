from urllib.request import urlopen


_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2009-21, Andreas Kloeckner"
author = "Andreas Kloeckner"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
ver_dic = {}
exec(compile(open("../pytools/version.py").read(), "../pytools/version.py", "exec"),
        ver_dic)
version = ".".join(str(x) for x in ver_dic["VERSION"])
release = ver_dic["VERSION_TEXT"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "loopy": ("https://documen.tician.de/loopy", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pymbolic": ("https://documen.tician.de/pymbolic", None),
    "pytest": ("https://docs.pytest.org/en/stable", None),
    "setuptools": ("https://setuptools.pypa.io/en/latest", None),
    "python": ("https://docs.python.org/3", None),
    "platformdirs": ("https://platformdirs.readthedocs.io/en/latest", None),
}

nitpicky = True
nitpick_ignore_regex = [
    ["py:class", r"typing_extensions\.(.+)"],
    ["py:class", r"ReadableBuffer"],
]

autodoc_type_aliases = {
    "GraphT": "pytools.graph.GraphT",
    "NodeT": "pytools.graph.NodeT",
}
