__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2014 Matt Wala
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

__doc__ = """
Dot helper functions
====================

.. autofunction:: dot_escape
.. autofunction:: show_dot
"""

import html
import logging
import os
from typing import Optional


logger = logging.getLogger(__name__)


# {{{ graphviz / dot interactive show

def dot_escape(s: str) -> str:
    """
    Escape the string *s* for compatibility with the
    `dot <http://graphviz.org/>`__ language, particularly
    backslashes and HTML tags.

    :arg s: The input string to escape.

    :returns: *s* with special characters escaped.
    """
    # "\" and HTML are significant in graphviz.
    return html.escape(s.replace("\\", "\\\\"))


def show_dot(dot_code: str, output_to: Optional[str] = None) -> Optional[str]:
    """
    Visualize the graph represented by *dot_code*.

    :arg dot_code: An instance of :class:`str` in the `dot <http://graphviz.org/>`__
        language to visualize.
    :arg output_to: An instance of :class:`str` that can be one of:

        - ``"xwindow"`` to visualize the graph as an
          `X window <https://en.wikipedia.org/wiki/X_Window_System>`_.
        - ``"browser"`` to visualize the graph as an SVG file in the
          system's default web-browser.
        - ``"svg"`` to store the dot code as an SVG file on the file system.
          Returns the path to the generated SVG file.

        Defaults to ``"xwindow"`` if X11 support is present, otherwise defaults
        to ``"browser"``.

    :returns: Depends on *output_to*. If ``"svg"``, returns the path to the
        generated SVG file, otherwise returns ``None``.
    """

    import subprocess
    from tempfile import mkdtemp
    temp_dir = mkdtemp(prefix="tmp_pytools_dot")

    dot_file_name = "code.dot"

    from os.path import join
    with open(join(temp_dir, dot_file_name), "w") as dotf:
        dotf.write(dot_code)

    # {{{ preprocess 'output_to'

    if output_to is None:
        with subprocess.Popen(["dot", "-T?"],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE
                              ) as proc:
            assert proc.stderr, ("Could not execute the 'dot' program. "
                                 "Please install the 'graphviz' package and "
                                 "make sure it is in your $PATH.")
            supported_formats = proc.stderr.read().decode()

            if " x11 " in supported_formats and "DISPLAY" in os.environ:
                output_to = "xwindow"
            else:
                output_to = "browser"

    # }}}

    if output_to == "xwindow":
        subprocess.check_call(["dot", "-Tx11", dot_file_name], cwd=temp_dir)
    elif output_to in ["browser", "svg"]:
        svg_file_name = "code.svg"
        subprocess.check_call(["dot", "-Tsvg", "-o", svg_file_name, dot_file_name],
                              cwd=temp_dir)

        full_svg_file_name = join(temp_dir, svg_file_name)
        logger.info("show_dot: svg written to '%s'", full_svg_file_name)

        if output_to == "svg":
            return full_svg_file_name
        else:
            assert output_to == "browser"

            from webbrowser import open as browser_open
            browser_open("file://" + full_svg_file_name)
    else:
        raise ValueError("`output_to` can be one of 'xwindow', 'browser', or 'svg',"
                         f" got '{output_to}'")

    return None
# }}}


# vim: foldmethod=marker
