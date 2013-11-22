from __future__ import division, with_statement

__copyright__ = "Copyright (C) 2009-2013 Andreas Kloeckner"

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


# loosely based on
# http://effbot.org/zone/python-code-generator.htm

class Indentation(object):
    def __init__(self, generator):
        self.generator = generator

    def __enter__(self):
        self.generator.indent()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.generator.dedent()


class PythonCodeGenerator(object):
    def __init__(self):
        self.preamble = []
        self.code = []
        self.level = 0

    def extend(self, sub_generator):
        self.code.extend(sub_generator.code)

    def extend_indent(self, sub_generator):
        with Indentation(self):
            for line in sub_generator.code:
                self.write(line)

    def get(self):
        result = "\n".join(self.code)
        if self.preamble:
            result = "\n".join(self.preamble) + "\n" + result
        return result

    def add_to_preamble(self, s):
        self.preamble.append(s)

    def __call__(self, s):
        if not s.strip():
            self.code.append("")
        else:
            self.code.append(" "*(4*self.level) + s)

    def indent(self):
        self.level += 1

    def dedent(self):
        if self.level == 0:
            raise RuntimeError("internal error in python code generator")
        self.level -= 1


class PythonFunctionGenerator(PythonCodeGenerator):
    def __init__(self, name, args):
        PythonCodeGenerator.__init__(self)
        self.name = name

        self("def %s(%s):" % (name, ", ".join(args)))
        self.indent()

    def get_function(self):
        result_dict = {}
        source_text = self.get()
        exec(compile(source_text, "<generated function %s>" % self.name, "exec"),
                result_dict)
        func = result_dict[self.name]
        result_dict["_MODULE_SOURCE_CODE"] = source_text
        return func
