# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
import io
import itertools
import sys

from orcinus.core.locations import Location

ANSI_COLOR_RED = "\033[31m" if sys.stderr.isatty() else ""
ANSI_COLOR_GREEN = "\x1b[32m" if sys.stderr.isatty() else ""
ANSI_COLOR_YELLOW = "\x1b[33m" if sys.stderr.isatty() else ""
ANSI_COLOR_BLUE = "\x1b[34m" if sys.stderr.isatty() else ""
ANSI_COLOR_MAGENTA = "\x1b[35m" if sys.stderr.isatty() else ""
ANSI_COLOR_CYAN = "\x1b[36m" if sys.stderr.isatty() else ""
ANSI_COLOR_RESET = "\x1b[0m" if sys.stderr.isatty() else ""


def load_source_content(location: Location, before=2, after=2):
    """ Load selected line and it's neighborhood lines """
    try:
        with open(location.filename, 'r', encoding='utf-8') as stream:
            at_before = max(0, location.begin.line - before)
            at_after = location.end.line + after

            idx = 0
            results = []
            for idx, line in itertools.islice(enumerate(stream), at_before, at_after):
                results.append((idx + 1, line.rstrip("\n")))
    except IOError:
        return []
    else:
        results.append([idx + 2, ""])
        return results


def show_source_lines(location: Location, before=2, after=2, columns=None):
    """
    Convert selected lines to error message, e.g.:

    ```
        1 : from module import system =
          : --------------------------^
    ```
    """
    stream = io.StringIO()
    columns = columns or 80

    strings = load_source_content(location, before, after)
    if not strings:
        return

    width = max(max(len(str(idx)) for idx, _ in strings), 5)
    for line, string in strings:
        s_line = str(line).rjust(width)

        stream.write(ANSI_COLOR_CYAN)
        stream.write(s_line)
        stream.write(" : ")
        stream.write(ANSI_COLOR_BLUE)
        for column, char in enumerate(string):
            column += 1
            is_error = False
            if location.begin.line == line:
                is_error = column >= location.begin.column
            if location.end.line == line:
                is_error = is_error and column <= location.end.column

            if is_error:
                stream.write(ANSI_COLOR_RED)
            else:
                stream.write(ANSI_COLOR_GREEN)
            stream.write(char)

        stream.write(ANSI_COLOR_RESET)
        stream.write("\n")

        # write error line
        if location.begin.line <= line <= location.end.line:
            stream.write("·" * width)
            stream.write(" : ")

            for column, char in itertools.chain(enumerate(string), ((len(string), None),)):
                column += 1

                is_error = False
                if location.begin.line == line:
                    is_error = column >= location.begin.column
                if location.end.line == line:
                    is_error = is_error and column <= location.end.column

                if is_error:
                    stream.write(ANSI_COLOR_RED)
                    stream.write("^")
                    stream.write(ANSI_COLOR_RESET)
                elif char is not None:
                    stream.write("·")
            stream.write("\n")

    return stream.getvalue()
