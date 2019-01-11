#!/usr/bin/env python
# Copyright (C) 2018 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
import os
import subprocess
import sys

import pytest

PYTHON_EXECUTABLE = sys.executable
OPT_EXECUTABLE = 'opt-6.0'
LLI_EXECUTABLE = 'lli-6.0'
BOOTSTRAP_SCRIPT = 'bootstrap.py'


def find_scripts(path):
    for path, _, filenames in os.walk(path):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)
            if ext == '.orx':
                yield os.path.join(path, basename)


TEST_FIXTURES = sorted(s for s in find_scripts('./tests'))


def execute(command, *, input=None, is_binary=False):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    stdout, stderr = process.communicate(input)
    if not is_binary:
        stdout, stderr = stdout.decode('utf-8').rstrip(), stderr.decode('utf-8').rstrip()
    if process.returncode:
        error = stderr if isinstance(stderr, str) else stderr.decode('utf-8')
        print(error, file=sys.stderr)
    return process.returncode, stdout, stderr


def get_build_options():
    # code, stdout, stderr = execute(['icu-config', '--ldflags', '--ldflags-icuio'])
    # if code:
    #     raise RuntimeError("Cannot select flags for ICU")

    library_path = None
    items = [
        # "-load", os.path.join(os.getcwd(), './dist/lib/libbootstrap-runtime.so')
    ]
    # for item in stdout.split(' '):
    #     item = item.strip()
    #     if not item:
    #         continue
    #
    #     if item.startswith('-L'):
    #         library_path = item[2:]
    #     elif item.startswith('-l'):
    #         name = item[2:]
    #         if library_path:
    #             filename = "{}.so".format(os.path.join(library_path, 'lib' + name))
    #         else:
    #             filename = "{}.so".format(os.path.join('lib' + name))
    #         items.append("-load")
    #         items.append(filename)
    return items


def compile_and_execute(filename, *, name, opt_level, arguments, input=None):
    # orcinus - generate LLVM IR
    code, assembly, stderr = execute([PYTHON_EXECUTABLE, BOOTSTRAP_SCRIPT, filename], is_binary=True)
    if code:
        return False, code, assembly, stderr

    # lli-6.0 - compile LLVM IR and execute
    flags = [
        f'-fake-argv0={name}'
    ]
    flags.extend(get_build_options())
    flags.extend(['-'])
    flags.extend(arguments)
    return (True,) + execute([LLI_EXECUTABLE, f'-O{opt_level}'] + flags, input=assembly)


def read_or_none(filename, *, default=None, type=None):
    if not filename:
        return default

    try:
        with open(filename, 'r+', encoding='utf-8') as stream:
            content = stream.read()
            content = content.rstrip()
            if type:
                content = type(content)
            return content
    except IOError:
        return default


def read_or_list(filename, *, type=None):
    value = read_or_none(filename, default='')
    if type:
        return [type(arg.strip()) for arg in value.split('\n')] if value else []
    return [arg.strip() for arg in value.split(' ')] if value else []


def source_name(fixture_value):
    root_path = os.path.dirname(__file__)
    fullname = os.path.join(root_path, fixture_value)
    basename = os.path.dirname(root_path)
    return os.path.relpath(fullname, basename)


@pytest.fixture(params=TEST_FIXTURES, ids=source_name)
def source_cases(request):
    fixture, _ = os.path.splitext(request.param)
    root_path = os.path.dirname(__file__)

    return (
        fixture,
        os.path.abspath(os.path.join(root_path, "{}.orx".format(fixture))),
        read_or_list(os.path.join(root_path, "{}.args".format(fixture)), type=str),
        read_or_none(os.path.join(root_path, "{}.in".format(fixture)), type=str),
        read_or_none(os.path.join(root_path, "{}.out".format(fixture)), type=str),
        read_or_none(os.path.join(root_path, "{}.err".format(fixture)), type=str),
        read_or_none(os.path.join(root_path, "{}.code".format(fixture)), default=0, type=int),
    )


def test_compile_and_execution(source_cases):
    name, filename, arguments, input, expected_output, expected_error, expected_code = source_cases

    # for opt_level in [0, 1, 2, 3]:  # Test on all optimization levels
    for opt_level in (0,):
        result = compile_and_execute(filename, name=name, opt_level=opt_level, arguments=arguments, input=input)
        result_full, result_code, result_output, result_error = result

        if expected_code is not None:
            assert result_code == expected_code
        if expected_error is not None:
            assert result_error == expected_error
        if expected_output is not None:
            assert result_output == expected_output
